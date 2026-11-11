"""Fee-aware paper ledger for spot BTC/JPY.

This module is intentionally isolated from exchange order submission. It moves
the simulated holding toward a strategy's target exposure using the latest
stored bid/ask, the shared :mod:`btc_trader.rebalance` rules, and one
independent ledger per product and trading mode.

Every decision — including "no trade" — is written back onto the signal's
``metadata_json["execution"]`` so a HOLD is explainable later (§36).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import (
    Fill,
    MarketTick,
    Order,
    PortfolioSnapshot,
    Position,
    StrategySignal,
)
from btc_trader.enums import ExitReason, OrderStatus, OrderType, Side, TradingMode
from btc_trader.rebalance import RebalancePlan, RebalanceRequest, plan_rebalance
from btc_trader.utils.timeutils import age_seconds, utcnow


@dataclass(frozen=True)
class PaperResetResult:
    """Counts and state created by a paper-ledger reset."""

    product_code: str
    orders_deleted: int
    fills_deleted: int
    positions_deleted: int
    snapshots_deleted: int


async def reset_paper_trading(session: AsyncSession, settings: Settings) -> PaperResetResult:
    """Clear simulated execution state and create a fresh zero-holding ledger.

    Market data and strategy signals deliberately remain available for analysis.
    Portfolio snapshots do not currently carry ``product_code``; this platform
    has one configured product, so all paper snapshots are reset together.
    """
    paper_orders = select(Order.id).where(
        Order.product_code == settings.product_code,
        Order.trading_mode == TradingMode.PAPER,
    )
    orders_deleted = await _count(
        session,
        select(func.count())
        .select_from(Order)
        .where(
            Order.product_code == settings.product_code, Order.trading_mode == TradingMode.PAPER
        ),
    )
    fills_deleted = await _count(
        session, select(func.count()).select_from(Fill).where(Fill.order_id.in_(paper_orders))
    )
    positions_deleted = await _count(
        session,
        select(func.count())
        .select_from(Position)
        .where(
            Position.product_code == settings.product_code,
            Position.trading_mode == TradingMode.PAPER,
        ),
    )
    snapshots_deleted = await _count(
        session,
        select(func.count())
        .select_from(PortfolioSnapshot)
        .where(PortfolioSnapshot.trading_mode == TradingMode.PAPER),
    )

    await session.execute(delete(Fill).where(Fill.order_id.in_(paper_orders)))
    await session.execute(
        delete(Order).where(
            Order.product_code == settings.product_code,
            Order.trading_mode == TradingMode.PAPER,
        )
    )
    await session.execute(
        delete(Position).where(
            Position.product_code == settings.product_code,
            Position.trading_mode == TradingMode.PAPER,
        )
    )
    await session.execute(
        delete(PortfolioSnapshot).where(PortfolioSnapshot.trading_mode == TradingMode.PAPER)
    )
    await session.flush()

    # Seed the fresh account immediately so the API/UI show zero BTC and the
    # configured starting cash without waiting for the next worker tick.
    await snapshot_latest(session, settings)
    return PaperResetResult(
        product_code=settings.product_code,
        orders_deleted=orders_deleted,
        fills_deleted=fills_deleted,
        positions_deleted=positions_deleted,
        snapshots_deleted=snapshots_deleted,
    )


async def rebalance_paper(
    session: AsyncSession,
    settings: Settings,
    signal: StrategySignal,
    target_exposure: Decimal,
    *,
    force: bool = False,
    reduction_reason: ExitReason = ExitReason.RISK_EXIT,
) -> Order | None:
    """Move the paper holding toward ``target_exposure``; return the filled order."""
    if settings.trading_mode is not TradingMode.PAPER:
        return None
    tick = await _latest_tick(session, settings.product_code)
    if tick is None:
        annotate_execution(signal, "blocked_no_market_data", "no stored ticker to price an order")
        return None
    bid = tick.best_bid or tick.ltp
    ask = tick.best_ask or tick.ltp
    if bid is None or ask is None:
        annotate_execution(signal, "blocked_no_price", "latest ticker has no bid/ask or last price")
        return None
    if settings.trading_kill_switch:
        annotate_execution(signal, "blocked_kill_switch", "TRADING_KILL_SWITCH forbids new orders")
        await snapshot(session, settings, bid)
        return None
    if age_seconds(tick.exchange_timestamp) > settings.max_market_data_age_seconds:
        annotate_execution(
            signal,
            "blocked_stale_market_data",
            f"latest ticker is older than {settings.max_market_data_age_seconds}s",
        )
        await snapshot(session, settings, bid)
        return None

    position = await _position(session, settings)
    state = await _latest_snapshot(session, settings)
    cash = state.jpy_balance if state else settings.paper_initial_jpy
    plan = plan_rebalance(
        RebalanceRequest(
            cash=cash,
            btc=position.btc_quantity if position else Decimal("0"),
            bid=bid,
            ask=ask,
            target_exposure=target_exposure,
            force=force,
            reduction_reason=reduction_reason,
            average_entry_price=position.average_entry_price if position else None,
            daily_loss_reached=await _daily_loss_reached(session, settings, position),
        ),
        settings,
    )
    if not plan.trades:
        annotate_execution(signal, "no_trade", plan.reason, plan)
        await snapshot(session, settings, bid)
        return None
    annotate_execution(signal, "filled", plan.reason, plan)
    assert plan.side is not None  # noqa: S101 - narrowed by plan.trades
    return await _fill(session, settings, signal, plan, cash, position, mark=bid)


def annotate_execution(
    signal: StrategySignal, status: str, detail: str, plan: RebalancePlan | None = None
) -> None:
    """Record what the ledger did with a signal under ``metadata_json["execution"]``."""
    execution: dict[str, str] = {"status": status, "detail": detail, "at": utcnow().isoformat()}
    if plan is not None:
        execution.update(
            {
                "equity_jpy": format(plan.equity, "f"),
                "current_exposure": format(plan.current_exposure, "f"),
                "target_exposure": format(plan.target_exposure, "f"),
            }
        )
    # Reassign rather than mutate: JSON columns only track whole-value changes.
    signal.metadata_json = {**(signal.metadata_json or {}), "execution": execution}


async def _fill(
    session: AsyncSession,
    settings: Settings,
    signal: StrategySignal,
    plan: RebalancePlan,
    cash: Decimal,
    position: Position | None,
    *,
    mark: Decimal,
) -> Order:
    side, price, size, fee = plan.side, plan.price, plan.size, plan.fee
    gross = price * size
    now = utcnow()
    order = Order(
        client_order_id=f"paper-{uuid4().hex}",
        exchange="paper",
        product_code=settings.product_code,
        side=side,
        order_type=OrderType.MARKET,
        requested_price=price,
        requested_size=size,
        filled_size=size,
        average_fill_price=price,
        status=OrderStatus.FILLED,
        strategy_name=signal.strategy_name,
        signal_id=signal.id,
        exit_reason=plan.exit_reason,
        trading_mode=TradingMode.PAPER,
        submitted_at=now,
        completed_at=now,
        decision_context={
            "fee": str(fee),
            "slippage_bps": str(settings.paper_slippage_bps),
            "equity_jpy": str(plan.equity),
            "exposure_before": str(plan.current_exposure),
            "target_exposure": str(plan.target_exposure),
            "reason": plan.reason,
        },
    )
    session.add(order)
    await session.flush()
    session.add(Fill(order_id=order.id, price=price, size=size, commission=fee, executed_at=now))
    if position is None:
        position = Position(
            product_code=settings.product_code,
            trading_mode=TradingMode.PAPER,
            btc_quantity=Decimal("0"),
            cost_basis_jpy=Decimal("0"),
            realized_pnl_jpy=Decimal("0"),
            unrealized_pnl_jpy=Decimal("0"),
        )
        session.add(position)
    if side is Side.BUY:
        position.cost_basis_jpy += gross + fee
        position.btc_quantity += size
        position.average_entry_price = position.cost_basis_jpy / position.btc_quantity
        cash -= gross + fee
    else:
        cost = position.average_entry_price * size if position.average_entry_price else Decimal("0")
        position.realized_pnl_jpy += gross - fee - cost
        position.cost_basis_jpy -= cost
        position.btc_quantity -= size
        if position.btc_quantity <= 0:
            position.btc_quantity = Decimal("0")
            position.cost_basis_jpy = Decimal("0")
            position.average_entry_price = None
        cash += gross - fee
    await session.flush()
    await snapshot(session, settings, mark, cash=cash, position=position)
    return order


async def snapshot(
    session: AsyncSession,
    settings: Settings,
    price: Decimal | None,
    *,
    cash: Decimal | None = None,
    position: Position | None = None,
) -> PortfolioSnapshot:
    position = position or await _position(session, settings)
    latest = await _latest_snapshot(session, settings)
    cash = (
        cash if cash is not None else (latest.jpy_balance if latest else settings.paper_initial_jpy)
    )
    btc = position.btc_quantity if position else Decimal("0")
    value = btc * price if price is not None else Decimal("0")
    unrealized = (
        (price - position.average_entry_price) * btc
        if position and position.average_entry_price and price
        else Decimal("0")
    )
    if position:
        position.unrealized_pnl_jpy = unrealized
    snapshot_row = PortfolioSnapshot(
        timestamp=utcnow(),
        trading_mode=TradingMode.PAPER,
        jpy_balance=cash,
        btc_balance=btc,
        btc_market_price=price,
        btc_value_jpy=value,
        total_value_jpy=cash + value,
        realized_pnl_jpy=position.realized_pnl_jpy if position else Decimal("0"),
        unrealized_pnl_jpy=unrealized,
    )
    session.add(snapshot_row)
    await session.flush()
    return snapshot_row


async def snapshot_latest(session: AsyncSession, settings: Settings) -> PortfolioSnapshot:
    """Mark the paper portfolio to the newest stored market price."""
    tick = await _latest_tick(session, settings.product_code)
    price = None if tick is None else (tick.best_bid or tick.ltp)
    return await snapshot(session, settings, price)


async def current_exposure(session: AsyncSession, settings: Settings) -> Decimal | None:
    """BTC share of the managed equity in the latest snapshot of the current mode.

    In live mode equity is capped at ``LIVE_CAPITAL_JPY``, as the bot sizes it.
    """
    latest = (
        await session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.trading_mode == settings.trading_mode)
            .order_by(PortfolioSnapshot.timestamp.desc(), PortfolioSnapshot.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None or latest.btc_market_price is None:
        return None
    value, cash = latest.btc_value_jpy, latest.jpy_balance
    if settings.trading_mode is TradingMode.LIVE:
        cash = max(Decimal("0"), min(cash, settings.live_capital_jpy - value))
    equity = cash + value
    return value / equity if equity > 0 else None


async def _count(session: AsyncSession, statement: Select[tuple[int]]) -> int:
    return int((await session.execute(statement)).scalar_one())


async def _position(session: AsyncSession, settings: Settings) -> Position | None:
    return (
        await session.execute(
            select(Position).where(
                Position.product_code == settings.product_code,
                Position.trading_mode == TradingMode.PAPER,
            )
        )
    ).scalar_one_or_none()


async def _latest_snapshot(session: AsyncSession, settings: Settings) -> PortfolioSnapshot | None:
    return (
        await session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.trading_mode == TradingMode.PAPER)
            .order_by(PortfolioSnapshot.timestamp.desc(), PortfolioSnapshot.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def _latest_tick(session: AsyncSession, product_code: str) -> MarketTick | None:
    return (
        await session.execute(
            select(MarketTick)
            .where(MarketTick.product_code == product_code)
            .order_by(MarketTick.exchange_timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def risk_day_start(settings: Settings, now: datetime | None = None) -> datetime:
    """Midnight of the current risk day in ``RISK_DAY_TIMEZONE``, as an aware datetime."""
    zone = ZoneInfo(settings.risk_day_timezone)
    local = (now or utcnow()).astimezone(zone)
    return datetime.combine(local.date(), time.min, tzinfo=zone)


async def _daily_loss_reached(
    session: AsyncSession, settings: Settings, position: Position | None
) -> bool:
    """Stop adding paper exposure once today's realised loss is exhausted."""
    if position is None or settings.max_daily_loss_jpy <= 0:
        return False
    baseline = (
        await session.execute(
            select(PortfolioSnapshot.realized_pnl_jpy)
            .where(
                PortfolioSnapshot.trading_mode == TradingMode.PAPER,
                PortfolioSnapshot.timestamp < risk_day_start(settings),
            )
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    realized_change = position.realized_pnl_jpy - (baseline or Decimal("0"))
    return max(Decimal("0"), -realized_change) >= settings.max_daily_loss_jpy
