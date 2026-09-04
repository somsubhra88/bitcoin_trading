"""Fee-aware paper ledger for spot BTC/JPY.

This module is intentionally isolated from exchange order submission. It models
fills against the latest stored bid/ask and maintains one independent ledger per
product and trading mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from uuid import uuid4

from sqlalchemy import delete, func, select
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
from btc_trader.enums import ExitReason, OrderStatus, OrderType, Side, TradingMode, TradingSignal
from btc_trader.utils.timeutils import age_seconds, ensure_utc, utcnow


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
    orders_deleted = int(
        (
            await session.execute(
                select(func.count())
                .select_from(Order)
                .where(
                    Order.product_code == settings.product_code,
                    Order.trading_mode == TradingMode.PAPER,
                )
            )
        ).scalar_one()
    )
    fills_deleted = int(
        (
            await session.execute(
                select(func.count()).select_from(Fill).where(Fill.order_id.in_(paper_orders))
            )
        ).scalar_one()
    )
    positions_deleted = int(
        (
            await session.execute(
                select(func.count())
                .select_from(Position)
                .where(
                    Position.product_code == settings.product_code,
                    Position.trading_mode == TradingMode.PAPER,
                )
            )
        ).scalar_one()
    )
    snapshots_deleted = int(
        (
            await session.execute(
                select(func.count())
                .select_from(PortfolioSnapshot)
                .where(PortfolioSnapshot.trading_mode == TradingMode.PAPER)
            )
        ).scalar_one()
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


def normalise_order_size(size: Decimal, settings: Settings) -> Decimal | None:
    """Apply the configured lot step and exchange minimum to a paper entry."""
    if size <= 0:
        return None
    step = settings.order_size_step_btc
    if step > 0:
        size = (size / step).to_integral_value(rounding=ROUND_DOWN) * step
    if settings.enforce_exchange_minimums and size < settings.min_order_size_btc:
        return None
    return size if size > 0 else None


async def execute_signal(
    session: AsyncSession, settings: Settings, signal: StrategySignal
) -> Order | None:
    """Apply one signal to the paper ledger; return a simulated filled order."""
    if settings.trading_mode is not TradingMode.PAPER:
        return None
    tick = await _latest_tick(session, settings.product_code)
    if tick is None:
        return None
    price = (tick.best_ask if signal.signal is TradingSignal.BUY else tick.best_bid) or tick.ltp
    confidence = signal.confidence
    if price is None or confidence is None or confidence < settings.paper_min_confidence:
        await snapshot(session, settings, price)
        return None
    if age_seconds(tick.exchange_timestamp) > settings.max_market_data_age_seconds:
        await snapshot(session, settings, price)
        return None
    if not _signal_data_is_fresh(signal, settings):
        await snapshot(session, settings, price)
        return None

    position = await _position(session, settings)
    state = await _latest_snapshot(session, settings)
    cash = state.jpy_balance if state else settings.paper_initial_jpy
    quantity = position.btc_quantity if position else Decimal("0")
    sell_mark = (tick.best_bid or tick.ltp) or price
    hard_stop = False
    if position is not None and quantity > 0:
        position.high_watermark_price = max(position.high_watermark_price or sell_mark, sell_mark)
        hard_stop = bool(
            position.average_entry_price
            and settings.paper_stop_loss_percent > 0
            and sell_mark
            <= position.average_entry_price
            * (Decimal("1") - settings.paper_stop_loss_percent / Decimal("100"))
        )
    trailing_stop = bool(
        position is not None
        and quantity > 0
        and position.high_watermark_price
        and settings.paper_trailing_stop_percent > 0
        and sell_mark
        <= position.high_watermark_price
        * (Decimal("1") - settings.paper_trailing_stop_percent / Decimal("100"))
    )

    if signal.signal is TradingSignal.BUY and quantity == 0:
        volatility_adjustment = _volatility_adjustment(signal, settings)
        allocation = settings.strategy_risk_fraction * (
            Decimal("0.5") + confidence * Decimal("0.5")
        )
        budget = min(
            settings.max_order_jpy,
            settings.max_position_jpy,
            cash * allocation * volatility_adjustment,
        )
        if budget <= 0 or await _daily_loss_reached(session, settings, position):
            await snapshot(session, settings, price)
            return None
        fill_price = price * (Decimal("1") + settings.paper_slippage_bps / Decimal("10000"))
        fee_rate = settings.taker_fee_percent / Decimal("100")
        raw_size = budget / (fill_price * (Decimal("1") + fee_rate))
        size = normalise_order_size(raw_size, settings)
        if size is None:
            signal.reason = (
                f"{signal.reason or 'BUY signal'}; paper entry blocked: "
                f"calculated {raw_size} BTC is below the configured minimum "
                f"of {settings.min_order_size_btc} BTC"
            )
            signal.metadata_json = {
                **(signal.metadata_json or {}),
                "paper_execution": "blocked_exchange_minimum",
                "calculated_size_btc": str(raw_size),
                "minimum_size_btc": str(settings.min_order_size_btc),
            }
            await snapshot(session, settings, price)
            return None
        return await _fill(
            session,
            settings,
            signal,
            Side.BUY,
            fill_price,
            size,
            cash,
            position,
            decision_context={
                "volatility_adjustment": str(volatility_adjustment),
                "allocation_fraction": str(allocation),
            },
        )

    if position is not None and quantity > 0:
        target = settings.min_profit_percent / Decimal("100")
        fill_price = sell_mark * (Decimal("1") - settings.paper_slippage_bps / Decimal("10000"))
        fee = fill_price * quantity * settings.taker_fee_percent / Decimal("100")
        entry_cost = (
            position.average_entry_price * quantity
            if position.average_entry_price
            else Decimal("0")
        )
        net_return = (
            (fill_price * quantity - fee - entry_cost) / entry_cost if entry_cost else Decimal("0")
        )
        # Risk exits are independent of the model signal. A BUY/HOLD emitted
        # during a drawdown must not suppress the stop.
        profit_exit = signal.signal in (TradingSignal.SELL, TradingSignal.HOLD) and net_return >= target
        if hard_stop or trailing_stop or profit_exit:
            return await _fill(
                session,
                settings,
                signal,
                Side.SELL,
                fill_price,
                quantity,
                cash,
                position,
                exit_reason=(
                    ExitReason.RISK_EXIT if hard_stop or trailing_stop else ExitReason.STRATEGY_EXIT
                ),
            )

    await snapshot(session, settings, price)
    return None


async def _fill(
    session: AsyncSession,
    settings: Settings,
    signal: StrategySignal,
    side: Side,
    price: Decimal,
    size: Decimal,
    cash: Decimal,
    position: Position | None,
    *,
    exit_reason: ExitReason | None = None,
    decision_context: dict[str, str] | None = None,
) -> Order:
    fee = price * size * settings.taker_fee_percent / Decimal("100")
    gross = price * size
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
        exit_reason=exit_reason,
        trading_mode=TradingMode.PAPER,
        submitted_at=utcnow(),
        completed_at=utcnow(),
        decision_context={
            "fee": str(fee),
            "slippage_bps": str(settings.paper_slippage_bps),
            **(decision_context or {}),
        },
    )
    session.add(order)
    await session.flush()
    session.add(
        Fill(order_id=order.id, price=price, size=size, commission=fee, executed_at=utcnow())
    )
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
        position.high_watermark_price = price
        cash -= gross + fee
    else:
        cost = position.average_entry_price * size if position.average_entry_price else Decimal("0")
        position.realized_pnl_jpy += gross - fee - cost
        position.cost_basis_jpy -= cost
        position.btc_quantity -= size
        position.average_entry_price = (
            position.cost_basis_jpy / position.btc_quantity if position.btc_quantity > 0 else None
        )
        if position.btc_quantity <= 0:
            position.high_watermark_price = None
        cash += gross - fee
    await session.flush()
    await snapshot(session, settings, price, cash=cash, position=position)
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
            .order_by(PortfolioSnapshot.timestamp.desc())
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


async def _daily_loss_reached(
    session: AsyncSession, settings: Settings, position: Position | None
) -> bool:
    """Stop opening new paper positions once today's realized loss is exhausted."""
    if position is None or settings.max_daily_loss_jpy <= 0:
        return False
    day_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    baseline = (
        await session.execute(
            select(PortfolioSnapshot.realized_pnl_jpy)
            .where(
                PortfolioSnapshot.trading_mode == TradingMode.PAPER,
                PortfolioSnapshot.timestamp < day_start,
            )
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    realized_change = position.realized_pnl_jpy - (baseline or Decimal("0"))
    return max(Decimal("0"), -realized_change) >= settings.max_daily_loss_jpy


def _volatility_adjustment(signal: StrategySignal, settings: Settings) -> Decimal:
    """Reduce capital allocation when observed return volatility exceeds target."""
    metadata = signal.metadata_json or {}
    try:
        observed = Decimal(str(metadata.get("volatility", "0")))
    except (InvalidOperation, ValueError, TypeError):
        observed = Decimal("0")
    target = settings.strategy_target_volatility_percent / Decimal("100")
    if observed <= 0:
        return Decimal("1")
    return min(Decimal("1"), target / observed)


def _signal_data_is_fresh(signal: StrategySignal, settings: Settings) -> bool:
    """Reject model decisions based on an order-book snapshot that went stale."""
    metadata = signal.metadata_json or {}
    source_timestamp = metadata.get("source_timestamp")
    if source_timestamp is None:
        return True
    try:
        timestamp = ensure_utc(datetime.fromisoformat(str(source_timestamp)))
    except (TypeError, ValueError):
        return False
    return age_seconds(timestamp) <= settings.max_market_data_age_seconds
