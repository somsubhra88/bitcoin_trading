"""Live execution on bitFlyer spot (BTC/JPY).

One call of :func:`rebalance_live` moves the real account toward the
strategy's target exposure with at most one order. The exchange is the source
of truth: balances, the fee tier and prices are read from bitFlyer on every
call, so deposits, manual trades and restarts are absorbed automatically.

Order lifecycle (PROJECT.md §30, §31):

1. every safety gate is checked (switches, capital, pending orders, circuit
   breaker, maintenance window, board state, fresh prices, daily loss);
2. the intended order is committed as ``SUBMITTING`` *before* any network I/O,
   so a crash can never leave an order the database does not know about;
3. it is submitted once as a marketable **limit IOC** order priced
   ``LIVE_MAX_SLIPPAGE_BPS`` through the touch with a one-minute expiry — it
   fills immediately against the book or not at all, and never at a price the
   bot did not accept;
4. the outcome is polled; anything unresolved is ``SUBMITTED`` or ``UNKNOWN``
   and is reconciled against the exchange before another order may be sent.

No order is ever retried automatically. A timeout makes the outcome
``UNKNOWN``; reconciliation finds the order on the exchange or, after a grace
period, marks it ``FAILED``. Too many failures halt trading until a human looks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import Fill, Order, PortfolioSnapshot, Position, StrategySignal
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.enums import ExitReason, OrderStatus, OrderType, Severity, Side, TradingMode
from btc_trader.errors import (
    ExchangeAuthenticationError,
    ExchangeError,
    ExchangeRateLimitError,
    ExchangeResponseError,
)
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.models import ExchangeOrder, OrderRequest, OwnExecution
from btc_trader.monitoring.alerts import send_alert
from btc_trader.monitoring.logging import get_logger
from btc_trader.paper import annotate_execution, risk_day_start
from btc_trader.rebalance import RebalanceRequest, normalise_order_size, plan_rebalance
from btc_trader.utils.timeutils import age_seconds, ensure_utc, utcnow

logger = get_logger(__name__)

COMPONENT = "live_execution"
#: bitFlyer's daily maintenance is 04:00-04:10 JST; keep a margin either side.
MAINTENANCE_START = time(3, 55)
MAINTENANCE_END = time(4, 15)
MAINTENANCE_ZONE = ZoneInfo("Asia/Tokyo")
#: bitFlyer never lists an order that expired without executing. Orders are
#: sent with a one-minute expiry, so an accepted order still unlisted after
#: this long is settled from its fills (normally: none).
ACCEPTED_ORDER_GRACE = timedelta(minutes=3)
#: An order whose acceptance id never arrived is searched for this long
#: before it is declared not executed.
UNKNOWN_ORDER_GRACE = timedelta(minutes=10)
POLL_INTERVAL_SECONDS = 1.0
_PRICE_TICK = Decimal("1")
_BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class LiveAccount:
    jpy: Decimal
    jpy_available: Decimal
    btc: Decimal
    btc_available: Decimal
    bid: Decimal
    ask: Decimal
    fee_rate: Decimal
    quoted_at: datetime

    def bot_cash(self, settings: Settings) -> Decimal:
        """JPY the bot may spend: available cash, within ``LIVE_CAPITAL_JPY``."""
        room = settings.live_capital_jpy - self.btc * self.bid
        return max(Decimal("0"), min(self.jpy_available, room))


def in_maintenance_window(now: datetime | None = None) -> bool:
    local = (now or utcnow()).astimezone(MAINTENANCE_ZONE).time()
    return MAINTENANCE_START <= local < MAINTENANCE_END


async def load_account(client: ExchangeClient, settings: Settings) -> LiveAccount:
    ticker = await client.get_ticker(settings.product_code)
    balances = {balance.currency_code: balance for balance in await client.get_balances()}
    fee_rate = await client.get_trading_commission(settings.product_code)
    if ticker.best_bid is None or ticker.best_ask is None:
        raise ExchangeResponseError("ticker has no bid/ask")
    base, quote = settings.product_code.split("_", 1)
    zero = Decimal("0")
    return LiveAccount(
        jpy=balances[quote].amount if quote in balances else zero,
        jpy_available=balances[quote].available if quote in balances else zero,
        btc=balances[base].amount if base in balances else zero,
        btc_available=balances[base].available if base in balances else zero,
        bid=ticker.best_bid,
        ask=ticker.best_ask,
        fee_rate=fee_rate,
        quoted_at=ticker.exchange_timestamp,
    )


async def rebalance_live(
    session: AsyncSession,
    settings: Settings,
    signal: StrategySignal,
    target_exposure: Decimal,
    *,
    client: ExchangeClient,
    force: bool = False,
    reduction_reason: ExitReason = ExitReason.RISK_EXIT,
) -> Order | None:
    """Move the real account toward ``target_exposure``; return the order sent."""
    blocked = await _gate(session, settings, client)
    if blocked is not None:
        annotate_execution(signal, *blocked)
        return None

    account = await load_account(client, settings)
    if age_seconds(account.quoted_at) > settings.max_market_data_age_seconds:
        annotate_execution(signal, "blocked_stale_market_data", "exchange ticker is stale")
        return None
    position = await _position(session, settings)
    await record_live_snapshot(session, settings, account, position)

    priced = settings.model_copy(update={"taker_fee_percent": account.fee_rate * 100})
    plan = plan_rebalance(
        RebalanceRequest(
            cash=account.bot_cash(settings),
            btc=account.btc_available,
            bid=account.bid,
            ask=account.ask,
            target_exposure=target_exposure,
            force=force,
            reduction_reason=reduction_reason,
            average_entry_price=position.average_entry_price if position else None,
            daily_loss_reached=await _daily_loss_reached(session, settings, account),
        ),
        priced,
    )
    if not plan.trades or plan.side is None:
        annotate_execution(signal, "no_trade", plan.reason, plan)
        return None

    slippage = settings.live_max_slippage_bps / _BPS
    if plan.side is Side.BUY:
        price = (account.ask * (1 + slippage)).quantize(_PRICE_TICK, ROUND_CEILING)
        # Worst case the whole order fills at the limit; bitFlyer takes the
        # commission in BTC, so JPY needed is exactly price x size.
        size = normalise_order_size(min(plan.size, account.jpy_available / price), settings, price)
    else:
        price = (account.bid * (1 - slippage)).quantize(_PRICE_TICK, ROUND_FLOOR)
        # Leave room for a commission charged in BTC on top of the sold size.
        sellable = account.btc_available / (1 + account.fee_rate)
        size = normalise_order_size(min(plan.size, sellable), settings, price)
    if size is None:
        annotate_execution(signal, "no_trade", "order rounds below the exchange minimum", plan)
        return None

    order = await submit_order(
        session,
        settings,
        client,
        OrderRequest(product_code=settings.product_code, side=plan.side, size=size, price=price),
        signal=signal,
        exit_reason=plan.exit_reason,
        context={
            "reason": plan.reason,
            "equity_jpy": str(plan.equity),
            "exposure_before": str(plan.current_exposure),
            "target_exposure": str(plan.target_exposure),
            "bid": str(account.bid),
            "ask": str(account.ask),
            "fee_rate": str(account.fee_rate),
            "limit_price": str(price),
        },
    )
    annotate_execution(signal, _execution_status(order), plan.reason, plan)
    return order


async def submit_order(
    session: AsyncSession,
    settings: Settings,
    client: ExchangeClient,
    request: OrderRequest,
    *,
    signal: StrategySignal | None = None,
    exit_reason: ExitReason | None = None,
    context: dict[str, str] | None = None,
) -> Order:
    """Persist, submit once, and track one order to a final state if possible."""
    order = Order(
        client_order_id=f"live-{uuid4().hex}",
        exchange=client.name,
        product_code=request.product_code,
        side=request.side,
        order_type=OrderType.LIMIT,
        requested_price=request.price,
        requested_size=request.size,
        filled_size=Decimal("0"),
        status=OrderStatus.SUBMITTING,
        strategy_name=signal.strategy_name if signal else None,
        signal_id=signal.id if signal else None,
        exit_reason=exit_reason,
        trading_mode=TradingMode.LIVE,
        decision_context={"time_in_force": request.time_in_force, **(context or {})},
        fills=[],
    )
    session.add(order)
    # Durable before the request leaves: a crash now is found by reconciliation.
    await session.commit()

    try:
        acceptance_id = await client.place_order(request)
    except (ExchangeResponseError, ExchangeAuthenticationError, ExchangeRateLimitError) as exc:
        # The exchange answered and did not accept the order.
        order.status = OrderStatus.REJECTED
        order.rejection_reason = str(exc)[:2000]
        order.completed_at = utcnow()
        await _event(session, Severity.ERROR, "live_order_rejected", order, str(exc))
        await session.commit()
        await send_alert(settings, Severity.ERROR, "Order rejected", _describe(order, str(exc)))
        return order
    except ExchangeError as exc:
        # Timeout or connection loss: the order may or may not exist.
        order.status = OrderStatus.UNKNOWN
        order.rejection_reason = str(exc)[:2000]
        await _event(session, Severity.WARNING, "live_order_unknown", order, str(exc))
        await session.commit()
        await send_alert(
            settings, Severity.WARNING, "Order outcome unknown — will reconcile", _describe(order)
        )
        return order

    order.exchange_order_id = acceptance_id
    order.status = OrderStatus.SUBMITTED
    order.submitted_at = utcnow()
    await session.commit()
    logger.info("live_order_submitted", client_order_id=order.client_order_id, side=request.side)

    deadline = asyncio.get_running_loop().time() + settings.live_order_poll_seconds
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        reported = await client.get_order(request.product_code, acceptance_id)
        if reported is not None and reported.is_final:
            await _finalize(session, settings, client, order, reported.state)
            break
    return order


async def reconcile_open_orders(
    session: AsyncSession, settings: Settings, client: ExchangeClient
) -> int:
    """Resolve every non-final live order against the exchange. Returns how many."""
    open_orders = list(
        (
            await session.execute(
                select(Order).where(
                    Order.trading_mode == TradingMode.LIVE,
                    Order.product_code == settings.product_code,
                    Order.status.in_([state for state in OrderStatus if state.is_open]),
                )
            )
        ).scalars()
    )
    resolved = 0
    for order in open_orders:
        age = utcnow() - ensure_utc(order.submitted_at or order.created_at)
        if order.exchange_order_id is None:
            match = await _find_on_exchange(session, settings, client, order)
            if match is not None:
                order.exchange_order_id = match.acceptance_id
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = match.ordered_at
            elif age > UNKNOWN_ORDER_GRACE:
                order.status = OrderStatus.FAILED
                order.completed_at = utcnow()
                order.rejection_reason = (
                    f"{order.rejection_reason or 'submission interrupted'}; not found on the "
                    "exchange after reconciliation, so it was not executed"
                )
                await _event(session, Severity.WARNING, "live_order_failed", order, "not found")
                resolved += 1
                continue
            else:
                continue
        reported = await client.get_order(order.product_code, order.exchange_order_id)
        if reported is not None and reported.is_final:
            await _finalize(session, settings, client, order, reported.state)
            resolved += 1
        elif reported is None and age > ACCEPTED_ORDER_GRACE:
            await _finalize(session, settings, client, order, "EXPIRED")
            resolved += 1
    await session.commit()
    return resolved


async def record_live_snapshot(
    session: AsyncSession,
    settings: Settings,
    account: LiveAccount,
    position: Position | None = None,
) -> PortfolioSnapshot:
    """Mark the real account to the bid and align the position with its balance."""
    position = position or await _position(session, settings)
    if position is not None:
        if account.btc <= 0:
            position.cost_basis_jpy = Decimal("0")
            position.average_entry_price = None
        elif position.average_entry_price:
            position.cost_basis_jpy = position.average_entry_price * account.btc
        position.btc_quantity = account.btc
        position.unrealized_pnl_jpy = (
            (account.bid - position.average_entry_price) * account.btc
            if position.average_entry_price
            else Decimal("0")
        )
    value = account.btc * account.bid
    row = PortfolioSnapshot(
        timestamp=utcnow(),
        trading_mode=TradingMode.LIVE,
        jpy_balance=account.jpy,
        btc_balance=account.btc,
        btc_market_price=account.bid,
        btc_value_jpy=value,
        total_value_jpy=account.jpy + value,
        realized_pnl_jpy=position.realized_pnl_jpy if position else Decimal("0"),
        unrealized_pnl_jpy=position.unrealized_pnl_jpy if position else Decimal("0"),
    )
    session.add(row)
    await session.flush()
    return row


async def snapshot_live(session: AsyncSession, settings: Settings, client: ExchangeClient) -> None:
    if settings.has_private_credentials:
        await record_live_snapshot(session, settings, await load_account(client, settings))


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
async def _gate(
    session: AsyncSession, settings: Settings, client: ExchangeClient
) -> tuple[str, str] | None:
    if not settings.live_orders_allowed:
        return (
            "blocked_live_gates",
            "live orders need TRADING_MODE=live, ENABLE_AUTOMATIC_TRADING, "
            "LIVE_TRADING_ACKNOWLEDGED, credentials, and the kill switch off",
        )
    if settings.live_capital_jpy <= 0:
        return "blocked_no_capital", "set LIVE_CAPITAL_JPY to the JPY the bot may manage"
    if in_maintenance_window():
        return "blocked_maintenance", "bitFlyer daily maintenance window (04:00-04:10 JST)"
    await reconcile_open_orders(session, settings, client)
    if await _open_order_count(session, settings):
        return "blocked_pending_order", "an earlier order is still being reconciled"
    failures = await _recent_failures(session, settings)
    if failures >= settings.live_max_failed_orders_per_day:
        await _halt(session, settings, failures)
        return (
            "blocked_circuit_breaker",
            f"{failures} failed or rejected orders in the last 24h; trading resumes on its "
            "own once fewer than LIVE_MAX_FAILED_ORDERS_PER_DAY remain in that window",
        )
    health = await client.get_health(settings.product_code)
    if not health.accepts_orders:
        return (
            "blocked_exchange_state",
            f"exchange health {health.state.value}, board {health.board_state}",
        )
    return None


async def _open_order_count(session: AsyncSession, settings: Settings) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(Order)
                .where(
                    Order.trading_mode == TradingMode.LIVE,
                    Order.product_code == settings.product_code,
                    Order.status.in_([state for state in OrderStatus if state.is_open]),
                )
            )
        ).scalar_one()
    )


async def _recent_failures(session: AsyncSession, settings: Settings) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(Order)
                .where(
                    Order.trading_mode == TradingMode.LIVE,
                    Order.product_code == settings.product_code,
                    Order.status.in_([OrderStatus.REJECTED, OrderStatus.FAILED]),
                    Order.created_at >= utcnow() - timedelta(hours=24),
                )
            )
        ).scalar_one()
    )


async def _halt(session: AsyncSession, settings: Settings, failures: int) -> None:
    """Record and announce the circuit breaker once per 24 hours."""
    events = await SystemEventRepository(session).recent(limit=50)
    cutoff = utcnow() - timedelta(hours=24)
    if any(
        event.event_type == "live_trading_halted" and ensure_utc(event.timestamp) >= cutoff
        for event in events
    ):
        return
    message = f"{failures} live orders failed or were rejected in 24h; new orders are paused"
    await SystemEventRepository(session).record(
        severity=Severity.CRITICAL,
        component=COMPONENT,
        event_type="live_trading_halted",
        message=message,
        metadata={"failures": failures},
    )
    await session.commit()
    await send_alert(settings, Severity.CRITICAL, "Live trading halted", message)


async def _daily_loss_reached(
    session: AsyncSession, settings: Settings, account: LiveAccount
) -> bool:
    """Equity fall since the start of the risk day (deposits/withdrawals count)."""
    if settings.max_daily_loss_jpy <= 0:
        return False
    baseline = (
        await session.execute(
            select(PortfolioSnapshot.total_value_jpy)
            .where(
                PortfolioSnapshot.trading_mode == TradingMode.LIVE,
                PortfolioSnapshot.timestamp < risk_day_start(settings),
            )
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if baseline is None:
        return False
    return baseline - (account.jpy + account.btc * account.bid) >= settings.max_daily_loss_jpy


# --------------------------------------------------------------------------- #
# settlement
# --------------------------------------------------------------------------- #
async def _finalize(
    session: AsyncSession,
    settings: Settings,
    client: ExchangeClient,
    order: Order,
    state: str,
) -> None:
    """Record fills, set the final status, and update the position."""
    assert order.exchange_order_id is not None  # noqa: S101 - only called once accepted
    executions = await client.get_executions(order.product_code, order.exchange_order_id)
    recorded = {fill.exchange_execution_id for fill in order.fills}
    new_fills = [item for item in executions if item.execution_id not in recorded]
    for item in new_fills:
        order.fills.append(
            Fill(
                exchange_execution_id=item.execution_id,
                price=item.price,
                size=item.size,
                # bitFlyer charges BTC; stored in JPY like every other money column.
                commission=item.commission * item.price,
                executed_at=item.executed_at,
            )
        )
    filled = sum((fill.size for fill in order.fills), Decimal("0"))
    order.filled_size = filled
    order.average_fill_price = (
        sum((fill.price * fill.size for fill in order.fills), Decimal("0")) / filled
        if filled > 0
        else None
    )
    order.completed_at = utcnow()
    if state == "REJECTED":
        order.status = OrderStatus.REJECTED
    elif state == "COMPLETED" or filled >= order.requested_size:
        order.status = OrderStatus.FILLED
    else:
        order.status = OrderStatus.CANCELLED
    order.raw_response = {"final_state": state, "executions": len(executions)}
    if new_fills:
        await _apply_to_position(session, settings, order, new_fills)
    await session.flush()
    if filled > 0:
        logger.info(
            "live_order_filled",
            client_order_id=order.client_order_id,
            side=order.side,
            size=str(filled),
            price=str(order.average_fill_price),
        )
        await send_alert(settings, Severity.INFO, f"{order.side.value} filled", _describe(order))
    elif state == "REJECTED":
        await _event(session, Severity.ERROR, "live_order_rejected", order, "rejected")
        await send_alert(settings, Severity.ERROR, "Order rejected", _describe(order))


async def _apply_to_position(
    session: AsyncSession, settings: Settings, order: Order, fills: list[OwnExecution]
) -> None:
    position = await _position(session, settings)
    if position is None:
        position = Position(
            product_code=settings.product_code,
            trading_mode=TradingMode.LIVE,
            btc_quantity=Decimal("0"),
            cost_basis_jpy=Decimal("0"),
            realized_pnl_jpy=Decimal("0"),
            unrealized_pnl_jpy=Decimal("0"),
        )
        session.add(position)
    for item in fills:
        if order.side is Side.BUY:
            received = item.size - item.commission
            position.cost_basis_jpy += item.price * item.size
            position.btc_quantity += received
            position.average_entry_price = (
                position.cost_basis_jpy / position.btc_quantity
                if position.btc_quantity > 0
                else None
            )
        else:
            average = position.average_entry_price or item.price
            position.realized_pnl_jpy += (item.price - average) * item.size - (
                item.commission * item.price
            )
            position.btc_quantity = max(Decimal("0"), position.btc_quantity - item.size)
            position.cost_basis_jpy = average * position.btc_quantity
            if position.btc_quantity == 0:
                position.average_entry_price = None


async def _find_on_exchange(
    session: AsyncSession, settings: Settings, client: ExchangeClient, order: Order
) -> ExchangeOrder | None:
    """Match an order whose acceptance id was never received (crash or timeout)."""
    known = set(
        (
            await session.execute(
                select(Order.exchange_order_id).where(
                    Order.trading_mode == TradingMode.LIVE, Order.exchange_order_id.is_not(None)
                )
            )
        ).scalars()
    )
    earliest = ensure_utc(order.created_at) - timedelta(minutes=1)
    for candidate in await client.get_recent_orders(order.product_code, count=20):
        if (
            candidate.acceptance_id not in known
            and candidate.side is order.side
            and candidate.size == order.requested_size
            and ensure_utc(candidate.ordered_at) >= earliest
        ):
            return candidate
    return None


async def _position(session: AsyncSession, settings: Settings) -> Position | None:
    return (
        await session.execute(
            select(Position).where(
                Position.product_code == settings.product_code,
                Position.trading_mode == TradingMode.LIVE,
            )
        )
    ).scalar_one_or_none()


async def _event(
    session: AsyncSession, severity: Severity, event_type: str, order: Order, detail: str
) -> None:
    await SystemEventRepository(session).record(
        severity=severity,
        component=COMPONENT,
        event_type=event_type,
        message=_describe(order, detail),
        metadata={"client_order_id": order.client_order_id, "status": order.status.value},
    )


def _describe(order: Order, detail: str | None = None) -> str:
    filled = (
        f", filled {order.filled_size} @ ¥{order.average_fill_price:,.0f}"
        if (order.filled_size and order.average_fill_price)
        else ""
    )
    text = (
        f"{order.side.value} {order.requested_size} BTC limit ¥{order.requested_price:,.0f} "
        f"[{order.status.value}{filled}]"
    )
    return f"{text}: {detail}" if detail else text


def _execution_status(order: Order) -> str:
    if order.status is OrderStatus.FILLED or order.filled_size > 0:
        return "filled"
    return {
        OrderStatus.REJECTED: "rejected",
        OrderStatus.UNKNOWN: "submitted_unknown",
        OrderStatus.SUBMITTED: "submitted",
        OrderStatus.CANCELLED: "not_filled",
    }.get(order.status, order.status.value.lower())


__all__ = [
    "LiveAccount",
    "in_maintenance_window",
    "load_account",
    "rebalance_live",
    "reconcile_open_orders",
    "record_live_snapshot",
    "snapshot_live",
    "submit_order",
]
