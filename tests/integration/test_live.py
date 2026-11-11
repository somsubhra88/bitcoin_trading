"""Live execution against a simulated bitFlyer account (PROJECT.md §26-§32, §48).

:class:`FakeExchangeClient` behaves like bitFlyer where it matters: IOC orders
fill at the touch or not at all, commission is charged in BTC, an order that
never executed is not listed, and a submission can be lost *after* the exchange
accepted it. Every unsafe path must end without a duplicate or blind order.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader import live
from btc_trader.config import Settings
from btc_trader.db.models import (
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    StrategySignal,
    SystemEvent,
)
from btc_trader.enums import (
    ExchangeHealthState,
    ExitReason,
    OrderStatus,
    Side,
    TradingMode,
    TradingSignal,
)
from btc_trader.live import in_maintenance_window, rebalance_live, reconcile_open_orders
from btc_trader.market_data.daily import store_external_bars
from btc_trader.market_data.history_provider import ExternalDailyBar
from btc_trader.paper import risk_day_start
from btc_trader.strategy_service import evaluate_and_store
from btc_trader.utils.timeutils import utcnow
from tests.fixtures.daily import bars_from_closes, price_path
from tests.fixtures.fake_exchange import FakeExchangeClient, make_ticker

ASK = Decimal("12001000")
BID = Decimal("11999000")
WEBHOOK = "https://hooks.example.test/alert"


@pytest.fixture(autouse=True)
def _fast_and_outside_maintenance(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(live, "POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(live, "in_maintenance_window", lambda now=None: False)
    yield


def _live(settings: Settings, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "trading_mode": TradingMode.LIVE,
        "enable_automatic_trading": True,
        "live_trading_acknowledged": True,
        "bitflyer_api_key": SecretStr("key"),
        "bitflyer_api_secret": SecretStr("secret"),
        "live_capital_jpy": Decimal("500000"),
        "max_position_jpy": Decimal("0"),
        "max_order_jpy": Decimal("0"),
        "live_order_poll_seconds": 1,
    }
    values.update(overrides)
    return settings.model_copy(update=values)


async def _signal(session: AsyncSession) -> StrategySignal:
    row = StrategySignal(
        strategy_name="trend_ensemble",
        strategy_version="2.0.0",
        product_code="BTC_JPY",
        timestamp=utcnow(),
        signal=TradingSignal.BUY,
        confidence=Decimal("1"),
        reference_price=ASK,
    )
    session.add(row)
    await session.flush()
    return row


async def _count(session: AsyncSession, model: type, *where: object) -> int:
    statement = select(func.count()).select_from(model)
    for clause in where:
        statement = statement.where(clause)  # type: ignore[arg-type]
    return int((await session.execute(statement)).scalar_one())


def _status(signal: StrategySignal) -> str:
    assert signal.metadata_json is not None
    return str(signal.metadata_json["execution"]["status"])


# --------------------------------------------------------------------------- #
# the happy paths
# --------------------------------------------------------------------------- #
async def test_a_buy_is_a_capped_limit_ioc_and_reconciles_to_the_balance(
    session: AsyncSession, settings: Settings
) -> None:
    exchange = FakeExchangeClient(jpy="1000000")
    signal = await _signal(session)

    order = await rebalance_live(
        session, _live(settings), signal, Decimal("0.5"), client=exchange, force=True
    )

    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.trading_mode is TradingMode.LIVE
    sent = exchange.orders[0].request
    assert sent.time_in_force == "IOC"
    assert sent.minute_to_expire == 1
    assert sent.price == (ASK * Decimal("1.003")).quantize(Decimal("1"), ROUND_CEILING)
    # LIVE_CAPITAL_JPY caps equity at ¥500k although ¥1M is available.
    assert ASK * sent.size == pytest.approx(Decimal("250000"), rel=Decimal("0.002"))
    assert order.filled_size == sent.size
    assert order.average_fill_price == ASK
    fill = (await session.execute(select(Fill))).scalar_one()
    assert fill.commission == (sent.size * Decimal("0.0015")).quantize(Decimal("1e-8")) * ASK
    position = (await session.execute(select(Position))).scalar_one()
    assert position.trading_mode is TradingMode.LIVE
    assert position.btc_quantity == exchange.btc  # commission came out of the BTC
    assert await _count(session, PortfolioSnapshot) >= 1
    assert _status(signal) == "filled"


async def test_a_stop_out_sells_what_the_btc_commission_allows(
    session: AsyncSession, settings: Settings
) -> None:
    exchange = FakeExchangeClient(jpy="0", btc="0.02")
    signal = await _signal(session)

    order = await rebalance_live(
        session,
        _live(settings),
        signal,
        Decimal("0"),
        client=exchange,
        reduction_reason=ExitReason.TRAILING_STOP,
    )

    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.side is Side.SELL
    assert order.exit_reason is ExitReason.TRAILING_STOP
    sent = exchange.orders[0].request
    assert sent.price == (BID * Decimal("0.997")).quantize(Decimal("1"), ROUND_FLOOR)
    # Selling the full 0.02 would fail when commission is charged on top in BTC.
    assert sent.size == Decimal("0.01997004")
    assert exchange.btc >= 0
    assert exchange.jpy == sent.size * BID


async def test_a_holding_within_the_band_is_left_alone(
    session: AsyncSession, settings: Settings
) -> None:
    exchange = FakeExchangeClient(jpy="250000", btc="0.0208")
    signal = await _signal(session)

    assert (
        await rebalance_live(session, _live(settings), signal, Decimal("0.5"), client=exchange)
        is None
    )
    assert "place_order" not in exchange.calls
    assert _status(signal) == "no_trade"


# --------------------------------------------------------------------------- #
# gates: every one must stop the order before it is sent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({"trading_mode": TradingMode.PAPER}, "blocked_live_gates"),
        ({"enable_automatic_trading": False}, "blocked_live_gates"),
        ({"live_trading_acknowledged": False}, "blocked_live_gates"),
        ({"trading_kill_switch": True}, "blocked_live_gates"),
        ({"bitflyer_api_secret": SecretStr("")}, "blocked_live_gates"),
        ({"live_capital_jpy": Decimal("0")}, "blocked_no_capital"),
    ],
)
async def test_configuration_gates_block_before_any_exchange_call(
    session: AsyncSession, settings: Settings, overrides: dict[str, object], status: str
) -> None:
    exchange = FakeExchangeClient(jpy="1000000")
    signal = await _signal(session)

    result = await rebalance_live(
        session, _live(settings, **overrides), signal, Decimal("0.5"), client=exchange, force=True
    )

    assert result is None
    assert exchange.calls == []
    assert _status(signal) == status


async def test_the_maintenance_window_blocks_orders(
    session: AsyncSession, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live, "in_maintenance_window", lambda now=None: True)
    exchange = FakeExchangeClient(jpy="1000000")
    signal = await _signal(session)

    await rebalance_live(session, _live(settings), signal, Decimal("0.5"), client=exchange)

    assert "place_order" not in exchange.calls
    assert _status(signal) == "blocked_maintenance"


def test_the_maintenance_window_is_04_00_to_04_10_jst_with_a_margin() -> None:
    from datetime import UTC, datetime

    assert in_maintenance_window(datetime(2026, 9, 1, 19, 0, tzinfo=UTC))  # 04:00 JST
    assert in_maintenance_window(datetime(2026, 9, 1, 18, 56, tzinfo=UTC))
    assert not in_maintenance_window(datetime(2026, 9, 1, 19, 16, tzinfo=UTC))
    assert not in_maintenance_window(datetime(2026, 9, 1, 0, 5, tzinfo=UTC))


@pytest.mark.parametrize(
    ("health", "board"),
    [
        (ExchangeHealthState.NO_ORDER, "RUNNING"),
        (ExchangeHealthState.SUPER_BUSY, "RUNNING"),
        (ExchangeHealthState.NORMAL, "CIRCUIT BREAK"),
    ],
)
async def test_an_unhealthy_exchange_blocks_orders(
    session: AsyncSession, settings: Settings, health: ExchangeHealthState, board: str
) -> None:
    exchange = FakeExchangeClient(jpy="1000000", health_state=health, board_state=board)
    signal = await _signal(session)

    await rebalance_live(session, _live(settings), signal, Decimal("0.5"), client=exchange)

    assert "place_order" not in exchange.calls
    assert _status(signal) == "blocked_exchange_state"


async def test_a_stale_exchange_quote_blocks_orders(
    session: AsyncSession, settings: Settings
) -> None:
    exchange = FakeExchangeClient(
        jpy="1000000", ticker=make_ticker(timestamp=utcnow() - timedelta(hours=1))
    )
    signal = await _signal(session)

    await rebalance_live(session, _live(settings), signal, Decimal("0.5"), client=exchange)

    assert "place_order" not in exchange.calls
    assert _status(signal) == "blocked_stale_market_data"


async def test_the_daily_loss_limit_pauses_buys_but_not_sells(
    session: AsyncSession, settings: Settings
) -> None:
    configured = _live(settings, max_daily_loss_jpy=Decimal("10000"))
    session.add(
        PortfolioSnapshot(
            timestamp=risk_day_start(configured) - timedelta(minutes=1),
            trading_mode=TradingMode.LIVE,
            jpy_balance=Decimal("600000"),
            btc_balance=Decimal("0"),
            btc_value_jpy=Decimal("0"),
            total_value_jpy=Decimal("600000"),
        )
    )
    exchange = FakeExchangeClient(jpy="400000", btc="0.01")
    signal = await _signal(session)

    blocked = await rebalance_live(
        session, configured, signal, Decimal("1"), client=exchange, force=True
    )
    sold = await rebalance_live(session, configured, signal, Decimal("0"), client=exchange)

    assert blocked is None
    assert sold is not None
    assert sold.side is Side.SELL


# --------------------------------------------------------------------------- #
# failures: never a duplicate, never a blind retry
# --------------------------------------------------------------------------- #
async def test_a_rejection_is_recorded_and_alerted_without_raising(
    session: AsyncSession, settings: Settings
) -> None:
    exchange = FakeExchangeClient(jpy="1000000")
    exchange.reject_orders = True
    signal = await _signal(session)

    with respx.mock(assert_all_called=False) as router:
        hook = router.post(WEBHOOK).mock(return_value=httpx.Response(200))
        order = await rebalance_live(
            session,
            _live(settings, alert_webhook_url=SecretStr(WEBHOOK)),
            signal,
            Decimal("0.5"),
            client=exchange,
            force=True,
        )

    assert order is not None
    assert order.status is OrderStatus.REJECTED
    assert order.rejection_reason is not None and "rejected" in order.rejection_reason
    assert exchange.calls.count("place_order") == 1
    assert _status(signal) == "rejected"
    assert await _count(session, SystemEvent, SystemEvent.event_type == "live_order_rejected") == 1
    assert hook.call_count == 1
    assert b"Order rejected" in hook.calls[0].request.content


async def test_repeated_failures_trip_the_circuit_breaker_once(
    session: AsyncSession, settings: Settings
) -> None:
    configured = _live(settings, live_max_failed_orders_per_day=2)
    exchange = FakeExchangeClient(jpy="1000000")
    exchange.reject_orders = True

    for _ in range(4):
        await rebalance_live(
            session, configured, await _signal(session), Decimal("0.5"), client=exchange, force=True
        )

    assert exchange.calls.count("place_order") == 2
    halted = SystemEvent.event_type == "live_trading_halted"
    assert await _count(session, SystemEvent, halted) == 1


async def test_a_lost_response_is_reconciled_instead_of_resent(
    session: AsyncSession, settings: Settings
) -> None:
    """The exchange accepted and filled the order, but the reply never arrived."""
    configured = _live(settings)
    exchange = FakeExchangeClient(jpy="1000000")
    exchange.timeout_orders = True
    signal = await _signal(session)

    first = await rebalance_live(
        session, configured, signal, Decimal("0.5"), client=exchange, force=True
    )
    assert first is not None
    assert first.status is OrderStatus.UNKNOWN
    assert first.exchange_order_id is None

    exchange.timeout_orders = False
    again = await rebalance_live(
        session, configured, signal, Decimal("0.5"), client=exchange, force=True
    )

    assert exchange.calls.count("place_order") == 1
    await session.refresh(first)
    assert first.status is OrderStatus.FILLED
    assert first.exchange_order_id == exchange.orders[0].acceptance_id
    assert first.filled_size == exchange.orders[0].request.size
    # Already at target once the lost fill is known: nothing new is sent.
    assert again is None
    assert _status(signal) == "no_trade"


async def test_an_unknown_order_never_seen_by_the_exchange_fails_after_the_grace(
    session: AsyncSession, settings: Settings
) -> None:
    configured = _live(settings)
    exchange = FakeExchangeClient(jpy="1000000")
    exchange.timeout_orders = True
    exchange.fill_ratio = Decimal("0")  # accepted but never executed: not listed
    order = await rebalance_live(
        session, configured, await _signal(session), Decimal("0.5"), client=exchange, force=True
    )
    assert order is not None and order.status is OrderStatus.UNKNOWN

    assert await reconcile_open_orders(session, configured, exchange) == 0
    order.created_at = utcnow() - live.UNKNOWN_ORDER_GRACE - timedelta(minutes=1)
    assert await reconcile_open_orders(session, configured, exchange) == 1

    assert order.status is OrderStatus.FAILED
    assert order.filled_size == 0


async def test_an_unfilled_ioc_blocks_until_settled_then_retries(
    session: AsyncSession, settings: Settings
) -> None:
    configured = _live(settings)
    exchange = FakeExchangeClient(jpy="1000000")
    exchange.fill_ratio = Decimal("0")
    signal = await _signal(session)

    order = await rebalance_live(
        session, configured, signal, Decimal("0.5"), client=exchange, force=True
    )
    assert order is not None and order.status is OrderStatus.SUBMITTED

    blocked = await rebalance_live(session, configured, signal, Decimal("0.5"), client=exchange)
    assert blocked is None
    assert _status(signal) == "blocked_pending_order"

    order.submitted_at = utcnow() - live.ACCEPTED_ORDER_GRACE - timedelta(minutes=1)
    exchange.fill_ratio = Decimal("1")
    retried = await rebalance_live(
        session, configured, signal, Decimal("0.5"), client=exchange, force=True
    )

    assert order.status is OrderStatus.CANCELLED
    assert order.filled_size == 0
    assert retried is not None and retried.status is OrderStatus.FILLED
    assert exchange.calls.count("place_order") == 2


async def test_a_partial_fill_keeps_what_executed(
    session: AsyncSession, settings: Settings
) -> None:
    exchange = FakeExchangeClient(jpy="1000000")
    exchange.fill_ratio = Decimal("0.5")
    signal = await _signal(session)

    order = await rebalance_live(
        session, _live(settings), signal, Decimal("0.5"), client=exchange, force=True
    )

    assert order is not None
    assert order.status is OrderStatus.CANCELLED
    assert 0 < order.filled_size < order.requested_size
    assert _status(signal) == "filled"


# --------------------------------------------------------------------------- #
# end to end through the strategy service
# --------------------------------------------------------------------------- #
async def test_the_daily_strategy_trades_live_once_per_decision(
    session: AsyncSession, settings: Settings
) -> None:
    configured = _live(settings, strategy_name="trend_ensemble")
    today = utcnow().date()
    bars = bars_from_closes(price_path(420), start=today - timedelta(days=420))
    await store_external_bars(
        session,
        configured,
        "bitbank",
        [
            ExternalDailyBar(bar.day, bar.open, bar.close, bar.close, bar.close, Decimal("1"))
            for bar in bars
        ],
    )
    exchange = FakeExchangeClient(jpy="1000000")

    first = await evaluate_and_store(session, configured, exchange_client=exchange)
    second = await evaluate_and_store(session, configured, exchange_client=exchange)

    assert first is not None and second is not None
    assert second.id == first.id
    assert exchange.calls.count("place_order") == 1
    order = (await session.execute(select(Order))).scalar_one()
    assert order.signal_id == first.id
    assert order.status is OrderStatus.FILLED


async def test_live_evaluation_requires_an_exchange_client(
    session: AsyncSession, settings: Settings
) -> None:
    configured = _live(settings, strategy_name="trend_ensemble")
    today = utcnow().date()
    bars = bars_from_closes(price_path(420), start=today - timedelta(days=420))
    await store_external_bars(
        session,
        configured,
        "bitbank",
        [ExternalDailyBar(b.day, b.open, b.close, b.close, b.close, Decimal("1")) for b in bars],
    )

    with pytest.raises(ValueError, match="exchange client"):
        await evaluate_and_store(session, configured)
