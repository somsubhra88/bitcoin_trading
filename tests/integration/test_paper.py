from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
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
from btc_trader.enums import ExitReason, OrderStatus, Side, TradingMode, TradingSignal
from btc_trader.market_data.daily import store_external_bars
from btc_trader.market_data.history_provider import ExternalDailyBar
from btc_trader.paper import rebalance_paper, reset_paper_trading, risk_day_start
from btc_trader.strategy_service import evaluate_and_store
from btc_trader.utils.timeutils import utcnow
from tests.fixtures.daily import bars_from_closes, price_path

PRICE = "10000000"


def _paper_settings(settings: Settings, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "trading_mode": TradingMode.PAPER,
        "max_market_data_age_seconds": 600,
        "max_position_jpy": Decimal("0"),
        "max_order_jpy": Decimal("0"),
    }
    values.update(overrides)
    return settings.model_copy(update=values)


async def _add_tick(
    session: AsyncSession,
    *,
    ltp: str = PRICE,
    bid: str = PRICE,
    ask: str = PRICE,
    timestamp: datetime | None = None,
) -> MarketTick:
    moment = timestamp or utcnow()
    tick = MarketTick(
        exchange="bitflyer",
        product_code="BTC_JPY",
        exchange_timestamp=moment,
        received_at=moment,
        ltp=Decimal(ltp),
        best_bid=Decimal(bid),
        best_ask=Decimal(ask),
    )
    session.add(tick)
    await session.flush()
    return tick


async def _signal(session: AsyncSession) -> StrategySignal:
    row = StrategySignal(
        strategy_name="trend_ensemble",
        strategy_version="2.0.0",
        product_code="BTC_JPY",
        timestamp=utcnow(),
        signal=TradingSignal.BUY,
        confidence=Decimal("1"),
        reference_price=Decimal(PRICE),
    )
    session.add(row)
    await session.flush()
    return row


async def _count(session: AsyncSession, model: type) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_data_only_never_creates_paper_orders(
    session: AsyncSession, settings: Settings
) -> None:
    await _add_tick(session)
    signal = await _signal(session)

    assert await rebalance_paper(session, settings, signal, Decimal("0.5")) is None
    assert await _count(session, Order) == 0
    assert await _count(session, PortfolioSnapshot) == 0


async def test_rebalance_buys_toward_the_target_and_marks_the_portfolio(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session)
    signal = await _signal(session)

    order = await rebalance_paper(session, paper, signal, Decimal("0.5"))

    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.side is Side.BUY
    assert await _count(session, Fill) == 1
    position = (await session.execute(select(Position))).scalar_one()
    fill_price = Decimal(PRICE) * Decimal("1.0005")  # 5 bps paper slippage
    assert position.btc_quantity == (Decimal("500000") / fill_price).quantize(
        Decimal("1e-8"), "ROUND_DOWN"
    )
    snapshot = (await session.execute(select(PortfolioSnapshot))).scalar_one()
    assert snapshot.jpy_balance + snapshot.btc_value_jpy < paper.paper_initial_jpy
    assert signal.metadata_json is not None
    assert signal.metadata_json["execution"]["status"] == "filled"


async def test_a_second_rebalance_to_the_same_target_does_not_churn(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session)
    signal = await _signal(session)
    assert await rebalance_paper(session, paper, signal, Decimal("0.5")) is not None

    again = await rebalance_paper(session, paper, signal, Decimal("0.5"), force=False)

    assert again is None
    assert await _count(session, Order) == 1
    assert signal.metadata_json is not None
    assert signal.metadata_json["execution"]["status"] == "no_trade"


async def test_a_stop_out_sells_everything_and_realises_pnl(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session)
    signal = await _signal(session)
    assert await rebalance_paper(session, paper, signal, Decimal("0.5")) is not None

    await _add_tick(session, ltp="9500000", bid="9500000", ask="9500000")
    order = await rebalance_paper(
        session, paper, signal, Decimal("0"), reduction_reason=ExitReason.TRAILING_STOP
    )

    assert order is not None
    assert order.side is Side.SELL
    assert order.exit_reason is ExitReason.TRAILING_STOP
    position = (await session.execute(select(Position))).scalar_one()
    assert position.btc_quantity == 0
    assert position.average_entry_price is None
    assert position.realized_pnl_jpy < 0


async def test_kill_switch_forbids_paper_orders(session: AsyncSession, settings: Settings) -> None:
    paper = _paper_settings(settings, trading_kill_switch=True)
    await _add_tick(session)
    signal = await _signal(session)

    assert await rebalance_paper(session, paper, signal, Decimal("0.5")) is None
    assert await _count(session, Order) == 0
    assert signal.metadata_json is not None
    assert signal.metadata_json["execution"]["status"] == "blocked_kill_switch"


async def test_stale_market_data_cannot_trigger_a_paper_order(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings, max_market_data_age_seconds=10)
    await _add_tick(session, timestamp=utcnow() - timedelta(minutes=1))
    signal = await _signal(session)

    assert await rebalance_paper(session, paper, signal, Decimal("0.5")) is None
    assert await _count(session, Order) == 0


async def test_exchange_minimum_block_is_explained(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings, paper_initial_jpy=Decimal("20000"))
    await _add_tick(session)
    signal = await _signal(session)

    assert await rebalance_paper(session, paper, signal, Decimal("0.25")) is None
    assert signal.metadata_json is not None
    assert "below the exchange minimum" in signal.metadata_json["execution"]["detail"]


async def test_paper_reset_starts_zero_holdings_and_preserves_analysis_history(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session)
    signal = await _signal(session)
    assert await rebalance_paper(session, paper, signal, Decimal("0.5")) is not None
    await session.commit()

    result = await reset_paper_trading(session, paper)

    assert result.orders_deleted == 1
    assert result.fills_deleted == 1
    assert result.positions_deleted == 1
    assert result.snapshots_deleted == 1
    assert await _count(session, Order) == 0
    assert await _count(session, StrategySignal) == 1
    snapshots = list((await session.execute(select(PortfolioSnapshot))).scalars())
    assert len(snapshots) == 1
    assert snapshots[0].btc_balance == Decimal("0")
    assert snapshots[0].total_value_jpy == paper.paper_initial_jpy


def test_the_risk_day_starts_at_local_midnight(settings: Settings) -> None:
    tokyo = settings.model_copy(update={"risk_day_timezone": "Asia/Tokyo"})
    moment = datetime(2026, 9, 1, 16, 30, tzinfo=UTC)  # 01:30 on 2 September in Tokyo

    start = risk_day_start(tokyo, moment)

    assert start.astimezone(UTC) == datetime(2026, 9, 1, 15, 0, tzinfo=UTC)


async def _store_rally(session: AsyncSession, settings: Settings, days: int = 420) -> None:
    today = utcnow().date()
    bars = bars_from_closes(
        price_path(days, drift=0.004, amplitude=0.02), start=today - timedelta(days=days)
    )
    await store_external_bars(
        session,
        settings,
        "bitbank",
        [
            ExternalDailyBar(bar.day, bar.open, bar.close, bar.close, bar.close, Decimal("1"))
            for bar in bars
        ],
    )
    last = bars[-1].close
    await _add_tick(session, ltp=str(last), bid=str(last), ask=str(last))


async def test_strategy_evaluation_drives_the_paper_ledger_once_per_day(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings, strategy_name="trend_ensemble")
    await _store_rally(session, paper)

    first = await evaluate_and_store(session, paper)
    second = await evaluate_and_store(session, paper)

    assert first is not None
    assert second is not None
    assert second.id == first.id
    assert await _count(session, StrategySignal) == 1
    assert await _count(session, Order) == 1
    order = (await session.execute(select(Order))).scalar_one()
    assert order.side is Side.BUY
    assert order.signal_id == first.id
    assert first.metadata_json is not None
    assert first.metadata_json["history"]["external_days"] == 420
    assert 0 < first.metadata_json["target_exposure"] <= 1


async def test_strategy_waits_for_daily_history(session: AsyncSession, settings: Settings) -> None:
    paper = _paper_settings(settings, strategy_name="trend_ensemble")
    await _store_rally(session, paper, days=40)

    assert await evaluate_and_store(session, paper) is None
    assert await _count(session, Order) == 0


async def test_a_stale_daily_bar_records_the_block_instead_of_trading(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings, strategy_name="trend_ensemble")
    today = utcnow().date()
    bars = bars_from_closes(price_path(400), start=today - timedelta(days=410))
    await store_external_bars(
        session,
        paper,
        "bitbank",
        [
            ExternalDailyBar(bar.day, bar.open, bar.close, bar.close, bar.close, Decimal("1"))
            for bar in bars
        ],
    )
    await _add_tick(session)

    signal = await evaluate_and_store(session, paper)

    assert signal is not None
    assert signal.metadata_json is not None
    assert signal.metadata_json["execution"]["status"] == "blocked_stale_daily_bar"
    assert await _count(session, Order) == 0


async def test_strategies_do_not_run_in_data_only_mode(
    session: AsyncSession, settings: Settings
) -> None:
    configured = settings.model_copy(update={"strategy_name": "trend_ensemble"})
    await _store_rally(session, configured)

    assert await evaluate_and_store(session, configured) is None
