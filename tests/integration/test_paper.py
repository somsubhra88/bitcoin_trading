from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import (
    Fill,
    MarketTick,
    Order,
    OrderBookSnapshot,
    PortfolioSnapshot,
    Position,
    StrategySignal,
)
from btc_trader.enums import ExitReason, OrderStatus, Side, TradingMode, TradingSignal
from btc_trader.paper import execute_signal, reset_paper_trading
from btc_trader.strategy_service import evaluate_and_store
from btc_trader.utils.timeutils import utcnow


def _paper_settings(settings: Settings, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "trading_mode": TradingMode.PAPER,
        "max_market_data_age_seconds": 600,
    }
    values.update(overrides)
    return settings.model_copy(update=values)


async def _add_tick(
    session: AsyncSession,
    *,
    ltp: str,
    bid: str,
    ask: str,
    timestamp=None,
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


async def _signal(
    session: AsyncSession, signal: TradingSignal, *, confidence: str = "1"
) -> StrategySignal:
    row = StrategySignal(
        strategy_name="test_strategy",
        strategy_version="1.0.0",
        product_code="BTC_JPY",
        timestamp=utcnow(),
        signal=signal,
        confidence=Decimal(confidence),
        reference_price=Decimal("100"),
    )
    session.add(row)
    await session.flush()
    return row


async def _count(session: AsyncSession, model: type) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_data_only_never_creates_paper_orders(
    session: AsyncSession, settings: Settings
) -> None:
    await _add_tick(session, ltp="100", bid="99", ask="101")
    signal = await _signal(session, TradingSignal.BUY)

    assert await execute_signal(session, settings, signal) is None
    assert await _count(session, Order) == 0
    assert await _count(session, PortfolioSnapshot) == 0


async def test_paper_buy_creates_a_filled_order_and_marks_portfolio(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session, ltp="100", bid="99", ask="101")
    signal = await _signal(session, TradingSignal.BUY)

    order = await execute_signal(session, paper, signal)

    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.side is Side.BUY
    assert await _count(session, Fill) == 1
    position = (await session.execute(select(Position))).scalar_one()
    assert position.trading_mode is TradingMode.PAPER
    assert position.btc_quantity > 0
    snapshot = (await session.execute(select(PortfolioSnapshot))).scalar_one()
    assert snapshot.jpy_balance < paper.paper_initial_jpy


async def test_paper_reset_starts_zero_holdings_and_preserves_analysis_history(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session, ltp="100", bid="99", ask="101")
    signal = await _signal(session, TradingSignal.BUY)
    assert await execute_signal(session, paper, signal) is not None
    await session.commit()

    result = await reset_paper_trading(session, paper)

    assert result.product_code == "BTC_JPY"
    assert result.orders_deleted == 1
    assert result.fills_deleted == 1
    assert result.positions_deleted == 1
    assert result.snapshots_deleted == 1
    assert await _count(session, Order) == 0
    assert await _count(session, Fill) == 0
    assert await _count(session, Position) == 0
    assert await _count(session, StrategySignal) == 1
    snapshots = list((await session.execute(select(PortfolioSnapshot))).scalars())
    assert len(snapshots) == 1
    assert snapshots[0].btc_balance == Decimal("0")
    assert snapshots[0].jpy_balance == paper.paper_initial_jpy
    assert snapshots[0].total_value_jpy == paper.paper_initial_jpy


async def test_hard_stop_exits_even_when_the_signal_is_not_sell(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(
        settings,
        min_profit_percent=Decimal("10"),
        paper_trailing_stop_percent=Decimal("0"),
        paper_stop_loss_percent=Decimal("1"),
    )
    await _add_tick(session, ltp="100", bid="99", ask="100")
    buy = await _signal(session, TradingSignal.BUY)
    assert await execute_signal(session, paper, buy) is not None

    await _add_tick(session, ltp="98", bid="98", ask="99")
    hold = await _signal(session, TradingSignal.HOLD)
    order = await execute_signal(session, paper, hold)

    assert order is not None
    assert order.side is Side.SELL
    assert order.exit_reason is ExitReason.RISK_EXIT
    assert (await session.execute(select(Position))).scalar_one().btc_quantity == 0


async def test_paper_entry_explains_exchange_minimum_block(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(
        settings,
        max_order_jpy=Decimal("0.05"),
        max_position_jpy=Decimal("0.05"),
    )
    await _add_tick(session, ltp="100", bid="99", ask="101")
    signal = await _signal(session, TradingSignal.BUY)

    assert await execute_signal(session, paper, signal) is None
    assert await _count(session, Order) == 0
    assert "below the configured minimum" in (signal.reason or "")
    assert signal.metadata_json is not None
    assert signal.metadata_json["paper_execution"] == "blocked_exchange_minimum"


async def test_strategy_evaluation_drives_the_paper_ledger(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings, strategy_name="ema_crossover")
    start = utcnow() - timedelta(seconds=20)
    for index in range(21):
        price = Decimal(100 + index)
        await _add_tick(
            session,
            ltp=str(price),
            bid=str(price),
            ask=str(price + 1),
            timestamp=start + timedelta(seconds=index),
        )

    signal = await evaluate_and_store(session, paper)

    assert signal is not None
    assert signal.signal is TradingSignal.BUY
    assert await _count(session, Order) == 1


async def test_low_threshold_momentum_signal_creates_a_paper_transaction(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(
        settings,
        strategy_name="momentum_regime",
        strategy_momentum_threshold_percent=Decimal("0.01"),
    )
    start = utcnow() - timedelta(seconds=11)
    for index in range(12):
        price = Decimal("100") if index < 11 else Decimal("100.02")
        await _add_tick(
            session,
            ltp=str(price),
            bid=str(price),
            ask=str(price + Decimal("0.01")),
            timestamp=start + timedelta(seconds=index),
        )

    signal = await evaluate_and_store(session, paper)

    assert signal is not None
    assert signal.signal is TradingSignal.BUY
    assert await _count(session, Order) == 1
    assert await _count(session, Fill) == 1


async def test_deeplob_strategy_creates_a_paper_transaction_from_order_books(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(
        settings,
        strategy_name="deeplob",
        strategy_deeplob_window=8,
        strategy_deeplob_horizon=2,
        strategy_deeplob_training_window=64,
        strategy_deeplob_levels=3,
        strategy_deeplob_threshold_percent=Decimal("0.01"),
        strategy_deeplob_min_quality_percent=Decimal("80"),
    )
    start = utcnow() - timedelta(seconds=79)
    for index in range(80):
        mid = Decimal("100") + Decimal(index) / Decimal("20")
        moment = start + timedelta(seconds=index)
        session.add(
            OrderBookSnapshot(
                exchange="bitflyer",
                product_code="BTC_JPY",
                received_at=moment,
                mid_price=mid,
                depth_levels=3,
                bids=[
                    {"price": str(mid - Decimal("0.5") - level), "size": "2"} for level in range(3)
                ],
                asks=[
                    {"price": str(mid + Decimal("0.5") + level), "size": "1"} for level in range(3)
                ],
            )
        )
    await _add_tick(session, ltp="103.95", bid="103.9", ask="104")
    await session.flush()

    signal = await evaluate_and_store(session, paper)

    assert signal is not None
    assert signal.strategy_name == "deeplob"
    assert signal.signal is TradingSignal.BUY
    assert await _count(session, Order) == 1
    assert await _count(session, Fill) == 1


async def test_paper_sell_respects_net_profit_target(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings)
    await _add_tick(session, ltp="100", bid="99", ask="100")
    buy = await _signal(session, TradingSignal.BUY)
    assert await execute_signal(session, paper, buy) is not None

    await _add_tick(session, ltp="100.5", bid="100.5", ask="101")
    sell = await _signal(session, TradingSignal.SELL)
    assert await execute_signal(session, paper, sell) is None
    assert await _count(session, Order) == 1

    await _add_tick(session, ltp="102", bid="102", ask="102.5")
    profitable_sell = await _signal(session, TradingSignal.SELL)
    order = await execute_signal(session, paper, profitable_sell)

    assert order is not None
    assert order.side is Side.SELL
    assert order.exit_reason is ExitReason.STRATEGY_EXIT
    position = (await session.execute(select(Position))).scalar_one()
    assert position.btc_quantity == 0


async def test_trailing_stop_can_exit_below_profit_target(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(
        settings,
        min_profit_percent=Decimal("10"),
        paper_trailing_stop_percent=Decimal("0.5"),
    )
    await _add_tick(session, ltp="100", bid="99", ask="100")
    buy = await _signal(session, TradingSignal.BUY)
    assert await execute_signal(session, paper, buy) is not None

    await _add_tick(session, ltp="102", bid="102", ask="103")
    hold = await _signal(session, TradingSignal.HOLD)
    assert await execute_signal(session, paper, hold) is None

    await _add_tick(session, ltp="101", bid="101", ask="102")
    stop = await _signal(session, TradingSignal.HOLD)
    order = await execute_signal(session, paper, stop)

    assert order is not None
    assert order.exit_reason is ExitReason.RISK_EXIT


async def test_stale_market_data_cannot_trigger_a_paper_order(
    session: AsyncSession, settings: Settings
) -> None:
    paper = _paper_settings(settings, max_market_data_age_seconds=10)
    old = utcnow() - timedelta(minutes=1)
    await _add_tick(session, ltp="100", bid="99", ask="101", timestamp=old)
    signal = await _signal(session, TradingSignal.BUY)

    assert await execute_signal(session, paper, signal) is None
    assert await _count(session, Order) == 0
