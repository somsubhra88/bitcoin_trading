from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from btc_trader.backtesting import run_backtest, run_walk_forward
from btc_trader.config import Settings
from btc_trader.db.models import MarketTick, OrderBookSnapshot


def _ticks(prices: list[str]) -> list[MarketTick]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    return [
        MarketTick(
            exchange="test",
            product_code="BTC_JPY",
            exchange_timestamp=start + timedelta(minutes=index),
            received_at=start + timedelta(minutes=index),
            ltp=Decimal(price),
            best_bid=Decimal(price),
            best_ask=Decimal(price),
        )
        for index, price in enumerate(prices)
    ]


def _backtest_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "strategy_name": "momentum_regime",
            "strategy_momentum_window": 12,
            "strategy_momentum_threshold_percent": Decimal("0.01"),
            "paper_min_confidence": Decimal("0.1"),
            "min_profit_percent": Decimal("0.1"),
        }
    )


def _order_books(count: int) -> list[OrderBookSnapshot]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    rows: list[OrderBookSnapshot] = []
    for index in range(count):
        if index < 70:
            mid = Decimal("100") + Decimal(index) / Decimal("20")
            bid_size, ask_size = "2", "1"
        else:
            mid = Decimal("103.5") - Decimal(index - 70) / Decimal("10")
            bid_size, ask_size = "1", "2"
        rows.append(
            OrderBookSnapshot(
                exchange="test",
                product_code="BTC_JPY",
                received_at=start + timedelta(minutes=index),
                mid_price=mid,
                depth_levels=3,
                bids=[
                    {"price": str(mid - Decimal("0.5") - level), "size": bid_size}
                    for level in range(3)
                ],
                asks=[
                    {"price": str(mid + Decimal("0.5") + level), "size": ask_size}
                    for level in range(3)
                ],
            )
        )
    return rows


def test_backtest_replays_signals_and_paper_fills(settings: Settings) -> None:
    prices = ["100"] * 11 + ["101"] + ["102"] * 11 + ["101.7"]

    result = run_backtest(_ticks(prices), _backtest_settings(settings))

    assert result.metrics.bars == 24
    assert result.metrics.buy_signals > 0
    assert result.metrics.sell_signals > 0
    assert result.metrics.trade_count == 1
    assert result.metrics.winning_trades == 1
    assert result.metrics.net_profit_jpy > 0
    assert len(result.equity_curve) == 24


def test_walk_forward_reports_unseen_test_windows(settings: Settings) -> None:
    prices = ["100"] * 11 + ["101"] + ["102"] * 11 + ["101.7"]
    result = run_walk_forward(
        _ticks(prices + prices + prices),
        _backtest_settings(settings),
        train_bars=24,
        test_bars=24,
    )

    assert len(result.windows) == 2
    assert result.windows[0].train_end < result.windows[0].test_start
    assert result.windows[0].test_metrics.bars == 24
    assert result.aggregate_metrics.bars == 48


def test_deeplob_backtest_and_walk_forward_use_order_book_history(settings: Settings) -> None:
    deeplob = settings.model_copy(
        update={
            "strategy_name": "deeplob",
            "strategy_deeplob_window": 8,
            "strategy_deeplob_horizon": 2,
            "strategy_deeplob_training_window": 32,
            "strategy_deeplob_levels": 3,
            "strategy_deeplob_threshold_percent": Decimal("0.01"),
            "strategy_deeplob_min_quality_percent": Decimal("80"),
            "paper_min_confidence": Decimal("0.1"),
            "min_profit_percent": Decimal("0.1"),
        }
    )
    books = _order_books(120)

    result = run_backtest([], deeplob, order_books=books)
    walk_forward = run_walk_forward(
        [],
        deeplob,
        order_books=books,
        train_bars=60,
        test_bars=30,
    )

    assert result.metrics.buy_signals > 0
    assert result.metrics.sell_signals > 0
    assert result.metrics.trade_count >= 1
    assert len(walk_forward.windows) == 2
    assert walk_forward.windows[0].train_end < walk_forward.windows[0].test_start
    assert "deeplob_threshold_percent" in walk_forward.windows[0].selected_parameters
