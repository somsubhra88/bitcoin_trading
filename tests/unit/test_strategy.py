from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from btc_trader.db.models import MarketTick
from btc_trader.enums import TradingSignal
from btc_trader.strategy import MIN_SAMPLES, StrategyParameters, evaluate


def _ticks(prices: list[str]) -> list[MarketTick]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    return [
        MarketTick(
            exchange="test",
            product_code="BTC_JPY",
            exchange_timestamp=start + timedelta(seconds=index),
            received_at=start + timedelta(seconds=index),
            ltp=Decimal(price),
        )
        for index, price in enumerate(prices)
    ]


@pytest.mark.parametrize("name", ["ema_crossover", "mean_reversion", "momentum_regime"])
def test_algorithms_wait_for_their_required_history(name: str) -> None:
    assert evaluate(name, _ticks(["100"] * (MIN_SAMPLES[name] - 1))) is None


@pytest.mark.parametrize(
    ("name", "prices", "expected"),
    [
        (
            "ema_crossover",
            [str(100 + index * 0.5) for index in range(21)],
            TradingSignal.BUY,
        ),
        (
            "mean_reversion",
            ["100"] * 19 + ["90"],
            TradingSignal.BUY,
        ),
        (
            "momentum_regime",
            ["100"] * 11 + ["101"],
            TradingSignal.BUY,
        ),
    ],
)
def test_each_algorithm_produces_a_decision(
    name: str, prices: list[str], expected: TradingSignal
) -> None:
    decision = evaluate(name, _ticks(prices))

    assert decision is not None
    assert decision.signal is expected
    assert decision.price == Decimal(prices[-1])
    assert Decimal("0") <= decision.confidence <= Decimal("1")
    assert decision.metadata


def test_none_is_an_explicit_disabled_strategy() -> None:
    assert evaluate("none", _ticks(["100"])) is None


def test_unknown_strategy_is_rejected_even_without_market_data() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        evaluate("unknown", [])


def test_momentum_threshold_is_configurable_for_low_volatility_markets() -> None:
    prices = ["100"] * 11 + ["100.02"]

    default = evaluate("momentum_regime", _ticks(prices))
    less_sensitive = evaluate(
        "momentum_regime",
        _ticks(prices),
        parameters=StrategyParameters(momentum_threshold_percent=Decimal("0.1")),
    )

    assert default is not None
    assert default.signal is TradingSignal.BUY
    assert default.confidence == Decimal("1")
    assert less_sensitive is not None
    assert less_sensitive.signal is TradingSignal.HOLD


def test_windows_are_configurable() -> None:
    parameters = StrategyParameters(
        fast_window=3,
        slow_window=5,
        mean_reversion_window=5,
        momentum_window=5,
        volatility_window=5,
    )
    decision = evaluate(
        "ema_crossover",
        _ticks(["100", "100", "100", "100", "101"]),
        parameters=parameters,
    )

    assert decision is not None
    assert decision.signal is TradingSignal.BUY
    assert "volatility" in decision.metadata


def test_regime_filter_blocks_mean_reversion_in_a_trend() -> None:
    prices = [str(100 + index) for index in range(20)]

    filtered = evaluate("mean_reversion", _ticks(prices))
    unfiltered = evaluate(
        "mean_reversion",
        _ticks(prices),
        parameters=StrategyParameters(regime_filter=False),
    )

    assert filtered is not None
    assert filtered.signal is TradingSignal.HOLD
    assert "trending tape" in filtered.reason
    assert unfiltered is not None
    assert unfiltered.signal is TradingSignal.SELL


def test_decision_reports_adaptive_threshold_diagnostics() -> None:
    decision = evaluate("momentum_regime", _ticks(["100"] * 11 + ["101"]))

    assert decision is not None
    assert decision.metadata["signal_threshold"] >= 0
    assert "baseline_volatility" in decision.metadata
    assert "return_autocorrelation" in decision.metadata
