"""Small, auditable strategy library used by paper mode and the dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from btc_trader.db.models import MarketTick
from btc_trader.enums import TradingSignal

STRATEGIES = {
    "none": "Disabled",
    "ema_crossover": "EMA crossover",
    "mean_reversion": "Mean reversion",
    "momentum_regime": "Momentum regime",
    "deeplob": "DeepLOB-style order book",
}
STRATEGY_VERSION = "1.1.0"

MIN_SAMPLES = {
    "ema_crossover": 21,
    "mean_reversion": 20,
    "momentum_regime": 12,
}


@dataclass(frozen=True, slots=True)
class StrategyParameters:
    """Runtime parameters shared by live evaluation and backtesting."""

    fast_window: int = 8
    slow_window: int = 21
    mean_reversion_window: int = 20
    momentum_window: int = 12
    volatility_window: int = 20
    ema_threshold_percent: Decimal = Decimal("0.05")
    mean_reversion_zscore_threshold: Decimal = Decimal("1.0")
    momentum_threshold_percent: Decimal = Decimal("0.01")
    edge_multiple: Decimal = Decimal("1.5")
    regime_filter: bool = True

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if (
            min(
                self.fast_window,
                self.slow_window,
                self.mean_reversion_window,
                self.momentum_window,
                self.volatility_window,
            )
            < 2
        ):
            raise ValueError("strategy windows must be at least 2 samples")
        if (
            self.ema_threshold_percent <= 0
            or self.mean_reversion_zscore_threshold <= 0
            or self.momentum_threshold_percent <= 0
            or self.edge_multiple < 0
        ):
            raise ValueError("strategy thresholds must be positive and edge_multiple non-negative")


@dataclass(frozen=True, slots=True)
class SignalDecision:
    signal: TradingSignal
    confidence: Decimal
    price: Decimal
    reason: str
    metadata: dict[str, str | float]


def evaluate(
    name: str,
    ticks: Sequence[MarketTick],
    *,
    parameters: StrategyParameters | None = None,
) -> SignalDecision | None:
    key = name.strip().lower()
    if key not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}")
    if key == "none":
        return None
    if key == "deeplob":
        raise ValueError("deeplob requires order-book snapshots; call evaluate_order_book")
    parameters = parameters or StrategyParameters()

    prices = [tick.ltp for tick in ticks if tick.ltp is not None]
    minimum_samples = {
        "ema_crossover": parameters.slow_window,
        "mean_reversion": parameters.mean_reversion_window,
        "momentum_regime": parameters.momentum_window,
    }[key]
    if len(prices) < minimum_samples:
        return None
    price = prices[-1]
    volatility = _return_volatility(prices, parameters.volatility_window)
    baseline_volatility = _robust_return_volatility(prices[:-1], parameters.volatility_window)
    return_autocorrelation = _return_autocorrelation(prices, parameters.volatility_window)

    if key == "ema_crossover":
        fast, slow = _ema(prices, parameters.fast_window), _ema(prices, parameters.slow_window)
        spread = (fast - slow) / slow if slow else 0.0
        threshold = _adaptive_threshold(
            parameters.ema_threshold_percent, baseline_volatility, parameters.edge_multiple
        )
        raw_signal = (
            TradingSignal.BUY
            if spread > threshold
            else TradingSignal.SELL
            if spread < -threshold
            else TradingSignal.HOLD
        )
        signal, regime_reason = _apply_regime_filter(
            key, raw_signal, return_autocorrelation, parameters.regime_filter
        )
        ema_metadata: dict[str, str | float] = {
            "fast_ema": fast,
            "slow_ema": slow,
            "volatility": float(volatility),
            "baseline_volatility": float(baseline_volatility),
            "signal_threshold": threshold,
            "signal_strength": abs(spread),
            "return_autocorrelation": return_autocorrelation,
        }
        return _decision(
            signal,
            price,
            0.0 if regime_reason else _normalised_confidence(abs(spread), threshold),
            f"EMA{parameters.fast_window} {fast:.0f} vs EMA{parameters.slow_window} {slow:.0f} "
            f"(threshold ±{threshold:.2%}){regime_reason or ''}",
            ema_metadata,
        )

    window = prices[-parameters.mean_reversion_window :]
    mean = sum(window, Decimal(0)) / len(window)
    deviation = _stddev(window, mean)
    z_score = float((price - mean) / Decimal(str(deviation))) if deviation else 0.0
    if key == "mean_reversion":
        threshold = float(parameters.mean_reversion_zscore_threshold)
        raw_signal = (
            TradingSignal.BUY
            if z_score < -threshold
            else TradingSignal.SELL
            if z_score > threshold
            else TradingSignal.HOLD
        )
        signal, regime_reason = _apply_regime_filter(
            key, raw_signal, return_autocorrelation, parameters.regime_filter
        )
        mean_reversion_metadata: dict[str, str | float] = {
            "mean": float(mean),
            "z_score": z_score,
            "volatility": float(volatility),
            "baseline_volatility": float(baseline_volatility),
            "signal_threshold": threshold,
            "signal_strength": abs(z_score),
            "return_autocorrelation": return_autocorrelation,
        }
        return _decision(
            signal,
            price,
            0.0 if regime_reason else _normalised_confidence(abs(z_score), threshold),
            f"price z-score {z_score:+.2f} around rolling mean (threshold ±{threshold:.2f})"
            f"{regime_reason or ''}",
            mean_reversion_metadata,
        )

    lookback = prices[-parameters.momentum_window :]
    momentum = float((lookback[-1] - lookback[0]) / lookback[0]) if lookback[0] else 0.0
    threshold = _adaptive_threshold(
        parameters.momentum_threshold_percent, baseline_volatility, parameters.edge_multiple
    )
    raw_signal = (
        TradingSignal.BUY
        if momentum > threshold
        else TradingSignal.SELL
        if momentum < -threshold
        else TradingSignal.HOLD
    )
    signal, regime_reason = _apply_regime_filter(
        key, raw_signal, return_autocorrelation, parameters.regime_filter
    )
    momentum_metadata: dict[str, str | float] = {
        "momentum": momentum,
        "volatility": float(volatility),
        "baseline_volatility": float(baseline_volatility),
        "signal_threshold": threshold,
        "signal_strength": abs(momentum),
        "return_autocorrelation": return_autocorrelation,
    }
    return _decision(
        signal,
        price,
        0.0 if regime_reason else _normalised_confidence(abs(momentum), threshold),
        f"{parameters.momentum_window}-sample momentum {momentum:+.2%} "
        f"(threshold ±{threshold:.2%}){regime_reason or ''}",
        momentum_metadata,
    )


def _adaptive_threshold(
    configured_percent: Decimal, baseline_volatility: Decimal, edge_multiple: Decimal
) -> float:
    """Keep the configured edge, or require more edge than recent market noise."""
    configured = float(configured_percent / Decimal("100"))
    volatility_floor = float(baseline_volatility * edge_multiple)
    return max(configured, volatility_floor)


def _apply_regime_filter(
    strategy_name: str,
    signal: TradingSignal,
    return_autocorrelation: float,
    enabled: bool,
) -> tuple[TradingSignal, str | None]:
    """Avoid trend entries in mean-reverting regimes and vice versa."""
    if not enabled or signal is TradingSignal.HOLD:
        return signal, None
    if strategy_name in {"ema_crossover", "momentum_regime"} and return_autocorrelation < -0.25:
        return TradingSignal.HOLD, "; regime filter: mean-reverting tape"
    if strategy_name == "mean_reversion" and return_autocorrelation > 0.25:
        return TradingSignal.HOLD, "; regime filter: trending tape"
    return signal, None


def _normalised_confidence(strength: float, threshold: float) -> float:
    return min(strength / threshold, 1.0) if threshold > 0 else 0.0


def _decision(
    signal: TradingSignal,
    price: Decimal,
    confidence: float,
    reason: str,
    metadata: dict[str, str | float],
) -> SignalDecision:
    return SignalDecision(
        signal, Decimal(str(round(max(0.0, min(confidence, 1.0)), 6))), price, reason, metadata
    )


def _ema(values: Sequence[Decimal], period: int) -> float:
    alpha = 2 / (period + 1)
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1 - alpha) * result
    return result


def _stddev(values: Sequence[Decimal], mean: Decimal) -> Decimal:
    variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values))
    return variance.sqrt()


def _return_volatility(values: Sequence[Decimal], window: int) -> Decimal:
    returns = _returns(values, window)
    if len(returns) < 2:
        return Decimal("0")
    mean = sum(returns, Decimal(0)) / Decimal(len(returns))
    return _stddev(returns, mean)


def _robust_return_volatility(values: Sequence[Decimal], window: int) -> Decimal:
    """Estimate baseline noise without allowing one price shock to dominate it."""
    returns = _returns(values, window)
    if len(returns) < 3:
        return Decimal("0")
    median = _median(returns)
    deviations = [abs(value - median) for value in returns]
    return _median(deviations) * Decimal("1.4826")


def _returns(values: Sequence[Decimal], window: int) -> list[Decimal]:
    prices = values[-window:]
    return [
        (current - previous) / previous for previous, current in pairwise(prices) if previous > 0
    ]


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


def _return_autocorrelation(values: Sequence[Decimal], window: int) -> float:
    """Lag-one return autocorrelation used as a light-weight regime signal."""
    returns = _returns(values, window)
    if len(returns) < 4:
        return 0.0
    mean = sum(returns, Decimal(0)) / Decimal(len(returns))
    denominator = sum((value - mean) ** 2 for value in returns)
    if denominator == 0:
        return 0.0
    covariance = sum(
        (previous - mean) * (current - mean) for previous, current in pairwise(returns)
    )
    return max(-1.0, min(1.0, float(covariance / denominator)))
