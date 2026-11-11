"""Daily trend-following ensemble with volatility-targeted sizing.

Why this strategy
-----------------
bitFlyer spot charges 0.15% per side for small accounts, so a round trip costs
roughly 0.35-0.40% once spread and slippage are included. Signals measured over
minutes rarely predict moves that large; the repository's earlier 5-minute
EMA/mean-reversion/momentum rules lost 60-85% over a year of real BTC/JPY data
for exactly that reason (``docs/strategy_research.md``).

The best-supported alternative in the literature is slow time-series momentum
with volatility-based sizing (Moskowitz, Ooi & Pedersen 2012; Moreira & Muir
2017; Zarattini, Pagani & Barbon 2025). This module implements the latter's
Bitcoin model:

* nine Donchian-channel sleeves on **daily closes** (5 ... 360 days);
* a sleeve goes long when the close reaches its N-day high, sets a trailing
  stop at the channel midpoint, ratchets that stop up only, and exits when a
  close falls below it;
* each long sleeve holds ``min(target_vol / realised_vol, max_exposure)`` of
  its capital share, so exposure falls when Bitcoin becomes more volatile.

Everything here is pure: given the same completed daily bars it returns the
same answer, which is what lets the backtest and the paper ledger share it.
Decisions use completed bars only; the caller executes them afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from btc_trader.config import STRATEGY_NAMES, Settings
from btc_trader.enums import TradingSignal

STRATEGIES = {
    "none": "Disabled",
    "trend_ensemble": "Trend ensemble (daily Donchian, volatility-targeted)",
}
assert set(STRATEGIES) == STRATEGY_NAMES  # noqa: S101 - import-time consistency check
STRATEGY_VERSION = "2.0.0"

#: Calendar days per year: Bitcoin trades every day.
DAYS_PER_YEAR = 365


@dataclass(frozen=True, slots=True)
class DailyBar:
    """One completed UTC day. ``source`` records where the close came from."""

    day: date
    open: Decimal
    close: Decimal
    source: str


@dataclass(frozen=True, slots=True)
class TrendParameters:
    horizons: tuple[int, ...] = (5, 10, 20, 30, 60, 90, 150, 250, 360)
    volatility_window: int = 90
    target_volatility: float = 0.25
    max_exposure: float = 1.0

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 2:
            raise ValueError("trend horizons must be at least 2 bars")
        if self.volatility_window < 10:
            raise ValueError("volatility window must be at least 10 bars")
        if self.target_volatility <= 0:
            raise ValueError("target volatility must be positive")
        if not 0 < self.max_exposure <= 1:
            raise ValueError("max exposure must be in (0, 1] for a spot account")

    @property
    def warmup_bars(self) -> int:
        """Bars needed before any exposure can be sized."""
        return self.volatility_window + 1


def parameters_from_settings(settings: Settings) -> TrendParameters:
    return TrendParameters(
        horizons=settings.trend_horizons,
        volatility_window=settings.strategy_volatility_window_days,
        target_volatility=float(settings.strategy_target_annual_volatility_percent / 100),
        max_exposure=float(settings.strategy_max_exposure_percent / 100),
    )


@dataclass(frozen=True, slots=True)
class SleeveState:
    horizon: int
    active: bool
    warming_up: bool
    stop: float | None
    channel_high: float | None
    channel_low: float | None
    entered: bool
    exited: bool


@dataclass(frozen=True, slots=True)
class TrendPath:
    """Full history of the ensemble, one entry per input bar."""

    days: list[date]
    closes: np.ndarray
    active: np.ndarray  # (horizons, bars) bool
    entered: np.ndarray  # (horizons, bars) bool
    exited: np.ndarray  # (horizons, bars) bool
    stops: np.ndarray  # (horizons, bars) float, nan when flat
    highs: np.ndarray  # (horizons, bars) float, nan while warming up
    lows: np.ndarray
    volatility: np.ndarray  # annualised, nan while warming up
    target: np.ndarray  # target BTC exposure in [0, max_exposure]
    parameters: TrendParameters = field(repr=False)


@dataclass(frozen=True, slots=True)
class TrendDecision:
    as_of: date
    close: Decimal
    target_exposure: Decimal
    previous_target_exposure: Decimal
    active_fraction: Decimal
    realized_volatility: float | None
    volatility_scale: float | None
    sleeves: tuple[SleeveState, ...]
    sleeves_changed: bool
    signal: TradingSignal
    confidence: Decimal
    reason: str
    metadata: dict[str, object]


def trend_path(bars: Sequence[DailyBar], parameters: TrendParameters) -> TrendPath:
    """Run every Donchian sleeve over the whole history in one pass."""
    closes = np.asarray([float(bar.close) for bar in bars], dtype=np.float64)
    count, sleeves = len(closes), len(parameters.horizons)
    active = np.zeros((sleeves, count), dtype=bool)
    entered = np.zeros((sleeves, count), dtype=bool)
    exited = np.zeros((sleeves, count), dtype=bool)
    stops = np.full((sleeves, count), np.nan)
    highs = np.full((sleeves, count), np.nan)
    lows = np.full((sleeves, count), np.nan)

    for row, horizon in enumerate(parameters.horizons):
        if count < horizon:
            continue
        windows = sliding_window_view(closes, horizon)
        highs[row, horizon - 1 :] = windows.max(axis=1)
        lows[row, horizon - 1 :] = windows.min(axis=1)
        on, stop = False, 0.0
        for index in range(horizon - 1, count):
            midpoint = (highs[row, index] + lows[row, index]) / 2
            if on:
                stop = max(stop, midpoint)
                if closes[index] < stop:
                    on = False
                    exited[row, index] = True
            elif closes[index] >= highs[row, index]:
                on, stop = True, midpoint
                entered[row, index] = True
            active[row, index] = on
            if on:
                stops[row, index] = stop

    volatility = _realised_volatility(closes, parameters.volatility_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.minimum(parameters.target_volatility / volatility, parameters.max_exposure)
    target = np.where(np.isfinite(scale), active.mean(axis=0) * scale, 0.0)
    target = np.clip(target, 0.0, parameters.max_exposure)
    return TrendPath(
        days=[bar.day for bar in bars],
        closes=closes,
        active=active,
        entered=entered,
        exited=exited,
        stops=stops,
        highs=highs,
        lows=lows,
        volatility=volatility,
        target=target,
        parameters=parameters,
    )


def evaluate_trend(
    bars: Sequence[DailyBar], parameters: TrendParameters | None = None
) -> TrendDecision | None:
    """Decision for the newest completed bar, or ``None`` before warm-up ends."""
    parameters = parameters or TrendParameters()
    if len(bars) < parameters.warmup_bars:
        return None
    path = trend_path(bars, parameters)
    return decision_at(path, len(bars) - 1, bars[-1].close)


def decision_at(path: TrendPath, index: int, close: Decimal) -> TrendDecision:
    parameters = path.parameters
    target = _exposure(path.target[index])
    previous = _exposure(path.target[index - 1]) if index > 0 else Decimal("0")
    volatility = float(path.volatility[index])
    volatility_known = math.isfinite(volatility) and volatility > 0
    scale = (
        min(parameters.target_volatility / volatility, parameters.max_exposure)
        if volatility_known
        else None
    )
    sleeves = tuple(
        SleeveState(
            horizon=horizon,
            active=bool(path.active[row, index]),
            warming_up=index + 1 < horizon,
            stop=_finite(path.stops[row, index]),
            channel_high=_finite(path.highs[row, index]),
            channel_low=_finite(path.lows[row, index]),
            entered=bool(path.entered[row, index]),
            exited=bool(path.exited[row, index]),
        )
        for row, horizon in enumerate(parameters.horizons)
    )
    active_count = sum(sleeve.active for sleeve in sleeves)
    active_fraction = (Decimal(active_count) / Decimal(len(sleeves))).quantize(Decimal("0.000001"))
    entries = [sleeve.horizon for sleeve in sleeves if sleeve.entered]
    exits = [sleeve.horizon for sleeve in sleeves if sleeve.exited]
    signal = (
        TradingSignal.BUY
        if target > previous
        else TradingSignal.SELL
        if target < previous
        else TradingSignal.HOLD
    )

    parts = [f"{active_count}/{len(sleeves)} trend sleeves long"]
    if entries:
        parts.append("entered " + ", ".join(f"{h}d" for h in entries))
    if exits:
        parts.append("stopped out " + ", ".join(f"{h}d" for h in exits))
    if volatility_known:
        parts.append(f"realised vol {volatility:.0%} vs target {parameters.target_volatility:.0%}")
    else:
        parts.append("volatility warming up")
    parts.append(
        f"target exposure {target:.1%}"
        + (f" (was {previous:.1%})" if target != previous else " (unchanged)")
    )

    metadata: dict[str, object] = {
        "model": "donchian_trend_ensemble",
        "as_of": path.days[index].isoformat(),
        "close": float(path.closes[index]),
        "target_exposure": float(target),
        "previous_target_exposure": float(previous),
        "active_fraction": float(active_fraction),
        "realized_volatility": volatility if volatility_known else None,
        "volatility_scale": scale,
        "target_volatility": parameters.target_volatility,
        "sleeves_changed": bool(entries or exits),
        "sleeves": [
            {
                "horizon": sleeve.horizon,
                "active": sleeve.active,
                "warming_up": sleeve.warming_up,
                "stop": sleeve.stop,
                "channel_high": sleeve.channel_high,
                "channel_low": sleeve.channel_low,
                "entered": sleeve.entered,
                "exited": sleeve.exited,
            }
            for sleeve in sleeves
        ],
    }
    return TrendDecision(
        as_of=path.days[index],
        close=close,
        target_exposure=target,
        previous_target_exposure=previous,
        active_fraction=active_fraction,
        realized_volatility=volatility if volatility_known else None,
        volatility_scale=scale,
        sleeves=sleeves,
        sleeves_changed=bool(entries or exits),
        signal=signal,
        confidence=active_fraction,
        reason="; ".join(parts),
        metadata=metadata,
    )


def _realised_volatility(closes: np.ndarray, window: int) -> np.ndarray:
    """Annualised std-dev of the last ``window`` daily log returns, per bar."""
    result = np.full(len(closes), np.nan)
    if len(closes) <= window:
        return result
    returns = np.diff(np.log(closes))
    rolling = sliding_window_view(returns, window).std(axis=1, ddof=1)
    result[window:] = rolling * math.sqrt(DAYS_PER_YEAR)
    return result


def _exposure(value: float) -> Decimal:
    # Exposures live in [0, 1], so normalize() cannot produce exponent notation.
    return Decimal(f"{float(value):.6f}").normalize()


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


__all__ = [
    "STRATEGIES",
    "STRATEGY_VERSION",
    "DailyBar",
    "SleeveState",
    "TrendDecision",
    "TrendParameters",
    "TrendPath",
    "decision_at",
    "evaluate_trend",
    "parameters_from_settings",
    "trend_path",
]
