"""Causal volatility- and trend-regime labels for conditional reporting.

Regime thresholds are quantiles of a **trailing** volatility estimate computed on
the *training* segment and then applied unchanged everywhere else. Two reasons:

* Re-estimating quantiles per segment would make "high volatility" mean something
  different in each fold, and the conditional table would not be comparable.
* Quantiles of the whole series would be a mild but real form of look-ahead — the
  label at bar ``t`` would depend on bars after ``t``.

The labels are used only for slicing results, never as a feature. If they were a
feature they would need the same causality proof the feature pipeline gets; as a
reporting device they need only to be honest about what they knew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from archetype_trader.data.types import MarketData
from archetype_trader.features.rolling import rolling_std
from archetype_trader.splits import Segment

VOL_REGIMES = ("low_vol", "mid_vol", "high_vol")
TREND_REGIMES = ("down", "flat", "up")


@dataclass(frozen=True)
class RegimeLabels:
    """Per-bar regime labels plus the train-fitted thresholds that produced them."""

    vol: np.ndarray
    trend: np.ndarray
    vol_thresholds: tuple[float, float]
    trend_thresholds: tuple[float, float]
    window: int

    def slice_masks(self, segment: Segment, kind: str = "vol") -> dict[str, np.ndarray]:
        """Boolean masks over ``[segment.start, segment.stop)`` for each regime."""
        labels = self.vol if kind == "vol" else self.trend
        names = VOL_REGIMES if kind == "vol" else TREND_REGIMES
        window = labels[segment.start : segment.stop]
        return {name: window == i for i, name in enumerate(names)}


def label_regimes(
    data: MarketData,
    train: Segment,
    *,
    window: int = 144,
    quantiles: tuple[float, float] = (0.33, 0.67),
) -> RegimeLabels:
    """Trailing-vol and trailing-momentum terciles, thresholds fitted on ``train``."""
    logret = np.zeros(len(data), dtype=np.float64)
    logret[1:] = np.diff(np.log(data.close))
    vol = np.nan_to_num(rolling_std(logret, window), nan=0.0)
    # Trailing momentum over the same window, so the two labels describe the same
    # amount of history.
    mom = np.zeros(len(data), dtype=np.float64)
    if window < len(data):
        mom[window:] = np.log(data.close[window:] / data.close[:-window])

    fit = slice(train.start, train.stop)
    vlo, vhi = np.quantile(vol[fit], quantiles)
    mlo, mhi = np.quantile(mom[fit], quantiles)
    vol_labels = np.digitize(vol, [vlo, vhi]).astype(np.int64)
    trend_labels = np.digitize(mom, [mlo, mhi]).astype(np.int64)
    return RegimeLabels(
        vol=vol_labels,
        trend=trend_labels,
        vol_thresholds=(float(vlo), float(vhi)),
        trend_thresholds=(float(mlo), float(mhi)),
        window=window,
    )


def conditional_metrics(
    returns: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    bars_per_year: int = 52_560,
) -> dict[str, float]:
    """Annualised Sharpe of ``returns`` restricted to each regime.

    The restricted series is not a tradable path — it skips bars — so only
    scale-free per-bar statistics are reported from it. Computing a drawdown on a
    gappy series would be meaningless, which is why Calmar is absent here.
    """
    scale = float(np.sqrt(bars_per_year))
    out: dict[str, float] = {}
    for name, mask in masks.items():
        m = mask[: returns.size]
        sub = returns[: m.size][m]
        if sub.size < 2:
            out[name] = 0.0
            continue
        sd = float(sub.std(ddof=1))
        out[name] = float(sub.mean() * scale / sd) if sd > 1e-12 else 0.0
        out[f"{name}_bars"] = float(sub.size)
        out[f"{name}_mean_log_return"] = float(sub.mean())
    return out
