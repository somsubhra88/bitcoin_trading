"""Block and stationary bootstrap confidence intervals for return-series metrics.

An i.i.d. bootstrap on 10-minute returns is wrong: volatility clusters, so
resampling single bars destroys exactly the dependence that drives drawdown and
tail risk, and produces intervals that are far too narrow. Both estimators here
resample *blocks*:

* **Circular block bootstrap** — fixed block length, wrapped at the end so every
  bar has equal probability of being sampled. Simple and adequate for Sharpe.
* **Stationary bootstrap** (Politis & Romano) — geometric block lengths, so the
  resampled series is strictly stationary and the result does not depend on an
  arbitrary block-boundary alignment. This is the default for drawdown-based
  metrics, which are sensitive to where blocks join.

A confidence interval here describes sampling variability of the metric **on this
one segment**. It is not a statement about future performance, and a wide
interval that includes zero is a negative result that must be reported as one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from archetype_trader.eval.metrics import max_drawdown

BootstrapKind = Literal["circular_block", "stationary"]


@dataclass(frozen=True)
class Interval:
    """Point estimate plus a percentile interval and the resampled distribution."""

    metric: str
    point: float
    lo: float
    hi: float
    mean: float
    std: float
    level: float
    n_samples: int
    kind: BootstrapKind
    block_bars: int
    #: Fraction of resamples at or below zero. A one-sided p-value for "no edge".
    frac_non_positive: float

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "metric": self.metric,
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "mean": self.mean,
            "std": self.std,
            "level": self.level,
            "n_samples": self.n_samples,
            "kind": self.kind,
            "block_bars": self.block_bars,
            "frac_non_positive": self.frac_non_positive,
        }

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def circular_block_indices(
    n: int, block: int, rng: np.random.Generator, *, size: int | None = None
) -> np.ndarray:
    """Indices of one circular block-bootstrap resample of length ``size or n``."""
    length = n if size is None else size
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block)
    idx = (starts[:, None] + offsets[None, :]).reshape(-1) % n
    return idx[:length]


def stationary_indices(
    n: int, mean_block: int, rng: np.random.Generator, *, size: int | None = None
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices with geometric block lengths.

    Implemented as a vectorised random walk: each step either continues the
    previous index or jumps to a fresh uniform one with probability
    ``1 / mean_block``. That is equivalent to drawing geometric block lengths and
    is O(n) with no Python loop over blocks.
    """
    length = n if size is None else size
    p = 1.0 / max(1, min(mean_block, n))
    jump = rng.random(length) < p
    jump[0] = True
    fresh = rng.integers(0, n, size=length)
    idx = np.empty(length, dtype=np.int64)
    cur = 0
    # np.where-style carry-forward needs the previous value, so this loop is the
    # honest implementation; length is the number of bars in one segment, not a
    # hot path.
    for i in range(length):
        cur = int(fresh[i]) if jump[i] else (cur + 1) % n
        idx[i] = cur
    return idx


def _resample(n: int, kind: BootstrapKind, block: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "circular_block":
        return circular_block_indices(n, block, rng)
    if kind == "stationary":
        return stationary_indices(n, block, rng)
    raise KeyError(f"unknown bootstrap kind {kind!r}")


def bootstrap_metric(
    returns: np.ndarray,
    fn: Callable[[np.ndarray], float],
    *,
    metric: str,
    n_samples: int = 1_000,
    block_bars: int = 144,
    kind: BootstrapKind = "stationary",
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Interval:
    """Bootstrap ``fn`` over resampled return series."""
    r = np.asarray(returns, dtype=np.float64)
    generator = rng or np.random.default_rng(0)
    point = float(fn(r))
    if r.size < 4:
        return Interval(
            metric=metric,
            point=point,
            lo=point,
            hi=point,
            mean=point,
            std=0.0,
            level=level,
            n_samples=0,
            kind=kind,
            block_bars=block_bars,
            frac_non_positive=float(point <= 0.0),
        )
    draws = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        draws[i] = fn(r[_resample(r.size, kind, block_bars, generator)])
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return Interval(
        metric=metric,
        point=point,
        lo=float(lo),
        hi=float(hi),
        mean=float(draws.mean()),
        std=float(draws.std(ddof=1)),
        level=level,
        n_samples=n_samples,
        kind=kind,
        block_bars=block_bars,
        frac_non_positive=float(np.mean(draws <= 0.0)),
    )


# --------------------------------------------------------------------------- #
# The metric functionals worth bootstrapping
# --------------------------------------------------------------------------- #
def total_log_return(r: np.ndarray) -> float:
    return float(r.sum())


def make_sharpe(bars_per_year: int) -> Callable[[np.ndarray], float]:
    scale = float(np.sqrt(bars_per_year))

    def sharpe(r: np.ndarray) -> float:
        sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
        return float(r.mean() * scale / sd) if sd > 1e-12 else 0.0

    return sharpe


def make_calmar(bars_per_year: int) -> Callable[[np.ndarray], float]:
    """Calmar on a resampled path.

    Drawdown is path-dependent, so the resampled equity curve — not the original
    one — is what gets measured. This is precisely why the stationary bootstrap is
    the default: block joins that break the volatility clustering would flatter
    the drawdown estimate.

    The numerator is the *log* annualised return, matching
    :func:`archetype_trader.eval.metrics.compute_metrics`. The two must agree or the
    interval would not bracket the point estimate it is printed next to.
    """

    def calmar(r: np.ndarray) -> float:
        if r.size == 0:
            return 0.0
        years = r.size / bars_per_year
        ann = float(r.sum() / years) if years > 1e-12 else 0.0
        mdd = max_drawdown(np.exp(np.concatenate([[0.0], np.cumsum(r)])))
        return float(ann / mdd) if mdd > 1e-12 else 0.0

    return calmar


def make_cvar(alpha: float) -> Callable[[np.ndarray], float]:
    def fn(r: np.ndarray) -> float:
        if r.size == 0:
            return 0.0
        q = float(np.quantile(r, alpha))
        tail = r[r <= q]
        return float(-(tail.mean() if tail.size else q))

    return fn


def bootstrap_suite(
    returns: np.ndarray,
    *,
    bars_per_year: int = 52_560,
    cvar_alpha: float = 0.05,
    n_samples: int = 1_000,
    block_bars: int = 144,
    kind: BootstrapKind = "stationary",
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, Interval]:
    """The four intervals the report always shows."""
    generator = rng or np.random.default_rng(0)
    specs: dict[str, Callable[[np.ndarray], float]] = {
        "total_log_return": total_log_return,
        "sharpe": make_sharpe(bars_per_year),
        "calmar": make_calmar(bars_per_year),
        "cvar": make_cvar(cvar_alpha),
    }
    return {
        name: bootstrap_metric(
            returns,
            fn,
            metric=name,
            n_samples=n_samples,
            block_bars=block_bars,
            kind=kind,
            level=level,
            rng=generator,
        )
        for name, fn in specs.items()
    }


def seed_distribution(
    values: list[float] | np.ndarray, *, name: str = "metric"
) -> dict[str, float]:
    """Summarise a metric across seeds.

    The median and the fraction of seeds above zero are the headline numbers, not
    the max: reporting the best seed is the single most common way a backtest
    overstates itself, so the max is recorded but never used for selection.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {f"{name}_n_seeds": 0.0}
    return {
        f"{name}_n_seeds": float(v.size),
        f"{name}_median": float(np.median(v)),
        f"{name}_mean": float(v.mean()),
        f"{name}_std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        f"{name}_min": float(v.min()),
        f"{name}_max": float(v.max()),
        f"{name}_q25": float(np.quantile(v, 0.25)),
        f"{name}_q75": float(np.quantile(v, 0.75)),
        f"{name}_frac_positive": float(np.mean(v > 0.0)),
    }
