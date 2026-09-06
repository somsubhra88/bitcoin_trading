"""Chronological purged walk-forward splits.

Three properties are enforced, not merely documented:

* **Chronological.** ``train < val < test`` in index order, always.
* **Purged with an embargo.** Consecutive segments are separated by
  ``embargo_bars = max_lookback + horizon (+ extra)``. A training observation's
  label depends on bars up to ``t + horizon``, and its features on bars back to
  ``t - max_lookback``, so anything closer than the embargo shares information
  with the next segment.
* **Non-overlapping test segments.** Concatenating per-fold test results can
  never double-count a bar.

Model selection reads ``val`` only. ``test`` is touched once, by the evaluation
stage, and never by a tuner.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class Segment:
    """Half-open bar range ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise ValueError(f"empty segment [{self.start}:{self.stop})")

    def __len__(self) -> int:
        return self.stop - self.start

    def __contains__(self, index: int) -> bool:
        return self.start <= index < self.stop

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "stop": self.stop, "bars": len(self)}


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold with its embargo already applied."""

    index: int
    train: Segment
    val: Segment
    test: Segment
    embargo_bars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "embargo_bars": self.embargo_bars,
            "train": self.train.as_dict(),
            "val": self.val.as_dict(),
            "test": self.test.as_dict(),
        }

    def check(self) -> None:
        """Assert the purge/embargo invariants. Raises on violation."""
        gap_tv = self.val.start - self.train.stop
        gap_vt = self.test.start - self.val.stop
        if gap_tv < self.embargo_bars:
            raise AssertionError(
                f"fold {self.index}: train->val gap {gap_tv} < embargo {self.embargo_bars}"
            )
        if gap_vt < self.embargo_bars:
            raise AssertionError(
                f"fold {self.index}: val->test gap {gap_vt} < embargo {self.embargo_bars}"
            )
        if not (self.train.stop <= self.val.start < self.val.stop <= self.test.start):
            raise AssertionError(f"fold {self.index}: segments are not chronological")


@dataclass(frozen=True)
class WalkForwardPlan:
    folds: tuple[Fold, ...]
    scheme: str
    embargo_bars: int
    n_bars: int
    valid_from: int

    def __iter__(self) -> Iterator[Fold]:
        return iter(self.folds)

    def __len__(self) -> int:
        return len(self.folds)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [f.as_dict() for f in self.folds]

    def check(self) -> None:
        """Validate every fold plus cross-fold test disjointness."""
        for fold in self.folds:
            fold.check()
            if fold.train.start < self.valid_from:
                raise AssertionError(
                    f"fold {fold.index}: train starts at {fold.train.start}, "
                    f"inside the feature warm-up region (valid_from={self.valid_from})"
                )
            if fold.test.stop > self.n_bars:
                raise AssertionError(
                    f"fold {fold.index}: test stops at {fold.test.stop} > {self.n_bars} bars"
                )
        for a, b in zip(self.folds, self.folds[1:], strict=False):
            if b.test.start < a.test.stop:
                raise AssertionError(
                    f"test segments of folds {a.index} and {b.index} overlap "
                    f"([{a.test.start}:{a.test.stop}) vs [{b.test.start}:{b.test.stop}))"
                )


def make_walk_forward(
    n_bars: int,
    *,
    valid_from: int,
    embargo_bars: int,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    n_folds: int,
    scheme: Literal["rolling", "anchored"] = "rolling",
) -> WalkForwardPlan:
    """Build a purged walk-forward plan.

    ``rolling`` slides a fixed-width training window forward; ``anchored`` keeps
    the training start pinned at ``valid_from`` and grows the window. Folds are
    stepped by ``test_bars`` so test segments tile the timeline without overlap.

    Raises when the series is too short for even one fold, with the exact
    shortfall — a silent reduction to zero folds is how leakage bugs hide.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if valid_from < 0 or valid_from >= n_bars:
        raise ValueError(f"valid_from={valid_from} outside [0, {n_bars})")

    span = train_bars + embargo_bars + val_bars + embargo_bars + test_bars
    needed = valid_from + span + (n_folds - 1) * test_bars
    if needed > n_bars:
        raise ValueError(
            f"series of {n_bars} bars is too short: need {needed} for {n_folds} fold(s) "
            f"(valid_from={valid_from}, train={train_bars}, val={val_bars}, test={test_bars}, "
            f"embargo={embargo_bars} applied twice). Shortfall: {needed - n_bars} bars."
        )

    folds: list[Fold] = []
    for k in range(n_folds):
        offset = valid_from + k * test_bars
        train_start = valid_from if scheme == "anchored" else offset
        train_stop = offset + train_bars
        val_start = train_stop + embargo_bars
        val_stop = val_start + val_bars
        test_start = val_stop + embargo_bars
        test_stop = test_start + test_bars
        fold = Fold(
            index=k,
            train=Segment(train_start, train_stop),
            val=Segment(val_start, val_stop),
            test=Segment(test_start, test_stop),
            embargo_bars=embargo_bars,
        )
        fold.check()
        folds.append(fold)

    plan = WalkForwardPlan(
        folds=tuple(folds),
        scheme=scheme,
        embargo_bars=embargo_bars,
        n_bars=n_bars,
        valid_from=valid_from,
    )
    plan.check()
    return plan


def plan_from_config(cfg: Any, n_bars: int, valid_from: int) -> WalkForwardPlan:
    """Build the plan implied by an :class:`ExperimentConfig`."""
    return make_walk_forward(
        n_bars,
        valid_from=valid_from,
        embargo_bars=cfg.embargo_bars,
        train_bars=cfg.splits.train_bars,
        val_bars=cfg.splits.val_bars,
        test_bars=cfg.splits.test_bars,
        n_folds=cfg.splits.n_folds,
        scheme=cfg.splits.scheme,
    )


def sample_starts(
    segment: Segment,
    *,
    horizon: int,
    lookback: int,
    n_samples: int,
    rng: np.random.Generator,
    replace: bool = False,
) -> np.ndarray:
    """Sample trajectory start indices fully contained in ``segment``.

    A start ``s`` is admissible when ``s - lookback + 1 >= segment.start`` (the
    observation window fits) and ``s + horizon < segment.stop`` (the trajectory
    and its final realised return fit). This is what keeps Phase-1 demonstrations
    from reaching across a fold boundary.
    """
    lo = segment.start + lookback - 1
    hi = segment.stop - horizon - 1
    if hi < lo:
        raise ValueError(
            f"segment [{segment.start}:{segment.stop}) cannot hold a trajectory with "
            f"lookback={lookback} and horizon={horizon}"
        )
    admissible = np.arange(lo, hi + 1)
    if not replace and n_samples > admissible.shape[0]:
        # Fall back to the full enumeration rather than silently drawing
        # duplicates the caller did not ask for.
        return admissible
    return rng.choice(admissible, size=n_samples, replace=replace)
