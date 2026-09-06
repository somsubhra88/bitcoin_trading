"""Phase-1 demonstration sampling. **Training segments only.**

Builds a dataset of ``(observation trajectory, hindsight-optimal action
sequence)`` pairs by sampling start indices inside a training segment and
solving each with the DP planner.

Two containment rules make a demonstration incapable of leaking:

* the observation window of the first step must fit inside the segment, and
* the planner's last realisable return (``open[s+h+1]``) must also fit.

Only ``start`` indices are stored, not materialised windows: 30k trajectories x
72 steps x ~75 features would be ~650 MB in float32, whereas the scaled feature
matrix is a few MB and gathering a batch is a single fancy-index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from archetype_trader.config import ExperimentConfig
from archetype_trader.data.types import MarketData
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.planner.dp import (
    default_cost_frac,
    dp_optimal_actions,
    plan_value,
    planning_returns,
)
from archetype_trader.splits import Segment
from archetype_trader.utils.logging import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class DemonstrationSet:
    """Hindsight-optimal demonstrations over a training segment.

    ``features`` is the *scaled* market matrix for the whole series; ``starts``
    indexes into it. ``obs_batch`` does the gather.
    """

    starts: np.ndarray  # (N,)
    actions: np.ndarray  # (N, h) indices into ``grid``
    returns: np.ndarray  # (N, h) realisable per-step log returns
    values: np.ndarray  # (N,) planner-optimal net log return
    n_changes: np.ndarray  # (N,)
    features: np.ndarray  # (T, F) scaled
    grid: tuple[float, ...]
    horizon: int
    cost_frac: float
    max_changes: int
    segment: tuple[int, int]
    meta: dict[str, Any]

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_actions(self) -> int:
        return len(self.grid)

    def obs_batch(self, rows: np.ndarray) -> np.ndarray:
        """``(B, h, F)`` observations. ``obs[b, t]`` is observable at bar ``s_b + t``."""
        idx = self.starts[rows][:, None] + np.arange(self.horizon)[None, :]
        return self.features[idx]

    def split(self, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Random train/val row split *within* the training segment.

        This is an optimisation split for early stopping, not a model-selection
        split: the real validation fold is a separate, later time range.
        """
        n = len(self)
        perm = rng.permutation(n)
        cut = max(1, int(n * (1.0 - val_fraction)))
        return perm[:cut], perm[cut:]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            starts=self.starts,
            actions=self.actions,
            returns=self.returns,
            values=self.values,
            n_changes=self.n_changes,
            features=self.features.astype(np.float32),
            grid=np.asarray(self.grid),
            horizon=self.horizon,
            cost_frac=self.cost_frac,
            max_changes=self.max_changes,
            segment=np.asarray(self.segment),
        )
        return p


def build_demonstrations(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    segment: Segment,
    *,
    rng: np.random.Generator,
    n_trajectories: int | None = None,
    max_changes: int | None = None,
) -> DemonstrationSet:
    """Sample and solve demonstrations inside ``segment``.

    ``segment`` must be a *training* segment. The function does not and cannot
    verify that on its own, so the caller (``pipeline.run_phase1``) is the single
    place that decides, and the walk-forward plan is what makes it correct.
    """
    horizon = cfg.phase1.horizon
    lookback = features.lookback
    n_traj = n_trajectories if n_trajectories is not None else cfg.phase1.n_trajectories
    changes = max_changes if max_changes is not None else cfg.planner.max_changes

    # A start needs its observation window behind it and horizon+1 bars of
    # tradable future ahead (see planning_returns).
    lo = max(segment.start + lookback - 1, features.valid_from)
    hi = segment.stop - horizon - 2
    if hi < lo:
        raise ValueError(
            f"training segment [{segment.start}:{segment.stop}) is too short for "
            f"horizon={horizon} and lookback={lookback}: admissible starts [{lo}, {hi}]"
        )
    admissible = np.arange(lo, hi + 1, dtype=np.int64)
    if n_traj <= admissible.shape[0]:
        starts = np.sort(rng.choice(admissible, size=n_traj, replace=False))
    else:
        # Fewer distinct starts than requested: use every one and sample the
        # remainder with replacement, so the requested count is still honoured
        # while every admissible start appears at least once.
        extra = rng.choice(admissible, size=n_traj - admissible.shape[0], replace=True)
        starts = np.sort(np.concatenate([admissible, extra]))

    grid = cfg.env.positions
    cost_frac = (
        cfg.planner.cost_bps * 1e-4
        if cfg.planner.cost_bps is not None
        else default_cost_frac(
            cfg.env.costs.taker_fee_bps,
            float(np.median(data.half_spread_frac(cfg.env.costs.fallback_half_spread_bps)) * 1e4),
        )
    )

    returns = planning_returns(data.open, starts, horizon)
    plan = dp_optimal_actions(
        returns, grid=grid, cost_frac=cost_frac, max_changes=changes, init_position=0.0
    )
    flat_value = plan_value(
        np.zeros_like(plan.action_indices), returns, grid=grid, cost_frac=cost_frac
    )
    if (plan.value < flat_value - 1e-9).any():
        # The all-flat sequence is always feasible, so the optimum can never be
        # worse. A violation means the DP is broken.
        raise AssertionError("planner returned a value below the always-flat baseline")

    scaled = scaler.transform(features.values).astype(np.float32)
    # Warm-up rows are NaN before scaling and stay NaN after; blank them so a
    # bug that reaches them fails a finite-check instead of producing NaN grads.
    scaled[: features.valid_from] = 0.0

    meta = {
        "n_trajectories": int(starts.shape[0]),
        "unique_starts": int(np.unique(starts).shape[0]),
        "admissible_starts": int(admissible.shape[0]),
        "mean_optimal_return": float(plan.value.mean()),
        "mean_flat_return": float(flat_value.mean()),
        "mean_changes": float(plan.n_changes.mean()),
        "action_histogram": np.bincount(plan.action_indices.ravel(), minlength=len(grid)).tolist(),
        "cost_frac": cost_frac,
        "max_changes": changes,
    }
    log_event(logger, "phase1.demonstrations", **meta)
    return DemonstrationSet(
        starts=starts,
        actions=plan.action_indices.astype(np.int64),
        returns=returns,
        values=plan.value,
        n_changes=plan.n_changes,
        features=scaled,
        grid=grid,
        horizon=horizon,
        cost_frac=cost_frac,
        max_changes=changes,
        segment=(segment.start, segment.stop),
        meta=meta,
    )
