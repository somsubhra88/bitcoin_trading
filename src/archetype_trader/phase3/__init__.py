"""Phase 3: bounded step-level adaptation of the selected archetype.

Import boundary that matters: :mod:`archetype_trader.phase3.adapter` is
inference-safe and :mod:`archetype_trader.phase3.hindsight` is training-only.
:mod:`archetype_trader.phase3.train` imports both, so nothing on the deployed
path may import it. This package's ``__init__`` re-exports from all three for
convenience in scripts and tests; the leakage test checks module imports, not
this file.
"""

from __future__ import annotations

from archetype_trader.phase3.adapter import (
    N_ADAPTER_SCALARS,
    AdapterController,
    AdapterDecision,
    OverrideAdapter,
    budget_mask,
    build_step_features,
    load_adapter,
    make_controller,
)
from archetype_trader.phase3.hindsight import (
    OverrideResult,
    adaptation_target,
    best_override,
    count_overrides,
)
from archetype_trader.phase3.train import (
    HorizonDataset,
    Phase3Result,
    RolloutBatch,
    UpdateStats,
    build_horizon_dataset,
    evaluate_adapter,
    rollout_batch,
    save_adapter,
    train_phase3,
)

__all__ = [
    "N_ADAPTER_SCALARS",
    "AdapterController",
    "AdapterDecision",
    "HorizonDataset",
    "OverrideAdapter",
    "OverrideResult",
    "Phase3Result",
    "RolloutBatch",
    "UpdateStats",
    "adaptation_target",
    "best_override",
    "budget_mask",
    "build_horizon_dataset",
    "build_step_features",
    "count_overrides",
    "evaluate_adapter",
    "load_adapter",
    "make_controller",
    "rollout_batch",
    "save_adapter",
    "train_phase3",
]
