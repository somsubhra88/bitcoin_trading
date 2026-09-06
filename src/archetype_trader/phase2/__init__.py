"""Phase 2: choose one archetype per horizon with a recurrent discrete policy."""

from __future__ import annotations

from archetype_trader.phase2.horizon_env import (
    HorizonEnv,
    HorizonOutcome,
    build_scaled_features,
)
from archetype_trader.phase2.selector import (
    RecurrentActorCritic,
    RecurrentQNetwork,
    SelectorPolicy,
    SelectorTrunk,
    build_selector,
)
from archetype_trader.phase2.train import (
    Phase2Result,
    label_array,
    load_selector,
    run_selector,
    save_selector,
    train_phase2,
)

__all__ = [
    "HorizonEnv",
    "HorizonOutcome",
    "Phase2Result",
    "RecurrentActorCritic",
    "RecurrentQNetwork",
    "SelectorPolicy",
    "SelectorTrunk",
    "build_scaled_features",
    "build_selector",
    "label_array",
    "load_selector",
    "run_selector",
    "save_selector",
    "train_phase2",
]
