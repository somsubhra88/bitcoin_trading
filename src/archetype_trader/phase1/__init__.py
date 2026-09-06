"""Phase 1: hindsight demonstrations and vector-quantised archetype learning."""

from __future__ import annotations

from archetype_trader.phase1.demonstrations import DemonstrationSet, build_demonstrations
from archetype_trader.phase1.diagnostics import (
    ArchetypeProfile,
    BehaviouralDistinctness,
    CodebookHealth,
    archetype_profiles,
    behavioural_distinctness,
    codebook_health,
    write_archetype_report,
)
from archetype_trader.phase1.models import (
    ArchetypeVQ,
    CausalActionDecoder,
    TrajectoryEncoder,
    VectorQuantizer,
    VQOutput,
)
from archetype_trader.phase1.train import (
    Phase1Result,
    assign_codes,
    code_distinctness,
    load_checkpoint,
    rollout_accuracy,
    save_checkpoint,
    train_phase1,
)

__all__ = [
    "ArchetypeProfile",
    "ArchetypeVQ",
    "BehaviouralDistinctness",
    "CausalActionDecoder",
    "CodebookHealth",
    "DemonstrationSet",
    "Phase1Result",
    "TrajectoryEncoder",
    "VQOutput",
    "VectorQuantizer",
    "archetype_profiles",
    "assign_codes",
    "behavioural_distinctness",
    "build_demonstrations",
    "code_distinctness",
    "codebook_health",
    "load_checkpoint",
    "rollout_accuracy",
    "save_checkpoint",
    "train_phase1",
    "write_archetype_report",
]
