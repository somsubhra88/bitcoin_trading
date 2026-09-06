"""The ablation matrix: which config knob is switched off, and what that isolates.

Every ablation is a *config patch* plus a variant, and the harness trains a fresh
stack per distinct patch. That is expensive and deliberate: turning off Phase-2's
auxiliary label term while reusing a Phase-2 policy trained *with* it would measure
nothing at all.

The reference row is ``full``. An ablation that beats the reference is a finding
about the reference, not a bug in the ablation, and must be reported as such — the
most common outcome in practice is that some penalty term the design argues for
turns out to cost performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Ablation:
    """One row of the matrix."""

    name: str
    variant: str
    #: Dotted config overrides applied before training. Empty for the reference.
    patch: dict[str, Any] = field(default_factory=dict)
    #: Disable the deterministic risk shield at evaluation time only.
    shield_enabled: bool = True
    isolates: str = ""
    #: Phases this row needs trained. Skipping unused phases is a large saving.
    needs_phase2: bool = True
    needs_phase3: bool = True

    def key(self) -> str:
        return self.name


REFERENCE = "full"

ABLATIONS: tuple[Ablation, ...] = (
    Ablation(
        name="full",
        variant="phase1_2_3",
        isolates="reference: all three phases, all penalties, shield on",
    ),
    # -- architecture ------------------------------------------------------- #
    Ablation(
        name="phase1_only",
        variant="phase1_only",
        needs_phase2=False,
        needs_phase3=False,
        isolates="value of the selector: one train-chosen archetype held throughout",
    ),
    Ablation(
        name="phase1_2",
        variant="phase1_2",
        needs_phase3=False,
        isolates="value of the step-level adapter",
    ),
    # -- risk shield -------------------------------------------------------- #
    Ablation(
        name="no_risk_shield",
        variant="phase1_2_3",
        shield_enabled=False,
        isolates="how much of the result is the deterministic shield rather than the policy",
    ),
    Ablation(
        name="shield_no_vol_target",
        variant="phase1_2_3",
        patch={"shield.vol_target_annual": None},
        isolates="volatility targeting inside the shield",
    ),
    # -- Phase-1 objective -------------------------------------------------- #
    Ablation(
        name="p1_two_changes",
        variant="phase1_2_3",
        patch={"planner.max_changes": 2},
        isolates="entry/exit planner ablation vs the single-change paper setting",
    ),
    Ablation(
        name="p1_no_ema_codebook",
        variant="phase1_2_3",
        patch={"phase1.ema_codebook": False},
        isolates="EMA codebook updates vs gradient-only codebook learning",
    ),
    # -- Phase-2 objective -------------------------------------------------- #
    Ablation(
        name="p2_no_aux_label",
        variant="phase1_2_3",
        patch={"phase2.aux_label_coef": 0.0},
        isolates="auxiliary cross-entropy toward the demonstration's archetype",
    ),
    Ablation(
        name="p2_no_cash_archetype",
        variant="phase1_2_3",
        patch={"phase2.include_cash_archetype": False},
        isolates="the explicit abstention action",
    ),
    Ablation(
        name="p2_dqn",
        variant="phase1_2_3",
        patch={"phase2.algo": "recurrent_dqn"},
        isolates="selector algorithm: recurrent DQN instead of recurrent PPO",
    ),
    # -- Phase-3 objective -------------------------------------------------- #
    Ablation(
        name="p3_no_regret",
        variant="phase1_2_3",
        patch={"phase3.regret_coef": 0.0},
        isolates="regret penalty against the hindsight-best override",
    ),
    Ablation(
        name="p3_no_imitation",
        variant="phase1_2_3",
        patch={"phase3.imitation_coef": 0.0},
        isolates="imitation of the training-only hindsight override target",
    ),
    Ablation(
        name="p3_no_cost_penalty",
        variant="phase1_2_3",
        patch={"phase3.cost_coef": 0.0},
        isolates="the adapter's turnover-cost penalty",
    ),
    Ablation(
        name="p3_budget_2",
        variant="phase1_2_3",
        patch={"phase3.override_budget": 2},
        isolates="override budget: two contiguous runs per horizon instead of one",
    ),
)

ABLATIONS_BY_NAME: dict[str, Ablation] = {a.name: a for a in ABLATIONS}

#: The cheap subset used by the smoke config and CI.
SMOKE_ABLATIONS: tuple[str, ...] = ("full", "phase1_only", "phase1_2", "no_risk_shield")


def resolve(names: tuple[str, ...] | None) -> tuple[Ablation, ...]:
    """Look up ablations by name, keeping the declared order and the reference first."""
    if names is None:
        return ABLATIONS
    unknown = [n for n in names if n not in ABLATIONS_BY_NAME]
    if unknown:
        raise KeyError(f"unknown ablations {unknown}; have {sorted(ABLATIONS_BY_NAME)}")
    ordered = [a for a in ABLATIONS if a.name in set(names)]
    return tuple(sorted(ordered, key=lambda a: (a.name != REFERENCE,)))
