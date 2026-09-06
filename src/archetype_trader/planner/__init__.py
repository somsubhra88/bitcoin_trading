"""Hindsight-optimal planner. **Training-only.**

This package sees future returns. It is used to generate Phase-1 demonstrations
and Phase-3 regret targets, both of which happen strictly inside training
segments.

The invariant that keeps this honest is an *import* invariant: no module on the
inference path (``phase1.models``, ``phase2.selector``, ``phase3.adapter``,
``inference``, ``paper``) may import ``archetype_trader.planner``, directly or
transitively. ``tests/archetype/test_leakage.py`` walks the import graph and
fails if that is ever violated, which is stronger than a runtime flag someone
could pass by accident.
"""

from __future__ import annotations

from archetype_trader.planner.dp import (
    PlanResult,
    brute_force_optimal,
    dp_optimal_actions,
    planning_returns,
)

#: Read by the leakage test as a machine-checkable marker of intent.
TRAINING_ONLY = True

__all__ = [
    "TRAINING_ONLY",
    "PlanResult",
    "brute_force_optimal",
    "dp_optimal_actions",
    "planning_returns",
]
