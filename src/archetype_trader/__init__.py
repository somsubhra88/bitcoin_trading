"""ArchetypeTrader: research-grade archetype-based RL trading for BTC/USDT.

Three-phase architecture:

* **Phase 1** learns a discrete codebook of reusable *strategy archetypes* from
  hindsight-optimal demonstrations produced by a dynamic-programming planner
  (training intervals only).
* **Phase 2** trains a high-level recurrent policy that picks one archetype per
  trading horizon from causal pre-horizon information.
* **Phase 3** trains a constrained adapter that may override the archetype's
  action a bounded number of times per horizon.

Everything downstream of Phase 1 is strictly causal: the planner and all
hindsight labels are gated behind :mod:`archetype_trader.planner` and are
unreachable from the inference path (see ``tests/archetype/test_leakage.py``).

This package is research and paper-trading software. It makes no claim of
profitability, and live execution is disabled unless explicitly enabled.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
