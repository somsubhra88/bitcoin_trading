"""Fee-aware dynamic-programming planner over a discrete position grid.

**Training-only. Sees the future by design.** See :mod:`archetype_trader.planner`.

Problem solved exactly
----------------------
Given per-step log returns :math:`r_0..r_{h-1}` for a unit long position, a grid
of admissible positions :math:`\\mathcal{P}`, a per-unit-turnover cost ``c``, and
a budget of at most ``max_changes`` position changes, maximise

.. math::

   \\sum_{t=0}^{h-1} \\Big[ a_t r_t + \\log(1 - c\\,|a_t - a_{t-1}|) \\Big]

with :math:`a_{-1}` the position held on entry. ``max_changes = 1`` reproduces
the "at most one position change" constraint; ``2`` is the entry/exit ablation.

Costs enter in log space (``log1p(-c|Δa|)``), matching the environment's
multiplicative accounting, so the planner is not optimising a different
objective from the one the agent is graded on.

State is ``(previous position, changes used)`` and the recursion is exact — this
is not a greedy heuristic, and :func:`brute_force_optimal` exists so the test
suite can prove it on small instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

_NEG_INF = -1.0e18


@dataclass(frozen=True)
class PlanResult:
    """Optimal plans for a batch of trajectories."""

    #: ``(N, h)`` indices into the position grid.
    action_indices: np.ndarray
    #: ``(N, h)`` the positions themselves.
    positions: np.ndarray
    #: ``(N,)`` optimal net log return over the horizon, costs included.
    value: np.ndarray
    #: ``(N,)`` number of position changes actually used.
    n_changes: np.ndarray
    grid: tuple[float, ...]
    cost_frac: float
    max_changes: int

    def __post_init__(self) -> None:
        if self.action_indices.shape != self.positions.shape:
            raise ValueError("action_indices and positions must have the same shape")
        if self.value.shape[0] != self.action_indices.shape[0]:
            raise ValueError("value and action_indices disagree on batch size")


def planning_returns(open_prices: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    """Per-step returns a plan can actually capture, given next-bar execution.

    A decision made on the close of bar ``s + t`` executes at ``open[s + t + 1]``
    and is unwound no earlier than ``open[s + t + 2]``. The realisable return for
    step ``t`` is therefore ``log(open[s+t+2] / open[s+t+1])`` — *not* the
    close-to-close return of bar ``s + t``, which no one can trade.

    Returns an ``(N, horizon)`` array. Requires ``max(starts) + horizon + 1`` to
    be a valid index.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    starts = np.asarray(starts, dtype=np.int64)
    needed = int(starts.max()) + horizon + 1
    if needed >= open_prices.shape[0]:
        raise IndexError(
            f"planning horizon reaches index {needed} but only {open_prices.shape[0]} bars exist"
        )
    if starts.min() < 0:
        raise IndexError(f"negative start index {int(starts.min())}")
    log_open = np.log(open_prices)
    # offsets[t] indexes open[s + t + 1]; the return for step t is the forward
    # difference at that point.
    idx = starts[:, None] + np.arange(horizon + 1, dtype=np.int64)[None, :] + 1
    return np.diff(log_open[idx], axis=1)


def dp_optimal_actions(
    returns: np.ndarray,
    *,
    grid: tuple[float, ...],
    cost_frac: float,
    max_changes: int,
    init_position: float = 0.0,
) -> PlanResult:
    """Exact DP solution for a batch of trajectories.

    ``returns`` is ``(N, h)`` from :func:`planning_returns`. Vectorised over the
    batch; the ``P * (K+1) * P`` inner loop is tiny (``P <= 3``, ``K <= 8``) and
    keeping it explicit is clearer than a five-dimensional broadcast.
    """
    returns = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    n, h = returns.shape
    pos = np.asarray(grid, dtype=np.float64)
    p_count = pos.shape[0]
    if p_count == 0:
        raise ValueError("position grid is empty")
    if max_changes < 1:
        raise ValueError(f"max_changes must be >= 1, got {max_changes}")
    if not 0.0 <= cost_frac < 1.0:
        raise ValueError(f"cost_frac must be in [0, 1), got {cost_frac}")

    init_idx = int(np.argmin(np.abs(pos - init_position)))
    if abs(pos[init_idx] - init_position) > 1e-12:
        raise ValueError(f"init_position {init_position} is not on the grid {grid}")

    turnover = np.abs(pos[:, None] - pos[None, :])
    if (cost_frac * turnover >= 1.0).any():
        raise ValueError(
            f"cost_frac={cost_frac} makes a full position change cost >= 100% of equity"
        )
    # (prev, new) log-cost of switching.
    switch_cost = np.log1p(-cost_frac * turnover)

    k_max = max_changes
    # V[n, p, k] = optimal value from the current step onward.
    value = np.zeros((n, p_count, k_max + 1), dtype=np.float64)
    choice = np.empty((h, n, p_count, k_max + 1), dtype=np.int8)

    for t in range(h - 1, -1, -1):
        step = pos[None, :] * returns[:, t : t + 1]  # (n, P) hold-value per new position
        new_value = np.empty_like(value)
        cand = np.empty((n, p_count), dtype=np.float64)
        for p in range(p_count):
            for k in range(k_max + 1):
                for j in range(p_count):
                    k_next = k + (1 if j != p else 0)
                    if k_next > k_max:
                        cand[:, j] = _NEG_INF
                    else:
                        cand[:, j] = switch_cost[p, j] + step[:, j] + value[:, j, k_next]
                best = np.argmax(cand, axis=1)
                choice[t, :, p, k] = best.astype(np.int8)
                new_value[:, p, k] = cand[np.arange(n), best]
        value = new_value

    # Forward pass: replay the stored argmax to recover the action sequence.
    rows = np.arange(n)
    prev = np.full(n, init_idx, dtype=np.int64)
    used = np.zeros(n, dtype=np.int64)
    actions = np.empty((n, h), dtype=np.int64)
    for t in range(h):
        j = choice[t, rows, prev, used].astype(np.int64)
        actions[:, t] = j
        used = used + (j != prev)
        prev = j

    total = value[rows, init_idx, 0]
    return PlanResult(
        action_indices=actions,
        positions=pos[actions],
        value=total,
        n_changes=used,
        grid=tuple(float(p) for p in pos),
        cost_frac=cost_frac,
        max_changes=max_changes,
    )


def plan_value(
    action_indices: np.ndarray,
    returns: np.ndarray,
    *,
    grid: tuple[float, ...],
    cost_frac: float,
    init_position: float = 0.0,
) -> np.ndarray:
    """Net log return of an arbitrary action sequence under the same objective.

    Used to score the *unmodified* archetype and Phase-3 overrides against the
    planner's optimum on identical terms.
    """
    pos = np.asarray(grid, dtype=np.float64)
    actions = np.atleast_2d(np.asarray(action_indices, dtype=np.int64))
    returns = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    if actions.shape != returns.shape:
        raise ValueError(f"actions {actions.shape} and returns {returns.shape} must match")
    taken = pos[actions]
    prev = np.concatenate(
        [np.full((actions.shape[0], 1), init_position, dtype=np.float64), taken[:, :-1]], axis=1
    )
    hold = (taken * returns).sum(axis=1)
    cost = np.log1p(-cost_frac * np.abs(taken - prev)).sum(axis=1)
    return hold + cost


def count_changes(
    action_indices: np.ndarray, *, grid: tuple[float, ...], init_position: float = 0.0
) -> np.ndarray:
    """Number of position changes in each sequence, counting entry from ``init``."""
    pos = np.asarray(grid, dtype=np.float64)
    actions = np.atleast_2d(np.asarray(action_indices, dtype=np.int64))
    taken = pos[actions]
    prev = np.concatenate(
        [np.full((actions.shape[0], 1), init_position, dtype=np.float64), taken[:, :-1]], axis=1
    )
    return (np.abs(taken - prev) > 1e-12).sum(axis=1)


def brute_force_optimal(
    returns: np.ndarray,
    *,
    grid: tuple[float, ...],
    cost_frac: float,
    max_changes: int,
    init_position: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive reference solution. ``O(P^h)`` — test-only, small ``h``.

    Exists so :func:`dp_optimal_actions` can be *proved* optimal rather than
    assumed. Returns ``(best_actions, best_values)``.
    """
    returns = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    n, h = returns.shape
    p_count = len(grid)
    if p_count**h > 200_000:
        raise ValueError(f"brute force refuses {p_count}^{h} sequences; use a smaller instance")

    all_seqs = np.array(list(product(range(p_count), repeat=h)), dtype=np.int64)
    changes = count_changes(all_seqs, grid=grid, init_position=init_position)
    feasible = all_seqs[changes <= max_changes]

    best_actions = np.empty((n, h), dtype=np.int64)
    best_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        vals = plan_value(
            feasible,
            np.repeat(returns[i : i + 1], feasible.shape[0], axis=0),
            grid=grid,
            cost_frac=cost_frac,
            init_position=init_position,
        )
        k = int(np.argmax(vals))
        best_actions[i] = feasible[k]
        best_values[i] = vals[k]
    return best_actions, best_values


def default_cost_frac(taker_fee_bps: float, half_spread_bps: float) -> float:
    """Per-unit-turnover cost the planner should assume.

    A position change of size ``|Δa|`` pays the fee plus one half-spread on that
    notional; impact is excluded because the planner has no size information and
    over-charging it would bias demonstrations toward inaction.
    """
    return (taker_fee_bps + half_spread_bps) * 1e-4
