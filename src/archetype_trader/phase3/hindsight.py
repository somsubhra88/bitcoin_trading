"""Best hindsight adaptation of a base action sequence. **Training-only.**

Phase 3 is allowed to deviate from the archetype's plan a bounded number of
times. To train it against something meaningful we need to know what the *best*
bounded deviation would have been — that is this module, and like
:mod:`archetype_trader.planner` it sees the future by construction.

Definition of the budget
------------------------
An **override** is a maximal contiguous run of steps whose action differs from
the base action. The budget caps the number of such runs, not the number of
differing bars. This is the definition that makes the useful behaviour cheap:
"the archetype says hold, but the trade has gone against us — exit now and stay
flat for the rest of the horizon" costs exactly one override, whereas counting
differing bars would charge it once per remaining bar and the adapter would
learn to never intervene.

State is ``(previous position, overrides used, whether the previous step was
inside an override)`` and the recursion is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from archetype_trader.planner.dp import plan_value

_NEG_INF = -1.0e18

#: Machine-checkable marker used by the leakage test: no module reachable from
#: the deployed path may import this one.
TRAINING_ONLY = True


@dataclass(frozen=True)
class OverrideResult:
    """Best bounded adaptation of a base plan, per trajectory."""

    #: ``(N, h)`` action indices of the best adapted sequence.
    actions: np.ndarray
    #: ``(N,)`` net log return of the adapted sequence.
    value: np.ndarray
    #: ``(N,)`` net log return of the unmodified base sequence.
    base_value: np.ndarray
    #: ``(N,)`` number of override runs used.
    n_overrides: np.ndarray
    budget: int
    grid: tuple[float, ...]
    cost_frac: float

    @property
    def headroom(self) -> np.ndarray:
        """``value - base_value``: how much the best adaptation could have added."""
        return self.value - self.base_value


def count_overrides(actions: np.ndarray, base_actions: np.ndarray) -> np.ndarray:
    """Number of maximal contiguous runs where ``actions`` differs from ``base``."""
    actions = np.atleast_2d(np.asarray(actions, dtype=np.int64))
    base = np.atleast_2d(np.asarray(base_actions, dtype=np.int64))
    if actions.shape != base.shape:
        raise ValueError(f"actions {actions.shape} and base {base.shape} must match")
    diff = actions != base
    starts = diff.copy()
    starts[:, 1:] &= ~diff[:, :-1]
    return starts.sum(axis=1).astype(np.int64)


def best_override(
    base_actions: np.ndarray,
    returns: np.ndarray,
    *,
    grid: tuple[float, ...],
    cost_frac: float,
    budget: int,
    init_position: float = 0.0,
) -> OverrideResult:
    """Exact DP for the best sequence within ``budget`` override runs of ``base``.

    ``base_actions`` and ``returns`` are both ``(N, h)``. Costs enter in log space
    exactly as in :mod:`archetype_trader.planner.dp`, so an override is charged
    the same turnover cost the environment would charge.
    """
    base = np.atleast_2d(np.asarray(base_actions, dtype=np.int64))
    rets = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    if base.shape != rets.shape:
        raise ValueError(f"base_actions {base.shape} and returns {rets.shape} must match")
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    pos = np.asarray(grid, dtype=np.float64)
    n, h = base.shape
    p_count = pos.shape[0]
    if not 0.0 <= cost_frac < 1.0:
        raise ValueError(f"cost_frac must be in [0, 1), got {cost_frac}")

    init_idx = int(np.argmin(np.abs(pos - init_position)))
    if abs(pos[init_idx] - init_position) > 1e-12:
        raise ValueError(f"init_position {init_position} is not on the grid {grid}")

    turnover = np.abs(pos[:, None] - pos[None, :])
    if (cost_frac * turnover >= 1.0).any():
        raise ValueError(f"cost_frac={cost_frac} makes a full position change cost >= 100%")
    switch_cost = np.log1p(-cost_frac * turnover)

    b_max = budget
    # value[n, prev_pos, overrides_used, in_override]
    value = np.zeros((n, p_count, b_max + 1, 2), dtype=np.float64)
    choice = np.empty((h, n, p_count, b_max + 1, 2), dtype=np.int8)
    rows = np.arange(n)

    for t in range(h - 1, -1, -1):
        step = pos[None, :] * rets[:, t : t + 1]  # (n, P)
        is_ovr = base[:, t][:, None] != np.arange(p_count)[None, :]  # (n, P)
        new_value = np.empty_like(value)
        cand = np.empty((n, p_count), dtype=np.float64)
        for p in range(p_count):
            for k in range(b_max + 1):
                for f in (0, 1):
                    for j in range(p_count):
                        ovr = is_ovr[:, j]
                        # A new run only starts when the previous step was not an
                        # override; continuing a run is free.
                        k_next = np.where(ovr & (f == 0), k + 1, k)
                        f_next = ovr.astype(np.int64)
                        feasible = k_next <= b_max
                        k_idx = np.clip(k_next, 0, b_max)
                        v = value[rows, j, k_idx, f_next]
                        cand[:, j] = np.where(
                            feasible, switch_cost[p, j] + step[:, j] + v, _NEG_INF
                        )
                    best = np.argmax(cand, axis=1)
                    choice[t, :, p, k, f] = best.astype(np.int8)
                    new_value[:, p, k, f] = cand[rows, best]
        value = new_value

    actions = np.empty((n, h), dtype=np.int64)
    prev = np.full(n, init_idx, dtype=np.int64)
    used = np.zeros(n, dtype=np.int64)
    in_ovr = np.zeros(n, dtype=np.int64)
    for t in range(h):
        j = choice[t, rows, prev, used, in_ovr].astype(np.int64)
        actions[:, t] = j
        ovr = (j != base[:, t]).astype(np.int64)
        used = used + ((ovr == 1) & (in_ovr == 0)).astype(np.int64)
        in_ovr = ovr
        prev = j

    total = value[rows, init_idx, 0, 0]
    base_val = plan_value(base, rets, grid=grid, cost_frac=cost_frac, init_position=init_position)
    result = OverrideResult(
        actions=actions,
        value=total,
        base_value=base_val,
        n_overrides=count_overrides(actions, base),
        budget=budget,
        grid=tuple(float(x) for x in pos),
        cost_frac=cost_frac,
    )
    if (result.headroom < -1e-9).any():
        # Keeping the base sequence is always feasible within any budget, so the
        # optimum can never be worse. A violation means the DP is broken.
        raise AssertionError("best_override returned a value below the base plan")
    if (result.n_overrides > budget).any():
        raise AssertionError("best_override exceeded the override budget")
    return result


def adaptation_target(
    best_actions: np.ndarray, base_actions: np.ndarray, *, retain_index: int
) -> np.ndarray:
    """Imitation labels for the adapter head.

    The head has ``A + 1`` outputs where the extra class means "retain the base
    action". Encoding retention explicitly (rather than as "predict the same
    index as the base") is what lets the budget be enforced by masking a single
    logit at inference.
    """
    best = np.atleast_2d(np.asarray(best_actions, dtype=np.int64))
    base = np.atleast_2d(np.asarray(base_actions, dtype=np.int64))
    return np.where(best == base, retain_index, best).astype(np.int64)


def brute_force_best_override(
    base_actions: np.ndarray,
    returns: np.ndarray,
    *,
    grid: tuple[float, ...],
    cost_frac: float,
    budget: int,
    init_position: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive reference for :func:`best_override`. ``O(P^h)`` — test-only."""
    from itertools import product

    base = np.atleast_2d(np.asarray(base_actions, dtype=np.int64))
    rets = np.atleast_2d(np.asarray(returns, dtype=np.float64))
    n, h = base.shape
    p_count = len(grid)
    if p_count**h > 200_000:
        raise ValueError(f"brute force refuses {p_count}^{h} sequences")

    all_seqs = np.array(list(product(range(p_count), repeat=h)), dtype=np.int64)
    best_actions = np.empty((n, h), dtype=np.int64)
    best_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        feasible = all_seqs[
            count_overrides(all_seqs, np.repeat(base[i : i + 1], all_seqs.shape[0], axis=0))
            <= budget
        ]
        vals = plan_value(
            feasible,
            np.repeat(rets[i : i + 1], feasible.shape[0], axis=0),
            grid=grid,
            cost_frac=cost_frac,
            init_position=init_position,
        )
        k = int(np.argmax(vals))
        best_actions[i] = feasible[k]
        best_values[i] = vals[k]
    return best_actions, best_values
