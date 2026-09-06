"""Codebook health and human-readable archetype profiles.

Two distinct jobs:

* :func:`codebook_health` answers "is the representation degenerate?" —
  utilisation, perplexity, dead codes, and action diversity. A ``K=10`` codebook
  that has collapsed onto two codes will still reconstruct actions acceptably,
  so reconstruction loss alone cannot detect it.
* :func:`archetype_profiles` answers "what did each archetype learn?" — return
  distribution, direction, turnover, holding pattern, volatility regime,
  drawdown and a feature profile, so an archetype can be described in words
  rather than by its index.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from archetype_trader.phase1.demonstrations import DemonstrationSet


@dataclass
class CodebookHealth:
    codebook_size: int
    n_assigned: int
    utilization: float
    perplexity: float
    normalized_perplexity: float
    dead_codes: list[int]
    usage_fractions: list[float]
    action_entropy_per_code: list[float]
    mean_action_entropy: float
    collapsed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def codebook_health(
    code_indices: np.ndarray,
    actions: np.ndarray,
    *,
    codebook_size: int,
    n_actions: int,
    min_perplexity: float = 2.0,
) -> CodebookHealth:
    """Assess collapse from assignment counts and per-code action diversity.

    ``code_indices`` ``(N,)``, ``actions`` ``(N, h)``.
    """
    counts = np.bincount(code_indices, minlength=codebook_size).astype(np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("no code assignments to assess")
    probs = counts / total
    nz = probs[probs > 0]
    perplexity = float(np.exp(-(nz * np.log(nz)).sum()))
    dead = np.flatnonzero(counts == 0).tolist()

    # Action entropy within each code: a code that emits a single action for
    # every trajectory carries no strategy, only a constant.
    entropies: list[float] = []
    for k in range(codebook_size):
        rows = code_indices == k
        if not rows.any():
            entropies.append(0.0)
            continue
        hist = np.bincount(actions[rows].ravel(), minlength=n_actions).astype(np.float64)
        p = hist / hist.sum()
        p = p[p > 0]
        entropies.append(float(-(p * np.log(p)).sum()))

    reasons: list[str] = []
    if perplexity < min_perplexity:
        reasons.append(f"perplexity {perplexity:.2f} < {min_perplexity}")
    if len(dead) > codebook_size // 2:
        reasons.append(f"{len(dead)}/{codebook_size} codes unused")
    if float(np.mean(entropies)) < 1e-3:
        reasons.append("archetypes emit a constant action")

    return CodebookHealth(
        codebook_size=codebook_size,
        n_assigned=int(codebook_size - len(dead)),
        utilization=float((counts > 0).mean()),
        perplexity=perplexity,
        normalized_perplexity=perplexity / codebook_size,
        dead_codes=dead,
        usage_fractions=probs.tolist(),
        action_entropy_per_code=entropies,
        mean_action_entropy=float(np.mean(entropies)),
        collapsed=bool(reasons),
        reasons=reasons,
    )


@dataclass
class BehaviouralDistinctness:
    """Does the *decoder* actually behave differently per code?

    Codebook perplexity only measures how the encoder spreads assignments. A
    decoder can ignore the code entirely and infer the action from the
    observation, which looks healthy by every metric above and makes Phase-2
    selection pointless. This measures the thing that matters: run every code
    over the same observations and compare the resulting action sequences.
    """

    codebook_size: int
    n_windows: int
    mean_pairwise_disagreement: float
    max_pairwise_disagreement: float
    #: Codes whose rollout is byte-identical to a lower-indexed code.
    duplicate_codes: list[int]
    n_distinct_behaviours: int
    long_fraction_per_code: list[float]
    #: Accuracy using the encoder's own code minus the best single fixed code.
    code_gain: float
    accuracy_own_code: float
    accuracy_best_fixed_code: float
    best_fixed_code: int
    degenerate: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def behavioural_distinctness(
    rollouts_by_code: np.ndarray,
    reference_actions: np.ndarray,
    own_code_rollout: np.ndarray,
    *,
    positions: np.ndarray,
    min_disagreement: float = 0.02,
    min_code_gain: float = 0.005,
) -> BehaviouralDistinctness:
    """``rollouts_by_code`` ``(K, B, h)`` action indices, one rollout per code.

    ``reference_actions`` ``(B, h)`` is the planner target; ``own_code_rollout``
    ``(B, h)`` is the rollout under the encoder's assigned code. ``positions`` is
    the position grid, used to report each code's long exposure.
    """
    k = int(rollouts_by_code.shape[0])
    pos = np.asarray(positions, dtype=np.float64)
    dis = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            d = float((rollouts_by_code[i] != rollouts_by_code[j]).mean())
            dis[i, j] = dis[j, i] = d

    duplicates: list[int] = []
    for i in range(1, k):
        if any(np.array_equal(rollouts_by_code[i], rollouts_by_code[j]) for j in range(i)):
            duplicates.append(i)

    acc_fixed = np.asarray(
        [float((rollouts_by_code[i] == reference_actions).mean()) for i in range(k)]
    )
    best_fixed = int(acc_fixed.argmax())
    acc_own = float((own_code_rollout == reference_actions).mean())
    gain = acc_own - float(acc_fixed[best_fixed])

    upper = dis[np.triu_indices(k, 1)]
    mean_dis = float(upper.mean()) if upper.size else 0.0
    reasons: list[str] = []
    if mean_dis < min_disagreement:
        reasons.append(f"mean pairwise action disagreement {mean_dis:.3f} < {min_disagreement}")
    if gain < min_code_gain:
        reasons.append(
            f"conditioning on the code adds only {gain:+.4f} accuracy over the best fixed code"
        )
    if len(duplicates) > k // 2:
        reasons.append(f"{len(duplicates)}/{k} codes duplicate another code's behaviour")

    return BehaviouralDistinctness(
        codebook_size=k,
        n_windows=int(rollouts_by_code.shape[1]),
        mean_pairwise_disagreement=mean_dis,
        max_pairwise_disagreement=float(dis.max()) if k > 1 else 0.0,
        duplicate_codes=duplicates,
        n_distinct_behaviours=k - len(duplicates),
        long_fraction_per_code=[float((pos[r] > 0).mean()) for r in rollouts_by_code],
        code_gain=gain,
        accuracy_own_code=acc_own,
        accuracy_best_fixed_code=float(acc_fixed[best_fixed]),
        best_fixed_code=best_fixed,
        degenerate=bool(reasons),
        reasons=reasons,
    )


@dataclass
class ArchetypeProfile:
    """Human-readable description of one archetype."""

    code: int
    n_trajectories: int
    usage_fraction: float
    mean_return: float
    median_return: float
    std_return: float
    p05_return: float
    p95_return: float
    win_rate: float
    mean_direction: float
    long_fraction: float
    short_fraction: float
    flat_fraction: float
    mean_turnover: float
    mean_changes: float
    mean_holding_bars: float
    entry_step_median: float
    mean_max_drawdown: float
    vol_regime_percentile: float
    mean_realized_vol: float
    top_features: list[dict[str, float]]
    representative_starts: list[int]

    def describe(self) -> str:
        """One-line natural-language summary used in the report."""
        if self.flat_fraction > 0.9:
            stance = "stays in cash"
        elif self.long_fraction > 0.6:
            stance = "holds long"
        elif self.short_fraction > 0.6:
            stance = "holds short"
        else:
            stance = "rotates direction"
        regime = (
            "high-volatility"
            if self.vol_regime_percentile > 0.66
            else "low-volatility"
            if self.vol_regime_percentile < 0.33
            else "mid-volatility"
        )
        return (
            f"archetype {self.code}: {stance} in {regime} regimes, "
            f"{self.mean_changes:.1f} position change(s), "
            f"median horizon return {self.median_return * 100:+.2f}%, "
            f"win rate {self.win_rate:.0%}, mean holding {self.mean_holding_bars:.0f} bars"
        )


def archetype_profiles(
    demos: DemonstrationSet,
    code_indices: np.ndarray,
    *,
    codebook_size: int,
    feature_names: tuple[str, ...],
    realized_vol: np.ndarray,
    n_top_features: int = 6,
    n_representatives: int = 5,
) -> list[ArchetypeProfile]:
    """Summarise each archetype over the demonstrations assigned to it.

    ``realized_vol`` is a per-bar trailing volatility series used to place each
    trajectory in a volatility regime.
    """
    positions = np.asarray(demos.grid, dtype=np.float64)[demos.actions]  # (N, h)
    per_step = positions * demos.returns
    cum = np.cumsum(per_step, axis=1)
    horizon_return = demos.values  # planner value already nets costs
    turnover = np.abs(np.diff(positions, axis=1, prepend=0.0)).sum(axis=1)
    holding = (np.abs(positions) > 1e-12).sum(axis=1)
    # First bar with a non-flat position; horizon when the plan never trades.
    nonflat = np.abs(positions) > 1e-12
    entry = np.where(nonflat.any(axis=1), nonflat.argmax(axis=1), demos.horizon)
    running_peak = np.maximum.accumulate(cum, axis=1)
    max_dd = (running_peak - cum).max(axis=1)

    traj_vol = realized_vol[demos.starts]
    vol_rank = _percentile_rank(traj_vol)

    profiles: list[ArchetypeProfile] = []
    global_feature_mean = demos.features.mean(axis=0)
    global_feature_std = demos.features.std(axis=0) + 1e-9

    for k in range(codebook_size):
        rows = np.flatnonzero(code_indices == k)
        if rows.size == 0:
            profiles.append(_empty_profile(k))
            continue
        pos_k = positions[rows]
        ret_k = horizon_return[rows]
        feat = demos.obs_batch(rows).reshape(-1, demos.n_features).mean(axis=0)
        z = (feat - global_feature_mean) / global_feature_std
        order = np.argsort(-np.abs(z))[:n_top_features]
        top = [{"feature": feature_names[i], "z": float(z[i])} for i in order]
        # Representatives: trajectories closest to this code's median return, so
        # the saved examples are typical rather than extreme.
        med = float(np.median(ret_k))
        rep = rows[np.argsort(np.abs(ret_k - med))[:n_representatives]]

        profiles.append(
            ArchetypeProfile(
                code=k,
                n_trajectories=int(rows.size),
                usage_fraction=float(rows.size / len(demos)),
                mean_return=float(ret_k.mean()),
                median_return=med,
                std_return=float(ret_k.std()),
                p05_return=float(np.percentile(ret_k, 5)),
                p95_return=float(np.percentile(ret_k, 95)),
                win_rate=float((ret_k > 0).mean()),
                mean_direction=float(pos_k.mean()),
                long_fraction=float((pos_k > 0).mean()),
                short_fraction=float((pos_k < 0).mean()),
                flat_fraction=float((np.abs(pos_k) < 1e-12).mean()),
                mean_turnover=float(turnover[rows].mean()),
                mean_changes=float(demos.n_changes[rows].mean()),
                mean_holding_bars=float(holding[rows].mean()),
                entry_step_median=float(np.median(entry[rows])),
                mean_max_drawdown=float(max_dd[rows].mean()),
                vol_regime_percentile=float(vol_rank[rows].mean()),
                mean_realized_vol=float(traj_vol[rows].mean()),
                top_features=top,
                representative_starts=[int(demos.starts[i]) for i in rep],
            )
        )
    return profiles


def _empty_profile(code: int) -> ArchetypeProfile:
    return ArchetypeProfile(
        code=code,
        n_trajectories=0,
        usage_fraction=0.0,
        mean_return=float("nan"),
        median_return=float("nan"),
        std_return=float("nan"),
        p05_return=float("nan"),
        p95_return=float("nan"),
        win_rate=float("nan"),
        mean_direction=float("nan"),
        long_fraction=float("nan"),
        short_fraction=float("nan"),
        flat_fraction=float("nan"),
        mean_turnover=float("nan"),
        mean_changes=float("nan"),
        mean_holding_bars=float("nan"),
        entry_step_median=float("nan"),
        mean_max_drawdown=float("nan"),
        vol_regime_percentile=float("nan"),
        mean_realized_vol=float("nan"),
        top_features=[],
        representative_starts=[],
    )


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Rank in [0, 1]; ties share the average rank."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.shape[0], dtype=np.float64)
    return ranks / max(x.shape[0] - 1, 1)


def write_archetype_report(
    profiles: list[ArchetypeProfile],
    health: CodebookHealth,
    path: str | Path,
    distinctness: BehaviouralDistinctness | None = None,
) -> Path:
    """Write a machine-readable JSON plus a human-readable Markdown sibling."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "codebook_health": health.to_dict(),
        "behavioural_distinctness": distinctness.to_dict() if distinctness else None,
        "archetypes": [asdict(x) for x in profiles],
    }
    p.write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# Archetype report",
        "",
        "## Codebook health",
        "",
        f"- codebook size: {health.codebook_size}",
        f"- utilisation: {health.utilization:.1%} ({health.n_assigned} codes assigned)",
        f"- perplexity: {health.perplexity:.2f} (normalised {health.normalized_perplexity:.2f})",
        f"- dead codes: {health.dead_codes or 'none'}",
        f"- mean per-code action entropy: {health.mean_action_entropy:.3f}",
        f"- collapsed: **{health.collapsed}**"
        + (f" — {'; '.join(health.reasons)}" if health.reasons else ""),
        "",
    ]
    if distinctness is not None:
        d = distinctness
        lines += [
            "## Behavioural distinctness (decoder rollouts per code)",
            "",
            f"- mean pairwise action disagreement: {d.mean_pairwise_disagreement:.3f} "
            f"(max {d.max_pairwise_disagreement:.3f})",
            f"- distinct behaviours: {d.n_distinct_behaviours}/{d.codebook_size} "
            f"(duplicates: {d.duplicate_codes or 'none'})",
            f"- accuracy with own code {d.accuracy_own_code:.3f} vs best fixed code "
            f"{d.best_fixed_code} at {d.accuracy_best_fixed_code:.3f} "
            f"(gain {d.code_gain:+.4f})",
            "- long fraction per code: " + ", ".join(f"{x:.2f}" for x in d.long_fraction_per_code),
            f"- degenerate: **{d.degenerate}**"
            + (f" — {'; '.join(d.reasons)}" if d.reasons else ""),
            "",
        ]
    lines += [
        "## Archetypes",
        "",
    ]
    for prof in profiles:
        if prof.n_trajectories == 0:
            lines.append(f"- archetype {prof.code}: **unused**")
            continue
        lines.append(f"- {prof.describe()}")
        lines.append(
            f"    - usage {prof.usage_fraction:.1%}, turnover {prof.mean_turnover:.2f}, "
            f"max DD {prof.mean_max_drawdown:.4f}, entry step {prof.entry_step_median:.0f}"
        )
        if prof.top_features:
            feats = ", ".join(f"{f['feature']} ({f['z']:+.2f} std)" for f in prof.top_features)
            lines.append(f"    - feature profile: {feats}")
    p.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return p
