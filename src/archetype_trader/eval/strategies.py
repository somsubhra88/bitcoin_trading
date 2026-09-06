"""Build and roll out the ArchetypeTrader variants and their ablations.

One fold, one seed, one training pass produces every variant, because they share
components: Phase 2 needs the Phase-1 decoder, Phase 3 needs both. Re-training per
variant would multiply cost and — worse — mean the Phase-1+2 row and the
Phase-1+2+3 row were built on *different* archetypes, so their difference would no
longer isolate the adapter.

Variants
--------
``phase1_only``
    No selector. The single archetype with the best **training-fold** outcome is
    held for the whole evaluation segment. Selecting it on training data is the
    point: a "best archetype" chosen on the evaluation segment would be a
    look-ahead result dressed up as an ablation.
``phase1_2``
    The Phase-2 selector choosing one archetype per horizon.
``phase1_2_3``
    Selector plus the Phase-3 step-level override adapter.

Ablations toggle individual loss terms and the risk shield. Each returns a
complete episode summary from the same environment, so the report cannot tell them
apart structurally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from archetype_trader.config import ExperimentConfig
from archetype_trader.data.types import MarketData
from archetype_trader.env.trading_env import TradingEnv
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.phase1.demonstrations import build_demonstrations
from archetype_trader.phase1.train import Phase1Result, train_phase1
from archetype_trader.phase2 import (
    HorizonEnv,
    Phase2Result,
    build_scaled_features,
    label_array,
    run_selector,
    train_phase2,
)
from archetype_trader.phase3 import (
    HorizonDataset,
    Phase3Result,
    build_horizon_dataset,
    make_controller,
    train_phase3,
)
from archetype_trader.splits import Fold, Segment
from archetype_trader.utils.logging import get_logger, log_event

logger = get_logger(__name__)

VARIANTS = ("phase1_only", "phase1_2", "phase1_2_3")


@dataclass
class TrainedStack:
    """Everything trained on one fold with one seed, plus the provenance to report it."""

    fold_index: int
    seed: int
    phase1: Phase1Result
    phase2: Phase2Result | None
    phase3: Phase3Result | None
    dataset: HorizonDataset | None
    scaled: np.ndarray
    best_train_archetype: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_archetypes(self) -> int:
        return self.phase1.model.codebook_size + (
            1 if self.meta.get("include_cash_archetype", True) else 0
        )


def _make_env(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    segment: Segment,
    *,
    shield_enabled: bool = True,
    env_override: Any = None,
) -> TradingEnv:
    env_cfg = env_override if env_override is not None else cfg.env
    shield_cfg = cfg.shield if shield_enabled else cfg.shield.model_copy(update={"enabled": False})
    start = max(
        segment.start, features.valid_from, features.lookback - 1, cfg.phase2.context_bars - 1
    )
    # The whole segment in one episode: a truncation mid-way would compare
    # strategies over different samples.
    env_cfg = env_cfg.model_copy(update={"max_episode_bars": max(2, segment.stop - start)})
    return TradingEnv(
        data,
        features,
        scaler,
        env_cfg,
        shield_cfg,
        start_index=start,
        end_index=segment.stop,
        bars_per_year=cfg.eval.bars_per_year,
        horizon_hint=cfg.phase1.horizon,
    )


def make_horizon_env(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    stack_or_model: Any,
    scaled: np.ndarray,
    segment: Segment,
    *,
    device: torch.device,
    override: Any = None,
    shadow_cost_frac: float = 0.0,
    shield_enabled: bool = True,
    env_override: Any = None,
) -> HorizonEnv:
    """Wrap a fresh :class:`TradingEnv` over ``segment`` in a :class:`HorizonEnv`."""
    model = getattr(stack_or_model, "model", stack_or_model)
    env = _make_env(
        cfg,
        data,
        features,
        scaler,
        segment,
        shield_enabled=shield_enabled,
        env_override=env_override,
    )
    return HorizonEnv(
        env,
        model,
        scaled,
        horizon=cfg.phase1.horizon,
        context_bars=cfg.phase2.context_bars,
        include_cash_archetype=cfg.phase2.include_cash_archetype,
        device=device,
        override=override,
        shadow_cost_frac=shadow_cost_frac,
    )


def _fixed_archetype_run(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    phase1: Phase1Result,
    scaled: np.ndarray,
    segment: Segment,
    archetype: int,
    *,
    device: torch.device,
    override: Any = None,
    shadow_cost_frac: float = 0.0,
    shield_enabled: bool = True,
    env_override: Any = None,
) -> dict[str, Any]:
    henv = make_horizon_env(
        cfg,
        data,
        features,
        scaler,
        phase1.model,
        scaled,
        segment,
        device=device,
        override=override,
        shadow_cost_frac=shadow_cost_frac,
        shield_enabled=shield_enabled,
        env_override=env_override,
    )
    henv.reset()
    while True:
        _, _, term, trunc, _ = henv.step(archetype)
        if term or trunc:
            break
    return henv.episode_stats()


def pick_best_train_archetype(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    phase1: Phase1Result,
    scaled: np.ndarray,
    train: Segment,
    *,
    device: torch.device,
) -> tuple[int, list[float]]:
    """Hold each archetype across the *training* segment; return the best index.

    Ranked on training-fold final equity. Deliberately not on validation: the
    Phase-1-only row is meant to be the honest "no selector" ablation, and choosing
    its single archetype on the fold it is scored on would make it a different,
    stronger, and dishonest strategy.
    """
    n = phase1.model.codebook_size + (1 if cfg.phase2.include_cash_archetype else 0)
    equities: list[float] = []
    for k in range(n):
        stats = _fixed_archetype_run(
            cfg, data, features, scaler, phase1, scaled, train, k, device=device
        )
        equities.append(float(stats["final_equity"]))
    best = int(np.argmax(equities))
    log_event(logger, "phase1_only.selected", archetype=best, train_equities=equities)
    return best, equities


def train_stack(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    fold: Fold,
    *,
    fold_index: int,
    seed: int,
    device: torch.device,
    with_phase2: bool = True,
    with_phase3: bool = True,
    phase3_overrides: dict[str, Any] | None = None,
) -> TrainedStack:
    """Train Phase 1 (+2, +3) on ``fold.train``. Nothing here touches val or test."""
    rng = np.random.default_rng(seed)
    demos = build_demonstrations(cfg, data, features, scaler, fold.train, rng=rng)
    p1 = train_phase1(cfg, demos, device=device, seed=seed)
    scaled = build_scaled_features(features, scaler)
    best_arch, train_equities = pick_best_train_archetype(
        cfg, data, features, scaler, p1, scaled, fold.train, device=device
    )

    p2: Phase2Result | None = None
    if with_phase2:
        henv = make_horizon_env(
            cfg, data, features, scaler, p1.model, scaled, fold.train, device=device
        )
        labels = label_array(len(data), demos.starts, p1.code_indices)
        p2 = train_phase2(cfg, henv, labels=labels, device=device, seed=seed)

    p3: Phase3Result | None = None
    ds: HorizonDataset | None = None
    if with_phase3:
        cfg3 = (
            cfg
            if not phase3_overrides
            else cfg.model_copy(update={"phase3": cfg.phase3.model_copy(update=phase3_overrides)})
        )
        ds = build_horizon_dataset(
            cfg3,
            data,
            features,
            scaler,
            fold.train,
            p1.model,
            rng=np.random.default_rng(seed + 1_000),
            device=device,
        )
        p3 = train_phase3(cfg3, ds, device=device, seed=seed)

    return TrainedStack(
        fold_index=fold_index,
        seed=seed,
        phase1=p1,
        phase2=p2,
        phase3=p3,
        dataset=ds,
        scaled=scaled,
        best_train_archetype=best_arch,
        meta={
            "include_cash_archetype": cfg.phase2.include_cash_archetype,
            "train_equities_by_archetype": train_equities,
            "phase1_perplexity": p1.health.perplexity,
            "phase1_utilization": p1.health.utilization,
            "phase1_collapsed": p1.health.collapsed,
            "phase1_distinct_behaviours": p1.distinctness.n_distinct_behaviours,
            "phase1_codebook_size": p1.distinctness.codebook_size,
            "phase2_best_objective": p2.best_objective if p2 else None,
            "phase3_best_val_improvement": p3.best_val_improvement if p3 else None,
            "phase3_overrides": dict(phase3_overrides or {}),
            "n_demonstrations": int(demos.starts.size),
        },
    )


def run_variant(
    variant: str,
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    stack: TrainedStack,
    segment: Segment,
    *,
    device: torch.device,
    shield_enabled: bool = True,
    env_override: Any = None,
) -> dict[str, Any]:
    """Roll one variant over ``segment`` and return its episode summary."""
    cost_frac = stack.dataset.cost_frac if stack.dataset is not None else 0.0
    if variant == "phase1_only":
        summary = _fixed_archetype_run(
            cfg,
            data,
            features,
            scaler,
            stack.phase1,
            stack.scaled,
            segment,
            stack.best_train_archetype,
            device=device,
            shield_enabled=shield_enabled,
            env_override=env_override,
        )
        summary["fixed_archetype"] = stack.best_train_archetype
    elif variant in ("phase1_2", "phase1_2_3"):
        if stack.phase2 is None:
            raise ValueError(f"variant {variant!r} needs a trained Phase-2 selector")
        override = None
        if variant == "phase1_2_3":
            if stack.phase3 is None or stack.dataset is None:
                raise ValueError("variant 'phase1_2_3' needs a trained Phase-3 adapter")
            override = make_controller(
                stack.phase3.model,
                grid=stack.dataset.grid,
                n_archetypes=stack.dataset.n_archetypes,
                budget=stack.dataset.budget,
                horizon=stack.dataset.horizon,
                device=device,
            )
        henv = make_horizon_env(
            cfg,
            data,
            features,
            scaler,
            stack.phase1.model,
            stack.scaled,
            segment,
            device=device,
            override=override,
            shadow_cost_frac=cost_frac,
            shield_enabled=shield_enabled,
            env_override=env_override,
        )
        summary = run_selector(henv, stack.phase2.policy, device=device, greedy=True)
    else:
        raise KeyError(f"unknown variant {variant!r}; have {VARIANTS}")
    summary["name"] = variant
    positions = np.asarray(summary["positions"], dtype=np.float64)
    summary["mean_position"] = float(positions.mean()) if positions.size else 0.0
    return summary
