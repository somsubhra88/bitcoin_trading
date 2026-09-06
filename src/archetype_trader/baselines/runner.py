"""Run every enabled baseline on one fold through one identical environment.

The whole point of this module is that no baseline gets a bespoke evaluation
path. All of them end at :meth:`TradingEnv.episode_summary`, so the metrics stage
downstream cannot tell a rule from a neural policy — which is the only way a
comparison table means anything.

Train/evaluate discipline: learned baselines fit on ``fold.train`` and are rolled
out on the requested evaluation segment. The rule baselines have nothing to fit,
which is exactly why they are the honest reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from archetype_trader.baselines.rl import (
    BarAlgo,
    BarRLResult,
    run_bar_policy,
    train_bar_baseline,
)
from archetype_trader.baselines.rules import SIGNALS, BaselineRun, run_rule_baseline
from archetype_trader.baselines.supervised import (
    ForecastModel,
    fit_forecaster,
    threshold_targets,
)
from archetype_trader.config import ExperimentConfig
from archetype_trader.data.types import MarketData
from archetype_trader.env.trading_env import TradingEnv
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.splits import Fold, Segment

BAR_ALGOS: tuple[BarAlgo, ...] = ("recurrent_ppo", "dqn", "sac_continuous")


@dataclass
class BaselineBundle:
    """Everything the evaluation stage needs from the baselines on one fold."""

    fold: int
    seed: int
    runs: dict[str, BaselineRun]
    #: Kept so the report can state how much the learned baselines actually saw.
    trained: dict[str, dict[str, Any]]

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for name, run in self.runs.items():
            s = run.summary
            rows.append(
                {
                    "name": name,
                    "final_equity": s["final_equity"],
                    "n_trades": s["n_trades"],
                    "fees": s["fees"],
                    "funding": s["funding"],
                    "mean_position": s.get("mean_position", 0.0),
                    "shield_intervention_rate": s["shield_intervention_rate"],
                    "n_liquidations": s["n_liquidations"],
                    "bankrupt": s["bankrupt"],
                }
            )
        return sorted(rows, key=lambda r: -float(r["final_equity"]))


def _eval_env(
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    cfg: ExperimentConfig,
    segment: Segment,
) -> TradingEnv:
    start = max(segment.start, features.valid_from, features.lookback - 1)
    return TradingEnv(
        data,
        features,
        scaler,
        cfg.env,
        cfg.shield,
        start_index=start,
        end_index=segment.stop,
        bars_per_year=cfg.eval.bars_per_year,
        # One pass over the segment, never truncated early: a baseline cut short
        # would be compared over a different sample than the strategy.
        horizon_hint=cfg.planner.horizon,
    )


def run_baselines(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    fold: Fold,
    segment: Segment,
    *,
    fold_index: int = 0,
    device: torch.device | None = None,
    seed: int = 0,
    names: tuple[str, ...] | None = None,
) -> BaselineBundle:
    """Fit (on ``fold.train``) and evaluate (on ``segment``) every enabled baseline."""
    dev = device or torch.device("cpu")
    wanted = names or cfg.baselines.enabled
    runs: dict[str, BaselineRun] = {}
    trained: dict[str, dict[str, Any]] = {}
    start = max(segment.start, features.valid_from, features.lookback - 1)

    for name in wanted:
        if name in SIGNALS:
            runs[name] = run_rule_baseline(
                name,
                data,
                features,
                scaler,
                cfg.env,
                cfg.shield,
                cfg.baselines,
                cfg.features,
                start_index=start,
                end_index=segment.stop,
            )
        elif name == "supervised_threshold":
            fitted = fit_forecaster(
                data,
                features,
                scaler,
                fold.train,
                cfg.baselines,
                cfg.features,
                horizon=cfg.planner.horizon,
                device=dev,
                seed=seed,
            )
            runs[name] = _run_supervised(cfg, data, features, scaler, segment, fitted, device=dev)
            trained[name] = {
                "train_rows": fitted.train_rows,
                "train_loss": fitted.train_loss,
                "val_loss": fitted.val_loss,
                "threshold": fitted.threshold,
                "train_sign_accuracy": fitted.train_sign_accuracy,
            }
        elif name in BAR_ALGOS:
            res = train_bar_baseline(
                name,  # type: ignore[arg-type]
                data,
                features,
                scaler,
                cfg.env,
                cfg.shield,
                cfg.baselines,
                fold.train,
                device=dev,
                seed=seed,
                bars_per_year=cfg.eval.bars_per_year,
            )
            env = _eval_env(data, features, scaler, cfg, segment)
            summary = run_bar_policy(res, env, device=dev)
            runs[name] = BaselineRun(
                name=name, summary=summary, targets=np.asarray(summary["positions"])
            )
            trained[name] = _bar_meta(res)
        else:
            raise KeyError(f"unknown baseline {name!r}")

    return BaselineBundle(fold=fold_index, seed=seed, runs=runs, trained=trained)


def _run_supervised(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    segment: Segment,
    fitted: ForecastModel,
    *,
    device: torch.device,
) -> BaselineRun:
    from archetype_trader.baselines.rules import run_targets

    targets = threshold_targets(
        fitted,
        features,
        scaler,
        allow_short=cfg.env.mode == "perp",
        device=device,
    )
    env = _eval_env(data, features, scaler, cfg, segment)
    start = max(segment.start, features.valid_from, features.lookback - 1)
    summary = run_targets(env, targets, start_index=start)
    summary["name"] = "supervised_threshold"
    summary["mean_position"] = float(np.mean(summary["positions"])) if summary["n_steps"] else 0.0
    return BaselineRun(name="supervised_threshold", summary=summary, targets=targets)


def _bar_meta(res: BarRLResult) -> dict[str, Any]:
    last = res.history[-1] if res.history else {}
    return {
        "n_env_steps": res.n_env_steps,
        "n_updates": len(res.history),
        "final_train_net_log_return": res.final_train_return,
        **{f"last_{k}": v for k, v in last.items()},
    }
