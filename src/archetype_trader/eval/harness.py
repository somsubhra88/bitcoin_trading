"""Walk-forward evaluation harness: folds x seeds x variants x baselines.

Discipline enforced here, not left to the caller:

* **The scaler, the demonstrations, the planner, and every model are fitted on
  ``fold.train`` only.** The harness never passes a validation or test segment to
  anything that fits.
* **Hyperparameter selection reads validation only.** The predeclared objective is
  ``cfg.eval.selection_metric``, evaluated as the *median across seeds* on the
  validation folds. Test metrics are computed once, at the end, and are never used
  to choose anything — the harness has no code path that could.
* **Every strategy ends at the same episode summary.** Baselines, variants and
  ablations are all reduced by :func:`compute_metrics`, so the comparison table is
  structurally incapable of giving one of them a different accounting.

The output is a :class:`WalkForwardReport` holding per-(fold, seed, strategy)
metrics, bootstrap intervals, stress sweeps, seed distributions and a robustness
verdict. Writing it to disk is :mod:`archetype_trader.eval.report`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from archetype_trader.baselines import run_baselines
from archetype_trader.config import ExperimentConfig
from archetype_trader.data.types import MarketData
from archetype_trader.eval.ablations import REFERENCE, Ablation, resolve
from archetype_trader.eval.bootstrap import Interval, bootstrap_suite, seed_distribution
from archetype_trader.eval.metrics import Metrics, compute_metrics, cost_attribution, max_drawdown
from archetype_trader.eval.regimes import RegimeLabels, conditional_metrics, label_regimes
from archetype_trader.eval.strategies import TrainedStack, run_variant, train_stack
from archetype_trader.eval.stress import StressPoint, collapse_verdict, stress_env_config
from archetype_trader.features.pipeline import FeatureMatrix, fit_scaler
from archetype_trader.splits import Fold, Segment, plan_from_config
from archetype_trader.utils.logging import get_logger, log_event

logger = get_logger(__name__)

_SELECTION_FIELD = {"calmar": "calmar", "sharpe": "sharpe", "composite": "composite"}


@dataclass
class Cell:
    """One (fold, seed, strategy, split) result."""

    fold: int
    seed: int
    strategy: str
    split: str
    metrics: Metrics
    attribution: dict[str, float] = field(default_factory=dict)
    regime: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict, repr=False)

    def row(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "seed": self.seed,
            "strategy": self.strategy,
            "split": self.split,
            **self.metrics.as_row(),
        }


@dataclass
class WalkForwardReport:
    """Everything the write-up needs, already reduced to scalars and small arrays."""

    config_hash: str
    cells: list[Cell]
    intervals: dict[str, dict[str, Interval]]
    stress: dict[str, dict[str, Any]]
    seed_stats: dict[str, dict[str, float]]
    selection: dict[str, Any]
    verdict: dict[str, Any]
    provenance: dict[str, Any]
    regimes: dict[str, Any] = field(default_factory=dict)
    #: Ablation name -> the patch keys that were already at the patched value.
    #: These rows are byte-identical to the reference by construction and must not
    #: be read as evidence that the knob does not matter. See
    #: :func:`vacuous_ablations`.
    vacuous: dict[str, list[str]] = field(default_factory=dict)
    equity_curves: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    return_series: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def table(self, split: str = "test") -> list[dict[str, Any]]:
        rows = [c.row() for c in self.cells if c.split == split]
        return sorted(rows, key=lambda r: (-float(r["calmar"]), -float(r["sharpe"])))

    def by_strategy(self, split: str, metric: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for c in self.cells:
            if c.split == split:
                out.setdefault(c.strategy, []).append(float(getattr(c.metrics, metric)))
        return out


def composite_objective(m: Metrics) -> float:
    """Weighted objective for ``selection_metric='composite'``.

    Declared here, in code, once, so it cannot be quietly re-weighted after seeing
    results. Net annualised return carries the sign; Sharpe rewards consistency;
    CVaR and turnover are charged. The weights are a judgement call and are printed
    in the report so a reader can disagree with the judgement rather than guess it.

    The return term is the *log* annualised return: the compounded one reaches 1e80
    on a three-day segment and would make every other term rounding error.
    """
    return (
        1.0 * m.annualized_log_return
        + 0.5 * m.sharpe
        - 2.0 * m.cvar * np.sqrt(m.bars_per_year)
        - 0.05 * m.turnover
    )


def selection_value(m: Metrics, metric: str) -> float:
    if metric == "composite":
        return composite_objective(m)
    if metric not in _SELECTION_FIELD:
        raise KeyError(f"unknown selection metric {metric!r}")
    return float(getattr(m, metric))


def _patched(cfg: ExperimentConfig, patch: dict[str, Any]) -> ExperimentConfig:
    """Apply dotted overrides to a validated config by round-tripping through a dict."""
    if not patch:
        return cfg
    from archetype_trader.config import _assign_dotted

    raw = cfg.model_dump(mode="json")
    for dotted, value in patch.items():
        _assign_dotted(raw, dotted, value)
    return ExperimentConfig.model_validate(raw)


def vacuous_ablations(
    cfg: ExperimentConfig, ablations: tuple[Ablation, ...]
) -> dict[str, list[str]]:
    """Ablations whose patch leaves this config unchanged, keyed by name.

    An ablation that patches a field to the value it already holds trains an
    identical stack and emits a row byte-identical to the reference. In a metrics
    table that row *looks* like evidence that the knob does not matter, when in
    fact the knob was never turned. The real case that motivated this:
    ``shield_no_vol_target`` sets ``shield.vol_target_annual`` to ``None``, which
    is already the default, so on any config that has not enabled volatility
    targeting the ablation is arithmetic rather than an experiment.

    Reported rather than raised. The matrix is still correct — it just contains
    fewer independent comparisons than it appears to, and the reader has to be
    told which.
    """
    base = cfg.model_dump(mode="json")
    out: dict[str, list[str]] = {}
    for ab in ablations:
        if not ab.patch:
            continue
        unchanged = [
            dotted
            for dotted, value in ab.patch.items()
            if _dotted_get(base, dotted) == _to_json_scalar(value)
        ]
        if len(unchanged) == len(ab.patch):
            out[ab.name] = unchanged
    return out


def _dotted_get(tree: dict[str, Any], dotted: str) -> Any:
    node: Any = tree
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _to_json_scalar(value: Any) -> Any:
    """Match ``model_dump(mode="json")``'s representation, where tuples are lists."""
    if isinstance(value, tuple):
        return [_to_json_scalar(v) for v in value]
    return value


_MISSING = object()


def _eval_segments(fold: Fold, splits: tuple[str, ...]) -> list[tuple[str, Segment]]:
    lookup = {"train": fold.train, "val": fold.val, "test": fold.test}
    return [(s, lookup[s]) for s in splits if s in lookup]


def evaluate(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    *,
    device: torch.device,
    ablation_names: tuple[str, ...] | None = None,
    splits: tuple[str, ...] = ("val", "test"),
    seeds: tuple[int, ...] | None = None,
    folds: tuple[int, ...] | None = None,
    with_baselines: bool = True,
    with_stress: bool = True,
    with_bootstrap: bool = True,
    provenance: dict[str, Any] | None = None,
) -> WalkForwardReport:
    """Run the whole matrix and reduce it to a :class:`WalkForwardReport`."""
    plan = plan_from_config(cfg, len(data), features.valid_from)
    plan.check()
    # ``eval.seeds`` and not ``cfg.seeds``: the latter is the training-time seed list,
    # and rl.md requires the *evaluation* seed count to be declared separately so a
    # cheap smoke run cannot silently become the evidence for a robustness claim.
    use_seeds = seeds or cfg.eval.seeds
    use_ablations = resolve(ablation_names)
    # Warned here, before the compute is spent, because the answer is knowable from
    # the config alone: these rows will duplicate the reference exactly.
    for name, keys in vacuous_ablations(cfg, use_ablations).items():
        logger.warning(
            "ablation %r is vacuous on this config: %s already hold the patched "
            "value, so the row will be identical to %r and is not evidence about "
            "the knob",
            name,
            keys,
            REFERENCE,
        )
    cells: list[Cell] = []
    equity: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}
    regime_table: dict[str, dict[str, float]] = {}
    regime_meta: dict[str, Any] = {}

    for f_idx, fold in enumerate(plan.folds):
        if folds is not None and f_idx not in folds:
            continue
        # Refit per fold: a scaler shared across folds would leak the later folds'
        # scale into the earlier ones.
        scaler = fit_scaler(features, fold.train.start, fold.train.stop)
        regimes = label_regimes(data, fold.train)
        regime_meta[f"fold{f_idx}"] = {
            "vol_thresholds": regimes.vol_thresholds,
            "trend_thresholds": regimes.trend_thresholds,
            "window": regimes.window,
        }
        for seed in use_seeds:
            for ab in use_ablations:
                acfg = _patched(cfg, ab.patch)
                # Trained fresh per ablation and released at the end of the iteration:
                # reusing a stack across patched configs would measure the wrong thing,
                # and holding every stack would keep N models alive for no reason.
                stack: TrainedStack = train_stack(
                    acfg,
                    data,
                    features,
                    scaler,
                    fold,
                    fold_index=f_idx,
                    seed=seed,
                    device=device,
                    with_phase2=ab.needs_phase2,
                    with_phase3=ab.needs_phase3,
                )
                log_event(
                    logger,
                    "harness.trained",
                    fold=f_idx,
                    seed=seed,
                    ablation=ab.name,
                    **{k: v for k, v in stack.meta.items() if not isinstance(v, list)},
                )
                for split, segment in _eval_segments(fold, splits):
                    summary = run_variant(
                        ab.variant,
                        acfg,
                        data,
                        features,
                        scaler,
                        stack,
                        segment,
                        device=device,
                        shield_enabled=ab.shield_enabled,
                    )
                    cell = _make_cell(acfg, summary, f_idx, seed, ab.name, split, segment, regimes)
                    cells.append(cell)
                    _record_series(equity, returns, cell)
                    if split == "test":
                        # First (fold, seed) wins, matching ``_record_series``: the
                        # regime bars and the plotted equity curve must describe the
                        # same run or the two figures contradict each other.
                        regime_table.setdefault(
                            ab.name,
                            {
                                k: v
                                for k, v in cell.regime.items()
                                if not k.endswith(("_bars", "_mean_log_return"))
                            },
                        )

            if with_baselines:
                bundle = run_baselines(
                    cfg,
                    data,
                    features,
                    scaler,
                    fold,
                    fold.test if "test" in splits else fold.val,
                    fold_index=f_idx,
                    device=device,
                    seed=seed,
                )
                split = "test" if "test" in splits else "val"
                segment = fold.test if split == "test" else fold.val
                for name, run in bundle.runs.items():
                    cell = _make_cell(
                        cfg,
                        run.summary,
                        f_idx,
                        seed,
                        name,
                        split,
                        segment,
                        regimes,
                        extra=bundle.trained.get(name, {}),
                    )
                    cells.append(cell)
                    _record_series(equity, returns, cell)
                    regime_table.setdefault(
                        name,
                        {
                            k: v
                            for k, v in cell.regime.items()
                            if not k.endswith(("_bars", "_mean_log_return"))
                        },
                    )

    # -- selection (validation only) ---------------------------------------- #
    selection = _select(cells, cfg.eval.selection_metric)

    # -- intervals, seeds, stress ------------------------------------------- #
    intervals: dict[str, dict[str, Interval]] = {}
    if with_bootstrap:
        rng = np.random.default_rng(cfg.seed)
        for strategy, series in returns.items():
            if not strategy.endswith("|test"):
                continue
            intervals[strategy.split("|")[0]] = bootstrap_suite(
                series,
                bars_per_year=cfg.eval.bars_per_year,
                cvar_alpha=cfg.eval.cvar_alpha,
                n_samples=cfg.eval.bootstrap_samples,
                block_bars=cfg.eval.bootstrap_block_bars,
                rng=rng,
            )

    seed_stats: dict[str, dict[str, float]] = {}
    for split in splits:
        for metric in ("sharpe", "calmar", "total_return", "max_drawdown"):
            for strategy, values in _group(cells, split, metric).items():
                seed_stats.setdefault(f"{strategy}|{split}", {}).update(
                    seed_distribution(values, name=metric)
                )

    stress: dict[str, dict[str, Any]] = {}
    if with_stress:
        first = 0 if folds is None else int(min(folds))
        stress = _stress_reference(
            cfg,
            data,
            features,
            plan.folds[first],
            fold_index=first,
            seed=use_seeds[0],
            device=device,
            split="test" if "test" in splits else "val",
        )

    verdict = robustness_verdict(cfg, cells, intervals, stress, seed_stats)
    return WalkForwardReport(
        config_hash=cfg.hash(),
        cells=cells,
        intervals=intervals,
        stress=stress,
        seed_stats=seed_stats,
        selection=selection,
        verdict=verdict,
        provenance=dict(provenance or {}),
        regimes={"thresholds": regime_meta, "test_sharpe_by_regime": regime_table},
        vacuous=vacuous_ablations(cfg, use_ablations),
        equity_curves=equity,
        return_series=returns,
    )


def _make_cell(
    cfg: ExperimentConfig,
    summary: dict[str, Any],
    fold: int,
    seed: int,
    strategy: str,
    split: str,
    segment: Segment,
    regimes: RegimeLabels,
    *,
    extra: dict[str, Any] | None = None,
) -> Cell:
    r = np.asarray(summary["net_log_returns"], dtype=np.float64)
    metrics = compute_metrics(
        summary,
        name=strategy,
        bars_per_year=cfg.eval.bars_per_year,
        cvar_alpha=cfg.eval.cvar_alpha,
        extra={
            **{
                k: summary[k]
                for k in (
                    "archetype_usage",
                    "switch_rate",
                    "cash_fraction",
                    "n_overrides",
                    "override_rate",
                    "within_horizon_override_improvement",
                    "fixed_archetype",
                    "shield_reason_counts",
                )
                if k in summary
            },
            **(extra or {}),
        },
    )
    return Cell(
        fold=fold,
        seed=seed,
        strategy=strategy,
        split=split,
        metrics=metrics,
        attribution=cost_attribution(summary),
        regime=conditional_metrics(
            r, regimes.slice_masks(segment, "vol"), bars_per_year=cfg.eval.bars_per_year
        ),
        summary=summary,
    )


def _record_series(
    equity: dict[str, np.ndarray], returns: dict[str, np.ndarray], cell: Cell
) -> None:
    """Keep one representative path per (strategy, split): the first seed and fold.

    The first, not the best. Plotting the best seed's equity curve next to a
    seed-distribution box plot would contradict the box plot on the same page.
    """
    key = f"{cell.strategy}|{cell.split}"
    if key in equity:
        return
    equity[key] = np.asarray(cell.summary["equity_curve"], dtype=np.float64)
    returns[key] = np.asarray(cell.summary["net_log_returns"], dtype=np.float64)


def _group(cells: list[Cell], split: str, metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for c in cells:
        if c.split == split:
            out.setdefault(c.strategy, []).append(float(getattr(c.metrics, metric)))
    return out


def _select(cells: list[Cell], metric: str) -> dict[str, Any]:
    """Pick the winning ablation on **validation** medians across seeds and folds."""
    scores: dict[str, list[float]] = {}
    for c in cells:
        if c.split != "val":
            continue
        scores.setdefault(c.strategy, []).append(selection_value(c.metrics, metric))
    if not scores:
        return {"metric": metric, "chosen": None, "reason": "no validation cells were produced"}
    medians = {k: float(np.median(v)) for k, v in scores.items()}
    chosen = max(medians, key=lambda k: medians[k])
    return {
        "metric": metric,
        "chosen": chosen,
        "median_by_strategy": medians,
        "n_observations": {k: len(v) for k, v in scores.items()},
        "note": "selected on validation folds only; test metrics never entered this choice",
    }


def _stress_reference(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    fold: Fold,
    *,
    fold_index: int,
    seed: int,
    device: torch.device,
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    """Friction sweep for the reference variant on one fold with one seed.

    One fold and one seed on purpose: the sweep multiplies training cost by the
    number of multipliers, and its question — "does the edge survive worse
    execution?" — is answered by the shape of the curve, not by its seed variance,
    which the seed distribution already reports.
    """
    segment = fold.test if split == "test" else fold.val
    scaler = fit_scaler(features, fold.train.start, fold.train.stop)
    stack = train_stack(
        cfg, data, features, scaler, fold, fold_index=fold_index, seed=seed, device=device
    )
    points: list[StressPoint] = []
    for mult in cfg.eval.stress_multipliers:
        env_cfg = stress_env_config(cfg.env, mult)
        summary = run_variant(
            "phase1_2_3",
            cfg,
            data,
            features,
            scaler,
            stack,
            segment,
            device=device,
            env_override=env_cfg,
        )
        points.append(StressPoint(multiplier=float(mult), summary=summary))
    verdict = collapse_verdict(points, initial_equity=cfg.env.initial_cash)
    return {
        REFERENCE: {
            "fold": fold_index,
            "seed": seed,
            "split": split,
            "points": [
                {
                    "multiplier": p.multiplier,
                    "final_equity": p.final_equity,
                    "n_trades": int(p.summary.get("n_trades", 0)),
                    "fees": float(p.summary.get("fees", 0.0)),
                    "funding": float(p.summary.get("funding", 0.0)),
                    "max_drawdown": max_drawdown(
                        np.asarray(p.summary["equity_curve"], dtype=np.float64)
                    ),
                }
                for p in points
            ],
            **verdict,
        }
    }


def robustness_verdict(
    cfg: ExperimentConfig,
    cells: list[Cell],
    intervals: dict[str, dict[str, Interval]],
    stress: dict[str, dict[str, Any]],
    seed_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Reject or accept the reference strategy against predeclared criteria.

    The criteria are fixed in code, checked mechanically, and every failure is
    listed. The design intent is that this function can and sometimes will return
    ``accepted=False`` for the strategy the repo exists to implement; that is the
    honest outcome, not a bug.
    """
    failures: list[str] = []
    checks: dict[str, Any] = {}
    test = [c for c in cells if c.split == "test" and c.strategy == REFERENCE]
    if not test:
        return {
            "accepted": False,
            "failures": ["no test-split results for the reference strategy"],
            "checks": {},
        }

    sharpes = np.asarray([c.metrics.sharpe for c in test], dtype=np.float64)
    returns = np.asarray([c.metrics.total_return for c in test], dtype=np.float64)
    turnovers = np.asarray([c.metrics.turnover for c in test], dtype=np.float64)

    # 1. Not a single lucky seed.
    frac_pos = float(np.mean(returns > 0.0))
    checks["frac_seeds_positive"] = frac_pos
    checks["n_test_observations"] = int(returns.size)
    if returns.size >= 3 and frac_pos < 0.6:
        failures.append(
            f"only {frac_pos:.0%} of (fold, seed) test runs were profitable; "
            "the result depends on which seed was drawn"
        )
    if returns.size < 3:
        failures.append(
            f"only {returns.size} test observation(s); rl.md requires >=10 seeds before "
            "any claim of robustness"
        )

    # 2. Not a single lucky segment.
    by_fold: dict[int, list[float]] = {}
    for c in test:
        by_fold.setdefault(c.fold, []).append(c.metrics.total_return)
    fold_medians = {k: float(np.median(v)) for k, v in by_fold.items()}
    checks["fold_median_total_return"] = fold_medians
    if len(fold_medians) >= 2 and sum(v > 0 for v in fold_medians.values()) < len(fold_medians):
        failures.append(
            "at least one walk-forward fold was unprofitable; the edge is segment-specific"
        )

    # 3. Costs are real and not omitted.
    residuals = [abs(c.attribution.get("residual", 0.0)) for c in test]
    checks["max_abs_cost_residual"] = float(max(residuals)) if residuals else 0.0
    if residuals and max(residuals) > 1e-6 * cfg.env.initial_cash:
        failures.append(
            f"cost attribution does not reconcile (max residual {max(residuals):.3e}); "
            "the net equity curve cannot be trusted"
        )
    if any(c.metrics.fees == 0.0 and c.metrics.n_trades > 0 for c in test):
        failures.append("trades were executed with zero fees; the cost model was bypassed")

    # 4. No impossible fills or broken accounting.
    if any(c.metrics.bankrupt for c in test):
        failures.append("the reference strategy went bankrupt on at least one test run")

    # 5. Turnover is not the whole story.
    checks["median_turnover"] = float(np.median(turnovers))
    if float(np.median(turnovers)) > 500.0:
        failures.append(
            f"median turnover {np.median(turnovers):.0f}x starting equity is implausible to "
            "execute at these sizes"
        )

    # 6. Statistical significance on the test path.
    ref = intervals.get(REFERENCE, {})
    if "sharpe" in ref:
        iv = ref["sharpe"]
        checks["sharpe_ci"] = [iv.lo, iv.hi]
        checks["sharpe_frac_non_positive"] = iv.frac_non_positive
        if iv.lo <= 0.0:
            failures.append(
                f"bootstrap {iv.level:.0%} CI for Sharpe includes zero "
                f"([{iv.lo:.2f}, {iv.hi:.2f}]); the edge is not distinguishable from noise"
            )

    # 7. Survives friction.
    ref_stress = stress.get(REFERENCE, {})
    if ref_stress:
        checks["survives_friction"] = ref_stress.get("survives_friction")
        if not ref_stress.get("survives_friction", False):
            failures.extend(f"friction stress: {r}" for r in ref_stress.get("reasons", ["failed"]))

    # 8. Beats the honest reference strategies.
    bh = [c.metrics for c in cells if c.split == "test" and c.strategy == "buy_and_hold"]
    if bh:
        ref_sharpe = float(np.median(sharpes))
        bh_sharpe = float(np.median([m.sharpe for m in bh]))
        checks["median_sharpe"] = ref_sharpe
        checks["buy_and_hold_median_sharpe"] = bh_sharpe
        if ref_sharpe <= bh_sharpe:
            failures.append(
                f"median test Sharpe {ref_sharpe:.2f} does not exceed buy-and-hold's "
                f"{bh_sharpe:.2f}; the machinery is not earning its complexity"
            )

    seed_key = f"{REFERENCE}|test"
    checks["seed_stats"] = seed_stats.get(seed_key, {})
    return {
        "accepted": not failures,
        "failures": failures,
        "checks": checks,
        "criteria": [
            ">=60% of (fold, seed) test runs profitable, and >=3 observations",
            "every walk-forward fold profitable at the median",
            "cost attribution reconciles and fees are actually charged",
            "no bankruptcy on any test run",
            "median turnover below 500x starting equity",
            "bootstrap Sharpe CI excludes zero",
            "positive net PnL up to 2x friction with no liquidation",
            "median test Sharpe above buy-and-hold",
        ],
    }


def collect_provenance(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Hashes and versions, gathered once so the report can state what it evaluated."""
    from datetime import UTC, datetime

    from archetype_trader.utils.provenance import (
        git_commit,
        package_versions,
        sha256_array,
    )

    versions = package_versions()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config_hash": cfg.hash(),
        "data_hash": sha256_array(
            data.ts.view(np.int64), data.open, data.high, data.low, data.close, data.volume
        ),
        "feature_hash": sha256_array(features.values),
        "n_bars": len(data),
        "n_features": int(features.values.shape[1]),
        "feature_names": list(features.names),
        "data_source": cfg.data.source,
        "bar_minutes": data.bar_minutes,
        "device": str(device),
        "torch_version": versions.get("torch", "unknown"),
        "python_version": versions.get("python", "unknown"),
        "versions": versions,
        "seeds": list(cfg.eval.seeds),
    }


def cells_to_rows(cells: list[Cell]) -> list[dict[str, Any]]:
    return [c.row() for c in cells]


def report_to_json(report: WalkForwardReport) -> dict[str, Any]:
    """JSON-safe view. Arrays are dropped, not truncated — they go to .npz."""
    return {
        "config_hash": report.config_hash,
        "rows": cells_to_rows(report.cells),
        "attribution": [
            {
                "fold": c.fold,
                "seed": c.seed,
                "strategy": c.strategy,
                "split": c.split,
                **c.attribution,
            }
            for c in report.cells
        ],
        "intervals": {
            strategy: {name: iv.as_row() for name, iv in ivs.items()}
            for strategy, ivs in report.intervals.items()
        },
        "stress": report.stress,
        "seed_stats": report.seed_stats,
        "selection": report.selection,
        "verdict": report.verdict,
        "regimes": report.regimes,
        "vacuous_ablations": report.vacuous,
        "provenance": report.provenance,
        "extras": [
            {
                "fold": c.fold,
                "seed": c.seed,
                "strategy": c.strategy,
                "split": c.split,
                **c.metrics.extra,
            }
            for c in report.cells
            if c.metrics.extra
        ],
    }
