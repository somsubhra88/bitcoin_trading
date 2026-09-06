"""Stage orchestration: the functions the CLI commands are thin wrappers around.

Every stage reads its inputs from disk and writes its outputs plus a
:class:`~archetype_trader.utils.provenance.RunManifest` next to them. That is
more IO than an in-process pipeline needs, and it is deliberate:

* **The split boundaries are recorded once, in one file, and re-read.** Stages 3
  to 5 never recompute which bars are training bars; they load
  ``features/splits.json``. A stage that recomputed its own boundaries could
  disagree with the one before it, and disagreeing boundaries are exactly how
  leakage gets in.
* **A stage can be re-run alone.** Phase 2 is the part worth iterating on; being
  able to re-run it against a fixed Phase 1 without re-solving 30,000 dynamic
  programs is the difference between a usable and an unusable loop.
* **Every artifact carries the hash of what produced it.** :func:`_require` fails
  loudly when a downstream stage is handed features built from different data, so
  a stale checkpoint cannot quietly contribute to a reported number.

The one thing to understand about the layout: :func:`run_phase1`,
:func:`run_phase2` and :func:`run_phase3` train **one** fold with **one** seed.
They exist for inspection and iteration. The numbers that get reported come from
:func:`run_evaluate`, which retrains the whole fold x seed x variant matrix
itself — because a reported metric must not depend on whichever checkpoint
happened to be lying in the run directory.

``run_phase1`` is also the single place that decides a segment is a *training*
segment (:func:`~archetype_trader.phase1.demonstrations.build_demonstrations`
cannot check that for itself), and it asserts it against the recorded plan
before the planner is allowed to run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from archetype_trader.config import ExperimentConfig, dump_config
from archetype_trader.data.loaders import load_market_data
from archetype_trader.data.quality import DataQualityReport
from archetype_trader.data.types import MarketData
from archetype_trader.features.pipeline import (
    FeatureBundle,
    FeatureMatrix,
    FeatureScaler,
    build_features,
    fit_scaler,
)
from archetype_trader.splits import Fold, Segment, WalkForwardPlan, plan_from_config
from archetype_trader.utils.logging import get_logger, log_event
from archetype_trader.utils.provenance import RunManifest, sha256_array
from archetype_trader.utils.seeding import resolve_device, seed_everything

logger = get_logger(__name__)

_MARKET_KEYS = ("ts", "open", "high", "low", "close", "volume")
_BOOK_KEYS = ("bid_px", "bid_sz", "ask_px", "ask_sz")


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunLayout:
    """Where every artifact lives. One place, so the CLI help can be accurate."""

    root: Path

    @staticmethod
    def for_config(cfg: ExperimentConfig) -> RunLayout:
        return RunLayout(cfg.run_dir())

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def market(self) -> Path:
        return self.data_dir / "market.npz"

    @property
    def quality(self) -> Path:
        return self.data_dir / "quality.json"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def features(self) -> Path:
        return self.features_dir / "features.npz"

    @property
    def definitions(self) -> Path:
        return self.features_dir / "definitions.json"

    @property
    def splits(self) -> Path:
        return self.features_dir / "splits.json"

    def scaler(self, fold: int) -> Path:
        return self.features_dir / f"scaler_fold{fold}.json"

    def checkpoint(self, phase: int, fold: int, seed: int) -> Path:
        return self.root / f"phase{phase}" / f"fold{fold}_seed{seed}.pt"

    def diagnostics(self, phase: int, fold: int, seed: int) -> Path:
        return self.root / f"phase{phase}" / f"fold{fold}_seed{seed}.json"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def paper_dir(self) -> Path:
        return self.root / "paper"

    def manifest(self, stage: str) -> Path:
        return self.root / "manifests" / f"{stage}.json"

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config.yaml"


def _require(path: Path, stage: str, command: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{stage} needs {path}, which does not exist. Run `{command}` first."
        )
    return path


def _write_manifest(
    layout: RunLayout,
    stage: str,
    cfg: ExperimentConfig,
    *,
    seed: int,
    data_hash: str = "",
    plan: WalkForwardPlan | None = None,
    artifacts: dict[str, Path] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    from archetype_trader.utils.provenance import sha256_file

    boundaries: list[dict[str, Any]] = []
    if plan is not None:
        boundaries = [
            {
                "fold": i,
                "train": [f.train.start, f.train.stop],
                "val": [f.val.start, f.val.stop],
                "test": [f.test.start, f.test.stop],
                "embargo_bars": f.embargo_bars,
            }
            for i, f in enumerate(plan.folds)
        ]
    manifest = RunManifest(
        stage=stage,
        seed=seed,
        config_hash=cfg.hash(),
        config=cfg.model_dump(mode="json"),
        data_hash=data_hash,
        split_boundaries=boundaries,
        artifact_hashes={
            k: sha256_file(v) for k, v in (artifacts or {}).items() if Path(v).exists()
        },
        metrics=metrics or {},
    )
    return manifest.write(layout.manifest(stage))


# --------------------------------------------------------------------------- #
# stage 1: ingest
# --------------------------------------------------------------------------- #
@dataclass
class IngestResult:
    data: MarketData
    report: DataQualityReport
    layout: RunLayout
    paths: dict[str, Path] = field(default_factory=dict)


def save_market_data(data: MarketData, path: Path) -> Path:
    """``.npz`` rather than Parquet: lossless for ``datetime64[ns]`` and the
    ``(T, levels)`` book arrays without a schema round-trip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "ts": data.ts.view(np.int64),
        **{k: getattr(data, k) for k in _MARKET_KEYS[1:]},
    }
    for k in _BOOK_KEYS:
        arr = getattr(data, k)
        if arr is not None:
            arrays[k] = arr
    np.savez_compressed(
        path,
        _meta=np.array(json.dumps({"symbol": data.symbol, "bar_minutes": data.bar_minutes})),
        **arrays,
    )
    return path


def load_saved_market_data(path: Path) -> MarketData:
    with np.load(path, allow_pickle=False) as blob:
        meta = json.loads(str(blob["_meta"]))
        kwargs: dict[str, Any] = {
            "ts": blob["ts"].astype("datetime64[ns]"),
            **{k: blob[k] for k in _MARKET_KEYS[1:]},
        }
        for k in _BOOK_KEYS:
            if k in blob.files:
                kwargs[k] = blob[k]
    return MarketData(**kwargs, symbol=meta["symbol"], bar_minutes=int(meta["bar_minutes"]))


def run_ingest(cfg: ExperimentConfig, *, strict: bool = True) -> IngestResult:
    """Load, normalise and validate the bars; write them plus a quality report."""
    layout = RunLayout.for_config(cfg)
    layout.root.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, layout.config_snapshot)

    data, report = load_market_data(cfg.data, strict=strict)
    paths = {
        "market": save_market_data(data, layout.market),
        "quality": Path(report.write(layout.quality)),
    }
    log_event(
        logger,
        "ingest.done",
        bars=len(data),
        start=report.start,
        end=report.end,
        levels=report.levels,
        gaps=len(report.gaps),
        missing_bars=report.n_gap_bars_missing,
        ok=report.ok,
        warnings=len(report.warnings),
    )
    _write_manifest(
        layout,
        "ingest",
        cfg,
        seed=cfg.seed,
        data_hash=report.data_hash,
        artifacts=paths,
        metrics={
            "n_bars": report.n_bars,
            "duplicates_removed": report.duplicates_removed,
            "n_gap_bars_missing": report.n_gap_bars_missing,
            "ok": report.ok,
        },
    )
    return IngestResult(data=data, report=report, layout=layout, paths=paths)


# --------------------------------------------------------------------------- #
# stage 2: features
# --------------------------------------------------------------------------- #
@dataclass
class FeaturesResult:
    data: MarketData
    features: FeatureMatrix
    plan: WalkForwardPlan
    scalers: list[FeatureScaler]
    layout: RunLayout
    paths: dict[str, Path] = field(default_factory=dict)


def run_features(cfg: ExperimentConfig) -> FeaturesResult:
    """Build features, plan the purged walk-forward, fit one scaler per fold.

    One scaler *per fold*, fitted on that fold's training range only. A single
    scaler shared across folds would carry the later folds' scale — their
    volatility regime, their price level — into the earlier ones, which is
    leakage that no later test can detect.
    """
    layout = RunLayout.for_config(cfg)
    _require(layout.market, "features", "archetype-trader ingest")
    data = load_saved_market_data(layout.market)

    features = build_features(data, cfg.features)
    plan = plan_from_config(cfg, len(data), features.valid_from)
    plan.check()

    layout.features_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        layout.features,
        values=features.values,
        _meta=np.array(
            json.dumps(
                {
                    "names": list(features.names),
                    "valid_from": features.valid_from,
                    "lookback": features.lookback,
                    "max_lookback": features.max_lookback,
                    "definition_hash": features.definition_hash,
                }
            )
        ),
    )
    layout.definitions.write_text(
        json.dumps(
            {
                "definition_hash": features.definition_hash,
                "n_features": features.n_features,
                "names": list(features.names),
                "valid_from": features.valid_from,
                "lookback": features.lookback,
                "max_lookback": features.max_lookback,
                "config": cfg.features.model_dump(mode="json"),
                "data_hash": _data_hash(data),
            },
            indent=2,
        ),
        "utf-8",
    )
    layout.splits.write_text(
        json.dumps(
            {
                "embargo_bars": cfg.embargo_bars,
                "n_bars": len(data),
                "valid_from": features.valid_from,
                "folds": [
                    {
                        "fold": i,
                        "train": [f.train.start, f.train.stop],
                        "val": [f.val.start, f.val.stop],
                        "test": [f.test.start, f.test.stop],
                        "embargo_bars": f.embargo_bars,
                    }
                    for i, f in enumerate(plan.folds)
                ],
            },
            indent=2,
        ),
        "utf-8",
    )

    scalers: list[FeatureScaler] = []
    paths: dict[str, Path] = {"features": layout.features, "splits": layout.splits}
    for i, fold in enumerate(plan.folds):
        sc = fit_scaler(features, fold.train.start, fold.train.stop)
        scalers.append(sc)
        paths[f"scaler_fold{i}"] = sc.save(layout.scaler(i))

    log_event(
        logger,
        "features.done",
        n_features=features.n_features,
        valid_from=features.valid_from,
        max_lookback=features.max_lookback,
        folds=len(plan.folds),
        embargo_bars=cfg.embargo_bars,
    )
    _write_manifest(
        layout,
        "features",
        cfg,
        seed=cfg.seed,
        data_hash=_data_hash(data),
        plan=plan,
        artifacts=paths,
        metrics={
            "n_features": features.n_features,
            "feature_hash": sha256_array(features.values),
            "definition_hash": features.definition_hash,
        },
    )
    return FeaturesResult(
        data=data, features=features, plan=plan, scalers=scalers, layout=layout, paths=paths
    )


def _data_hash(data: MarketData) -> str:
    """Hash every market input, including optional order-book snapshots."""
    return data.data_hash()


def load_bundle(
    cfg: ExperimentConfig, *, fold_index: int
) -> tuple[FeatureBundle, Fold, WalkForwardPlan]:
    """Re-read the feature stage's artifacts for one fold.

    The split boundaries come from ``features/splits.json`` and are *verified*
    against a freshly computed plan rather than trusted: a config edit between
    stages that moved a boundary would otherwise train on bars the recorded plan
    calls validation.
    """
    layout = RunLayout.for_config(cfg)
    _require(layout.features, "this stage", "archetype-trader features")
    _require(layout.splits, "this stage", "archetype-trader features")
    data = load_saved_market_data(_require(layout.market, "this stage", "archetype-trader ingest"))

    with np.load(layout.features, allow_pickle=False) as blob:
        meta = json.loads(str(blob["_meta"]))
        values = blob["values"]
    features = FeatureMatrix(
        values=values,
        names=tuple(meta["names"]),
        valid_from=int(meta["valid_from"]),
        lookback=int(meta["lookback"]),
        max_lookback=int(meta["max_lookback"]),
        definition_hash=str(meta["definition_hash"]),
    )

    recorded = json.loads(layout.splits.read_text())
    definition_path = _require(layout.definitions, "this stage", "archetype-trader features")
    definitions = json.loads(definition_path.read_text())
    recorded_data_hash = definitions.get("data_hash")
    current_data_hash = _data_hash(data)
    if recorded_data_hash != current_data_hash:
        raise ValueError(
            "the market data recorded by `features` no longer matches the current "
            "market artifact. Re-run `archetype-trader ingest` and `features`."
        )
    plan = plan_from_config(cfg, len(data), features.valid_from)
    plan.check()
    live = [
        {
            "fold": i,
            "train": [f.train.start, f.train.stop],
            "val": [f.val.start, f.val.stop],
            "test": [f.test.start, f.test.stop],
            "embargo_bars": f.embargo_bars,
        }
        for i, f in enumerate(plan.folds)
    ]
    if live != recorded["folds"]:
        raise ValueError(
            "the split boundaries recorded by `features` no longer match this config. "
            "Re-run `archetype-trader features` rather than training across a moved "
            f"boundary.\n  recorded: {recorded['folds']}\n  current:  {live}"
        )
    if fold_index >= len(plan.folds):
        raise IndexError(f"fold {fold_index} does not exist ({len(plan.folds)} folds planned)")

    scaler = FeatureScaler.load(
        _require(layout.scaler(fold_index), "this stage", "archetype-trader features")
    )
    return (
        FeatureBundle(data=data, features=features, scaler=scaler),
        plan.folds[fold_index],
        plan,
    )


def _assert_training_segment(segment: Segment, fold: Fold) -> None:
    """The guard that keeps the planner off validation and test bars.

    The planner is the only component in the repository that indexes the future,
    so this is the assertion that matters most. It is cheap and it is checked
    every time a demonstration set or a hindsight dataset is built.
    """
    if (segment.start, segment.stop) != (fold.train.start, fold.train.stop):
        raise AssertionError(
            f"refusing to run the training-only planner on segment {segment}; "
            f"fold.train is {fold.train}. The planner must never see val or test bars."
        )
    if fold.train.stop > fold.val.start:
        raise AssertionError(
            f"train {fold.train} overlaps val {fold.val}; the walk-forward plan is broken"
        )


# --------------------------------------------------------------------------- #
# stage 3: phase 1
# --------------------------------------------------------------------------- #
def run_phase1(
    cfg: ExperimentConfig, *, fold_index: int = 0, seed: int | None = None
) -> dict[str, Any]:
    """Solve training-segment demonstrations and fit the archetype codebook."""
    from archetype_trader.phase1.demonstrations import build_demonstrations
    from archetype_trader.phase1.train import save_checkpoint, train_phase1

    use_seed = cfg.seed if seed is None else seed
    device = resolve_device(cfg.device)
    seed_everything(use_seed)
    bundle, fold, plan = load_bundle(cfg, fold_index=fold_index)
    layout = RunLayout.for_config(cfg)

    _assert_training_segment(fold.train, fold)
    demos = build_demonstrations(
        cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        fold.train,
        rng=np.random.default_rng(use_seed),
    )
    if int(demos.starts.max()) + cfg.phase1.horizon + 2 > fold.val.start:
        raise AssertionError(
            f"a demonstration reaches bar {int(demos.starts.max()) + cfg.phase1.horizon + 2}, "
            f"past val.start={fold.val.start}"
        )

    result = train_phase1(cfg, demos, device=device, seed=use_seed)
    ckpt = save_checkpoint(result, demos, layout.checkpoint(1, fold_index, use_seed), cfg=cfg)

    diag = {
        "fold": fold_index,
        "seed": use_seed,
        "n_demonstrations": int(demos.starts.size),
        "last_demo_bar": int(demos.starts.max()) + cfg.phase1.horizon + 2,
        "val_start": fold.val.start,
        "health": result.health.to_dict(),
        "distinctness": result.distinctness.to_dict(),
        "history": result.history_table(),
        "meta": result.meta,
    }
    layout.diagnostics(1, fold_index, use_seed).parent.mkdir(parents=True, exist_ok=True)
    layout.diagnostics(1, fold_index, use_seed).write_text(json.dumps(diag, indent=2, default=str))

    log_event(
        logger,
        "phase1.done",
        fold=fold_index,
        seed=use_seed,
        perplexity=result.health.perplexity,
        utilization=result.health.utilization,
        collapsed=result.health.collapsed,
        distinct_behaviours=result.distinctness.n_distinct_behaviours,
        code_gain=result.distinctness.code_gain,
        checkpoint=str(ckpt),
    )
    _write_manifest(
        layout,
        f"phase1_fold{fold_index}_seed{use_seed}",
        cfg,
        seed=use_seed,
        data_hash=_data_hash(bundle.data),
        plan=plan,
        artifacts={"checkpoint": ckpt},
        metrics={
            "perplexity": result.health.perplexity,
            "utilization": result.health.utilization,
            "collapsed": result.health.collapsed,
            "code_gain": result.distinctness.code_gain,
        },
    )
    return {"checkpoint": ckpt, "diagnostics": diag}


# --------------------------------------------------------------------------- #
# stage 4: phase 2
# --------------------------------------------------------------------------- #
def run_phase2(
    cfg: ExperimentConfig, *, fold_index: int = 0, seed: int | None = None
) -> dict[str, Any]:
    """Train the archetype selector against the frozen Phase-1 decoder."""
    from archetype_trader.eval.strategies import make_horizon_env
    from archetype_trader.phase1.demonstrations import build_demonstrations
    from archetype_trader.phase1.train import load_checkpoint
    from archetype_trader.phase2.horizon_env import build_scaled_features
    from archetype_trader.phase2.train import label_array, save_selector, train_phase2

    use_seed = cfg.seed if seed is None else seed
    device = resolve_device(cfg.device)
    seed_everything(use_seed)
    bundle, fold, plan = load_bundle(cfg, fold_index=fold_index)
    layout = RunLayout.for_config(cfg)

    p1_path = _require(
        layout.checkpoint(1, fold_index, use_seed),
        "phase2",
        f"archetype-trader phase1 --fold {fold_index} --seed {use_seed}",
    )
    model, blob = load_checkpoint(p1_path, device=device)
    _check_grid(blob, cfg, p1_path)

    scaled = build_scaled_features(bundle.features, bundle.scaler)
    henv = make_horizon_env(
        cfg, bundle.data, bundle.features, bundle.scaler, model, scaled, fold.train, device=device
    )
    # The auxiliary imitation target needs the demonstration labels, which are a
    # training-segment artifact. Re-solving them here rather than caching them in
    # the Phase-1 checkpoint keeps the checkpoint free of planner output.
    _assert_training_segment(fold.train, fold)
    demos = build_demonstrations(
        cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        fold.train,
        rng=np.random.default_rng(use_seed),
    )
    from archetype_trader.phase1.train import assign_codes

    codes = assign_codes(model, demos, device=device, batch_size=cfg.phase1.batch_size)
    labels = label_array(len(bundle.data), demos.starts, codes)

    result = train_phase2(cfg, henv, labels=labels, device=device, seed=use_seed)
    ckpt = save_selector(result, henv, layout.checkpoint(2, fold_index, use_seed), cfg=cfg)

    diag = {
        "fold": fold_index,
        "seed": use_seed,
        "algo": cfg.phase2.algo,
        "best_objective": result.best_objective,
        "history": result.history_table(),
        "meta": result.meta,
    }
    layout.diagnostics(2, fold_index, use_seed).parent.mkdir(parents=True, exist_ok=True)
    layout.diagnostics(2, fold_index, use_seed).write_text(json.dumps(diag, indent=2, default=str))

    last = result.history[-1] if result.history else None
    log_event(
        logger,
        "phase2.done",
        fold=fold_index,
        seed=use_seed,
        algo=cfg.phase2.algo,
        best_objective=result.best_objective,
        switch_rate=None if last is None else last.switch_rate,
        cash_fraction=None if last is None else last.cash_fraction,
        checkpoint=str(ckpt),
    )
    _write_manifest(
        layout,
        f"phase2_fold{fold_index}_seed{use_seed}",
        cfg,
        seed=use_seed,
        data_hash=_data_hash(bundle.data),
        plan=plan,
        artifacts={"checkpoint": ckpt, "phase1": p1_path},
        metrics={"best_objective": result.best_objective, "algo": cfg.phase2.algo},
    )
    return {"checkpoint": ckpt, "diagnostics": diag}


# --------------------------------------------------------------------------- #
# stage 5: phase 3
# --------------------------------------------------------------------------- #
def run_phase3(
    cfg: ExperimentConfig, *, fold_index: int = 0, seed: int | None = None
) -> dict[str, Any]:
    """Train the step-level override adapter against hindsight-optimal overrides."""
    from archetype_trader.phase1.train import load_checkpoint
    from archetype_trader.phase3.train import build_horizon_dataset, save_adapter, train_phase3

    use_seed = cfg.seed if seed is None else seed
    device = resolve_device(cfg.device)
    seed_everything(use_seed)
    bundle, fold, plan = load_bundle(cfg, fold_index=fold_index)
    layout = RunLayout.for_config(cfg)

    p1_path = _require(
        layout.checkpoint(1, fold_index, use_seed),
        "phase3",
        f"archetype-trader phase1 --fold {fold_index} --seed {use_seed}",
    )
    model, blob = load_checkpoint(p1_path, device=device)
    _check_grid(blob, cfg, p1_path)

    _assert_training_segment(fold.train, fold)
    ds = build_horizon_dataset(
        cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        fold.train,
        model,
        rng=np.random.default_rng(use_seed + 1_000),
        device=device,
        include_cash_archetype=cfg.phase2.include_cash_archetype,
    )
    result = train_phase3(cfg, ds, device=device, seed=use_seed)
    ckpt = save_adapter(result, ds, layout.checkpoint(3, fold_index, use_seed), cfg=cfg)

    diag = {
        "fold": fold_index,
        "seed": use_seed,
        "best_val_improvement": result.best_val_improvement,
        "dataset_meta": result.dataset_meta,
        "history": result.history_table(),
        "meta": result.meta,
    }
    layout.diagnostics(3, fold_index, use_seed).parent.mkdir(parents=True, exist_ok=True)
    layout.diagnostics(3, fold_index, use_seed).write_text(json.dumps(diag, indent=2, default=str))

    last = result.history[-1] if result.history else None
    log_event(
        logger,
        "phase3.done",
        fold=fold_index,
        seed=use_seed,
        best_val_improvement=result.best_val_improvement,
        override_rate=None if last is None else last.val_override_rate,
        checkpoint=str(ckpt),
    )
    _write_manifest(
        layout,
        f"phase3_fold{fold_index}_seed{use_seed}",
        cfg,
        seed=use_seed,
        data_hash=_data_hash(bundle.data),
        plan=plan,
        artifacts={"checkpoint": ckpt, "phase1": p1_path},
        metrics={"best_val_improvement": result.best_val_improvement},
    )
    return {"checkpoint": ckpt, "diagnostics": diag}


def _check_grid(blob: dict[str, Any], cfg: ExperimentConfig, path: Path) -> None:
    """A checkpoint trained on a different action grid cannot be reused."""
    grid = blob.get("grid")
    if grid is None:
        return
    want = list(cfg.env.positions)
    if len(grid) != len(want) or any(abs(a - b) > 1e-12 for a, b in zip(grid, want, strict=True)):
        raise ValueError(
            f"{path} was trained on action grid {grid} but this config asks for {want}. "
            "Re-run phase1 or restore env.position_levels/leverage/mode."
        )


# --------------------------------------------------------------------------- #
# stage 6: evaluate
# --------------------------------------------------------------------------- #
def run_evaluate(
    cfg: ExperimentConfig,
    *,
    ablations: tuple[str, ...] | None = None,
    splits: tuple[str, ...] = ("val", "test"),
    seeds: tuple[int, ...] | None = None,
    folds: tuple[int, ...] | None = None,
    with_baselines: bool = True,
    with_stress: bool = True,
    with_bootstrap: bool = True,
    make_plots: bool | None = None,
) -> dict[str, Any]:
    """Run the whole walk-forward matrix and write the report.

    This stage retrains from scratch. It does not read the Phase 1-3 checkpoints,
    on purpose: a reported number must be reproducible from the config and the
    seed alone, not contingent on which checkpoints happen to be on disk.
    """
    from archetype_trader.eval.harness import collect_provenance, evaluate
    from archetype_trader.eval.report import write_report

    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    layout = RunLayout.for_config(cfg)
    _require(layout.features, "evaluate", "archetype-trader features")
    bundle, _fold, plan = load_bundle(cfg, fold_index=0)

    provenance = collect_provenance(cfg, bundle.data, bundle.features, device=device)
    report = evaluate(
        cfg,
        bundle.data,
        bundle.features,
        device=device,
        ablation_names=ablations,
        splits=splits,
        seeds=seeds,
        folds=folds,
        with_baselines=with_baselines,
        with_stress=with_stress,
        with_bootstrap=with_bootstrap,
        provenance=provenance,
    )
    paths = write_report(
        report,
        layout.eval_dir,
        initial_equity=cfg.env.initial_cash,
        title=f"ArchetypeTrader walk-forward evaluation - {cfg.name}",
        make_plots=cfg.eval.make_plots if make_plots is None else make_plots,
    )
    log_event(
        logger,
        "evaluate.done",
        cells=len(report.cells),
        accepted=report.verdict.get("accepted"),
        selected=report.selection.get("chosen"),
        report=str(paths.get("report_md")),
    )
    _write_manifest(
        layout,
        "evaluate",
        cfg,
        seed=cfg.seed,
        data_hash=_data_hash(bundle.data),
        plan=plan,
        artifacts=paths,
        metrics={
            "n_cells": len(report.cells),
            "verdict": report.verdict,
            "selection": report.selection,
        },
    )
    return {"report": report, "paths": paths}


# --------------------------------------------------------------------------- #
# stage 7: paper trading
# --------------------------------------------------------------------------- #
def run_paper(
    cfg: ExperimentConfig,
    *,
    fold_index: int = 0,
    seed: int | None = None,
    split: str = "test",
    max_bars: int | None = None,
    enable_live: bool = False,
    exchange: str = "paper",
    testnet: bool = True,
    confirm_phrase: str | None = None,
    limits: Any = None,
    with_phase3: bool = True,
) -> dict[str, Any]:
    """Replay a held-out segment through the deployment path.

    The pre-flight gate runs first and can refuse. ``enable_live=True`` is
    *requested*, never granted here: :mod:`archetype_trader.live.safety` ships an
    empty exchange allowlist, so a live session is blocked in this repository.
    """
    from archetype_trader.live.agent import build_agent, load_stack
    from archetype_trader.live.broker import PaperBroker
    from archetype_trader.live.runner import PaperRunner
    from archetype_trader.live.safety import LIVE_WARNING, LiveLimits, preflight
    from archetype_trader.phase2.horizon_env import build_scaled_features

    use_seed = cfg.seed if seed is None else seed
    mode = "live" if enable_live else "paper"
    if enable_live:
        print(LIVE_WARNING)
    report = preflight(
        cfg,
        mode=mode,
        enable_live=enable_live,
        exchange=exchange,
        testnet=testnet,
        limits=limits or LiveLimits(),
        confirm_phrase=confirm_phrase,
    )
    for w in report.warnings:
        logger.warning("pre-flight warning: %s", w)
    report.raise_if_blocked()

    device = resolve_device(cfg.device)
    seed_everything(use_seed)
    bundle, fold, plan = load_bundle(cfg, fold_index=fold_index)
    layout = RunLayout.for_config(cfg)

    segment = {"train": fold.train, "val": fold.val, "test": fold.test}[split]
    stack = load_stack(
        cfg=cfg,
        phase1_path=_require(
            layout.checkpoint(1, fold_index, use_seed),
            "paper",
            f"archetype-trader phase1 --fold {fold_index} --seed {use_seed}",
        ),
        phase2_path=_optional(layout.checkpoint(2, fold_index, use_seed)),
        phase3_path=_optional(layout.checkpoint(3, fold_index, use_seed)) if with_phase3 else None,
        device=device,
    )
    agent = build_agent(stack, device=device)

    start = max(
        segment.start,
        bundle.features.valid_from,
        bundle.features.lookback - 1,
        cfg.phase2.context_bars - 1,
    )
    env_cfg = cfg.env.model_copy(update={"max_episode_bars": max(2, segment.stop - start)})
    broker = PaperBroker(
        bundle.data,
        env_cfg,
        start_index=start,
        end_index=segment.stop,
        symbol=cfg.data.symbol,
        bars_per_year=cfg.eval.bars_per_year,
        vol_lookback=cfg.shield.vol_lookback,
    )
    layout.paper_dir.mkdir(parents=True, exist_ok=True)
    runner = PaperRunner(
        agent,
        broker,
        bundle.data,
        build_scaled_features(bundle.features, bundle.scaler),
        env_cfg,
        cfg.shield,
        run_id=f"{cfg.name}-f{fold_index}-s{use_seed}-{split}",
        bars_per_year=cfg.eval.bars_per_year,
        horizon_hint=cfg.phase1.horizon,
        journal_path=layout.paper_dir / f"journal_fold{fold_index}_seed{use_seed}_{split}.jsonl",
        limits=limits,
    )
    result = runner.run(max_bars=max_bars)

    out = layout.paper_dir / f"result_fold{fold_index}_seed{use_seed}_{split}.json"
    out.write_text(
        json.dumps(
            {
                "run_id": runner.run_id,
                "split": split,
                "segment": [segment.start, segment.stop],
                "bars": result.bars,
                "final_equity": result.final_equity,
                "metrics": result.metrics.as_dict(),
                "agent_stats": result.agent_stats,
                "summary": {
                    k: v for k, v in result.summary.items() if not isinstance(v, np.ndarray)
                },
                "preflight": {
                    "mode": report.mode,
                    "approved": report.approved,
                    "warnings": list(report.warnings),
                    "notes": report.notes,
                },
            },
            indent=2,
            default=str,
        ),
        "utf-8",
    )
    log_event(
        logger,
        "paper.done",
        split=split,
        bars=result.bars,
        final_equity=result.final_equity,
        orders=result.metrics.orders_submitted,
        rejections=result.metrics.rejections,
        shield_interventions=result.metrics.shield_interventions,
        reconciliation_breaches=result.metrics.reconciliation_breaches,
        journal=str(result.journal_path),
    )
    _write_manifest(
        layout,
        f"paper_fold{fold_index}_seed{use_seed}_{split}",
        cfg,
        seed=use_seed,
        data_hash=_data_hash(bundle.data),
        plan=plan,
        artifacts={"result": out},
        metrics=result.metrics.as_dict(),
    )
    return {"result": result, "paths": {"result": out, "journal": result.journal_path}}


def _optional(path: Path) -> Path | None:
    return path if path.exists() else None


__all__ = [
    "FeaturesResult",
    "IngestResult",
    "RunLayout",
    "load_bundle",
    "load_saved_market_data",
    "run_evaluate",
    "run_features",
    "run_ingest",
    "run_paper",
    "run_phase1",
    "run_phase2",
    "run_phase3",
    "save_market_data",
]


def _unused_torch_guard() -> None:  # pragma: no cover
    """``torch`` is imported at module scope so a missing extra fails here, at
    import time, rather than three stages into a long run."""
    _ = torch.__version__
