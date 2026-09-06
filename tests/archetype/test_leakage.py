"""Leakage tests. These are the ones that matter.

Every other test in this package checks that the code does what it says. These
check the single property that, if violated, makes every number the repository
produces meaningless — and violated *silently*, because a look-ahead leak shows up
as excellent performance, not as an error.

Four independent lines of defence, each tested here:

1. **Static** (:func:`test_deployment_path_cannot_import_the_planner`): the modules
   on the deployment path do not, transitively, import the two modules that index
   the future. This is the strongest guarantee available, because it holds without
   executing anything.
2. **Structural** (the segment-boundary tests): the planner is called only on
   ``fold.train``, and the assertion that enforces it actually fires.
3. **Numerical** (:func:`test_features_never_use_a_future_bar` and
   :func:`test_every_shipped_baseline_signal_is_causal`): perturbing the future
   moves no earlier feature or signal value, and the check that proves it is
   itself shown to fire on a deliberately peeking signal.
4. **Provenance** (:func:`test_load_bundle_rejects_moved_boundaries`): a run cannot
   train across boundaries that have moved since ``features`` recorded them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "archetype_trader"

#: The two modules that legitimately index the future. Both are training-only.
#: ``planner.dp`` solves the best action sequence over a *known* price path;
#: ``phase3.hindsight`` does the same for the best override. Neither may be
#: reachable from anything that runs at inference.
FUTURE_INDEXING = {"archetype_trader.planner", "archetype_trader.phase3.hindsight"}

#: Modules that run when a decision is made against data that has not happened
#: yet. `live/` is the whole deployment surface; `phase*/…` entries are the model
#: definitions and the step-level controller, which the agent loads and calls.
DEPLOYMENT_PATH = (
    "archetype_trader.live.agent",
    "archetype_trader.live.broker",
    "archetype_trader.live.runner",
    "archetype_trader.live.safety",
    "archetype_trader.live",
    "archetype_trader.phase1.models",
    "archetype_trader.phase2.selector",
    "archetype_trader.phase3.adapter",
    "archetype_trader.risk.shield",
    "archetype_trader.env.execution",
    "archetype_trader.env.ledger",
    "archetype_trader.features.pipeline",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_of(path: Path) -> set[str]:
    """Every ``archetype_trader.*`` module this file imports, at any indentation.

    Parses rather than greps, so a module name inside a docstring or a comment
    cannot produce a false positive, and a deferred import inside a function
    cannot produce a false negative — a function-local
    ``import archetype_trader.planner`` is exactly as dangerous as a top-level one.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = _module_name(path)
    base = pkg.rsplit(".", 1)[0] if "." in pkg else pkg
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("archetype_trader"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against this module's package
                anchor = base.rsplit(".", node.level - 1)[0] if node.level > 1 else base
                mod = f"{anchor}.{node.module}" if node.module else anchor
            else:
                mod = node.module or ""
            if not mod.startswith("archetype_trader"):
                continue
            found.add(mod)
            # `from archetype_trader.phase3 import hindsight` imports a submodule,
            # not a name, and must be resolved as one.
            found.update(f"{mod}.{a.name}" for a in node.names)
    return found


def _import_graph() -> dict[str, set[str]]:
    return {_module_name(p): _imports_of(p) for p in sorted(SRC.rglob("*.py"))}


def _reachable(graph: dict[str, set[str]], roots: tuple[str, ...]) -> dict[str, list[str]]:
    """Transitive closure with the path that reached each node, for the failure message."""
    paths: dict[str, list[str]] = {}
    stack = [(r, [r]) for r in roots if r in graph]
    while stack:
        mod, path = stack.pop()
        for dep in graph.get(mod, set()):
            # A dependency may name a module or a symbol inside one; keep the
            # longest prefix that is a real module so `…hindsight.best_override`
            # still resolves to `…hindsight`.
            target = dep
            while target and target not in graph:
                target, _, _ = target.rpartition(".")
            if not target or target in paths:
                continue
            paths[target] = [*path, target]
            stack.append((target, paths[target]))
    return paths


def test_deployment_path_cannot_import_the_planner() -> None:
    graph = _import_graph()
    assert graph, "no modules found; the source layout moved"
    reachable = _reachable(graph, DEPLOYMENT_PATH)
    violations = {m: p for m, p in reachable.items() if m in FUTURE_INDEXING}
    assert not violations, (
        "a deployment-path module can reach a future-indexing module:\n"
        + "\n".join(f"  {m}: {' -> '.join(p)}" for m, p in sorted(violations.items()))
    )


def test_the_import_graph_test_can_actually_fail() -> None:
    """A guard that never fires is decoration. Prove this one detects a real edge.

    ``phase3.train`` legitimately imports ``phase3.hindsight`` — it needs the
    hindsight target to compute regret. Rooting the search there must therefore
    produce exactly the violation the deployment-path test asserts is absent.
    """
    graph = _import_graph()
    reachable = _reachable(graph, ("archetype_trader.phase3.train",))
    assert FUTURE_INDEXING & set(reachable), (
        "phase3.train no longer reaches any future-indexing module, so "
        "test_deployment_path_cannot_import_the_planner is now vacuous"
    )


def test_every_declared_deployment_module_exists() -> None:
    """Guard against the list above rotting into a set of typos that vacuously pass."""
    graph = _import_graph()
    missing = [m for m in (*DEPLOYMENT_PATH, *FUTURE_INDEXING) if m not in graph]
    assert not missing, missing


# --------------------------------------------------------------------------- #
# Numerical: perturb the future, nothing earlier may move
# --------------------------------------------------------------------------- #
def test_every_shipped_baseline_signal_is_causal(bundle) -> None:
    from archetype_trader.baselines.rules import SIGNALS, assert_causal_signal

    for name, fn in SIGNALS.items():
        assert_causal_signal(fn, bundle.data, bundle.cfg.baselines, bundle.cfg.features, name=name)


def test_assert_causal_signal_catches_a_peeking_signal(bundle) -> None:
    """The guard must fire on a deliberately forward-looking signal.

    Without this, ``test_every_shipped_baseline_signal_is_causal`` could be passing
    because the check is broken rather than because the signals are clean. The
    peeking signal below is the classic off-by-one — trade bar ``t`` on the return
    *of* bar ``t`` — which is what turns a backtest into a fortune-teller.
    """
    from archetype_trader.baselines.rules import assert_causal_signal

    def peeking(data, cfg, fcfg):
        fwd = np.diff(data.close, prepend=data.close[0])
        return np.sign(fwd)

    with pytest.raises(AssertionError, match="not causal"):
        assert_causal_signal(
            peeking, bundle.data, bundle.cfg.baselines, bundle.cfg.features, name="peeking"
        )


def test_features_never_use_a_future_bar(bundle) -> None:
    """Perturb the future; no feature row strictly before the splice may move.

    Rebuilds the entire feature matrix on a series whose second half has been
    scaled by 1.5, and demands the earlier rows match to within floating-point
    equality. This catches leaks no single-feature check would: a centred rolling
    window, a whole-series normalisation, a backfilled gap, a global quantile.

    Note the boundary is ``[:splice]``, exclusive, one row tighter than the
    ``[:probe + 1]`` that :func:`baselines.rules.assert_causal_signal` uses. The
    two contracts genuinely differ: a *feature* at bar ``t`` is observable at the
    close of ``t`` and may use ``close[t]``, so perturbing ``close[splice]`` is
    allowed to move ``logret_1[splice]``. A *signal* at ``t`` is a position held
    *during* bar ``t`` and may only use closes through ``t-1``.
    """
    from dataclasses import replace

    from archetype_trader.features.pipeline import build_features

    splice = len(bundle.data) // 2
    bumped = replace(
        bundle.data,
        open=bundle.data.open.copy(),
        high=bundle.data.high.copy(),
        low=bundle.data.low.copy(),
        close=bundle.data.close.copy(),
        volume=bundle.data.volume.copy(),
    )
    for arr in (bumped.open, bumped.high, bumped.low, bumped.close, bumped.volume):
        arr[splice:] *= 1.5
    rebuilt = build_features(bumped, bundle.cfg.features)

    base = np.nan_to_num(bundle.features.values[:splice])
    other = np.nan_to_num(rebuilt.values[:splice])
    assert base.shape == other.shape
    bad = np.argwhere(np.abs(base - other) > 1e-12)
    offenders = [(int(r), bundle.features.names[int(c)]) for r, c in bad[:5]]
    assert bad.size == 0, (
        f"{len(bad)} feature cells before bar {splice} changed when bars "
        f">= {splice} were perturbed; first offenders: {offenders}"
    )


# --------------------------------------------------------------------------- #
# Structural: the planner runs on training bars only
# --------------------------------------------------------------------------- #
def test_planner_guard_rejects_a_validation_segment(bundle) -> None:
    from archetype_trader.pipeline import _assert_training_segment

    fold = bundle.fold
    _assert_training_segment(fold.train, fold)  # the legitimate call

    for name in ("val", "test"):
        with pytest.raises(AssertionError, match="training-only planner"):
            _assert_training_segment(getattr(fold, name), fold)


def test_demonstrations_never_reach_the_validation_segment(bundle) -> None:
    """The horizon, not just the start index, must land before ``val.start``.

    A demonstration starting one bar before the embargo still *reads* ``horizon``
    bars forward, so checking only the start index would pass a plan that leaks a
    whole horizon of validation data into the training labels.
    """
    from archetype_trader.phase1.demonstrations import build_demonstrations

    cfg, fold = bundle.cfg, bundle.fold
    demos = build_demonstrations(
        cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        fold.train,
        rng=np.random.default_rng(0),
    )
    assert demos.starts.size > 0
    last_bar_read = int(demos.starts.max()) + cfg.phase1.horizon + 2
    assert last_bar_read < fold.val.start, (
        f"a demonstration reads up to bar {last_bar_read}, but validation starts at "
        f"{fold.val.start}"
    )


def test_every_fold_embargo_covers_lookback_plus_horizon(bundle) -> None:
    cfg, plan = bundle.cfg, bundle.plan
    required = cfg.features.max_lookback + cfg.planner.horizon
    for i, f in enumerate(plan.folds):
        assert f.embargo_bars >= required, (i, f.embargo_bars, required)
        assert f.train.stop + f.embargo_bars <= f.val.start, (i, f)
        assert f.val.stop + f.embargo_bars <= f.test.start, (i, f)
        assert f.train.stop <= f.val.start < f.val.stop <= f.test.start, (i, f)


# --------------------------------------------------------------------------- #
# Provenance: boundaries cannot move underneath a training run
# --------------------------------------------------------------------------- #
def test_load_bundle_rejects_moved_boundaries(tmp_path, cfg) -> None:
    """Change a split parameter after ``features`` ran; training must refuse.

    Without this, editing ``train_bars`` and re-running ``phase1`` would train on
    one set of boundaries while the recorded manifest claimed another — and the
    scaler on disk would have been fitted to the old ones.
    """
    from archetype_trader.pipeline import load_bundle, run_features, run_ingest

    moved = cfg.model_copy(update={"output_dir": str(tmp_path)})
    run_ingest(moved)
    run_features(moved)
    load_bundle(moved, fold_index=0)  # unchanged config loads fine

    shifted = moved.model_copy(
        update={
            "splits": moved.splits.model_copy(update={"train_bars": moved.splits.train_bars - 50})
        }
    )
    with pytest.raises(ValueError, match="no longer match this config"):
        load_bundle(shifted, fold_index=0)


def test_load_bundle_rejects_changed_market_artifact(tmp_path, cfg) -> None:
    """Feature artifacts must not be paired with a revised market file."""
    from dataclasses import replace

    from archetype_trader.pipeline import (
        RunLayout,
        load_bundle,
        load_saved_market_data,
        run_features,
        run_ingest,
        save_market_data,
    )

    moved = cfg.model_copy(update={"output_dir": str(tmp_path)})
    run_ingest(moved)
    run_features(moved)
    layout = RunLayout.for_config(moved)
    market = load_saved_market_data(layout.market)
    changed_close = market.close.copy()
    changed_close[-1] *= 1.01
    save_market_data(replace(market, close=changed_close), layout.market)

    with pytest.raises(ValueError, match="market data recorded by `features`"):
        load_bundle(moved, fold_index=0)
