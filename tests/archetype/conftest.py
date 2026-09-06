"""Fixtures for the ArchetypeTrader suite.

Two design choices worth stating.

**The heavy fixtures are session-scoped.** Training a three-phase stack costs tens
of seconds even at smoke sizes, and a dozen tests need the *same* trained stack to
say anything about parity or determinism. Re-training per test would both slow the
suite by an order of magnitude and weaken it: parity between two independently
trained stacks is a different, weaker claim than parity between two views of one.

**The config here is deliberately tiny and deliberately honest.** ``n_bars=3000``
with a 4-code book and 4 Phase-1 epochs is far too small to learn anything; that
is the point. These tests assert *plumbing* invariants — accounting identities,
causality, idempotency, determinism — which must hold regardless of whether the
model learned. No test in this package asserts that the strategy makes money, and
none should: the synthetic generator's signal-to-noise ratio is a fixture
parameter, so any such assertion would be testing the generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from archetype_trader.config import ExperimentConfig

torch = pytest.importorskip("torch", reason="ArchetypeTrader needs the `research` extra")
pytest.importorskip("gymnasium", reason="ArchetypeTrader needs the `research` extra")

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
#: Smallest config that still exercises every branch: perp mode (so the {-1,0,+1}
#: grid, funding and liquidation checks are live), ``position_levels=2`` (so the
#: grid is not symmetric-trivial), and one fold (so ``fold.train`` boundaries are
#: unambiguous in the leakage assertions).
SMOKE_CONFIG: dict[str, Any] = {
    "name": "pytest_archetype",
    "seed": 3,
    "seeds": [0],
    "output_dir": "runs/pytest",
    "data": {"source": "synthetic", "synthetic": {"n_bars": 3000, "seed": 5}},
    "features": {
        "lookback": 24,
        "return_windows": [1, 3, 6],
        "vol_windows": [12, 36],
        "ma_windows": [12, 36],
        "momentum_windows": [6, 24],
        "macd": [6, 12, 4],
        "downside_vol_window": 12,
    },
    "splits": {"n_folds": 1, "train_bars": 1400, "val_bars": 400, "test_bars": 500},
    "env": {"mode": "perp", "leverage": 2.0, "position_levels": 2, "max_episode_bars": 600},
    "shield": {"max_leverage": 2.0, "max_gross_exposure": 2.0, "max_net_exposure": 2.0},
    "planner": {"horizon": 24},
    # 12 epochs over 400 trajectories, not the 4-over-200 that would run faster.
    # At 4 epochs the Phase-1 decoder is behaviourally degenerate (``code_gain``
    # goes *negative*) and the Phase-2 selector answers by choosing the explicit
    # cash archetype for the entire test segment. Every parity assertion then
    # compares two all-zero series and passes for the wrong reason — a flat run
    # has no fills, no fees and a constant equity curve, so it cannot detect a
    # mispriced fill or a mis-timed fee. 12/400 is the smallest setting measured
    # to leave cash on this fixture (133 of 480 bars hold a position, across two
    # distinct archetypes); ``test_the_reference_run_actually_trades`` in
    # test_live.py fails if that ever regresses.
    "phase1": {
        "horizon": 24,
        "n_trajectories": 400,
        "codebook_size": 4,
        "code_dim": 8,
        "hidden_dim": 32,
        "epochs": 12,
        "batch_size": 64,
    },
    "phase2": {
        "context_bars": 24,
        "hidden_dim": 32,
        "updates": 4,
        "horizons_per_update": 6,
        "epochs_per_update": 2,
        "minibatch_size": 4,
    },
    "phase3": {"hidden_dim": 32, "epochs": 6, "n_episodes": 60, "batch_size": 32},
    "eval": {"seeds": [0], "bootstrap_samples": 50, "make_plots": False},
    "baselines": {"enabled": []},
}


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    from archetype_trader.config import ExperimentConfig

    return ExperimentConfig.model_validate(SMOKE_CONFIG)


# --------------------------------------------------------------------------- #
# Data / features / splits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bundle:
    """Everything downstream of ``features``, assembled once."""

    cfg: ExperimentConfig
    data: Any
    features: Any
    plan: Any
    fold: Any
    scaler: Any


@pytest.fixture(scope="session")
def bundle(cfg: ExperimentConfig) -> Bundle:
    from archetype_trader.data.synthetic import generate_synthetic
    from archetype_trader.features.pipeline import build_features, fit_scaler
    from archetype_trader.splits import plan_from_config

    data = generate_synthetic(
        cfg.data.synthetic,
        bar_minutes=cfg.data.bar_minutes,
        orderbook_levels=cfg.data.orderbook_levels,
        symbol=cfg.data.symbol,
    )
    features = build_features(data, cfg.features)
    plan = plan_from_config(cfg, len(data), features.valid_from)
    fold = plan.folds[0]
    # Fitted on the training segment only. A scaler fitted on the whole series is
    # the cheapest possible leak and the hardest to notice downstream.
    scaler = fit_scaler(features, fold.train.start, fold.train.stop)
    return Bundle(cfg=cfg, data=data, features=features, plan=plan, fold=fold, scaler=scaler)


@pytest.fixture(scope="session")
def data(bundle: Bundle) -> Any:
    return bundle.data


@pytest.fixture(scope="session")
def features(bundle: Bundle) -> Any:
    return bundle.features


@pytest.fixture(scope="session")
def scaler(bundle: Bundle) -> Any:
    return bundle.scaler


@pytest.fixture(scope="session")
def fold(bundle: Bundle) -> Any:
    return bundle.fold


# --------------------------------------------------------------------------- #
# Trained stack
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def stack(bundle: Bundle) -> Any:
    """A trained Phase-1+2+3 stack. Session-scoped: this is the slow fixture."""
    from archetype_trader.eval.strategies import train_stack
    from archetype_trader.utils.seeding import seed_everything

    seed_everything(bundle.cfg.seed)
    return train_stack(
        bundle.cfg,
        bundle.data,
        bundle.features,
        bundle.scaler,
        bundle.fold,
        fold_index=0,
        seed=0,
        device=DEVICE,
        with_phase2=True,
        with_phase3=True,
    )
