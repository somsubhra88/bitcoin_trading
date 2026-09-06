"""Reference strategies the three-phase model is measured against.

Nothing in here imports the planner or any hindsight module: a baseline that
could see the future would make the comparison table meaningless in the one
direction that flatters nobody.
"""

from __future__ import annotations

from archetype_trader.baselines.rl import (
    BarActorCritic,
    BarAlgo,
    BarQNetwork,
    BarRLResult,
    BarTrunk,
    ContinuousCritic,
    ReplayBuffer,
    SACModules,
    SquashedGaussianActor,
    run_bar_policy,
    train_bar_baseline,
    train_bar_dqn,
    train_bar_ppo,
    train_bar_sac,
)
from archetype_trader.baselines.rules import (
    SIGNALS,
    BaselineRun,
    assert_causal_signal,
    buy_and_hold_signal,
    cash_signal,
    ema_cross_signal,
    macd_signal,
    run_rule_baseline,
    run_targets,
    snap_to_grid,
    vol_target_trend_signal,
)
from archetype_trader.baselines.runner import BAR_ALGOS, BaselineBundle, run_baselines
from archetype_trader.baselines.supervised import (
    ForecastMLP,
    ForecastModel,
    fit_forecaster,
    forward_log_return,
    threshold_targets,
    training_rows,
)

__all__ = [
    "BAR_ALGOS",
    "SIGNALS",
    "BarActorCritic",
    "BarAlgo",
    "BarQNetwork",
    "BarRLResult",
    "BarTrunk",
    "BaselineBundle",
    "BaselineRun",
    "ContinuousCritic",
    "ForecastMLP",
    "ForecastModel",
    "ReplayBuffer",
    "SACModules",
    "SquashedGaussianActor",
    "assert_causal_signal",
    "buy_and_hold_signal",
    "cash_signal",
    "ema_cross_signal",
    "fit_forecaster",
    "forward_log_return",
    "macd_signal",
    "run_bar_policy",
    "run_baselines",
    "run_rule_baseline",
    "run_targets",
    "snap_to_grid",
    "threshold_targets",
    "train_bar_baseline",
    "train_bar_dqn",
    "train_bar_ppo",
    "train_bar_sac",
    "training_rows",
    "vol_target_trend_signal",
]
