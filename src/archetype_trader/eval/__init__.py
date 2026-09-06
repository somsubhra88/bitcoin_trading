"""Evaluation: metrics, bootstrap intervals, stress tests, ablations, reporting.

The package has one rule that shapes everything in it: **no strategy gets a bespoke
evaluation path.** Baselines, the three ArchetypeTrader variants and every ablation
all produce a :meth:`TradingEnv.episode_summary` dict, and every number in the
report is derived from those dicts by :func:`compute_metrics`. A comparison table
built this way cannot flatter one strategy through a different accounting.

The second rule is that the robustness verdict is computed, not written. See
:func:`robustness_verdict` for the predeclared criteria; it returns
``accepted=False`` whenever they are not met, including for the reference strategy.
"""

from __future__ import annotations

from archetype_trader.eval.ablations import (
    ABLATIONS,
    ABLATIONS_BY_NAME,
    REFERENCE,
    SMOKE_ABLATIONS,
    Ablation,
    resolve,
)
from archetype_trader.eval.bootstrap import (
    BootstrapKind,
    Interval,
    bootstrap_metric,
    bootstrap_suite,
    circular_block_indices,
    make_calmar,
    make_cvar,
    make_sharpe,
    seed_distribution,
    stationary_indices,
    total_log_return,
)
from archetype_trader.eval.harness import (
    Cell,
    WalkForwardReport,
    collect_provenance,
    composite_objective,
    evaluate,
    report_to_json,
    robustness_verdict,
    selection_value,
)
from archetype_trader.eval.metrics import (
    Metrics,
    compute_metrics,
    cost_attribution,
    cvar,
    max_drawdown,
    rolling_metric,
    underwater,
)
from archetype_trader.eval.regimes import (
    TREND_REGIMES,
    VOL_REGIMES,
    RegimeLabels,
    conditional_metrics,
    label_regimes,
)
from archetype_trader.eval.report import (
    DISCLAIMER,
    render_markdown,
    write_csv,
    write_plots,
    write_report,
)
from archetype_trader.eval.strategies import (
    VARIANTS,
    TrainedStack,
    make_horizon_env,
    pick_best_train_archetype,
    run_variant,
    train_stack,
)
from archetype_trader.eval.stress import (
    StressPoint,
    collapse_verdict,
    scale_costs,
    stress_env_config,
)

__all__ = [
    "ABLATIONS",
    "ABLATIONS_BY_NAME",
    "DISCLAIMER",
    "REFERENCE",
    "SMOKE_ABLATIONS",
    "TREND_REGIMES",
    "VARIANTS",
    "VOL_REGIMES",
    "Ablation",
    "BootstrapKind",
    "Cell",
    "Interval",
    "Metrics",
    "RegimeLabels",
    "StressPoint",
    "TrainedStack",
    "WalkForwardReport",
    "bootstrap_metric",
    "bootstrap_suite",
    "circular_block_indices",
    "collapse_verdict",
    "collect_provenance",
    "composite_objective",
    "compute_metrics",
    "conditional_metrics",
    "cost_attribution",
    "cvar",
    "evaluate",
    "label_regimes",
    "make_calmar",
    "make_cvar",
    "make_horizon_env",
    "make_sharpe",
    "max_drawdown",
    "pick_best_train_archetype",
    "render_markdown",
    "report_to_json",
    "resolve",
    "robustness_verdict",
    "rolling_metric",
    "run_variant",
    "scale_costs",
    "seed_distribution",
    "selection_value",
    "stationary_indices",
    "stress_env_config",
    "total_log_return",
    "train_stack",
    "underwater",
    "write_csv",
    "write_plots",
    "write_report",
]
