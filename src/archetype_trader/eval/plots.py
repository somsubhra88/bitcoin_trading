"""Diagnostic plots. Matplotlib only, Agg backend, no seaborn, no styling opinions.

Every function takes already-computed arrays and writes one PNG. They are
deliberately dumb: a plotting helper that recomputes a metric is a second
implementation of that metric, and the two will disagree eventually.

Plots are optional (``eval.make_plots``). If matplotlib is missing the module
imports fine and every function raises on call, so a headless environment without
it can still run the numeric evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised by absence, not presence
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def _require_mpl() -> None:
    if not HAVE_MPL:
        raise RuntimeError(
            "matplotlib is not installed; set eval.make_plots=false or install the 'plots' extra"
        )


def _save(fig: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_equity(
    curves: dict[str, np.ndarray], path: str | Path, *, title: str = "Equity (net of all costs)"
) -> Path:
    """Overlaid equity curves on a log y-axis.

    Log scale, always: on a linear axis a 4x winner compresses every other curve
    into a flat line at the bottom, and the comparison the plot exists for becomes
    unreadable.
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, eq in curves.items():
        ax.plot(np.asarray(eq, dtype=np.float64), label=name, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("bar")
    ax.set_ylabel("equity")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, path)


def plot_underwater(curves: dict[str, np.ndarray], path: str | Path) -> Path:
    """Drawdown-from-peak, plotted downward so deeper is visually lower."""
    _require_mpl()
    from archetype_trader.eval.metrics import underwater

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, eq in curves.items():
        ax.plot(-underwater(np.asarray(eq, dtype=np.float64)), label=name, linewidth=1.0)
    ax.set_xlabel("bar")
    ax.set_ylabel("drawdown")
    ax.set_title("Underwater curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, path)


def plot_rolling_risk(
    returns: dict[str, np.ndarray],
    path: str | Path,
    *,
    window: int = 720,
    bars_per_year: int = 52_560,
) -> Path:
    """Trailing rolling Sharpe and volatility, stacked."""
    _require_mpl()
    from archetype_trader.eval.metrics import rolling_metric

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, r in returns.items():
        arr = np.asarray(r, dtype=np.float64)
        axes[0].plot(
            rolling_metric(arr, window, bars_per_year=bars_per_year, metric="sharpe"),
            label=name,
            linewidth=1.0,
        )
        axes[1].plot(
            rolling_metric(arr, window, bars_per_year=bars_per_year, metric="vol"),
            label=name,
            linewidth=1.0,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(f"rolling Sharpe ({window} bars)")
    axes[1].set_ylabel("rolling ann. vol")
    axes[1].set_xlabel("bar")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    return _save(fig, path)


def plot_positions(
    proposed: np.ndarray, executed: np.ndarray, path: str | Path, *, name: str = "strategy"
) -> Path:
    """Proposed versus executed position. The gap is the risk shield, visualised."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.step(np.arange(len(proposed)), proposed, where="post", label="proposed", linewidth=0.9)
    ax.step(np.arange(len(executed)), executed, where="post", label="executed", linewidth=0.9)
    ax.set_xlabel("bar")
    ax.set_ylabel("target position")
    ax.set_title(f"{name}: proposed vs executed (gap = risk shield)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_gross_vs_net(
    gross_cum: np.ndarray, net_cum: np.ndarray, path: str | Path, *, name: str = "strategy"
) -> Path:
    """Cumulative gross versus net log return. The widening gap is total friction."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gross_cum, label="gross (no costs)", linewidth=1.1)
    ax.plot(net_cum, label="net (all costs)", linewidth=1.1)
    ax.fill_between(
        np.arange(len(net_cum)), net_cum, gross_cum[: len(net_cum)], alpha=0.2, label="friction"
    )
    ax.set_xlabel("bar")
    ax.set_ylabel("cumulative log return")
    ax.set_title(f"{name}: gross vs net")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_cost_attribution(attribution: dict[str, float], path: str | Path) -> Path:
    """Waterfall-ish bar chart of where gross PnL went."""
    _require_mpl()
    order = ("gross_pnl", "fees", "funding", "execution_cost", "net_pnl")
    keys = [k for k in order if k in attribution]
    vals = [attribution[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]
    ax.bar(keys, vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("quote currency")
    ax.set_title(f"Cost attribution (residual {attribution.get('residual', 0.0):+.2e})")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)


def plot_archetype_usage(
    usage: np.ndarray, path: str | Path, *, cash_index: int | None = None
) -> Path:
    """How often the selector chose each archetype. Flat-to-one bar = collapse."""
    _require_mpl()
    u = np.asarray(usage, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [
        "tab:gray" if cash_index is not None and i == cash_index else "tab:blue"
        for i in range(u.size)
    ]
    ax.bar(np.arange(u.size), u, color=colors)
    ax.axhline(1.0 / max(u.size, 1), color="black", linestyle="--", linewidth=0.8, label="uniform")
    ax.set_xlabel("archetype")
    ax.set_ylabel("selection frequency")
    ax.set_title("Archetype usage (grey = cash)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_codebook_health(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Perplexity, utilisation and dead-code count across Phase-1 epochs."""
    _require_mpl()
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(ep, [h.get("perplexity", np.nan) for h in history], marker="o", linewidth=1.0)
    axes[0].set_ylabel("perplexity")
    axes[1].plot(ep, [h.get("utilization", np.nan) for h in history], marker="o", linewidth=1.0)
    axes[1].set_ylabel("utilisation")
    axes[1].set_ylim(0, 1.05)
    axes[2].plot(ep, [h.get("dead_codes", np.nan) for h in history], marker="o", linewidth=1.0)
    axes[2].set_ylabel("dead codes")
    axes[2].set_xlabel("epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Codebook health")
    return _save(fig, path)


def plot_stress(
    equity_by_multiplier: dict[float, float], path: str | Path, *, initial_equity: float
) -> Path:
    """Net PnL against friction multiplier. A cliff here is a rejection."""
    _require_mpl()
    mults = sorted(equity_by_multiplier)
    pnl = [equity_by_multiplier[m] - initial_equity for m in mults]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mults, pnl, marker="o", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(mults)
    ax.set_xticklabels([f"{m:g}x" for m in mults])
    ax.set_xlabel("friction multiplier")
    ax.set_ylabel("net PnL")
    ax.set_title("Friction stress test")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_seed_distribution(
    values_by_name: dict[str, list[float]], path: str | Path, *, metric: str = "sharpe"
) -> Path:
    """Box plot across seeds. One tall box = the result is a seed, not a strategy."""
    _require_mpl()
    names = list(values_by_name)
    data = [np.asarray(values_by_name[n], dtype=np.float64) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4))
    ax.boxplot(data, tick_labels=names, showmeans=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across seeds")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, path)


def plot_regime_conditioned(
    table: dict[str, dict[str, float]], path: str | Path, *, metric: str = "sharpe"
) -> Path:
    """Grouped bars of one metric per strategy per volatility regime."""
    _require_mpl()
    strategies = list(table)
    regimes = sorted({r for row in table.values() for r in row})
    width = 0.8 / max(len(regimes), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(strategies)), 4))
    x = np.arange(len(strategies))
    for i, regime in enumerate(regimes):
        vals = [table[s].get(regime, np.nan) for s in strategies]
        ax.bar(x + i * width, vals, width=width, label=regime)
    ax.set_xticks(x + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(strategies, rotation=25, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by volatility regime")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, path)
