"""Performance and risk metrics computed from an environment episode summary.

All return-based metrics are derived from the **net** log-return series the
environment produced, so fees, spread, slippage, funding and every shield
intervention are already inside them. There is no separate "gross" path that
could accidentally become the headline number; gross PnL is reported only as a
cost-attribution line.

Log versus simple returns: log returns are what the environment produces and what
compounds additively, so total return is ``expm1(sum)``. Sharpe and Sortino are
computed on log returns too — at 10-minute resolution the difference from simple
returns is third-order, and mixing conventions between the ratio's numerator and
denominator is a much bigger error than the one being avoided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class Metrics:
    """One strategy over one segment. Every field is a scalar so it tables cleanly."""

    name: str
    n_steps: int
    bars_per_year: int
    total_return: float
    annualized_return: float
    #: ``total_log_return / years``. Reported alongside ``annualized_return``
    #: because on a segment of a few hundred bars the compounded figure is
    #: astronomically large and unreadable, while this one stays interpretable.
    #: Calmar is built on this; see :func:`compute_metrics`.
    annualized_log_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    cvar: float
    var: float
    downside_vol: float
    turnover: float
    fees: float
    funding: float
    execution_cost: float
    n_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    exposure: float
    mean_position: float
    max_leverage: float
    n_liquidations: int
    n_partial_fills: int
    n_rejected: int
    bankrupt: bool
    shield_intervention_rate: float
    final_equity: float
    gross_pnl: float
    net_pnl: float
    cost_drag: float
    #: Free-form extras (archetype usage, codebook health, override rates).
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra")
        return d


def max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough fraction, in ``[0, 1]``. ``0.0`` for a monotone curve."""
    if equity.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / np.maximum(peak, _EPS)))


def cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """``(VaR, CVaR)`` at level ``alpha``, both reported as positive losses.

    Empirical, not parametric: fitting a normal to 10-minute crypto returns would
    understate exactly the tail the metric exists to measure.
    """
    if returns.size == 0:
        return 0.0, 0.0
    q = float(np.quantile(returns, alpha))
    tail = returns[returns <= q]
    c = float(tail.mean()) if tail.size else q
    return -q, -c


def _ratio(numer: float, denom: float) -> float:
    """Guarded ratio. Returns 0.0 when the denominator vanishes.

    A zero rather than an inf: an infinite Sharpe from a flat equity curve would
    sort to the top of every comparison table, which is the opposite of useful.
    """
    return float(numer / denom) if abs(denom) > _EPS else 0.0


def compute_metrics(
    summary: dict[str, Any],
    *,
    name: str | None = None,
    bars_per_year: int = 52_560,
    cvar_alpha: float = 0.05,
    extra: dict[str, Any] | None = None,
) -> Metrics:
    """Turn a :meth:`TradingEnv.episode_summary` dict into a :class:`Metrics`."""
    r = np.asarray(summary["net_log_returns"], dtype=np.float64)
    eq = np.asarray(summary["equity_curve"], dtype=np.float64)
    pos = np.asarray(summary["positions"], dtype=np.float64)
    n = int(r.size)
    scale = float(np.sqrt(bars_per_year))

    total_log = float(r.sum())
    total_return = float(np.expm1(total_log))
    years = n / bars_per_year if n else 0.0
    ann_log_return = float(total_log / years) if years > _EPS else 0.0
    # ``expm1`` of a few hundred overflows to inf, and a test segment of 450
    # 10-minute bars is ~3 days, so compounding it to a year legitimately produces
    # numbers like 1e83. Clipped so the column is finite and sortable; the
    # unclipped, honest version of the same quantity is ``annualized_log_return``.
    ann_return = float(np.expm1(min(ann_log_return, 700.0))) if years > _EPS else 0.0
    vol = float(r.std(ddof=1)) if n > 1 else 0.0
    ann_vol = vol * scale
    neg = r[r < 0.0]
    # Downside deviation is taken over *all* bars, not only losing ones: dividing
    # the squared losses by the loss count would make a strategy that rarely
    # trades look less risky purely because it has fewer observations.
    dd_vol = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) * scale if n else 0.0
    mdd = max_drawdown(eq)
    var_a, cvar_a = cvar(r, cvar_alpha)

    wins = r[r > 0.0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-neg.mean()) if neg.size else 0.0
    turnover_notional = float(summary.get("turnover_notional", 0.0))
    initial_equity = float(eq[0]) if eq.size else 0.0

    return Metrics(
        name=name or str(summary.get("name", "strategy")),
        n_steps=n,
        bars_per_year=bars_per_year,
        total_return=total_return,
        annualized_return=ann_return,
        annualized_log_return=ann_log_return,
        annualized_vol=ann_vol,
        sharpe=_ratio(float(r.mean()) * scale, vol) if n > 1 else 0.0,
        sortino=_ratio(float(r.mean()) * scale, dd_vol / scale) if n else 0.0,
        # Calmar on the *log* annualised return, not the compounded one. Textbook
        # Calmar uses the compounded figure, but the two differ only by the monotone
        # map ``expm1``, so rankings over a fixed segment length are identical —
        # and this form does not saturate at 1e308 on short segments, which the
        # compounded one does.
        calmar=_ratio(ann_log_return, mdd),
        max_drawdown=mdd,
        cvar=cvar_a,
        var=var_a,
        downside_vol=dd_vol,
        # Turnover as multiples of starting equity: comparable across strategies
        # whose equity paths diverge, unlike a per-bar notional sum.
        turnover=_ratio(turnover_notional, initial_equity),
        fees=float(summary.get("fees", 0.0)),
        funding=float(summary.get("funding", 0.0)),
        execution_cost=float(summary.get("execution_cost", 0.0)),
        n_trades=int(summary.get("n_trades", 0)),
        hit_rate=float((r > 0.0).mean()) if n else 0.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=_ratio(avg_win, avg_loss),
        exposure=float(np.mean(np.abs(pos) > _EPS)) if pos.size else 0.0,
        mean_position=float(pos.mean()) if pos.size else 0.0,
        max_leverage=float(np.max(np.abs(pos))) if pos.size else 0.0,
        n_liquidations=int(summary.get("n_liquidations", 0)),
        n_partial_fills=int(summary.get("n_partial_fills", 0)),
        n_rejected=int(summary.get("n_rejected", 0)),
        bankrupt=bool(summary.get("bankrupt", False)),
        shield_intervention_rate=float(summary.get("shield_intervention_rate", 0.0)),
        final_equity=float(eq[-1]) if eq.size else 0.0,
        gross_pnl=float(summary.get("gross_pnl", 0.0)),
        net_pnl=float(eq[-1] - eq[0]) if eq.size else 0.0,
        cost_drag=float(summary.get("fees", 0.0))
        + float(summary.get("funding", 0.0))
        + float(summary.get("execution_cost", 0.0)),
        extra=dict(extra or {}),
    )


def cost_attribution(summary: dict[str, Any]) -> dict[str, float]:
    """Split the gap between gross and net PnL into its named components.

    The residual line is the point of this function: if it is not close to zero,
    some cost is being double-counted or missed, and the equity curve cannot be
    trusted.
    """
    eq = np.asarray(summary["equity_curve"], dtype=np.float64)
    net = float(eq[-1] - eq[0]) if eq.size else 0.0
    gross = float(summary.get("gross_pnl", 0.0))
    fees = float(summary.get("fees", 0.0))
    funding = float(summary.get("funding", 0.0))
    execution = float(summary.get("execution_cost", 0.0))
    return {
        "gross_pnl": gross,
        "fees": -fees,
        "funding": -funding,
        "execution_cost": -execution,
        "net_pnl": net,
        "residual": net - (gross - fees - funding - execution),
    }


def rolling_metric(
    returns: np.ndarray, window: int, *, bars_per_year: int = 52_560, metric: str = "sharpe"
) -> np.ndarray:
    """Trailing rolling risk metric, NaN over the warm-up.

    Trailing rather than centred: a centred window at bar ``t`` would summarise
    bars after ``t``, which is fine for a plot but invites the number to be read
    as something a live system could have known.
    """
    n = returns.size
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n < window:
        return out
    scale = float(np.sqrt(bars_per_year))
    view = np.lib.stride_tricks.sliding_window_view(returns, window)
    if metric == "sharpe":
        mu = view.mean(axis=1) * scale
        sd = view.std(axis=1, ddof=1)
        vals = np.divide(mu, sd, out=np.zeros_like(mu), where=np.abs(sd) > _EPS)
    elif metric == "vol":
        vals = view.std(axis=1, ddof=1) * scale
    elif metric == "downside_vol":
        vals = np.sqrt(np.mean(np.minimum(view, 0.0) ** 2, axis=1)) * scale
    else:
        raise KeyError(f"unknown rolling metric {metric!r}")
    out[window - 1 :] = vals
    return out


def underwater(equity: np.ndarray) -> np.ndarray:
    """Fractional drawdown at every bar, for the underwater plot."""
    if equity.size == 0:
        return equity
    peak = np.maximum.accumulate(equity)
    return 1.0 - equity / np.maximum(peak, _EPS)
