"""Deterministic historical simulation of the trend strategy.

The backtest uses exactly the code paths the paper ledger uses:
:func:`btc_trader.strategy.trend_path` for decisions and
:func:`btc_trader.rebalance.plan_rebalance` for sizing, limits and costs.
It never writes to the database or calls an exchange.

Timing is strictly causal: the decision computed from the close of day ``t``
is filled at the open of day ``t + 1`` (bitFlyer trades around the clock, so
that open is the first price available after the close is known), and equity is
then marked at the close of ``t + 1``.

The strategy's parameters are fixed in advance (they follow the published
model), so there is no in-sample tuning step to walk forward. Robustness is
reported instead as a calendar-year breakdown against buy-and-hold and as a
transaction-cost stress test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import numpy as np

from btc_trader.config import Settings
from btc_trader.enums import ExitReason, Side
from btc_trader.rebalance import RebalanceRequest, plan_rebalance
from btc_trader.strategy import (
    DAYS_PER_YEAR,
    STRATEGIES,
    DailyBar,
    TrendParameters,
    parameters_from_settings,
    trend_path,
)

COST_STRESS_MULTIPLIERS = (Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("4"))


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    name: str
    start: date
    end: date
    days: int
    initial_equity_jpy: Decimal
    final_equity_jpy: Decimal
    net_profit_jpy: Decimal
    total_return_percent: float
    cagr_percent: float
    annual_volatility_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_percent: float
    calmar_ratio: float
    trade_count: int
    turnover_per_year: float
    fees_paid_jpy: Decimal
    average_exposure_percent: float
    time_in_market_percent: float


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    day: date
    side: str
    price: Decimal
    size: Decimal
    fee: Decimal
    exposure_before: Decimal
    target_exposure: Decimal
    exit_reason: str | None


@dataclass(frozen=True, slots=True)
class PeriodReturn:
    year: int
    strategy_return_percent: float
    benchmark_return_percent: float
    strategy_max_drawdown_percent: float
    benchmark_max_drawdown_percent: float


@dataclass(frozen=True, slots=True)
class CostStressResult:
    cost_multiplier: float
    cagr_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    fees_paid_jpy: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy_name: str
    metrics: BacktestMetrics
    benchmark: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: list[dict[str, str]]
    periods: list[PeriodReturn]
    cost_stress: list[CostStressResult]
    parameters: dict[str, object]


@dataclass(slots=True)
class _Run:
    days: list[date]
    equity: list[Decimal]
    exposure: list[float]
    trades: list[BacktestTrade]
    fees: Decimal
    turnover: Decimal


def run_backtest(
    bars: Sequence[DailyBar],
    settings: Settings,
    *,
    strategy_name: str | None = None,
    start: date | None = None,
    end: date | None = None,
    cost_multiplier: Decimal = Decimal("1"),
    apply_jpy_limits: bool = False,
    include_cost_stress: bool = True,
) -> BacktestResult:
    """Replay the trend strategy over completed daily bars."""
    name = (strategy_name or settings.strategy_name).strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}")
    if name == "none":
        raise ValueError("select a strategy to backtest (trend_ensemble)")
    parameters = parameters_from_settings(settings)
    ordered = sorted(bars, key=lambda bar: bar.day)
    if end is not None:
        ordered = [bar for bar in ordered if bar.day <= end]
    path = trend_path(ordered, parameters)
    first = _first_index(ordered, parameters, start)
    if first is None or len(ordered) - first < 2:
        raise ValueError(
            f"not enough daily history: {len(ordered)} bars stored, the strategy needs "
            f"{parameters.warmup_bars} for warm-up plus at least 2 to trade. Load daily "
            "history first (POST /api/v1/market/daily-history/sync)."
        )

    sleeves_changed = (path.entered | path.exited).any(axis=0)
    stopped_out = path.exited.any(axis=0)

    def simulate(multiplier: Decimal, *, benchmark: bool = False) -> _Run:
        targets = np.ones(len(ordered)) if benchmark else path.target
        return _simulate(
            ordered,
            targets,
            sleeves_changed,
            stopped_out,
            first,
            settings,
            multiplier,
            apply_jpy_limits=apply_jpy_limits,
        )

    strategy_run = simulate(cost_multiplier)
    benchmark_run = simulate(cost_multiplier, benchmark=True)
    metrics = _metrics(name, strategy_run)
    benchmark = _metrics("buy_and_hold", benchmark_run)
    stress: list[CostStressResult] = []
    if include_cost_stress:
        for multiplier in COST_STRESS_MULTIPLIERS:
            run = strategy_run if multiplier == cost_multiplier else simulate(multiplier)
            stressed = _metrics(name, run)
            stress.append(
                CostStressResult(
                    cost_multiplier=float(multiplier),
                    cagr_percent=stressed.cagr_percent,
                    sharpe_ratio=stressed.sharpe_ratio,
                    max_drawdown_percent=stressed.max_drawdown_percent,
                    fees_paid_jpy=stressed.fees_paid_jpy,
                )
            )
    curve = [
        {
            "date": day.isoformat(),
            "equity_jpy": format(equity, "f"),
            "benchmark_equity_jpy": format(bench, "f"),
            "exposure": f"{exposure:.6f}",
        }
        for day, equity, bench, exposure in zip(
            strategy_run.days,
            strategy_run.equity,
            benchmark_run.equity,
            strategy_run.exposure,
            strict=True,
        )
    ]
    return BacktestResult(
        strategy_name=name,
        metrics=metrics,
        benchmark=benchmark,
        trades=strategy_run.trades,
        equity_curve=curve,
        periods=_periods(strategy_run, benchmark_run),
        cost_stress=stress,
        parameters=_parameter_summary(parameters, settings, cost_multiplier, apply_jpy_limits),
    )


def _first_index(
    bars: Sequence[DailyBar], parameters: TrendParameters, start: date | None
) -> int | None:
    """First bar whose *previous* close already has a sized decision."""
    first = parameters.warmup_bars
    if start is not None:
        first = max(first, next((i for i, bar in enumerate(bars) if bar.day >= start), len(bars)))
    return first if first < len(bars) else None


def _simulate(
    bars: Sequence[DailyBar],
    targets: np.ndarray,
    sleeves_changed: np.ndarray,
    stopped_out: np.ndarray,
    first: int,
    settings: Settings,
    cost_multiplier: Decimal,
    *,
    apply_jpy_limits: bool,
) -> _Run:
    cash = settings.paper_initial_jpy
    btc = Decimal("0")
    cost_basis = Decimal("0")
    realised_by_day: dict[date, Decimal] = {}
    run = _Run(
        days=[bars[first - 1].day],
        equity=[cash],
        exposure=[0.0],
        trades=[],
        fees=Decimal("0"),
        turnover=Decimal("0"),
    )
    for index in range(first, len(bars)):
        bar, decided = bars[index], index - 1
        price = bar.open if bar.open > 0 else bars[decided].close
        average = cost_basis / btc if btc > 0 else None
        daily_loss = (
            settings.max_daily_loss_jpy > 0
            and -realised_by_day.get(bar.day, Decimal("0")) >= settings.max_daily_loss_jpy
        )
        plan = plan_rebalance(
            RebalanceRequest(
                cash=cash,
                btc=btc,
                bid=price,
                ask=price,
                target_exposure=Decimal(str(round(float(targets[decided]), 6))),
                force=bool(sleeves_changed[decided]),
                reduction_reason=(
                    ExitReason.TRAILING_STOP if stopped_out[decided] else ExitReason.RISK_EXIT
                ),
                average_entry_price=average,
                daily_loss_reached=daily_loss,
                cost_multiplier=cost_multiplier,
                apply_jpy_limits=apply_jpy_limits,
            ),
            settings,
        )
        if plan.trades:
            gross = plan.price * plan.size
            if plan.side is Side.BUY:
                cash -= gross + plan.fee
                btc += plan.size
                cost_basis += gross + plan.fee
            else:
                released = (average or Decimal("0")) * plan.size
                realised_by_day[bar.day] = (
                    realised_by_day.get(bar.day, Decimal("0")) + gross - plan.fee - released
                )
                cash += gross - plan.fee
                btc -= plan.size
                cost_basis = cost_basis - released if btc > 0 else Decimal("0")
            run.fees += plan.fee
            run.turnover += gross / plan.equity if plan.equity > 0 else Decimal("0")
            run.trades.append(
                BacktestTrade(
                    day=bar.day,
                    side=plan.side.value if plan.side else "",
                    price=plan.price,
                    size=plan.size,
                    fee=plan.fee,
                    exposure_before=plan.current_exposure,
                    target_exposure=plan.target_exposure,
                    exit_reason=plan.exit_reason.value if plan.exit_reason else None,
                )
            )
        equity = cash + btc * bar.close
        run.days.append(bar.day)
        run.equity.append(equity)
        run.exposure.append(float(btc * bar.close / equity) if equity > 0 else 0.0)
    return run


def _metrics(name: str, run: _Run) -> BacktestMetrics:
    equity = np.asarray([float(value) for value in run.equity])
    returns = equity[1:] / equity[:-1] - 1 if len(equity) > 1 else np.zeros(0)
    days = len(returns)
    years = days / DAYS_PER_YEAR if days else 0.0
    total = equity[-1] / equity[0] - 1 if equity[0] > 0 else 0.0
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1 if years > 0 and equity[-1] > 0 else 0.0
    deviation = float(returns.std()) if days > 1 else 0.0
    downside = returns[returns < 0]
    downside_deviation = float(np.sqrt((downside**2).sum() / days)) if days else 0.0
    mean = float(returns.mean()) if days else 0.0
    max_drawdown = _max_drawdown(equity)
    exposure = np.asarray(run.exposure[1:]) if len(run.exposure) > 1 else np.zeros(1)
    return BacktestMetrics(
        name=name,
        start=run.days[0],
        end=run.days[-1],
        days=days,
        initial_equity_jpy=run.equity[0],
        final_equity_jpy=run.equity[-1],
        net_profit_jpy=run.equity[-1] - run.equity[0],
        total_return_percent=total * 100,
        cagr_percent=cagr * 100,
        annual_volatility_percent=deviation * math.sqrt(DAYS_PER_YEAR) * 100,
        sharpe_ratio=mean / deviation * math.sqrt(DAYS_PER_YEAR) if deviation else 0.0,
        sortino_ratio=(
            mean / downside_deviation * math.sqrt(DAYS_PER_YEAR) if downside_deviation else 0.0
        ),
        max_drawdown_percent=max_drawdown * 100,
        calmar_ratio=cagr / max_drawdown if max_drawdown else 0.0,
        trade_count=len(run.trades),
        turnover_per_year=float(run.turnover) / years if years else 0.0,
        fees_paid_jpy=run.fees,
        average_exposure_percent=float(exposure.mean()) * 100,
        time_in_market_percent=float((exposure > 0).mean()) * 100,
    )


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1 - equity / peaks))


def _periods(strategy: _Run, benchmark: _Run) -> list[PeriodReturn]:
    years = sorted({day.year for day in strategy.days[1:]})
    result: list[PeriodReturn] = []
    for year in years:
        indices = [i for i, day in enumerate(strategy.days) if day.year == year]
        base = max(indices[0] - 1, 0)
        selected = [base, *indices]
        strategy_slice = np.asarray([float(strategy.equity[i]) for i in selected])
        benchmark_slice = np.asarray([float(benchmark.equity[i]) for i in selected])
        result.append(
            PeriodReturn(
                year=year,
                strategy_return_percent=(strategy_slice[-1] / strategy_slice[0] - 1) * 100,
                benchmark_return_percent=(benchmark_slice[-1] / benchmark_slice[0] - 1) * 100,
                strategy_max_drawdown_percent=_max_drawdown(strategy_slice) * 100,
                benchmark_max_drawdown_percent=_max_drawdown(benchmark_slice) * 100,
            )
        )
    return result


def _parameter_summary(
    parameters: TrendParameters,
    settings: Settings,
    cost_multiplier: Decimal,
    apply_jpy_limits: bool,
) -> dict[str, object]:
    return {
        "horizons": list(parameters.horizons),
        "volatility_window_days": parameters.volatility_window,
        "target_annual_volatility_percent": parameters.target_volatility * 100,
        "max_exposure_percent": parameters.max_exposure * 100,
        "rebalance_band_percent": float(settings.strategy_rebalance_band_percent),
        "taker_fee_percent": float(settings.taker_fee_percent),
        "slippage_bps": float(settings.paper_slippage_bps),
        "cost_multiplier": float(cost_multiplier),
        "apply_jpy_limits": apply_jpy_limits,
        "initial_equity_jpy": format(settings.paper_initial_jpy, "f"),
    }


__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "BacktestTrade",
    "CostStressResult",
    "PeriodReturn",
    "run_backtest",
]
