"""Deterministic historical simulation for strategy validation.

Backtests use the same strategy evaluator and paper-fill assumptions as the
worker, but never write to the database or call an exchange. Walk-forward
validation selects parameters on a past window and reports results only on the
following unseen window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import Any

from btc_trader.config import Settings
from btc_trader.db.models import MarketTick, OrderBookSnapshot
from btc_trader.deeplob import (
    evaluate_order_book,
    snapshot_quotes,
)
from btc_trader.deeplob import (
    parameters_from_settings as deeplob_parameters,
)
from btc_trader.enums import TradingSignal
from btc_trader.paper import normalise_order_size
from btc_trader.strategy import STRATEGIES, SignalDecision, StrategyParameters, evaluate


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    net_pnl_jpy: Decimal
    return_percent: Decimal


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    strategy_name: str
    bars: int
    initial_cash_jpy: Decimal
    final_equity_jpy: Decimal
    net_profit_jpy: Decimal
    total_return_percent: Decimal
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate_percent: Decimal
    max_drawdown_percent: Decimal
    sharpe_ratio: float
    buy_signals: int
    sell_signals: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    selected_parameters: dict[str, str | int]
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    strategy_name: str
    train_bars: int
    test_bars: int
    windows: list[WalkForwardWindow]
    aggregate_metrics: BacktestMetrics


@dataclass(frozen=True, slots=True)
class _Observation:
    timestamp: datetime
    mark: Decimal
    bid: Decimal | None
    ask: Decimal | None
    ltp: Decimal | None
    decision: SignalDecision | None
    proposed_decision: SignalDecision | None = None


def strategy_parameters(settings: Settings) -> StrategyParameters:
    return StrategyParameters(
        fast_window=settings.strategy_fast_window,
        slow_window=settings.strategy_slow_window,
        mean_reversion_window=settings.strategy_mean_reversion_window,
        momentum_window=settings.strategy_momentum_window,
        volatility_window=settings.strategy_volatility_window,
        ema_threshold_percent=settings.strategy_ema_threshold_percent,
        mean_reversion_zscore_threshold=settings.strategy_mean_reversion_zscore_threshold,
        momentum_threshold_percent=settings.strategy_momentum_threshold_percent,
        edge_multiple=settings.strategy_edge_multiple,
        regime_filter=settings.strategy_regime_filter,
    )


def run_backtest(
    ticks: Sequence[MarketTick],
    settings: Settings,
    *,
    strategy_name: str | None = None,
    order_books: Sequence[OrderBookSnapshot] | None = None,
) -> BacktestResult:
    """Replay stored market data with the same fee-aware paper execution rules."""
    name = (strategy_name or settings.strategy_name).strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}")
    observations = _observations(ticks, order_books, settings, name)
    cash = settings.paper_initial_jpy
    initial_cash = cash
    quantity = Decimal("0")
    cost_basis = Decimal("0")
    entry_price = Decimal("0")
    entry_timestamp: datetime | None = None
    high_watermark = Decimal("0")
    realized_by_day: dict[Any, Decimal] = {}
    trades: list[BacktestTrade] = []
    equity_curve: list[dict[str, str]] = []
    buy_signals = 0
    sell_signals = 0

    for observation in observations:
        decision = observation.decision
        mark = observation.mark

        if (
            decision is not None
            and decision.signal is TradingSignal.BUY
            and quantity == 0
            and decision.confidence >= settings.paper_min_confidence
            and not _daily_loss_reached(
                realized_by_day, observation.timestamp, settings.max_daily_loss_jpy
            )
        ):
            # Evaluate risk exits before acting on the strategy signal. A
            # BUY/HOLD cannot veto a hard or trailing stop.
                    buy_price = _fill_price(observation, TradingSignal.BUY, settings)
                    if buy_price is not None:
                        allocation = settings.strategy_risk_fraction * (
                            Decimal("0.5") + decision.confidence * Decimal("0.5")
                        )
                        budget = min(
                            settings.max_order_jpy,
                            settings.max_position_jpy,
                            cash * allocation * _volatility_adjustment(decision.metadata, settings),
                        )
                        fee_rate = settings.taker_fee_percent / Decimal("100")
                        raw_size = budget / (buy_price * (Decimal("1") + fee_rate))
                        size = normalise_order_size(raw_size, settings)
                        if size is not None:
                            fee = buy_price * size * fee_rate
                            cash -= buy_price * size + fee
                            quantity = size
                            cost_basis = buy_price * size + fee
                            entry_price = cost_basis / size
                            entry_timestamp = observation.timestamp
                            high_watermark = buy_price

        if observation.proposed_decision is not None:
            if observation.proposed_decision.signal is TradingSignal.BUY:
                buy_signals += 1
            elif observation.proposed_decision.signal is TradingSignal.SELL:
                sell_signals += 1

        if quantity > 0 and entry_timestamp is not None:
            high_watermark = max(high_watermark, mark)
            sell_price = _fill_price(observation, TradingSignal.SELL, settings)
            if sell_price is not None:
                fee = sell_price * quantity * settings.taker_fee_percent / Decimal("100")
                net_pnl = sell_price * quantity - fee - cost_basis
                target = settings.min_profit_percent / Decimal("100")
                net_return = net_pnl / cost_basis if cost_basis else Decimal("0")
                trailing = (
                    settings.paper_trailing_stop_percent > 0
                    and sell_price
                    <= high_watermark
                    * (Decimal("1") - settings.paper_trailing_stop_percent / Decimal("100"))
                )
                hard_stop = (
                    settings.paper_stop_loss_percent > 0
                    and sell_price
                    <= entry_price
                    * (Decimal("1") - settings.paper_stop_loss_percent / Decimal("100"))
                )
                profit_exit = decision is not None and decision.signal in (
                    TradingSignal.SELL,
                    TradingSignal.HOLD,
                ) and net_return >= target
                if hard_stop or trailing or profit_exit:
                    cash += sell_price * quantity - fee
                    trades.append(
                        BacktestTrade(
                            entry_timestamp=entry_timestamp,
                            exit_timestamp=observation.timestamp,
                            entry_price=entry_price,
                            exit_price=sell_price,
                            size=quantity,
                            net_pnl_jpy=net_pnl,
                            return_percent=net_return * Decimal("100"),
                        )
                    )
                    realized_by_day[observation.timestamp.date()] = (
                        realized_by_day.get(observation.timestamp.date(), Decimal("0"))
                        + net_pnl
                    )
                    quantity = Decimal("0")
                    cost_basis = Decimal("0")
                    entry_price = Decimal("0")
                    entry_timestamp = None
                    high_watermark = Decimal("0")

        equity = cash + quantity * mark
        equity_curve.append(
            {
                "timestamp": observation.timestamp.isoformat(),
                "equity_jpy": format(equity, "f"),
            }
        )

    last_mark = observations[-1].mark if observations else None
    if observations and quantity > 0 and entry_timestamp is not None:
        # A finite backtest must settle inventory at its final observable mark;
        # otherwise the reported equity contains an unclosed position and the
        # trade statistics omit its realised costs.
        last_observation = observations[-1]
        sell_price = _fill_price(last_observation, TradingSignal.SELL, settings)
        fee = sell_price * quantity * settings.taker_fee_percent / Decimal("100")
        net_pnl = sell_price * quantity - fee - cost_basis
        net_return = net_pnl / cost_basis if cost_basis else Decimal("0")
        cash += sell_price * quantity - fee
        trades.append(
            BacktestTrade(
                entry_timestamp=entry_timestamp,
                exit_timestamp=last_observation.timestamp,
                entry_price=entry_price,
                exit_price=sell_price,
                size=quantity,
                net_pnl_jpy=net_pnl,
                return_percent=net_return * Decimal("100"),
            )
        )
        quantity = Decimal("0")
    final_equity = cash + quantity * last_mark if last_mark is not None else cash
    metrics = _metrics(
        name,
        len(observations),
        initial_cash,
        final_equity,
        trades,
        equity_curve,
        buy_signals,
        sell_signals,
    )
    return BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve)


def run_walk_forward(
    ticks: Sequence[MarketTick],
    settings: Settings,
    *,
    strategy_name: str | None = None,
    order_books: Sequence[OrderBookSnapshot] | None = None,
    train_bars: int = 500,
    test_bars: int = 100,
    step_bars: int | None = None,
) -> WalkForwardResult:
    """Tune sensitivity on each train window and score the next test window."""
    if train_bars < 1 or test_bars < 1:
        raise ValueError("train_bars and test_bars must be positive")
    name = (strategy_name or settings.strategy_name).strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}")
    if name == "deeplob":
        if not order_books:
            raise ValueError("DeepLOB walk-forward validation requires order-book snapshots")
        dataset: list[Any] = sorted(order_books, key=lambda snapshot: snapshot.received_at)
    else:
        dataset = sorted(
            (tick for tick in ticks if tick.ltp is not None),
            key=lambda tick: tick.exchange_timestamp,
        )
    step = step_bars or test_bars
    if step < 1:
        raise ValueError("step_bars must be positive")

    windows: list[WalkForwardWindow] = []
    start = 0
    while start + train_bars + test_bars <= len(dataset):
        train = dataset[start : start + train_bars]
        test_start = start + train_bars
        test = dataset[test_start : test_start + test_bars]
        candidates = _candidate_settings(settings, name)
        candidate_results = [
            run_backtest(
                train if name != "deeplob" else [],
                candidate,
                strategy_name=name,
                order_books=train if name == "deeplob" else None,
            )
            for candidate in candidates
        ]
        selected_index = max(
            range(len(candidate_results)),
            key=lambda index: candidate_results[index].metrics.net_profit_jpy,
        )
        selected_settings = candidates[selected_index]
        train_result = candidate_results[selected_index]
        test_result = run_backtest(
            test if name != "deeplob" else [],
            selected_settings,
            strategy_name=name,
            order_books=test if name == "deeplob" else None,
        )
        train_start = _observation_timestamp(train[0])
        train_end = _observation_timestamp(train[-1])
        test_timestamp_start = _observation_timestamp(test[0])
        test_timestamp_end = _observation_timestamp(test[-1])
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_timestamp_start,
                test_end=test_timestamp_end,
                selected_parameters={
                    **_parameter_summary(strategy_parameters(selected_settings)),
                    **_deeplob_parameter_summary(selected_settings),
                },
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
            )
        )
        start += step

    aggregate = _aggregate_metrics([window.test_metrics for window in windows], name)
    return WalkForwardResult(
        strategy_name=name,
        train_bars=train_bars,
        test_bars=test_bars,
        windows=windows,
        aggregate_metrics=aggregate,
    )


def _candidate_settings(settings: Settings, strategy_name: str) -> list[Settings]:
    if strategy_name == "ema_crossover":
        return [
            settings.model_copy(
                update={
                    "strategy_ema_threshold_percent": settings.strategy_ema_threshold_percent
                    * factor
                }
            )
            for factor in (Decimal("0.5"), Decimal("1"), Decimal("2"))
        ]
    if strategy_name == "mean_reversion":
        return [
            settings.model_copy(
                update={
                    "strategy_mean_reversion_zscore_threshold": (
                        settings.strategy_mean_reversion_zscore_threshold * factor
                    )
                }
            )
            for factor in (Decimal("0.75"), Decimal("1"), Decimal("1.25"))
        ]
    if strategy_name == "momentum_regime":
        return [
            settings.model_copy(
                update={
                    "strategy_momentum_threshold_percent": (
                        settings.strategy_momentum_threshold_percent * factor
                    )
                }
            )
            for factor in (Decimal("0.5"), Decimal("1"), Decimal("2"))
        ]
    if strategy_name == "deeplob":
        return [
            settings.model_copy(
                update={
                    "strategy_deeplob_threshold_percent": (
                        settings.strategy_deeplob_threshold_percent * factor
                    )
                }
            )
            for factor in (Decimal("0.5"), Decimal("1"), Decimal("2"))
        ]
    return [settings]


def _parameter_summary(parameters: StrategyParameters) -> dict[str, str | int]:
    return {
        "fast_window": parameters.fast_window,
        "slow_window": parameters.slow_window,
        "mean_reversion_window": parameters.mean_reversion_window,
        "momentum_window": parameters.momentum_window,
        "volatility_window": parameters.volatility_window,
        "ema_threshold_percent": format(parameters.ema_threshold_percent, "f"),
        "mean_reversion_zscore_threshold": format(parameters.mean_reversion_zscore_threshold, "f"),
        "momentum_threshold_percent": format(parameters.momentum_threshold_percent, "f"),
        "edge_multiple": format(parameters.edge_multiple, "f"),
        "regime_filter": "enabled" if parameters.regime_filter else "disabled",
    }


def _deeplob_parameter_summary(settings: Settings) -> dict[str, str | int]:
    parameters = deeplob_parameters(settings)
    return {
        "deeplob_window": parameters.window,
        "deeplob_horizon": parameters.horizon,
        "deeplob_training_window": parameters.training_window,
        "deeplob_levels": parameters.levels,
        "deeplob_threshold_percent": format(parameters.threshold_percent, "f"),
        "deeplob_min_quality_percent": format(parameters.min_quality_percent, "f"),
    }


def _observations(
    ticks: Sequence[MarketTick],
    order_books: Sequence[OrderBookSnapshot] | None,
    settings: Settings,
    name: str,
) -> list[_Observation]:
    if name == "deeplob":
        if not order_books:
            raise ValueError("DeepLOB backtesting requires order-book snapshots")
        ordered_books = sorted(order_books, key=lambda snapshot: snapshot.received_at)
        observations = [
            _Observation(
                timestamp=snapshot.received_at,
                mark=quotes[0],
                bid=quotes[1],
                ask=quotes[2],
                ltp=quotes[0],
                decision=evaluate_order_book(
                    ordered_books[: index + 1],
                    parameters=deeplob_parameters(settings),
                ),
            )
            for index, snapshot in enumerate(ordered_books)
            if (quotes := snapshot_quotes(snapshot)) is not None
        ]
        return _delay_decisions(observations)

    ordered_ticks = sorted(
        (tick for tick in ticks if tick.ltp is not None),
        key=lambda tick: tick.exchange_timestamp,
    )
    parameters = strategy_parameters(settings)
    observations = [
        _Observation(
            timestamp=tick.exchange_timestamp,
            mark=tick.best_bid or tick.ltp,  # type: ignore[arg-type]
            bid=tick.best_bid,
            ask=tick.best_ask,
            ltp=tick.ltp,
            decision=evaluate(name, ordered_ticks[: index + 1], parameters=parameters),
        )
        for index, tick in enumerate(ordered_ticks)
        if tick.best_bid or tick.ltp
    ]
    return _delay_decisions(observations)


def _delay_decisions(observations: list[_Observation]) -> list[_Observation]:
    """Execute a decision on the next bar, after its source quote is known."""
    if not observations:
        return observations
    delayed: list[_Observation] = []
    previous: SignalDecision | None = None
    for observation in observations:
        delayed.append(
            _Observation(
                timestamp=observation.timestamp,
                mark=observation.mark,
                bid=observation.bid,
                ask=observation.ask,
                ltp=observation.ltp,
                decision=previous,
                proposed_decision=observation.decision,
            )
        )
        previous = observation.decision
    return delayed


def _observation_timestamp(observation: MarketTick | OrderBookSnapshot) -> datetime:
    return (
        observation.received_at
        if isinstance(observation, OrderBookSnapshot)
        else observation.exchange_timestamp
    )


def _fill_price(observation: _Observation, signal: TradingSignal, settings: Settings) -> Decimal:
    price = (observation.ask if signal is TradingSignal.BUY else observation.bid) or observation.ltp
    if price is None:
        price = observation.mark
    if price is None:
        raise ValueError("backtest observation has no executable price")
    direction = Decimal("1") if signal is TradingSignal.BUY else Decimal("-1")
    return price * (Decimal("1") + direction * settings.paper_slippage_bps / Decimal("10000"))


def _volatility_adjustment(metadata: dict[str, Any] | None, settings: Settings) -> Decimal:
    try:
        observed = Decimal(str((metadata or {}).get("volatility", "0")))
    except (InvalidOperation, TypeError, ValueError):
        observed = Decimal("0")
    target = settings.strategy_target_volatility_percent / Decimal("100")
    return Decimal("1") if observed <= 0 else min(Decimal("1"), target / observed)


def _daily_loss_reached(
    realized_by_day: dict[Any, Decimal], timestamp: datetime, maximum: Decimal
) -> bool:
    if maximum <= 0:
        return False
    return max(Decimal("0"), -realized_by_day.get(timestamp.date(), Decimal("0"))) >= maximum


def _metrics(
    name: str,
    bars: int,
    initial_cash: Decimal,
    final_equity: Decimal,
    trades: list[BacktestTrade],
    equity_curve: list[dict[str, str]],
    buy_signals: int,
    sell_signals: int,
) -> BacktestMetrics:
    winners = sum(trade.net_pnl_jpy > 0 for trade in trades)
    losers = sum(trade.net_pnl_jpy <= 0 for trade in trades)
    trade_count = len(trades)
    win_rate = (
        Decimal(winners) / Decimal(trade_count) * Decimal("100") if trade_count else Decimal("0")
    )
    total_return = (
        (final_equity - initial_cash) / initial_cash * Decimal("100")
        if initial_cash
        else Decimal("0")
    )
    equities = [Decimal(point["equity_jpy"]) for point in equity_curve]
    max_drawdown = _max_drawdown(equities)
    sharpe = _sharpe(equities)
    return BacktestMetrics(
        strategy_name=name,
        bars=bars,
        initial_cash_jpy=initial_cash,
        final_equity_jpy=final_equity,
        net_profit_jpy=final_equity - initial_cash,
        total_return_percent=total_return,
        trade_count=trade_count,
        winning_trades=winners,
        losing_trades=losers,
        win_rate_percent=win_rate,
        max_drawdown_percent=max_drawdown,
        sharpe_ratio=sharpe,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
    )


def _aggregate_metrics(metrics: list[BacktestMetrics], name: str) -> BacktestMetrics:
    if not metrics:
        return _metrics(name, 0, Decimal("0"), Decimal("0"), [], [], 0, 0)
    initial = sum((metric.initial_cash_jpy for metric in metrics), Decimal("0"))
    final = sum((metric.final_equity_jpy for metric in metrics), Decimal("0"))
    trades = sum((metric.trade_count for metric in metrics), 0)
    winners = sum((metric.winning_trades for metric in metrics), 0)
    losers = sum((metric.losing_trades for metric in metrics), 0)
    return BacktestMetrics(
        strategy_name=name,
        bars=sum(metric.bars for metric in metrics),
        initial_cash_jpy=initial,
        final_equity_jpy=final,
        net_profit_jpy=final - initial,
        total_return_percent=(
            (final - initial) / initial * Decimal("100") if initial else Decimal("0")
        ),
        trade_count=trades,
        winning_trades=winners,
        losing_trades=losers,
        win_rate_percent=(
            Decimal(winners) / Decimal(trades) * Decimal("100") if trades else Decimal("0")
        ),
        max_drawdown_percent=max(metric.max_drawdown_percent for metric in metrics),
        sharpe_ratio=sum(metric.sharpe_ratio for metric in metrics) / len(metrics),
        buy_signals=sum(metric.buy_signals for metric in metrics),
        sell_signals=sum(metric.sell_signals for metric in metrics),
    )


def _max_drawdown(equities: list[Decimal]) -> Decimal:
    if not equities:
        return Decimal("0")
    peak = equities[0]
    maximum = Decimal("0")
    for equity in equities:
        peak = max(peak, equity)
        if peak:
            maximum = max(maximum, (peak - equity) / peak * Decimal("100"))
    return maximum


def _sharpe(equities: list[Decimal]) -> float:
    if len(equities) < 3:
        return 0.0
    returns = [
        float((current - previous) / previous)
        for previous, current in pairwise(equities)
        if previous > 0
    ]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    deviation = math.sqrt(variance)
    return mean / deviation * math.sqrt(len(returns)) if deviation else 0.0
