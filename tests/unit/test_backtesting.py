from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from btc_trader.backtesting import run_backtest
from btc_trader.config import Settings
from btc_trader.enums import Side
from btc_trader.strategy import DailyBar
from tests.fixtures.daily import bars_from_closes, price_path, up_then_down


def _settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"strategy_name": "trend_ensemble"})


def test_backtest_trades_the_trend_and_reports_a_benchmark(settings: Settings) -> None:
    result = run_backtest(up_then_down(), _settings(settings))

    assert result.metrics.trade_count > 0
    assert result.benchmark.trade_count == 1
    assert result.metrics.start == result.benchmark.start
    assert len(result.equity_curve) == result.metrics.days + 1
    assert Decimal(result.equity_curve[-1]["equity_jpy"]) == result.metrics.final_equity_jpy
    # The strategy is flat at the end of the decline; buy-and-hold is not.
    assert float(result.equity_curve[-1]["exposure"]) == 0.0
    assert result.metrics.max_drawdown_percent < result.benchmark.max_drawdown_percent


def test_fills_happen_at_the_next_open_not_the_signal_close(settings: Settings) -> None:
    """Look-ahead guard: a decision from close t is executed at open t+1."""
    bars = up_then_down()
    gapped = [DailyBar(bar.day, bar.open * Decimal("1.01"), bar.close, bar.source) for bar in bars]
    tuned = _settings(settings).model_copy(update={"paper_slippage_bps": Decimal("0")})

    result = run_backtest(gapped, tuned)

    opens = {bar.day: bar.open for bar in gapped}
    for trade in result.trades:
        assert trade.price == opens[trade.day]


def test_backtest_accounting_reconciles(settings: Settings) -> None:
    bars = up_then_down()
    result = run_backtest(bars, _settings(settings), include_cost_stress=False)

    cash = settings.paper_initial_jpy
    btc = Decimal("0")
    for trade in result.trades:
        if trade.side == Side.BUY.value:
            cash -= trade.price * trade.size + trade.fee
            btc += trade.size
        else:
            cash += trade.price * trade.size - trade.fee
            btc -= trade.size
    assert btc >= 0
    assert cash >= 0
    assert cash + btc * bars[-1].close == result.metrics.final_equity_jpy
    assert sum(trade.fee for trade in result.trades) == result.metrics.fees_paid_jpy


def test_higher_costs_cannot_improve_the_result(settings: Settings) -> None:
    result = run_backtest(up_then_down(), _settings(settings))

    stress = {item.cost_multiplier: item for item in result.cost_stress}
    assert set(stress) == {0.5, 1.0, 2.0, 4.0}
    assert stress[0.5].cagr_percent >= stress[1.0].cagr_percent >= stress[4.0].cagr_percent
    assert stress[4.0].fees_paid_jpy > stress[1.0].fees_paid_jpy


def test_calendar_periods_cover_every_simulated_year(settings: Settings) -> None:
    result = run_backtest(up_then_down(), _settings(settings))

    years = [period.year for period in result.periods]
    assert years == sorted(set(years))
    assert years[0] == result.metrics.start.year or years[0] == result.metrics.start.year + 1
    assert years[-1] == result.metrics.end.year


def test_start_date_limits_the_simulation_but_keeps_warm_history(settings: Settings) -> None:
    bars = up_then_down()
    start = bars[450].day

    result = run_backtest(bars, _settings(settings), start=start, include_cost_stress=False)

    assert result.metrics.start == start - timedelta(days=1)
    assert result.metrics.days == len(bars) - 450


def test_short_history_is_rejected_with_guidance(settings: Settings) -> None:
    with pytest.raises(ValueError, match="not enough daily history"):
        run_backtest(bars_from_closes(price_path(50)), _settings(settings))


def test_a_disabled_strategy_cannot_be_backtested(settings: Settings) -> None:
    with pytest.raises(ValueError, match="select a strategy"):
        run_backtest(up_then_down(), settings, strategy_name="none")
    with pytest.raises(ValueError, match="unknown strategy"):
        run_backtest(up_then_down(), settings, strategy_name="ema_crossover")
