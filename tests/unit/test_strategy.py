from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from btc_trader.config import Settings, parse_horizons
from btc_trader.enums import TradingSignal
from btc_trader.strategy import (
    TrendParameters,
    evaluate_trend,
    parameters_from_settings,
    trend_path,
)
from tests.fixtures.daily import bars_from_closes, price_path, up_then_down

SHORT = TrendParameters(horizons=(5, 10, 20), volatility_window=20)


def test_no_decision_before_volatility_warm_up() -> None:
    bars = bars_from_closes(price_path(SHORT.warmup_bars - 1))

    assert evaluate_trend(bars, SHORT) is None
    assert evaluate_trend(bars_from_closes(price_path(SHORT.warmup_bars)), SHORT) is not None


def test_a_steady_rally_puts_every_sleeve_long_with_volatility_sized_exposure() -> None:
    bars = bars_from_closes(price_path(200, drift=0.01, amplitude=0.0))

    decision = evaluate_trend(bars, SHORT)

    assert decision is not None
    assert all(sleeve.active for sleeve in decision.sleeves)
    assert decision.active_fraction == Decimal("1")
    # A noiseless rally has ~zero volatility, so sizing saturates at the cap.
    assert decision.target_exposure == Decimal("1")
    assert decision.as_of == bars[-1].day


def test_exposure_is_target_volatility_over_realised_volatility() -> None:
    bars = bars_from_closes(price_path(300, drift=0.006, amplitude=0.05))
    parameters = TrendParameters(horizons=(5, 10, 20), volatility_window=20, target_volatility=0.2)

    decision = evaluate_trend(bars, parameters)

    assert decision is not None
    assert decision.realized_volatility is not None
    expected = float(decision.active_fraction) * min(0.2 / decision.realized_volatility, 1.0)
    assert float(decision.target_exposure) == pytest.approx(expected, abs=1e-6)


def test_higher_target_volatility_never_lowers_exposure() -> None:
    bars = bars_from_closes(price_path(300, drift=0.006, amplitude=0.05))
    low = trend_path(bars, TrendParameters(horizons=(5, 10, 20), volatility_window=20))
    high = trend_path(
        bars, TrendParameters(horizons=(5, 10, 20), volatility_window=20, target_volatility=0.5)
    )

    assert np.all(high.target >= low.target - 1e-12)
    assert np.all(high.target <= 1.0)


def test_a_decline_stops_every_sleeve_out_and_targets_cash() -> None:
    bars = up_then_down()

    decision = evaluate_trend(bars, TrendParameters())

    assert decision is not None
    assert not any(sleeve.active for sleeve in decision.sleeves)
    assert decision.target_exposure == Decimal("0")
    assert "0/9 trend sleeves long" in decision.reason


def test_trailing_stops_only_ratchet_upwards_while_long() -> None:
    path = trend_path(up_then_down(), TrendParameters())

    for row in range(len(path.parameters.horizons)):
        stops = path.stops[row]
        for index in range(1, len(stops)):
            if path.active[row, index] and path.active[row, index - 1]:
                assert stops[index] >= stops[index - 1]


def test_exits_happen_only_on_a_close_below_the_stop() -> None:
    path = trend_path(up_then_down(), TrendParameters())
    midpoints = (path.highs + path.lows) / 2

    rows, columns = np.nonzero(path.exited)
    assert len(rows) > 0
    for row, column in zip(rows, columns, strict=True):
        effective_stop = max(path.stops[row, column - 1], midpoints[row, column])
        assert path.closes[column] < effective_stop
    # And the converse: a long sleeve whose close holds above its stop stays long.
    held = path.active[:, 1:] & path.active[:, :-1]
    closes = np.broadcast_to(path.closes[1:], held.shape)
    assert np.all(closes[held] >= path.stops[:, 1:][held])


def test_decisions_never_depend_on_future_bars() -> None:
    """Look-ahead guard: a prefix of the history yields the same decisions."""
    bars = up_then_down()
    full = trend_path(bars, TrendParameters())

    for cut in (100, 250, 361, 450, len(bars) - 1):
        prefix = trend_path(bars[: cut + 1], TrendParameters())
        assert prefix.target[cut] == pytest.approx(full.target[cut])
        assert np.array_equal(prefix.active[:, cut], full.active[:, cut])


def test_signal_reports_the_direction_of_the_target_change() -> None:
    rally = bars_from_closes(price_path(60, drift=0.01, amplitude=0.0))
    decline = bars_from_closes(
        [float(rally[-1].close) * (0.97**i) for i in range(1, 30)], start=rally[-1].day
    )[1:]
    bars = rally + decline

    signals = [evaluate_trend(bars[:n], SHORT) for n in range(SHORT.warmup_bars, len(bars) + 1)]

    observed = {decision.signal for decision in signals if decision is not None}
    assert TradingSignal.SELL in observed
    for decision in signals:
        assert decision is not None
        if decision.target_exposure > decision.previous_target_exposure:
            assert decision.signal is TradingSignal.BUY
        elif decision.target_exposure < decision.previous_target_exposure:
            assert decision.signal is TradingSignal.SELL
        else:
            assert decision.signal is TradingSignal.HOLD


def test_long_horizons_report_warm_up_until_their_history_exists() -> None:
    decision = evaluate_trend(bars_from_closes(price_path(150)), TrendParameters())

    assert decision is not None
    warming = {sleeve.horizon for sleeve in decision.sleeves if sleeve.warming_up}
    assert warming == {250, 360}


def test_parameters_come_from_settings(settings: Settings) -> None:
    parameters = parameters_from_settings(
        settings.model_copy(
            update={
                "strategy_target_annual_volatility_percent": Decimal("40"),
                "strategy_max_exposure_percent": Decimal("60"),
            }
        )
    )

    assert parameters.horizons == (5, 10, 20, 30, 60, 90, 150, 250, 360)
    assert parameters.volatility_window == 90
    assert parameters.target_volatility == pytest.approx(0.4)
    assert parameters.max_exposure == pytest.approx(0.6)


def test_horizon_parsing_sorts_and_rejects_nonsense() -> None:
    assert parse_horizons("20, 5,10,5") == (5, 10, 20)
    for bad in ("", "a,b", "1", "5,2000"):
        with pytest.raises(ValueError):
            parse_horizons(bad)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        TrendParameters(max_exposure=1.5)
    with pytest.raises(ValueError):
        TrendParameters(volatility_window=5)
    with pytest.raises(ValueError):
        TrendParameters(target_volatility=0)
