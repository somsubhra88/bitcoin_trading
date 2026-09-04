"""OHLCV aggregation and provenance (PROJECT.md §9, §39, §40).

The load-bearing assertion in this module is that tick-sampled bars are never
indistinguishable from exchange OHLC: they carry a different source, zero volume
and ``is_true_ohlc is False``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from btc_trader.db.models import MarketExecution, MarketTick
from btc_trader.enums import CandleSource, Side
from btc_trader.market_data.candles import (
    INTERVALS,
    candles_from_executions,
    candles_from_ticks,
    resolve_interval,
)

START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _execution(offset_seconds: int, price: str, size: str = "0.01") -> MarketExecution:
    return MarketExecution(
        exchange="bitflyer",
        product_code="BTC_JPY",
        exchange_execution_id=str(1000 + offset_seconds),
        side=Side.BUY,
        price=Decimal(price),
        size=Decimal(size),
        executed_at=START + timedelta(seconds=offset_seconds),
        received_at=START + timedelta(seconds=offset_seconds),
    )


def _tick(offset_seconds: int, ltp: str | None) -> MarketTick:
    return MarketTick(
        exchange="bitflyer",
        product_code="BTC_JPY",
        exchange_timestamp=START + timedelta(seconds=offset_seconds),
        received_at=START + timedelta(seconds=offset_seconds),
        ltp=Decimal(ltp) if ltp is not None else None,
    )


# --------------------------------------------------------------------------- #
def test_interval_table_matches_the_spec() -> None:
    assert set(INTERVALS) == {"1m", "5m", "15m", "1h", "4h", "1d"}


def test_resolve_interval_is_case_insensitive() -> None:
    assert resolve_interval(" 5M ") == timedelta(minutes=5)


def test_resolve_interval_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown interval"):
        resolve_interval("3m")


# --------------------------------------------------------------------------- #
# executions -> true OHLCV
# --------------------------------------------------------------------------- #
def test_ohlc_comes_from_first_and_last_prints_not_from_extremes() -> None:
    bars = candles_from_executions(
        [
            _execution(0, "100"),
            _execution(10, "120"),
            _execution(20, "90"),
            _execution(30, "110"),
        ],
        product_code="BTC_JPY",
        interval="5m",
    )
    assert len(bars) == 1
    bar = bars[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (
        Decimal("100"),
        Decimal("120"),
        Decimal("90"),
        Decimal("110"),
    )
    assert bar.trade_count == 4
    assert bar.volume == Decimal("0.04")
    assert bar.source is CandleSource.EXECUTIONS
    assert bar.is_true_ohlc is True
    assert bar.bucket_start == START


def test_out_of_order_input_is_sorted_before_aggregation() -> None:
    ordered = candles_from_executions(
        [_execution(0, "100"), _execution(60, "200")], product_code="BTC_JPY", interval="5m"
    )
    shuffled = candles_from_executions(
        [_execution(60, "200"), _execution(0, "100")], product_code="BTC_JPY", interval="5m"
    )
    assert shuffled == ordered
    assert shuffled[0].open == Decimal("100")
    assert shuffled[0].close == Decimal("200")


def test_prints_are_split_across_buckets_and_returned_oldest_first() -> None:
    bars = candles_from_executions(
        [_execution(0, "100"), _execution(400, "150"), _execution(700, "120")],
        product_code="BTC_JPY",
        interval="5m",
    )
    assert [bar.bucket_start for bar in bars] == [
        START,
        START + timedelta(minutes=5),
        START + timedelta(minutes=10),
    ]
    assert [bar.close for bar in bars] == [Decimal("100"), Decimal("150"), Decimal("120")]


def test_volume_sums_without_float_error() -> None:
    bars = candles_from_executions(
        [_execution(index, "100", "0.1") for index in range(10)],
        product_code="BTC_JPY",
        interval="5m",
    )
    assert bars[0].volume == Decimal("1.0")  # 0.1 * 10, exactly


def test_no_executions_means_no_bars() -> None:
    assert candles_from_executions([], product_code="BTC_JPY", interval="5m") == []


# --------------------------------------------------------------------------- #
# ticks -> explicitly-labelled sampled bars
# --------------------------------------------------------------------------- #
def test_sampled_bars_are_labelled_and_carry_zero_volume() -> None:
    bars = candles_from_ticks(
        [_tick(0, "100"), _tick(60, "130"), _tick(120, "110")],
        product_code="BTC_JPY",
        interval="5m",
    )
    assert len(bars) == 1
    bar = bars[0]
    assert bar.source is CandleSource.SAMPLED_TICKS
    assert bar.is_true_ohlc is False
    assert bar.volume == Decimal("0")
    assert (bar.open, bar.high, bar.low, bar.close) == (
        Decimal("100"),
        Decimal("130"),
        Decimal("100"),
        Decimal("110"),
    )
    assert bar.trade_count == 3  # observations, not trades


def test_ticks_without_a_price_are_skipped_not_defaulted() -> None:
    """§9: a missing LTP must not become a fabricated zero."""
    bars = candles_from_ticks(
        [_tick(0, None), _tick(60, "100"), _tick(120, None)],
        product_code="BTC_JPY",
        interval="5m",
    )
    assert len(bars) == 1
    assert bars[0].trade_count == 1
    assert bars[0].open == bars[0].close == Decimal("100")


def test_all_prices_missing_means_no_bars() -> None:
    assert candles_from_ticks([_tick(0, None)], product_code="BTC_JPY", interval="5m") == []


def test_a_single_sample_produces_a_flat_bar() -> None:
    """With a 5-minute poll a "1m" sampled bar holds one observation (§40)."""
    bars = candles_from_ticks([_tick(0, "12000000")], product_code="BTC_JPY", interval="1m")
    bar = bars[0]
    assert bar.open == bar.high == bar.low == bar.close == Decimal("12000000")
    assert bar.volume == Decimal("0")
    assert bar.is_true_ohlc is False


def test_sampled_and_execution_bars_are_never_equal() -> None:
    """Same prices, same bucket — provenance still separates them."""
    from_prints = candles_from_executions(
        [_execution(0, "100", "0")], product_code="BTC_JPY", interval="5m"
    )
    from_samples = candles_from_ticks([_tick(0, "100")], product_code="BTC_JPY", interval="5m")
    assert from_prints[0].source is not from_samples[0].source
    assert from_prints[0] != from_samples[0]
