"""Chart bars from sampled ticker observations (PROJECT.md §40)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from btc_trader.db.models import MarketTick
from btc_trader.market_data.candles import INTERVALS, candles_from_ticks

START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _tick(offset_seconds: int, ltp: str | None) -> MarketTick:
    return MarketTick(
        exchange="bitflyer",
        product_code="BTC_JPY",
        exchange_timestamp=START + timedelta(seconds=offset_seconds),
        received_at=START + timedelta(seconds=offset_seconds),
        ltp=Decimal(ltp) if ltp is not None else None,
    )


def test_interval_table_matches_the_spec() -> None:
    assert set(INTERVALS) == {"1m", "5m", "15m", "1h", "4h", "1d"}


def test_unknown_intervals_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown interval"):
        candles_from_ticks([], "3m")


def test_ohlc_follows_sample_order_not_extremes() -> None:
    bars = candles_from_ticks(
        [_tick(120, "110"), _tick(0, "100"), _tick(60, "130"), _tick(90, "95")], "5m"
    )

    assert len(bars) == 1
    bar = bars[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (
        Decimal("100"),
        Decimal("130"),
        Decimal("95"),
        Decimal("110"),
    )
    assert bar.samples == 4
    assert bar.bucket_start == START


def test_samples_split_across_buckets_oldest_first() -> None:
    bars = candles_from_ticks([_tick(0, "100"), _tick(400, "150"), _tick(700, "120")], "5m")

    assert [bar.bucket_start for bar in bars] == [
        START,
        START + timedelta(minutes=5),
        START + timedelta(minutes=10),
    ]
    assert [bar.close for bar in bars] == [Decimal("100"), Decimal("150"), Decimal("120")]


def test_ticks_without_a_price_are_skipped_not_defaulted() -> None:
    """§9: a missing LTP must not become a fabricated zero."""
    bars = candles_from_ticks([_tick(0, None), _tick(60, "100"), _tick(120, None)], "5m")

    assert len(bars) == 1
    assert bars[0].samples == 1
    assert candles_from_ticks([_tick(0, None)], "5m") == []
