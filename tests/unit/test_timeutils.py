"""UTC time handling (PROJECT.md §38)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from btc_trader.utils.timeutils import (
    age_seconds,
    ensure_utc,
    floor_to_interval,
    parse_bitflyer_timestamp,
    to_display_timezone,
    utcnow,
)


def test_utcnow_is_aware_and_utc() -> None:
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # bitFlyer sends a variable number of fractional digits, and no offset.
        ("2026-09-01T17:47:23.553", datetime(2026, 9, 1, 17, 47, 23, 553000, tzinfo=UTC)),
        ("2026-09-01T17:46:45.98", datetime(2026, 9, 1, 17, 46, 45, 980000, tzinfo=UTC)),
        ("2026-09-01T17:46:45", datetime(2026, 9, 1, 17, 46, 45, tzinfo=UTC)),
        ("2026-09-01T17:46:45Z", datetime(2026, 9, 1, 17, 46, 45, tzinfo=UTC)),
        ("  2026-09-01T17:46:45.5  ", datetime(2026, 9, 1, 17, 46, 45, 500000, tzinfo=UTC)),
    ],
)
def test_parse_bitflyer_timestamp(raw: str, expected: datetime) -> None:
    parsed = parse_bitflyer_timestamp(raw)
    assert parsed == expected
    assert parsed.tzinfo is not None


def test_parse_bitflyer_timestamp_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        parse_bitflyer_timestamp("not-a-timestamp")


def test_naive_input_is_treated_as_utc() -> None:
    assert ensure_utc(datetime(2026, 9, 1, 12, 0)) == datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_offset_input_is_converted_not_relabelled() -> None:
    tokyo = datetime(2026, 9, 1, 21, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert ensure_utc(tokyo) == datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_to_display_timezone_keeps_the_instant() -> None:
    moment = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    displayed = to_display_timezone(moment, "Asia/Tokyo")
    assert displayed.hour == 21
    assert displayed == moment


def test_age_seconds_uses_an_injected_reference() -> None:
    moment = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    assert age_seconds(moment, now=moment + timedelta(seconds=90)) == 90.0


def test_age_seconds_across_timezones() -> None:
    """Age must not depend on the tzinfo the caller happened to use."""
    moment = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    reference = datetime(2026, 9, 1, 21, 5, tzinfo=timezone(timedelta(hours=9)))
    assert age_seconds(moment, now=reference) == 300.0


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (timedelta(minutes=1), datetime(2026, 9, 1, 12, 34, tzinfo=UTC)),
        (timedelta(minutes=5), datetime(2026, 9, 1, 12, 30, tzinfo=UTC)),
        (timedelta(minutes=15), datetime(2026, 9, 1, 12, 30, tzinfo=UTC)),
        (timedelta(hours=1), datetime(2026, 9, 1, 12, 0, tzinfo=UTC)),
        (timedelta(hours=4), datetime(2026, 9, 1, 12, 0, tzinfo=UTC)),
        (timedelta(days=1), datetime(2026, 9, 1, 0, 0, tzinfo=UTC)),
    ],
)
def test_floor_to_interval(interval: timedelta, expected: datetime) -> None:
    moment = datetime(2026, 9, 1, 12, 34, 56, 789000, tzinfo=UTC)
    assert floor_to_interval(moment, interval) == expected


def test_floor_to_interval_is_epoch_anchored() -> None:
    """Buckets must be identical in every process, not relative to "now"."""
    assert floor_to_interval(datetime(1970, 1, 1, 0, 3, 30, tzinfo=UTC), timedelta(minutes=5)) == (
        datetime(1970, 1, 1, 0, 0, tzinfo=UTC)
    )


def test_floor_to_interval_is_idempotent() -> None:
    interval = timedelta(minutes=15)
    once = floor_to_interval(datetime(2026, 9, 1, 12, 34, tzinfo=UTC), interval)
    assert floor_to_interval(once, interval) == once


def test_floor_to_interval_rejects_a_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        floor_to_interval(utcnow(), timedelta(0))
