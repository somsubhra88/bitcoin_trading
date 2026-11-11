"""Dashboard value formatting (PROJECT.md §13, §15, §38).

The dashboard is the one place where money becomes text, so the rules that
matter are: a JSON *string* price is parsed exactly, a missing value renders as
a dash rather than a zero (§9), and UTC instants are shown in the configured
timezone (§38).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
import pytest

from dashboard.app import (
    CANDLE_INTERVALS,
    DEFAULT_INTERVAL_FOR_PERIOD,
    PERIOD_LABELS,
    _as_float,
    _decimal,
    _detail,
    _duration,
    _icon,
    _local,
    _moment,
    _money,
    _percent,
    _period_start,
    _quantity,
)

TOKYO = ZoneInfo("Asia/Tokyo")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12411935.0", Decimal("12411935.0")),
        ("0.023552", Decimal("0.023552")),
        ("-1", Decimal("-1")),
        (0, Decimal(0)),
    ],
)
def test_decimal_parses_api_money_exactly(value: object, expected: Decimal) -> None:
    """§13: the API sends money as a string precisely so no float appears here."""
    parsed = _decimal(value)

    assert parsed == expected
    assert isinstance(parsed, Decimal)


@pytest.mark.parametrize("value", [None, "", "n/a", "12,000", object()])
def test_unparseable_money_is_none_not_zero(value: object) -> None:
    """§9: a missing or broken field must never be rendered as a real number."""
    assert _decimal(value) is None


def test_a_big_price_keeps_every_digit() -> None:
    """A float round-trip of this value would lose the last digits."""
    assert _decimal("12345678901234.123456789") == Decimal("12345678901234.123456789")


def test_moment_normalises_to_utc() -> None:
    assert _moment("2026-09-01T17:47:23.553000Z") == datetime(
        2026, 9, 1, 17, 47, 23, 553000, tzinfo=UTC
    )


@pytest.mark.parametrize("value", [None, "", "not-a-date"])
def test_a_broken_timestamp_is_none(value: object) -> None:
    assert _moment(value) is None


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_local_renders_in_the_configured_timezone() -> None:
    """§38: stored UTC, displayed local — 17:47 UTC is 02:47 the next day in Tokyo."""
    assert _local("2026-09-01T17:47:23Z", TOKYO) == "2026-09-02 02:47:23"
    assert _local("2026-09-01T17:47:23Z", TOKYO, with_seconds=False) == "2026-09-02 02:47"


def test_local_renders_a_missing_time_as_a_dash() -> None:
    assert _local(None, TOKYO) == "—"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12411935.0", "¥12,411,935"),
        ("12411935.5", "¥12,411,936"),  # half-even rounding, display only
        ("-5000", "¥-5,000"),
        (None, "—"),
    ],
)
def test_money_is_grouped_and_prefixed(value: object, expected: str) -> None:
    assert _money(value) == expected


def test_money_can_keep_decimal_places() -> None:
    assert _money("1234.5678", places=2) == "¥1,234.57"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0.023552", "0.023552"), ("1.10000000", "1.1"), (None, "—")],
)
def test_quantity_drops_trailing_zeros(value: object, expected: str) -> None:
    assert _quantity(value) == expected


def test_a_zero_quantity_is_shown_at_full_precision() -> None:
    """A flat position is a fact worth showing as 0.00000000, not as "0"."""
    assert _quantity("0") == "0.00000000"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.5", "+1.50%"), ("-0.074", "-0.07%"), ("0", "+0.00%"), (None, "—")],
)
def test_percent_always_carries_a_sign(value: object, expected: str) -> None:
    assert _percent(value) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "—"), (0, "0s"), (59.9, "59s"), (60, "1m 0s"), (3599, "59m 59s"), (7325, "2h 2m")],
)
def test_duration_is_human_readable(seconds: float | None, expected: str) -> None:
    assert _duration(seconds) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [("ok", "🟢"), ("OK", "🟢"), ("degraded", "🟡"), ("error", "🔴"), ("disabled", "⚪")],
)
def test_status_icons_are_case_insensitive(state: str, expected: str) -> None:
    assert _icon(state) == expected


@pytest.mark.parametrize("state", [None, "something-new"])
def test_an_unknown_state_gets_the_neutral_icon(state: str | None) -> None:
    assert _icon(state) == "⚪"


# --------------------------------------------------------------------------- #
# chart plumbing
# --------------------------------------------------------------------------- #
def test_chart_coordinates_are_floats() -> None:
    """Plotly cannot plot Decimals; this is display maths only (§13)."""
    assert _as_float("12411935.0") == pytest.approx(12411935.0)
    assert _as_float(None) is None


def test_every_period_has_a_default_interval_that_the_api_accepts() -> None:
    assert set(DEFAULT_INTERVAL_FOR_PERIOD) == set(PERIOD_LABELS)
    assert set(DEFAULT_INTERVAL_FOR_PERIOD.values()) <= set(CANDLE_INTERVALS)


@pytest.mark.parametrize(
    ("period", "span"),
    [
        ("1h", timedelta(hours=1)),
        ("6h", timedelta(hours=6)),
        ("24h", timedelta(days=1)),
        ("7d", timedelta(days=7)),
        ("30d", timedelta(days=30)),
    ],
)
def test_period_start_is_that_far_back(period: str, span: timedelta) -> None:
    start = _period_start(period)

    assert start is not None
    assert abs((datetime.now(UTC) - start) - span) < timedelta(seconds=5)


def test_the_all_period_has_no_lower_bound() -> None:
    assert _period_start("all") is None


# --------------------------------------------------------------------------- #
# error surfacing
# --------------------------------------------------------------------------- #
def test_an_api_error_detail_is_extracted() -> None:
    response = httpx.Response(422, json={"detail": "unknown period '17y'"})

    assert _detail(response) == "unknown period '17y'"


def test_a_non_json_error_body_falls_back_to_text() -> None:
    response = httpx.Response(500, text="Internal Server Error")

    assert _detail(response) == "Internal Server Error"


def test_an_error_detail_is_truncated() -> None:
    response = httpx.Response(500, json={"detail": "x" * 5000})

    assert len(_detail(response)) == 200
