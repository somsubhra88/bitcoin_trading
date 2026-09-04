"""UTC-only time helpers (PROJECT.md §38).

Naive datetimes never enter the database.  bitFlyer returns ISO-8601 timestamps
in UTC *without* an offset suffix, so parsing must attach UTC explicitly rather
than letting Python produce a naive value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime.

    A naive input is *assumed* to already be UTC — true for every bitFlyer
    timestamp and for values round-tripped through SQLite.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_bitflyer_timestamp(value: str) -> datetime:
    """Parse a bitFlyer timestamp such as ``2026-09-01T17:19:23.81``.

    bitFlyer emits a variable number of fractional-second digits and sometimes
    omits the fraction entirely; ``fromisoformat`` on 3.11+ handles both, but
    the value is always naive UTC and must be localised.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unparseable bitFlyer timestamp: {value!r}") from exc
    return ensure_utc(parsed)


def to_display_timezone(value: datetime, timezone_name: str) -> datetime:
    """Convert an aware UTC datetime into a display timezone (e.g. Asia/Tokyo)."""
    return ensure_utc(value).astimezone(ZoneInfo(timezone_name))


def age_seconds(value: datetime, *, now: datetime | None = None) -> float:
    """Seconds elapsed since ``value``."""
    reference = now or utcnow()
    return (ensure_utc(reference) - ensure_utc(value)).total_seconds()


def floor_to_interval(value: datetime, interval: timedelta) -> datetime:
    """Floor an aware datetime to the start of its ``interval`` bucket.

    Buckets are anchored at the Unix epoch so they are stable across processes.
    """
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    reference = ensure_utc(value)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    step = int(interval.total_seconds())
    elapsed = int((reference - epoch).total_seconds())
    return epoch + timedelta(seconds=elapsed - (elapsed % step))
