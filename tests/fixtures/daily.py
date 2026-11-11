"""Deterministic daily-bar fixtures for the trend strategy."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from btc_trader.strategy import DailyBar

START_DAY = date(2024, 1, 1)


def price_path(
    days: int, *, drift: float = 0.004, amplitude: float = 0.02, start: float = 10_000_000
) -> list[float]:
    """A noisy but reproducible path: geometric drift plus a fixed oscillation."""
    return [
        start * math.exp(drift * index + amplitude * math.sin(index * 1.7) * math.cos(index * 0.3))
        for index in range(days)
    ]


def bars_from_closes(
    closes: list[float], *, start: date = START_DAY, source: str = "bitbank"
) -> list[DailyBar]:
    """Bars whose open is the previous close, as on a 24/7 market."""
    bars: list[DailyBar] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        bars.append(
            DailyBar(
                day=start + timedelta(days=index),
                open=Decimal(str(round(previous))),
                close=Decimal(str(round(close))),
                source=source,
            )
        )
        previous = close
    return bars


def up_then_down(up_days: int = 400, down_days: int = 120) -> list[DailyBar]:
    """A long rally followed by a decline steep enough to stop every sleeve out."""
    rally = price_path(up_days)
    decline = [rally[-1] * math.exp(-0.01 * (index + 1)) for index in range(down_days)]
    return bars_from_closes(rally + decline)


def day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)
