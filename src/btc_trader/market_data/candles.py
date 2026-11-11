"""OHLC aggregation of stored ticker samples for the price chart (PROJECT.md §40).

These bars are built from periodic last-price samples, not exchange trade
prints: with a 5-minute interval a "1m" bar holds at most one observation and
volume is unknown. They are for display only; the strategy uses daily bars from
:mod:`btc_trader.market_data.daily`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from btc_trader.db.models import MarketTick
from btc_trader.utils.timeutils import floor_to_interval

#: Supported interval labels and their durations.
INTERVALS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


@dataclass(slots=True)
class SampledCandle:
    bucket_start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    samples: int


def candles_from_ticks(ticks: Sequence[MarketTick], interval: str) -> list[SampledCandle]:
    """Fold LTP samples into bars of ``interval`` (oldest first)."""
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; expected one of {sorted(INTERVALS)}")
    step = INTERVALS[interval]
    buckets: dict[datetime, SampledCandle] = {}
    for tick in sorted(ticks, key=lambda item: item.exchange_timestamp):
        if tick.ltp is None:
            continue
        start = floor_to_interval(tick.exchange_timestamp, step)
        bar = buckets.get(start)
        if bar is None:
            buckets[start] = SampledCandle(start, tick.ltp, tick.ltp, tick.ltp, tick.ltp, 1)
        else:
            bar.high = max(bar.high, tick.ltp)
            bar.low = min(bar.low, tick.ltp)
            bar.close = tick.ltp
            bar.samples += 1
    return [buckets[key] for key in sorted(buckets)]
