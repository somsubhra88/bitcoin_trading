"""OHLCV aggregation (PROJECT.md §40).

Two provenances, never conflated:

* :data:`CandleSource.EXECUTIONS` — true OHLCV built from individual trade
  prints, with real traded volume;
* :data:`CandleSource.SAMPLED_TICKS` — bars derived from periodic LTP samples.
  With a 5-minute polling interval a "1m sampled candle" contains at most one
  observation, so these bars are *not* exchange OHLC and every consumer must
  label them as sampled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, get_settings
from btc_trader.db.models import MarketExecution, MarketTick
from btc_trader.enums import CandleSource
from btc_trader.market_data.repository import MarketDataRepository
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


def resolve_interval(interval: str) -> timedelta:
    key = interval.strip().lower()
    if key not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; expected one of {sorted(INTERVALS)}")
    return INTERVALS[key]


@dataclass(frozen=True, slots=True)
class Candle:
    """An OHLCV bar with explicit provenance."""

    product_code: str
    interval: str
    source: CandleSource
    bucket_start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int

    @property
    def is_true_ohlc(self) -> bool:
        """``False`` for tick-sampled bars, which must be labelled as sampled."""
        return self.source is CandleSource.EXECUTIONS


@dataclass(slots=True)
class _Accumulator:
    bucket_start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int

    def update(self, price: Decimal, size: Decimal) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.trade_count += 1


def _aggregate(
    observations: Iterable[tuple[datetime, Decimal, Decimal]],
    *,
    product_code: str,
    interval: str,
    source: CandleSource,
) -> list[Candle]:
    """Fold time-ordered ``(timestamp, price, size)`` triples into bars."""
    step = resolve_interval(interval)
    buckets: dict[datetime, _Accumulator] = {}

    for timestamp, price, size in observations:
        bucket_start = floor_to_interval(timestamp, step)
        accumulator = buckets.get(bucket_start)
        if accumulator is None:
            buckets[bucket_start] = _Accumulator(
                bucket_start=bucket_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
                trade_count=1,
            )
        else:
            accumulator.update(price, size)

    return [
        Candle(
            product_code=product_code,
            interval=interval,
            source=source,
            bucket_start=accumulator.bucket_start,
            open=accumulator.open,
            high=accumulator.high,
            low=accumulator.low,
            close=accumulator.close,
            volume=accumulator.volume,
            trade_count=accumulator.trade_count,
        )
        for accumulator in sorted(buckets.values(), key=lambda item: item.bucket_start)
    ]


def candles_from_executions(
    executions: Sequence[MarketExecution], *, product_code: str, interval: str
) -> list[Candle]:
    """True OHLCV from trade prints."""
    ordered = sorted(executions, key=lambda execution: execution.executed_at)
    return _aggregate(
        ((item.executed_at, item.price, item.size) for item in ordered),
        product_code=product_code,
        interval=interval,
        source=CandleSource.EXECUTIONS,
    )


def candles_from_ticks(
    ticks: Sequence[MarketTick], *, product_code: str, interval: str
) -> list[Candle]:
    """Sampled bars from LTP observations.

    Volume is always zero: an LTP sample carries no traded size, and inventing
    one would misrepresent the data (PROJECT.md §9, §40).
    """
    ordered = sorted(
        (tick for tick in ticks if tick.ltp is not None),
        key=lambda tick: tick.exchange_timestamp,
    )
    return _aggregate(
        (
            (tick.exchange_timestamp, tick.ltp, Decimal("0"))
            for tick in ordered
            if tick.ltp is not None
        ),
        product_code=product_code,
        interval=interval,
        source=CandleSource.SAMPLED_TICKS,
    )


class CandleService:
    """Aggregates stored observations on demand.

    Execution data is preferred; tick sampling is the explicitly-labelled
    fallback when no executions are stored for the window.
    """

    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._repository = MarketDataRepository(session)

    async def get_candles(
        self,
        *,
        product_code: str | None = None,
        interval: str = "5m",
        start: datetime | None = None,
        end: datetime | None = None,
        prefer_source: CandleSource | None = None,
    ) -> list[Candle]:
        code = product_code or self._settings.product_code
        resolve_interval(interval)  # validate early

        if prefer_source is not CandleSource.SAMPLED_TICKS:
            executions = await self._repository.executions_between(code, start=start, end=end)
            if executions:
                return candles_from_executions(executions, product_code=code, interval=interval)
            if prefer_source is CandleSource.EXECUTIONS:
                return []

        ticks = await self._repository.ticks_between(code, start=start, end=end)
        return candles_from_ticks(ticks, product_code=code, interval=interval)
