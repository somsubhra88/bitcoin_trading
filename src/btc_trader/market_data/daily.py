"""Daily bar series for the trend strategy.

Two inputs, merged per UTC day:

* **own bars** — materialised from this platform's bitFlyer ticks once a day
  has completed (``candles``: exchange=bitflyer, interval=1d,
  source=sampled_ticks). They are preferred whenever the day was sampled well
  enough to trust its close;
* **external bars** — completed daily OHLC from the configured history
  provider (source=exchange_ohlc), used to bootstrap the year of history the
  strategy needs and to fill days the collector missed.

bitFlyer and bitbank prices differ by a small, slowly varying basis. When both
exist for overlapping days, external closes are rescaled by the median
own/external ratio so a splice cannot fabricate a breakout.

Only completed days are ever returned: today's partial bar would let an
intraday price masquerade as a daily close.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from statistics import median

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import Candle, MarketTick
from btc_trader.enums import CandleSource
from btc_trader.market_data.history_provider import (
    ExternalDailyBar,
    create_daily_history_provider,
)
from btc_trader.market_data.repository import _insert_ignore
from btc_trader.monitoring.logging import get_logger
from btc_trader.strategy import DailyBar
from btc_trader.utils.timeutils import utcnow

logger = get_logger(__name__)

DAILY = "1d"
OWN_EXCHANGE = "bitflyer"
#: Overlapping days used to estimate the bitFlyer/external basis.
BASIS_WINDOW_DAYS = 30
#: An own bar needs at least this share of the expected samples in its day.
MIN_SAMPLE_COVERAGE = Decimal("0.5")
#: Days of ticks examined per sync; bounds the work after a long outage.
MAX_SYNC_DAYS = 400


@dataclass(frozen=True, slots=True)
class DailySeries:
    bars: list[DailyBar]
    own_days: int
    external_days: int
    basis_ratio: Decimal | None

    @property
    def last_day(self) -> date | None:
        return self.bars[-1].day if self.bars else None


def _day_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


async def store_external_bars(
    session: AsyncSession, settings: Settings, exchange: str, bars: Sequence[ExternalDailyBar]
) -> int:
    """Idempotently insert completed external daily bars. Returns rows inserted."""
    if not bars:
        return 0
    existing_days = {
        _as_utc(value).date()
        for value in (
            await session.execute(
                select(Candle.bucket_start).where(
                    Candle.exchange == exchange,
                    Candle.product_code == settings.product_code,
                    Candle.interval == DAILY,
                    Candle.source == CandleSource.EXCHANGE_OHLC,
                    Candle.bucket_start >= _day_start(min(bar.day for bar in bars)),
                    Candle.bucket_start <= _day_start(max(bar.day for bar in bars)),
                )
            )
        ).scalars()
    }
    rows = [
        {
            "exchange": exchange,
            "product_code": settings.product_code,
            "interval": DAILY,
            "source": CandleSource.EXCHANGE_OHLC,
            "bucket_start": _day_start(bar.day),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "trade_count": 0,
        }
        for bar in bars
        if bar.day not in existing_days
    ]
    if rows:
        statement = _insert_ignore(
            session, Candle, ["exchange", "product_code", "interval", "source", "bucket_start"]
        )
        await session.execute(statement, rows)
    return len(rows)


async def sync_external_history(
    session: AsyncSession, settings: Settings, *, years: int | None = None
) -> int:
    """Fetch the configured provider's daily bars for recent calendar years."""
    provider = create_daily_history_provider(settings)
    if provider is None:
        return 0
    span = years or settings.daily_history_years
    current_year = utcnow().year
    inserted = 0
    try:
        for year in range(current_year - span + 1, current_year + 1):
            bars = await provider.fetch_year(year)
            inserted += await store_external_bars(session, settings, provider.name, bars)
    finally:
        await provider.aclose()
    logger.info("daily_history_synced", provider=provider.name, years=span, inserted=inserted)
    return inserted


async def refresh_external_history(session: AsyncSession, settings: Settings) -> int:
    """Cheap top-up: a full backfill when history is short, else this year only."""
    if settings.daily_history_provider == "none":
        return 0
    count = int(
        (
            await session.execute(
                select(func.count())
                .select_from(Candle)
                .where(
                    Candle.product_code == settings.product_code,
                    Candle.interval == DAILY,
                    Candle.source == CandleSource.EXCHANGE_OHLC,
                )
            )
        ).scalar_one()
    )
    needed = max(settings.trend_horizons[-1], settings.strategy_volatility_window_days) + 30
    # Two years so that on 1 January the just-completed 31 December is fetched.
    years = None if count < needed else 2
    return await sync_external_history(session, settings, years=years)


async def sync_own_daily_bars(
    session: AsyncSession, settings: Settings, *, today: date | None = None
) -> int:
    """Materialise completed UTC days of bitFlyer ticks into daily candles."""
    today = today or utcnow().date()
    last_stored = (
        await session.execute(
            select(func.max(Candle.bucket_start)).where(
                Candle.exchange == OWN_EXCHANGE,
                Candle.product_code == settings.product_code,
                Candle.interval == DAILY,
                Candle.source == CandleSource.SAMPLED_TICKS,
            )
        )
    ).scalar_one_or_none()
    first_tick = (
        await session.execute(
            select(func.min(MarketTick.exchange_timestamp)).where(
                MarketTick.product_code == settings.product_code,
                MarketTick.exchange == OWN_EXCHANGE,
            )
        )
    ).scalar_one_or_none()
    if first_tick is None:
        return 0
    start = (
        _as_utc(last_stored).date() + timedelta(days=1)
        if last_stored is not None
        else _as_utc(first_tick).date()
    )
    start = max(start, today - timedelta(days=MAX_SYNC_DAYS))
    inserted = 0
    day = start
    while day < today:
        ticks = list(
            (
                await session.execute(
                    select(MarketTick.exchange_timestamp, MarketTick.ltp)
                    .where(
                        MarketTick.product_code == settings.product_code,
                        MarketTick.exchange == OWN_EXCHANGE,
                        MarketTick.exchange_timestamp >= _day_start(day),
                        MarketTick.exchange_timestamp < _day_start(day + timedelta(days=1)),
                        MarketTick.ltp.is_not(None),
                    )
                    .order_by(MarketTick.exchange_timestamp)
                )
            ).all()
        )
        if ticks:
            prices = [price for _, price in ticks]
            statement = _insert_ignore(
                session, Candle, ["exchange", "product_code", "interval", "source", "bucket_start"]
            ).values(
                exchange=OWN_EXCHANGE,
                product_code=settings.product_code,
                interval=DAILY,
                source=CandleSource.SAMPLED_TICKS,
                bucket_start=_day_start(day),
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=Decimal("0"),
                trade_count=len(prices),
            )
            await session.execute(statement)
            inserted += 1
        day += timedelta(days=1)
    return inserted


async def load_daily_series(
    session: AsyncSession, settings: Settings, *, today: date | None = None
) -> DailySeries:
    """Merged completed daily bars, oldest first."""
    today = today or utcnow().date()
    rows = list(
        (
            await session.execute(
                select(Candle)
                .where(
                    Candle.product_code == settings.product_code,
                    Candle.interval == DAILY,
                    Candle.bucket_start < _day_start(today),
                    Candle.source.in_([CandleSource.SAMPLED_TICKS, CandleSource.EXCHANGE_OHLC]),
                )
                .order_by(Candle.bucket_start)
            )
        ).scalars()
    )
    minimum_samples = Decimal(86_400) / Decimal(settings.market_data_interval_seconds)
    minimum_samples *= MIN_SAMPLE_COVERAGE
    own: dict[date, Candle] = {}
    external: dict[date, Candle] = {}
    for row in rows:
        day = _as_utc(row.bucket_start).date()
        if row.source is CandleSource.SAMPLED_TICKS and row.exchange == OWN_EXCHANGE:
            if row.trade_count >= minimum_samples:
                own[day] = row
        elif row.source is CandleSource.EXCHANGE_OHLC:
            external.setdefault(day, row)

    overlap = sorted(set(own) & set(external))[-BASIS_WINDOW_DAYS:]
    ratio = (
        Decimal(str(median(float(own[d].close / external[d].close) for d in overlap)))
        if overlap
        else None
    )
    scale = ratio or Decimal("1")
    bars: list[DailyBar] = []
    for day in sorted(set(own) | set(external)):
        if day in own:
            candle = own[day]
            bars.append(DailyBar(day, candle.open, candle.close, OWN_EXCHANGE))
        else:
            candle = external[day]
            bars.append(DailyBar(day, candle.open * scale, candle.close * scale, candle.exchange))
    own_days = sum(bar.source == OWN_EXCHANGE for bar in bars)
    return DailySeries(
        bars=bars,
        own_days=own_days,
        external_days=len(bars) - own_days,
        basis_ratio=ratio,
    )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


__all__ = [
    "DailySeries",
    "load_daily_series",
    "refresh_external_history",
    "store_external_bars",
    "sync_external_history",
    "sync_own_daily_bars",
]
