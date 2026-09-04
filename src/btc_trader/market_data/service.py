"""Read-side market-data service.

The API and the dashboard consume this service, which reads from the database
only — never from the exchange (PROJECT.md §2.1, §16).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, get_settings
from btc_trader.db.models import CollectorRun, MarketTick, OrderBookSnapshot, SystemEvent
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.market_data.repository import CollectorRunRepository, MarketDataRepository
from btc_trader.utils.timeutils import age_seconds, utcnow

#: Dashboard/API period selectors (PROJECT.md §15) -> lookback window.
PERIODS: dict[str, timedelta | None] = {
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}


def resolve_period(period: str) -> timedelta | None:
    """Translate a period label into a lookback window."""
    key = period.strip().lower()
    if key not in PERIODS:
        raise ValueError(f"unknown period {period!r}; expected one of {sorted(PERIODS)}")
    return PERIODS[key]


@dataclass(slots=True)
class MarketOverview:
    """Everything the dashboard "Market Overview" panel needs."""

    product_code: str
    exchange: str | None
    latest_timestamp: datetime | None
    data_age_seconds: float | None
    is_stale: bool
    ltp: Decimal | None
    best_bid: Decimal | None
    best_ask: Decimal | None
    spread: Decimal | None
    spread_percent: Decimal | None
    volume_24h: Decimal | None
    change_absolute: Decimal | None
    change_percent: Decimal | None
    change_window: str
    stored_tick_count: int
    first_stored_timestamp: datetime | None


@dataclass(slots=True)
class CollectorStatus:
    last_run_at: datetime | None
    last_run_success: bool | None
    last_successful_run_at: datetime | None
    last_run_duration_ms: int | None
    last_error_type: str | None
    last_error_message: str | None
    interval_seconds: int
    seconds_since_last_success: float | None


class MarketDataService:
    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._repository = MarketDataRepository(session)
        self._runs = CollectorRunRepository(session)
        self._events = SystemEventRepository(session)

    # ------------------------------------------------------------------ #
    async def latest_tick(self, product_code: str | None = None) -> MarketTick | None:
        return await self._repository.latest_tick(product_code or self._settings.product_code)

    async def latest_order_book(self, product_code: str | None = None) -> OrderBookSnapshot | None:
        return await self._repository.latest_order_book(product_code or self._settings.product_code)

    async def history(
        self,
        *,
        product_code: str | None = None,
        period: str = "24h",
        limit: int = 5000,
    ) -> list[MarketTick]:
        code = product_code or self._settings.product_code
        window = resolve_period(period)
        start = utcnow() - window if window else None
        return await self._repository.ticks_between(code, start=start, limit=limit)

    async def overview(
        self, *, product_code: str | None = None, change_window: str = "24h"
    ) -> MarketOverview:
        code = product_code or self._settings.product_code
        latest = await self._repository.latest_tick(code)
        count = await self._repository.count_ticks(code)
        first_timestamp, _ = await self._repository.tick_time_range(code)

        if latest is None:
            return MarketOverview(
                product_code=code,
                exchange=None,
                latest_timestamp=None,
                data_age_seconds=None,
                is_stale=True,
                ltp=None,
                best_bid=None,
                best_ask=None,
                spread=None,
                spread_percent=None,
                volume_24h=None,
                change_absolute=None,
                change_percent=None,
                change_window=change_window,
                stored_tick_count=count,
                first_stored_timestamp=first_timestamp,
            )

        age = age_seconds(latest.exchange_timestamp)
        spread = latest.spread
        spread_percent = None
        if spread is not None and latest.best_bid:
            spread_percent = (spread / latest.best_bid) * Decimal(100)

        window = resolve_period(change_window)
        baseline: Decimal | None = None
        if window is not None:
            baseline = await self._repository.price_at_or_before(
                code, latest.exchange_timestamp - window
            )
        elif first_timestamp is not None:
            baseline = await self._repository.price_at_or_before(code, first_timestamp)

        change_absolute = None
        change_percent = None
        if baseline is not None and latest.ltp is not None and baseline != 0:
            change_absolute = latest.ltp - baseline
            change_percent = (change_absolute / baseline) * Decimal(100)

        return MarketOverview(
            product_code=code,
            exchange=latest.exchange,
            latest_timestamp=latest.exchange_timestamp,
            data_age_seconds=age,
            is_stale=age > self._settings.max_market_data_age_seconds,
            ltp=latest.ltp,
            best_bid=latest.best_bid,
            best_ask=latest.best_ask,
            spread=spread,
            spread_percent=spread_percent,
            volume_24h=latest.volume,
            change_absolute=change_absolute,
            change_percent=change_percent,
            change_window=change_window,
            stored_tick_count=count,
            first_stored_timestamp=first_timestamp,
        )

    async def collector_status(self) -> CollectorStatus:
        last_run: CollectorRun | None = await self._runs.last_run()
        last_success: CollectorRun | None = await self._runs.last_successful_run()
        return CollectorStatus(
            last_run_at=last_run.started_at if last_run else None,
            last_run_success=last_run.success if last_run else None,
            last_successful_run_at=last_success.started_at if last_success else None,
            last_run_duration_ms=last_run.duration_ms if last_run else None,
            last_error_type=last_run.error_type if last_run else None,
            last_error_message=last_run.error_message if last_run else None,
            interval_seconds=self._settings.market_data_interval_seconds,
            seconds_since_last_success=(
                age_seconds(last_success.started_at) if last_success else None
            ),
        )

    async def recent_collector_runs(self, limit: int = 20) -> list[CollectorRun]:
        return await self._runs.recent_runs(limit=limit)

    async def recent_events(self, limit: int = 25) -> list[SystemEvent]:
        return await self._events.recent(limit=limit)

    async def last_error_event(self) -> SystemEvent | None:
        return await self._events.last_error()

    async def stored_execution_count(self, product_code: str | None = None) -> int:
        return await self._repository.count_executions(product_code or self._settings.product_code)
