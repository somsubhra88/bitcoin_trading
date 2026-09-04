"""Market-data collection (PROJECT.md §10).

One implementation, three callers: the scheduler, ``POST /api/v1/market/collect``
and ``python -m btc_trader.scripts.collect_once``.  Collection logic is never
duplicated (§11).

Guarantees:

* the ticker is always attempted; the order book and executions are optional
  extras whose failure degrades the run but never loses the tick;
* run bookkeeping uses its own transactions, so a failed or partially-failed
  collection still leaves an auditable ``collector_runs`` row and a
  ``system_events`` entry;
* the error is re-raised to the caller after being recorded — the scheduler
  swallows it so one bad request can never kill the job (§10.8).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, get_settings
from btc_trader.db.models import CollectorRun
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.db.session import session_scope
from btc_trader.enums import Severity
from btc_trader.errors import ExchangeError, MarketDataError
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.registry import create_exchange_client
from btc_trader.market_data.repository import CollectorRunRepository, MarketDataRepository
from btc_trader.monitoring.logging import get_logger

logger = get_logger(__name__)

Trigger = Literal["scheduled", "manual", "cli"]

COMPONENT = "market_data_collector"


@dataclass(slots=True)
class CollectionResult:
    """Outcome of one collection attempt."""

    success: bool
    trigger: Trigger
    product_code: str
    exchange: str
    started_at: datetime
    duration_ms: int
    tick_inserted: bool = False
    tick_duplicate: bool = False
    executions_fetched: int = 0
    executions_inserted: int = 0
    order_book_inserted: bool = False
    ltp: Decimal | None = None
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    exchange_timestamp: datetime | None = None
    degraded: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    def as_log_context(self) -> dict[str, object]:
        return {
            "trigger": self.trigger,
            "product_code": self.product_code,
            "duration_ms": self.duration_ms,
            "tick_inserted": self.tick_inserted,
            "tick_duplicate": self.tick_duplicate,
            "executions_fetched": self.executions_fetched,
            "executions_inserted": self.executions_inserted,
            "order_book_inserted": self.order_book_inserted,
            "ltp": format(self.ltp, "f") if self.ltp is not None else None,
            "degraded": self.degraded or None,
        }


@dataclass(slots=True)
class _StoreOutcome:
    tick_inserted: bool
    executions_fetched: int
    executions_inserted: int
    order_book_inserted: bool
    degraded: list[str]
    ltp: Decimal | None
    best_bid: Decimal | None
    best_ask: Decimal | None
    exchange_timestamp: datetime


class MarketDataCollector:
    """Collects one observation set per invocation.

    The exchange client may be injected (tests, or a long-lived client shared by
    the scheduler); when omitted a client is created on first use and closed by
    :meth:`aclose`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        exchange_client: ExchangeClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = exchange_client
        self._owns_client = exchange_client is None

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> ExchangeClient:
        if self._client is None:
            self._client = create_exchange_client(self._settings)
        return self._client

    # ------------------------------------------------------------------ #
    async def collect(self, trigger: Trigger = "scheduled") -> CollectionResult:
        """Fetch, normalise and store one set of market observations."""
        settings = self._settings
        client = self._get_client()
        product_code = settings.product_code
        started_perf = time.perf_counter()

        run_id, started_at = await self._start_run(client.name, product_code, trigger)

        try:
            outcome = await self._fetch_and_store(client, product_code)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_perf) * 1000)
            await self._record_failure(run_id, exc, product_code, trigger, duration_ms)
            logger.error(
                "market_data_collection_failed",
                trigger=trigger,
                product_code=product_code,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if isinstance(exc, MarketDataError):
                raise
            raise MarketDataError(f"market data collection failed: {exc}") from exc

        duration_ms = int((time.perf_counter() - started_perf) * 1000)
        await self._record_success(run_id, outcome, product_code)

        result = CollectionResult(
            success=True,
            trigger=trigger,
            product_code=product_code,
            exchange=client.name,
            started_at=started_at,
            duration_ms=duration_ms,
            tick_inserted=outcome.tick_inserted,
            tick_duplicate=not outcome.tick_inserted,
            executions_fetched=outcome.executions_fetched,
            executions_inserted=outcome.executions_inserted,
            order_book_inserted=outcome.order_book_inserted,
            ltp=outcome.ltp,
            best_bid=outcome.best_bid,
            best_ask=outcome.best_ask,
            exchange_timestamp=outcome.exchange_timestamp,
            degraded=outcome.degraded,
        )
        logger.info("market_data_collected", **result.as_log_context())
        return result

    # ------------------------------------------------------------------ #
    async def _fetch_and_store(self, client: ExchangeClient, product_code: str) -> _StoreOutcome:
        settings = self._settings
        degraded: list[str] = []

        # --- ticker: mandatory (PROJECT.md §9) --------------------------
        ticker = await client.get_ticker(product_code)

        # --- order book: optional --------------------------------------
        book = None
        if settings.collect_order_book:
            try:
                book = await client.get_order_book(product_code)
            except ExchangeError as exc:
                degraded.append("order_book")
                logger.warning(
                    "order_book_collection_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        # --- executions: optional --------------------------------------
        executions = None
        if settings.collect_executions:
            try:
                # Polling the newest page loses trades when activity exceeds the
                # page size. Continue from our high-water mark instead.
                async with session_scope(settings) as session:
                    after_id = await MarketDataRepository(session).latest_execution_id(
                        product_code, client.name
                    )
                executions = await client.get_recent_executions(
                    product_code,
                    limit=settings.executions_fetch_limit,
                    after_id=after_id,
                )
            except ExchangeError as exc:
                degraded.append("executions")
                logger.warning(
                    "executions_collection_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        async with session_scope(settings) as session:
            repository = MarketDataRepository(session)
            tick_inserted = await repository.insert_tick(ticker)
            order_book_inserted = (
                await repository.insert_order_book_snapshot(book, settings.order_book_depth_levels)
                if book is not None
                else False
            )
            executions_inserted = (
                await repository.insert_executions(executions) if executions else 0
            )

        return _StoreOutcome(
            tick_inserted=tick_inserted,
            executions_fetched=len(executions) if executions else 0,
            executions_inserted=executions_inserted,
            order_book_inserted=order_book_inserted,
            degraded=degraded,
            ltp=ticker.ltp,
            best_bid=ticker.best_bid,
            best_ask=ticker.best_ask,
            exchange_timestamp=ticker.exchange_timestamp,
        )

    # ------------------------------------------------------------------ #
    # run bookkeeping — each step in its own transaction so it survives a
    # failure of the collection itself.
    # ------------------------------------------------------------------ #
    async def _start_run(
        self, exchange: str, product_code: str, trigger: Trigger
    ) -> tuple[int, datetime]:
        async with session_scope(self._settings) as session:
            run = await CollectorRunRepository(session).start(
                exchange=exchange, product_code=product_code, trigger=trigger
            )
            return run.id, run.started_at

    async def _record_success(self, run_id: int, outcome: _StoreOutcome, product_code: str) -> None:
        async with session_scope(self._settings) as session:
            run = await self._load_run(session, run_id)
            if run is not None:
                await CollectorRunRepository(session).finish(
                    run,
                    success=True,
                    ticks_inserted=int(outcome.tick_inserted),
                    executions_inserted=outcome.executions_inserted,
                    order_books_inserted=int(outcome.order_book_inserted),
                    error_message=(
                        "; ".join(f"degraded:{item}" for item in outcome.degraded) or None
                    ),
                )
            if outcome.degraded:
                await SystemEventRepository(session).record(
                    severity=Severity.WARNING,
                    component=COMPONENT,
                    event_type="collection_degraded",
                    message=f"collection succeeded without: {', '.join(outcome.degraded)}",
                    metadata={"product_code": product_code, "degraded": outcome.degraded},
                )

    async def _record_failure(
        self,
        run_id: int,
        exc: BaseException,
        product_code: str,
        trigger: Trigger,
        duration_ms: int,
    ) -> None:
        try:
            async with session_scope(self._settings) as session:
                run = await self._load_run(session, run_id)
                if run is not None:
                    await CollectorRunRepository(session).finish(
                        run,
                        success=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                await SystemEventRepository(session).record(
                    severity=Severity.ERROR,
                    component=COMPONENT,
                    event_type="collector_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    metadata={
                        "product_code": product_code,
                        "trigger": trigger,
                        "duration_ms": duration_ms,
                    },
                )
        except Exception as bookkeeping_error:  # noqa: BLE001 - must not mask the cause
            logger.error(
                "collector_failure_not_recorded",
                original_error=str(exc),
                bookkeeping_error=str(bookkeeping_error),
            )

    @staticmethod
    async def _load_run(session: AsyncSession, run_id: int) -> CollectorRun | None:
        statement = select(CollectorRun).where(CollectorRun.id == run_id)
        return (await session.execute(statement)).scalar_one_or_none()


async def collect_market_data(
    trigger: Trigger = "scheduled",
    *,
    settings: Settings | None = None,
    exchange_client: ExchangeClient | None = None,
) -> CollectionResult:
    """Module-level entry point matching PROJECT.md §10."""
    collector = MarketDataCollector(settings, exchange_client=exchange_client)
    try:
        return await collector.collect(trigger)
    finally:
        await collector.aclose()
