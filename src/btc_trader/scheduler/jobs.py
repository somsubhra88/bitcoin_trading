"""APScheduler wiring (PROJECT.md §14).

Every job is registered with ``max_instances=1`` and ``coalesce=True``, so a
collection that outlives its interval is never run concurrently with itself and
a backlog of missed runs collapses into a single execution.

Jobs never raise: an exception inside a job is logged and recorded, because a
propagating error would take the scheduler thread's job with it (§10.8).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, JobEvent
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from btc_trader.config import Settings, get_settings
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.db.session import session_scope
from btc_trader.enums import Severity, TradingMode
from btc_trader.exchange.base import ExchangeClient
from btc_trader.market_data.collector import MarketDataCollector
from btc_trader.monitoring.logging import get_logger
from btc_trader.paper import snapshot_latest
from btc_trader.runtime_settings import get_runtime_settings
from btc_trader.strategy_service import evaluate_and_store
from btc_trader.utils.timeutils import utcnow

logger = get_logger(__name__)

COMPONENT = "scheduler"

JOB_MARKET_DATA = "market_data_collection_job"
JOB_STRATEGY = "strategy_evaluation_job"
JOB_PORTFOLIO_SNAPSHOT = "portfolio_snapshot_job"
JOB_RECONCILIATION = "exchange_reconciliation_job"

#: Tolerance for a late job before APScheduler drops the run, seconds.
MISFIRE_GRACE_SECONDS = 60


class SchedulerService:
    """Owns the APScheduler instance and the long-lived exchange client.

    ``exchange_client`` is injectable for the same reason the collector's is: it
    lets the scheduling behaviour be exercised without any network traffic, and
    a caller that already holds a client can share it.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        exchange_client: ExchangeClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._exchange_client = exchange_client
        self._scheduler = AsyncIOScheduler(
            executors={"default": AsyncIOExecutor()},
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": MISFIRE_GRACE_SECONDS,
            },
            timezone="UTC",
        )
        self._collector = MarketDataCollector(self._settings, exchange_client=exchange_client)
        self._started = False

    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self._started and self._scheduler.running

    def describe(self) -> list[dict[str, Any]]:
        """Registered jobs and their next fire times (dashboard/system status)."""
        if not self._started:
            return []
        return [
            {
                "id": job.id,
                "name": job.name,
                "trigger": str(job.trigger),
                "next_run_time": (job.next_run_time.isoformat() if job.next_run_time else None),
            }
            for job in self._scheduler.get_jobs()
        ]

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._started:
            return
        self._register_jobs()
        self._scheduler.add_listener(self._on_job_problem, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        self._scheduler.start()
        self._started = True
        logger.info(
            "scheduler_started",
            market_data_interval_seconds=self._settings.market_data_interval_seconds,
            trading_mode=self._settings.trading_mode.value,
            jobs=[job["id"] for job in self.describe()],
        )

    async def shutdown(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False
            logger.info("scheduler_stopped")
        await self._collector.aclose()

    # ------------------------------------------------------------------ #
    def _register_jobs(self) -> None:
        settings = self._settings

        self._scheduler.add_job(
            self._safe(self._collect_market_data, JOB_MARKET_DATA),
            trigger=IntervalTrigger(seconds=settings.market_data_interval_seconds),
            id=JOB_MARKET_DATA,
            name="Collect BTC/JPY market data",
            replace_existing=True,
            # Collect once immediately so a fresh deployment has data at once.
            next_run_time=utcnow(),
        )

        if settings.trading_mode is TradingMode.PAPER:
            self._scheduler.add_job(
                self._safe(self._evaluate_strategy, JOB_STRATEGY),
                trigger=IntervalTrigger(seconds=settings.market_data_interval_seconds),
                id=JOB_STRATEGY,
                name="Evaluate trading strategy",
                replace_existing=True,
                next_run_time=utcnow(),
            )
            self._scheduler.add_job(
                self._safe(self._snapshot_portfolio, JOB_PORTFOLIO_SNAPSHOT),
                trigger=IntervalTrigger(seconds=settings.portfolio_snapshot_interval_seconds),
                id=JOB_PORTFOLIO_SNAPSHOT,
                name="Mark paper portfolio to market",
                replace_existing=True,
                next_run_time=utcnow(),
            )

    def _safe(
        self, job: Callable[[], Awaitable[None]], job_id: str
    ) -> Callable[[], Awaitable[None]]:
        """Wrap a job so no exception can escape into the scheduler."""

        async def runner() -> None:
            try:
                await job()
            except Exception as exc:  # noqa: BLE001 - boundary: log, record, continue
                logger.error(
                    "scheduled_job_failed",
                    job_id=job_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                await _record_event(
                    self._settings,
                    severity=Severity.ERROR,
                    event_type="scheduled_job_failed",
                    message=f"{job_id}: {type(exc).__name__}: {exc}",
                    metadata={"job_id": job_id},
                )

        runner.__name__ = f"safe_{job_id}"
        return runner

    async def _collect_market_data(self) -> None:
        await self._refresh_runtime_settings()
        await self._collector.collect(trigger="scheduled")

    async def _evaluate_strategy(self) -> None:
        await self._refresh_runtime_settings()
        async with session_scope(self._settings) as session:
            await evaluate_and_store(session, self._settings)

    async def _snapshot_portfolio(self) -> None:
        await self._refresh_runtime_settings()
        async with session_scope(self._settings) as session:
            await snapshot_latest(session, self._settings)

    async def _refresh_runtime_settings(self) -> None:
        """Pull dashboard changes and reschedule without restarting the worker."""
        async with session_scope(self._settings) as session:
            current = await get_runtime_settings(session, self._settings)
        old = self._settings
        if current.model_dump() == old.model_dump():
            return
        self._settings = current
        await self._collector.aclose()
        self._collector = MarketDataCollector(current, exchange_client=self._exchange_client)
        if self._started:
            self._scheduler.reschedule_job(
                JOB_MARKET_DATA,
                trigger=IntervalTrigger(seconds=current.market_data_interval_seconds),
            )
            portfolio_job = self._scheduler.get_job(JOB_PORTFOLIO_SNAPSHOT)
            if current.trading_mode is TradingMode.PAPER and portfolio_job is None:
                self._scheduler.add_job(
                    self._safe(self._snapshot_portfolio, JOB_PORTFOLIO_SNAPSHOT),
                    trigger=IntervalTrigger(seconds=current.portfolio_snapshot_interval_seconds),
                    id=JOB_PORTFOLIO_SNAPSHOT,
                    name="Mark paper portfolio to market",
                    replace_existing=True,
                    next_run_time=utcnow(),
                )
            elif current.trading_mode is not TradingMode.PAPER and portfolio_job:
                self._scheduler.remove_job(JOB_PORTFOLIO_SNAPSHOT)
            elif portfolio_job:
                self._scheduler.reschedule_job(
                    JOB_PORTFOLIO_SNAPSHOT,
                    trigger=IntervalTrigger(seconds=current.portfolio_snapshot_interval_seconds),
                )
        logger.info(
            "runtime_settings_refreshed",
            market_data_interval_seconds=current.market_data_interval_seconds,
        )

    def _on_job_problem(self, event: JobEvent) -> None:
        """APScheduler-level listener for missed runs and executor errors."""
        logger.warning(
            "scheduler_job_event",
            job_id=event.job_id,
            code=event.code,
            exception=str(getattr(event, "exception", "")) or None,
        )


async def _record_event(
    settings: Settings,
    *,
    severity: Severity,
    event_type: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort system-event write; a logging failure must not cascade."""
    try:
        async with session_scope(settings) as session:
            await SystemEventRepository(session).record(
                severity=severity,
                component=COMPONENT,
                event_type=event_type,
                message=message,
                metadata=metadata,
            )
    except Exception as exc:  # noqa: BLE001 - observability path, never fatal
        logger.error("system_event_write_failed", event_type=event_type, error=str(exc))
