"""Scheduling behaviour (PROJECT.md §14, §10.8).

The scheduler is started for real (``AsyncIOScheduler`` on the test event loop)
with a fake exchange client, so job registration, the immediate first run and the
"a job must never raise" boundary are all exercised without network traffic.

Timing is kept to fractions of a second by pointing the interval trigger at a
short interval; assertions wait on observable state (rows in the database), never
on a fixed sleep long enough to be flaky.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, reload_settings
from btc_trader.db.models import CollectorRun, MarketTick, RuntimeSetting, SystemEvent
from btc_trader.enums import Severity, TradingMode
from btc_trader.scheduler.jobs import (
    JOB_DAILY_HISTORY,
    JOB_MARKET_DATA,
    JOB_PORTFOLIO_SNAPSHOT,
    JOB_STRATEGY,
    MISFIRE_GRACE_SECONDS,
    SchedulerService,
)
from tests.fixtures.fake_exchange import FakeExchangeClient


async def _count(session: AsyncSession, model: type) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def _wait_for(session: AsyncSession, model: type, minimum: int = 1) -> int:
    """Poll until ``model`` has at least ``minimum`` rows, or fail after 5s."""
    deadline = asyncio.get_running_loop().time() + 5.0
    while True:
        count = await _count(session, model)
        if count >= minimum:
            return count
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"{model.__name__} never reached {minimum} rows (saw {count})")
        await asyncio.sleep(0.02)


async def _wait_for_successful_run(session: AsyncSession) -> None:
    """Poll until collection bookkeeping confirms a successful run."""
    deadline = asyncio.get_running_loop().time() + 5.0
    while True:
        successful = await session.scalar(
            select(func.count()).select_from(CollectorRun).where(CollectorRun.success.is_(True))
        )
        if successful:
            return
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail("collector never recorded a successful run")
        await asyncio.sleep(0.02)


@pytest.fixture
async def scheduler(database: Settings) -> AsyncIterator[SchedulerService]:
    """A stopped scheduler with a fake client; the test decides when to start."""
    service = SchedulerService(database, exchange_client=FakeExchangeClient())
    try:
        yield service
    finally:
        await service.shutdown()


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
async def test_nothing_is_reported_before_start(scheduler: SchedulerService) -> None:
    assert scheduler.running is False
    assert scheduler.describe() == []


async def test_only_data_jobs_are_registered_in_data_only_mode(
    scheduler: SchedulerService,
) -> None:
    """§14: data-only mode registers no strategy or portfolio jobs."""
    scheduler.start()

    jobs = scheduler.describe()
    assert [job["id"] for job in jobs] == [JOB_MARKET_DATA, JOB_DAILY_HISTORY]
    assert scheduler.running is True
    assert jobs[0]["next_run_time"] is not None
    assert "interval" in jobs[0]["trigger"]


async def test_paper_mode_registers_strategy_and_portfolio_jobs(database: Settings) -> None:
    settings = database.model_copy(
        update={"trading_mode": TradingMode.PAPER, "strategy_name": "none"}
    )
    service = SchedulerService(settings, exchange_client=FakeExchangeClient())
    try:
        service.start()
        assert {job["id"] for job in service.describe()} == {
            JOB_MARKET_DATA,
            JOB_DAILY_HISTORY,
            JOB_STRATEGY,
            JOB_PORTFOLIO_SNAPSHOT,
        }
    finally:
        await service.shutdown()


async def test_switching_to_paper_at_runtime_starts_strategy_evaluation(
    scheduler: SchedulerService, session: AsyncSession
) -> None:
    """Regression: a dashboard switch to paper mode used to leave no strategy job."""
    scheduler.start()
    session.add(RuntimeSetting(key="trading_mode", value="paper"))
    await session.commit()

    await scheduler._refresh_runtime_settings()

    ids = {job["id"] for job in scheduler.describe()}
    assert {JOB_STRATEGY, JOB_PORTFOLIO_SNAPSHOT} <= ids

    await session.execute(
        RuntimeSetting.__table__.update().values(value="data_only")  # type: ignore[attr-defined]
    )
    await session.commit()
    await scheduler._refresh_runtime_settings()

    assert {job["id"] for job in scheduler.describe()} == {JOB_MARKET_DATA, JOB_DAILY_HISTORY}


async def test_a_deployment_kill_switch_beats_a_saved_dashboard_value(
    database: Settings, session: AsyncSession
) -> None:
    """TRADING_KILL_SWITCH=true in the environment is an emergency stop."""
    session.add(RuntimeSetting(key="trading_kill_switch", value=False))
    session.add(RuntimeSetting(key="market_data_interval_seconds", value=60))
    await session.commit()
    service = SchedulerService(
        database.model_copy(update={"trading_kill_switch": True}),
        exchange_client=FakeExchangeClient(),
    )
    try:
        await service._refresh_runtime_settings()

        assert service._settings.trading_kill_switch is True
        assert service._settings.market_data_interval_seconds == 60
    finally:
        await service.shutdown()


async def test_a_removed_dashboard_override_falls_back_to_the_environment(
    scheduler: SchedulerService, session: AsyncSession
) -> None:
    session.add(RuntimeSetting(key="market_data_interval_seconds", value=60))
    await session.commit()
    await scheduler._refresh_runtime_settings()
    assert scheduler._settings.market_data_interval_seconds == 60

    await session.execute(RuntimeSetting.__table__.delete())  # type: ignore[attr-defined]
    await session.commit()
    await scheduler._refresh_runtime_settings()

    assert scheduler._settings.market_data_interval_seconds == 300


async def test_the_interval_comes_from_configuration(
    monkeypatch: pytest.MonkeyPatch, database: Settings
) -> None:
    monkeypatch.setenv("MARKET_DATA_INTERVAL_SECONDS", "900")
    settings = reload_settings()
    service = SchedulerService(settings, exchange_client=FakeExchangeClient())
    try:
        service.start()

        assert "0:15:00" in service.describe()[0]["trigger"]
    finally:
        await service.shutdown()


async def test_dashboard_interval_change_reschedules_running_worker(
    scheduler: SchedulerService, session: AsyncSession
) -> None:
    scheduler.start()
    session.add(RuntimeSetting(key="market_data_interval_seconds", value=60))
    await session.commit()

    await scheduler._refresh_runtime_settings()

    assert "0:01:00" in scheduler.describe()[0]["trigger"]


async def test_the_job_cannot_overlap_itself(scheduler: SchedulerService) -> None:
    """§14: a collection that outruns its interval must not run concurrently."""
    scheduler.start()

    job = scheduler._scheduler.get_job(JOB_MARKET_DATA)
    assert job is not None
    assert job.max_instances == 1
    assert job.coalesce is True
    assert job.misfire_grace_time == MISFIRE_GRACE_SECONDS


async def test_start_is_idempotent(scheduler: SchedulerService) -> None:
    scheduler.start()
    scheduler.start()

    assert [job["id"] for job in scheduler.describe()] == [JOB_MARKET_DATA, JOB_DAILY_HISTORY]


async def test_shutdown_stops_reporting_jobs(scheduler: SchedulerService) -> None:
    scheduler.start()
    await scheduler.shutdown()

    assert scheduler.running is False
    assert scheduler.describe() == []


async def test_shutdown_before_start_is_harmless(scheduler: SchedulerService) -> None:
    await scheduler.shutdown()

    assert scheduler.running is False


async def test_an_injected_client_survives_shutdown(database: Settings) -> None:
    """The caller that supplied the client owns its lifetime."""
    client = FakeExchangeClient()
    service = SchedulerService(database, exchange_client=client)

    service.start()
    await service.shutdown()

    assert client.closed is False


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
async def test_the_first_collection_runs_immediately(
    scheduler: SchedulerService, session: AsyncSession
) -> None:
    """A fresh deployment must not wait a whole interval for its first data."""
    scheduler.start()

    await _wait_for(session, MarketTick)

    run = (await session.execute(select(CollectorRun))).scalars().first()
    assert run is not None
    assert run.trigger == "scheduled"
    assert run.success is True


async def test_a_failing_job_records_the_failure_and_keeps_the_scheduler_alive(
    database: Settings, session: AsyncSession
) -> None:
    """§10.8: one bad request must never kill the job."""
    client = FakeExchangeClient(fail_ticker=True)
    service = SchedulerService(database, exchange_client=client)
    try:
        service.start()

        await _wait_for(session, SystemEvent)

        # The collector recorded the failed run...
        failed = (await session.execute(select(CollectorRun))).scalars().first()
        assert failed is not None
        assert failed.success is False
        # ...and the scheduler is still up, ready for the next interval.
        assert service.running is True

        events = list((await session.execute(select(SystemEvent))).scalars())
        assert any(event.severity is Severity.ERROR for event in events)
        assert {"collector_failed", "scheduled_job_failed"} & {e.event_type for e in events}
    finally:
        await service.shutdown()


async def test_the_scheduler_recovers_once_the_exchange_does(
    database: Settings, session: AsyncSession
) -> None:
    retry_interval_seconds = 0.2
    client = FakeExchangeClient(fail_ticker=True)
    service = SchedulerService(database, exchange_client=client)
    try:
        service.start()
        # Shorten the interval so the retry lands inside the test's patience.
        service._scheduler.reschedule_job(
            JOB_MARKET_DATA, trigger="interval", seconds=retry_interval_seconds
        )
        await _wait_for(session, CollectorRun)

        client.fail_ticker = False
        await _wait_for_successful_run(session)

        runs = list((await session.execute(select(CollectorRun))).scalars())
        assert False in [run.success for run in runs]
        assert True in [run.success for run in runs]
    finally:
        await service.shutdown()


async def test_a_history_provider_outage_is_recorded_not_raised(
    scheduler: SchedulerService, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from btc_trader.errors import ExchangeTransientError
    from btc_trader.scheduler import jobs

    async def unavailable(*_: object, **__: object) -> int:
        raise ExchangeTransientError("bitbank unreachable")

    monkeypatch.setattr(jobs, "refresh_external_history", unavailable)
    scheduler._settings = scheduler._settings.model_copy(update={"daily_history_auto_sync": True})
    monkeypatch.setattr(
        scheduler, "_refresh_runtime_settings", lambda: asyncio.sleep(0), raising=True
    )

    await scheduler._sync_daily_history()

    events = list(
        (
            await session.execute(
                select(SystemEvent).where(SystemEvent.event_type == "daily_history_refresh_failed")
            )
        ).scalars()
    )
    assert len(events) == 1
    assert "bitbank unreachable" in events[0].message


async def test_live_mode_marks_the_real_account(database: Settings, session: AsyncSession) -> None:
    from decimal import Decimal

    from pydantic import SecretStr

    from btc_trader.db.models import PortfolioSnapshot

    live = database.model_copy(
        update={
            "trading_mode": TradingMode.LIVE,
            "enable_automatic_trading": True,
            "live_trading_acknowledged": True,
            "bitflyer_api_key": SecretStr("key"),
            "bitflyer_api_secret": SecretStr("secret"),
        }
    )
    exchange = FakeExchangeClient(jpy="100000", btc="0.01")
    service = SchedulerService(live, exchange_client=exchange)
    service._refresh_runtime_settings = lambda: asyncio.sleep(0)  # type: ignore[method-assign]
    try:
        service.start()
        assert {JOB_STRATEGY, JOB_PORTFOLIO_SNAPSHOT} <= {job["id"] for job in service.describe()}

        await service._snapshot_portfolio()
    finally:
        await service.shutdown()

    snapshot = (
        await session.execute(
            select(PortfolioSnapshot).where(PortfolioSnapshot.trading_mode == TradingMode.LIVE)
        )
    ).scalar_one()
    assert snapshot.jpy_balance == Decimal("100000")
    assert snapshot.btc_balance == Decimal("0.01")
    assert exchange.closed is False  # injected clients belong to the caller


async def test_a_failing_job_alerts_once_and_again_on_recovery(
    scheduler: SchedulerService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from btc_trader.scheduler import jobs

    alerts: list[str] = []

    async def record(_settings: object, _severity: object, title: str, _detail: str) -> bool:
        alerts.append(title)
        return True

    monkeypatch.setattr(jobs, "send_alert", record)
    outcomes = iter([RuntimeError("boom"), RuntimeError("boom"), None])

    async def flaky() -> None:
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome

    runner = scheduler._safe(flaky, "flaky_job")
    for _ in range(3):
        await runner()

    assert alerts == ["flaky_job failing", "flaky_job recovered"]
