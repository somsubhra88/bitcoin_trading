"""End-to-end collection behaviour (PROJECT.md §10, §11).

Exercised with :class:`FakeExchangeClient` against a real database, so the
storage and bookkeeping paths are the production ones — only the network is
replaced. The rules under test:

* each run stores one ticker observation, deduplicated by exchange timestamp;
* every run, successful or not, leaves an auditable ``collector_runs`` row, and
  every failure also leaves a ``system_events`` row (§10.8).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import CollectorRun, MarketTick, SystemEvent
from btc_trader.db.session import session_scope
from btc_trader.enums import Severity
from btc_trader.errors import MarketDataError
from btc_trader.market_data.collector import MarketDataCollector, collect_market_data
from tests.fixtures.fake_exchange import BASE_TIME, FakeExchangeClient, make_ticker


async def _count(session: AsyncSession, model: type) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def _rows(session: AsyncSession, model: type) -> list:
    return list((await session.execute(select(model))).scalars())


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
async def test_a_successful_collection_stores_everything(
    database: Settings, session: AsyncSession
) -> None:
    client = FakeExchangeClient()

    result = await collect_market_data("manual", settings=database, exchange_client=client)

    assert result.success is True
    assert result.trigger == "manual"
    assert result.exchange == "fake"
    assert result.tick_inserted is True
    assert result.tick_duplicate is False
    assert result.ltp == Decimal("12000000")
    assert result.duration_ms >= 0

    assert await _count(session, MarketTick) == 1
    # No warning or error was worth recording.
    assert await _count(session, SystemEvent) == 0


async def test_the_run_is_recorded_for_the_dashboard(
    database: Settings, session: AsyncSession
) -> None:
    await collect_market_data("cli", settings=database, exchange_client=FakeExchangeClient())

    run = (await session.execute(select(CollectorRun))).scalar_one()
    assert run.success is True
    assert run.trigger == "cli"
    assert run.product_code == "BTC_JPY"
    assert run.exchange == "fake"
    assert run.ticks_inserted == 1
    assert run.finished_at is not None
    assert run.error_type is None


async def test_only_the_configured_endpoints_are_called(database: Settings) -> None:
    client = FakeExchangeClient()

    await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert client.calls == ["get_ticker"]


async def test_an_injected_client_is_not_closed_by_the_collector(database: Settings) -> None:
    """The scheduler shares one long-lived client across runs."""
    client = FakeExchangeClient()

    await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert client.closed is False


# --------------------------------------------------------------------------- #
# repeated collection
# --------------------------------------------------------------------------- #
async def test_collecting_the_same_data_twice_inserts_nothing_new(
    database: Settings, session: AsyncSession
) -> None:
    """§10.4: an unchanged ticker is a duplicate, not an error."""
    client = FakeExchangeClient()

    first = await collect_market_data("scheduled", settings=database, exchange_client=client)
    second = await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert first.tick_inserted is True
    assert second.tick_inserted is False
    assert second.tick_duplicate is True
    assert second.success is True

    assert await _count(session, MarketTick) == 1
    assert await _count(session, CollectorRun) == 2


async def test_new_observations_accumulate(database: Settings, session: AsyncSession) -> None:
    client = FakeExchangeClient()
    await collect_market_data("scheduled", settings=database, exchange_client=client)

    client.ticker = make_ticker(timestamp=BASE_TIME + timedelta(seconds=300))

    result = await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert result.tick_inserted is True
    assert await _count(session, MarketTick) == 2


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
async def test_a_ticker_failure_fails_the_run_and_is_recorded(
    database: Settings, session: AsyncSession
) -> None:
    """§9: without a ticker there is nothing to store, so the run must fail loudly."""
    client = FakeExchangeClient(fail_ticker=True)

    with pytest.raises(MarketDataError):
        await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert await _count(session, MarketTick) == 0

    run = (await session.execute(select(CollectorRun))).scalar_one()
    assert run.success is False
    assert run.error_type == "ExchangeError"
    assert run.error_message is not None and "ticker unavailable" in run.error_message
    assert run.finished_at is not None

    event = (await session.execute(select(SystemEvent))).scalar_one()
    assert event.severity is Severity.ERROR
    assert event.event_type == "collector_failed"
    assert event.component == "market_data_collector"


async def test_a_failed_run_does_not_stop_the_next_one(
    database: Settings, session: AsyncSession
) -> None:
    client = FakeExchangeClient(fail_ticker=True)
    with pytest.raises(MarketDataError):
        await collect_market_data("scheduled", settings=database, exchange_client=client)

    client.fail_ticker = False
    result = await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert result.success is True
    runs = await _rows(session, CollectorRun)
    assert [run.success for run in runs] == [False, True]


# --------------------------------------------------------------------------- #
# the collector object itself
# --------------------------------------------------------------------------- #
async def test_the_collector_can_be_reused(database: Settings, session: AsyncSession) -> None:
    collector = MarketDataCollector(database, exchange_client=FakeExchangeClient())
    try:
        await collector.collect("scheduled")
        await collector.collect("manual")
    finally:
        await collector.aclose()

    assert await _count(session, CollectorRun) == 2


async def test_aclose_leaves_an_injected_client_alone(database: Settings) -> None:
    client = FakeExchangeClient()
    collector = MarketDataCollector(database, exchange_client=client)

    await collector.aclose()

    assert client.closed is False


async def test_collection_uses_its_own_transaction_per_step(database: Settings) -> None:
    """The run row must be visible even though the collection is still in flight."""
    client = FakeExchangeClient(fail_ticker=True)
    with pytest.raises(MarketDataError):
        await collect_market_data("scheduled", settings=database, exchange_client=client)

    # A brand-new session sees the committed bookkeeping.
    async with session_scope(database) as fresh:
        assert await _count(fresh, CollectorRun) == 1
        assert await _count(fresh, SystemEvent) == 1
