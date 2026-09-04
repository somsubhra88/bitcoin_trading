"""End-to-end collection behaviour (PROJECT.md §10, §11).

Exercised with :class:`FakeExchangeClient` against a real database, so the
storage, bookkeeping and degradation paths are the production ones — only the
network is replaced.  The rules under test:

* the ticker is mandatory; a failure there fails the run;
* the order book and executions are optional — losing them degrades a run that
  still keeps the tick (§10.6);
* every run, successful or not, leaves an auditable ``collector_runs`` row, and
  every failure or degradation also leaves a ``system_events`` row (§10.8).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, reload_settings
from btc_trader.db.models import (
    CollectorRun,
    MarketExecution,
    MarketTick,
    OrderBookSnapshot,
    SystemEvent,
)
from btc_trader.db.session import session_scope
from btc_trader.enums import Severity
from btc_trader.errors import MarketDataError
from btc_trader.market_data.collector import MarketDataCollector, collect_market_data
from tests.fixtures.fake_exchange import (
    BASE_TIME,
    FakeExchangeClient,
    make_executions,
    make_ticker,
)


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
    assert result.degraded == []
    assert result.tick_inserted is True
    assert result.tick_duplicate is False
    assert result.executions_fetched == 3
    assert result.executions_inserted == 3
    assert result.order_book_inserted is True
    assert result.ltp == Decimal("12000000")
    assert result.duration_ms >= 0

    assert await _count(session, MarketTick) == 1
    assert await _count(session, MarketExecution) == 3
    assert await _count(session, OrderBookSnapshot) == 1
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
    assert run.executions_inserted == 3
    assert run.order_books_inserted == 1
    assert run.finished_at is not None
    assert run.error_type is None


async def test_only_the_configured_endpoints_are_called(database: Settings) -> None:
    client = FakeExchangeClient()

    await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert client.calls == ["get_ticker", "get_order_book", "get_recent_executions"]


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
    assert second.executions_fetched == 3
    assert second.executions_inserted == 0

    assert await _count(session, MarketTick) == 1
    assert await _count(session, MarketExecution) == 3
    assert await _count(session, CollectorRun) == 2


async def test_new_observations_accumulate(database: Settings, session: AsyncSession) -> None:
    client = FakeExchangeClient()
    await collect_market_data("scheduled", settings=database, exchange_client=client)

    client.ticker = make_ticker(timestamp=BASE_TIME + timedelta(seconds=300))
    client.executions = make_executions(3, first_id=1002)  # overlaps by one

    result = await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert result.tick_inserted is True
    assert result.executions_inserted == 2
    assert await _count(session, MarketTick) == 2
    assert await _count(session, MarketExecution) == 5


# --------------------------------------------------------------------------- #
# degradation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("flag", "label"),
    [("fail_order_book", "order_book"), ("fail_executions", "executions")],
)
async def test_an_optional_endpoint_failure_degrades_but_keeps_the_tick(
    database: Settings, session: AsyncSession, flag: str, label: str
) -> None:
    client = FakeExchangeClient(**{flag: True})

    result = await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert result.success is True
    assert result.degraded == [label]
    assert result.tick_inserted is True
    assert await _count(session, MarketTick) == 1

    run = (await session.execute(select(CollectorRun))).scalar_one()
    assert run.success is True
    assert run.error_message == f"degraded:{label}"

    event = (await session.execute(select(SystemEvent))).scalar_one()
    assert event.severity is Severity.WARNING
    assert event.event_type == "collection_degraded"
    assert label in event.message


async def test_both_optional_endpoints_can_fail_at_once(
    database: Settings, session: AsyncSession
) -> None:
    client = FakeExchangeClient(fail_order_book=True, fail_executions=True)

    result = await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert result.success is True
    assert result.degraded == ["order_book", "executions"]
    assert result.order_book_inserted is False
    assert result.executions_inserted == 0
    assert await _count(session, MarketTick) == 1
    assert await _count(session, OrderBookSnapshot) == 0
    assert await _count(session, MarketExecution) == 0


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


async def test_the_order_book_is_never_fetched_when_the_ticker_fails(database: Settings) -> None:
    client = FakeExchangeClient(fail_ticker=True)

    with pytest.raises(MarketDataError):
        await collect_market_data("scheduled", settings=database, exchange_client=client)

    assert client.calls == ["get_ticker"]


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
# configuration
# --------------------------------------------------------------------------- #
async def test_optional_collection_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch, database: Settings, session: AsyncSession
) -> None:
    monkeypatch.setenv("COLLECT_ORDER_BOOK", "false")
    monkeypatch.setenv("COLLECT_EXECUTIONS", "false")
    settings = reload_settings()
    client = FakeExchangeClient()

    result = await collect_market_data("scheduled", settings=settings, exchange_client=client)

    assert client.calls == ["get_ticker"]
    assert result.degraded == []
    assert result.order_book_inserted is False
    assert result.executions_inserted == 0
    assert await _count(session, MarketTick) == 1


async def test_the_configured_fetch_limit_is_honoured(
    monkeypatch: pytest.MonkeyPatch, database: Settings
) -> None:
    monkeypatch.setenv("EXECUTIONS_FETCH_LIMIT", "2")
    settings = reload_settings()
    client = FakeExchangeClient(executions=make_executions(5, first_id=2000))

    result = await collect_market_data("scheduled", settings=settings, exchange_client=client)

    assert result.executions_fetched == 2
    assert result.executions_inserted == 2


async def test_the_configured_depth_is_stored(
    monkeypatch: pytest.MonkeyPatch, database: Settings, session: AsyncSession
) -> None:
    monkeypatch.setenv("ORDER_BOOK_DEPTH_LEVELS", "1")
    settings = reload_settings()

    await collect_market_data("scheduled", settings=settings, exchange_client=FakeExchangeClient())

    snapshot = (await session.execute(select(OrderBookSnapshot))).scalar_one()
    assert snapshot.depth_levels == 1
    assert len(snapshot.bids) == 1
    assert len(snapshot.asks) == 1


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
