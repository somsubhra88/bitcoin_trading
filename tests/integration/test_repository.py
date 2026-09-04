"""Persistence and idempotency (PROJECT.md §10.4, §12.2, §51.22).

The contract under test: collecting the same observation twice must never create
a second row, and the returned counts must be the number of rows that really
landed — the dashboard's "inserted" figures and the backfill anchor both depend
on that being exact.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest import mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.db.models import MarketExecution, MarketTick, OrderBookSnapshot
from btc_trader.exchange.models import Execution, OrderBook, OrderBookLevel, Ticker
from btc_trader.market_data import repository as repository_module
from btc_trader.market_data.repository import CollectorRunRepository, MarketDataRepository
from tests.fixtures.fake_exchange import BASE_TIME, make_executions, make_ticker

EXCHANGE = "bitflyer"


@pytest.fixture
def repo(session: AsyncSession) -> MarketDataRepository:
    return MarketDataRepository(session)


async def _count(session: AsyncSession, model: type) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


def _ticker(
    offset: int = 0, *, ltp: str | None = "12000000", product_code: str = "BTC_JPY"
) -> Ticker:
    """A ticker ``offset`` seconds after :data:`BASE_TIME`, optionally without an LTP."""
    ticker = make_ticker(
        exchange=EXCHANGE,
        product_code=product_code,
        timestamp=BASE_TIME + timedelta(seconds=offset),
    )
    return ticker.model_copy(update={"ltp": Decimal(ltp) if ltp is not None else None})


def _executions(count: int, *, first_id: int, product_code: str = "BTC_JPY") -> list[Execution]:
    return make_executions(
        count, exchange=EXCHANGE, product_code=product_code, first_id=first_id, start=BASE_TIME
    )


# --------------------------------------------------------------------------- #
# ticks
# --------------------------------------------------------------------------- #
async def test_insert_tick_stores_every_field(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    ticker = _ticker()
    assert await repo.insert_tick(ticker) is True

    stored = (await session.execute(select(MarketTick))).scalar_one()
    assert stored.exchange == ticker.exchange
    assert stored.product_code == ticker.product_code
    assert stored.exchange_timestamp == ticker.exchange_timestamp
    assert stored.ltp == ticker.ltp
    assert stored.best_bid == ticker.best_bid
    assert stored.best_ask == ticker.best_ask
    assert stored.tick_id == ticker.tick_id
    assert stored.raw_payload is not None


async def test_reinserting_the_same_tick_is_a_no_op(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    """Polling an unchanged ticker must not grow the table."""
    ticker = _ticker()
    assert await repo.insert_tick(ticker) is True
    assert await repo.insert_tick(ticker) is False
    assert await _count(session, MarketTick) == 1


async def test_a_later_timestamp_is_a_new_tick(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    await repo.insert_tick(_ticker())
    assert await repo.insert_tick(_ticker(1)) is True
    assert await _count(session, MarketTick) == 2


async def test_the_same_timestamp_on_another_market_is_a_new_tick(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    """The natural key includes the market, so BTC_JPY and ETH_JPY never collide."""
    await repo.insert_tick(_ticker())
    assert await repo.insert_tick(_ticker(product_code="ETH_JPY")) is True
    assert await _count(session, MarketTick) == 2


async def test_latest_tick_and_counts(repo: MarketDataRepository) -> None:
    for offset in range(3):
        await repo.insert_tick(_ticker(offset, ltp=str(offset)))
    await repo.insert_tick(_ticker(product_code="ETH_JPY"))

    latest = await repo.latest_tick("BTC_JPY")
    assert latest is not None
    assert latest.exchange_timestamp == BASE_TIME + timedelta(seconds=2)
    assert await repo.count_ticks("BTC_JPY") == 3
    assert await repo.count_ticks("ETH_JPY") == 1
    assert await repo.latest_tick("BTC_JPY", exchange="nowhere") is None


async def test_ticks_between_filters_and_orders(repo: MarketDataRepository) -> None:
    for offset in range(5):
        await repo.insert_tick(_ticker(offset, ltp=str(offset)))

    window = await repo.ticks_between(
        "BTC_JPY", start=BASE_TIME + timedelta(seconds=1), end=BASE_TIME + timedelta(seconds=3)
    )
    assert [tick.ltp for tick in window] == [Decimal(1), Decimal(2), Decimal(3)]

    descending = await repo.ticks_between("BTC_JPY", ascending=False)
    assert [tick.ltp for tick in descending] == [Decimal(n) for n in (4, 3, 2, 1, 0)]


async def test_ticks_between_limit_keeps_the_newest_rows(repo: MarketDataRepository) -> None:
    """A limit must trim the *oldest* history, not the latest prices."""
    for offset in range(10):
        await repo.insert_tick(_ticker(offset, ltp=str(offset)))

    window = await repo.ticks_between("BTC_JPY", limit=3)
    assert [tick.ltp for tick in window] == [Decimal(7), Decimal(8), Decimal(9)]


async def test_tick_time_range(repo: MarketDataRepository) -> None:
    assert await repo.tick_time_range("BTC_JPY") == (None, None)
    for offset in (0, 5, 2):
        await repo.insert_tick(_ticker(offset))
    first, last = await repo.tick_time_range("BTC_JPY")
    assert first == BASE_TIME
    assert last == BASE_TIME + timedelta(seconds=5)


async def test_price_at_or_before_ignores_the_future_and_null_prices(
    repo: MarketDataRepository,
) -> None:
    await repo.insert_tick(_ticker(0, ltp="12000000"))
    await repo.insert_tick(_ticker(10, ltp=None))
    await repo.insert_tick(_ticker(20, ltp="12500000"))

    at_five = await repo.price_at_or_before("BTC_JPY", BASE_TIME + timedelta(seconds=5))
    assert at_five == Decimal("12000000")
    # The nearest tick at t+15 has no LTP, so the baseline falls further back.
    at_fifteen = await repo.price_at_or_before("BTC_JPY", BASE_TIME + timedelta(seconds=15))
    assert at_fifteen == Decimal("12000000")
    before_everything = await repo.price_at_or_before("BTC_JPY", BASE_TIME - timedelta(days=1))
    assert before_everything is None


# --------------------------------------------------------------------------- #
# executions
# --------------------------------------------------------------------------- #
async def test_insert_executions_returns_the_number_of_new_rows(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    assert await repo.insert_executions(_executions(3, first_id=100)) == 3
    assert await _count(session, MarketExecution) == 3


async def test_insert_executions_is_idempotent(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    batch = _executions(3, first_id=100)
    assert await repo.insert_executions(batch) == 3
    assert await repo.insert_executions(batch) == 0
    assert await _count(session, MarketExecution) == 3


async def test_overlapping_batches_only_insert_the_new_tail(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    """The realistic polling case: consecutive fetches share most of their rows."""
    assert await repo.insert_executions(_executions(5, first_id=100)) == 5
    assert await repo.insert_executions(_executions(5, first_id=103)) == 3
    assert await _count(session, MarketExecution) == 8


async def test_duplicates_inside_one_batch_are_collapsed(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    """ON CONFLICT cannot deduplicate a single statement, so the code must."""
    batch = _executions(2, first_id=100)
    assert await repo.insert_executions([*batch, *batch]) == 2
    assert await _count(session, MarketExecution) == 2


async def test_an_empty_batch_is_free(repo: MarketDataRepository) -> None:
    assert await repo.insert_executions([]) == 0


async def test_the_same_execution_id_on_another_market_is_kept(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    await repo.insert_executions(_executions(2, first_id=100))
    inserted = await repo.insert_executions(_executions(2, first_id=100, product_code="ETH_JPY"))
    assert inserted == 2
    assert await _count(session, MarketExecution) == 4


async def test_executions_between_filters_and_orders(repo: MarketDataRepository) -> None:
    await repo.insert_executions(_executions(5, first_id=100))

    window = await repo.executions_between(
        "BTC_JPY", start=BASE_TIME + timedelta(seconds=1), end=BASE_TIME + timedelta(seconds=2)
    )
    assert [row.exchange_execution_id for row in window] == ["101", "102"]

    newest_first = await repo.executions_between("BTC_JPY", ascending=False, limit=2)
    assert [row.exchange_execution_id for row in newest_first] == ["104", "103"]
    assert await repo.count_executions("BTC_JPY") == 5


async def test_oldest_stored_execution_id_sorts_numerically(repo: MarketDataRepository) -> None:
    """Ids live in a text column: "99" must still rank below "1000"."""
    await repo.insert_executions(_executions(1, first_id=99))
    await repo.insert_executions(_executions(1, first_id=1000))
    await repo.insert_executions(_executions(1, first_id=100))

    assert await repo.oldest_stored_execution_id("BTC_JPY", EXCHANGE) == 99


async def test_oldest_stored_execution_id_is_scoped_per_market(repo: MarketDataRepository) -> None:
    await repo.insert_executions(_executions(1, first_id=500))
    await repo.insert_executions(_executions(1, first_id=1, product_code="ETH_JPY"))

    assert await repo.oldest_stored_execution_id("BTC_JPY", EXCHANGE) == 500
    assert await repo.oldest_stored_execution_id("BTC_JPY", "other") is None
    assert await repo.oldest_stored_execution_id("XRP_JPY", EXCHANGE) is None


async def test_a_non_numeric_execution_id_does_not_break_the_anchor(
    repo: MarketDataRepository,
) -> None:
    """Another exchange could use opaque ids; backfill must degrade, not crash."""
    await repo.insert_executions(
        [
            Execution(
                exchange=EXCHANGE,
                product_code="BTC_JPY",
                exchange_execution_id="abc",
                side=None,
                price=Decimal("12000000"),
                size=Decimal("0.01"),
                executed_at=BASE_TIME,
                received_at=BASE_TIME,
            )
        ]
    )
    assert await repo.oldest_stored_execution_id("BTC_JPY", EXCHANGE) is None


# --------------------------------------------------------------------------- #
# order book
# --------------------------------------------------------------------------- #
def _book(*, received_at: datetime = BASE_TIME, mid_price: str = "12000000") -> OrderBook:
    return OrderBook(
        exchange=EXCHANGE,
        product_code="BTC_JPY",
        received_at=received_at,
        mid_price=Decimal(mid_price),
        bids=tuple(
            OrderBookLevel(price=Decimal(11999999 - index), size=Decimal("0.1"))
            for index in range(3)
        ),
        asks=tuple(
            OrderBookLevel(price=Decimal(12000001 + index), size=Decimal("0.1"))
            for index in range(3)
        ),
    )


async def test_order_book_snapshot_is_truncated_to_the_configured_depth(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    assert await repo.insert_order_book_snapshot(_book(), depth_levels=2) is True

    stored = (await session.execute(select(OrderBookSnapshot))).scalar_one()
    assert stored.depth_levels == 2
    assert len(stored.bids) == 2
    assert len(stored.asks) == 2
    # Prices survive as strings, never as binary floats (§13).
    assert stored.bids[0]["price"] == "11999999"
    assert stored.mid_price == Decimal("12000000")


async def test_reinserting_a_snapshot_with_the_same_received_at_is_a_no_op(
    repo: MarketDataRepository, session: AsyncSession
) -> None:
    assert await repo.insert_order_book_snapshot(_book(), depth_levels=2) is True
    assert await repo.insert_order_book_snapshot(_book(), depth_levels=2) is False
    assert await _count(session, OrderBookSnapshot) == 1


async def test_latest_order_book(repo: MarketDataRepository) -> None:
    await repo.insert_order_book_snapshot(_book(), depth_levels=3)
    await repo.insert_order_book_snapshot(
        _book(received_at=BASE_TIME + timedelta(seconds=5), mid_price="12345678"), depth_levels=3
    )

    latest = await repo.latest_order_book("BTC_JPY")
    assert latest is not None
    assert latest.mid_price == Decimal("12345678")
    assert await repo.latest_order_book("ETH_JPY") is None


# --------------------------------------------------------------------------- #
# collector runs
# --------------------------------------------------------------------------- #
async def test_collector_run_lifecycle(session: AsyncSession) -> None:
    runs = CollectorRunRepository(session)
    run = await runs.start(exchange=EXCHANGE, product_code="BTC_JPY", trigger="manual")
    assert run.id is not None
    assert run.finished_at is None

    await runs.finish(run, success=True, ticks_inserted=1, executions_inserted=7)
    assert run.success is True
    assert run.finished_at is not None
    assert run.duration_ms is not None and run.duration_ms >= 0
    assert run.executions_inserted == 7

    assert (await runs.last_run()) is run
    assert (await runs.last_successful_run()) is run


async def test_last_successful_run_skips_failures(session: AsyncSession) -> None:
    runs = CollectorRunRepository(session)
    good = await runs.start(exchange=EXCHANGE, product_code="BTC_JPY", trigger="schedule")
    await runs.finish(good, success=True, ticks_inserted=1)
    bad = await runs.start(exchange=EXCHANGE, product_code="BTC_JPY", trigger="schedule")
    await runs.finish(bad, success=False, error_type="ExchangeTransientError", error_message="503")

    assert (await runs.last_run()) is bad
    assert (await runs.last_successful_run()) is good
    assert len(await runs.recent_runs()) == 2
    assert len(await runs.recent_runs(limit=1)) == 1


async def test_a_long_error_message_is_truncated(session: AsyncSession) -> None:
    runs = CollectorRunRepository(session)
    run = await runs.start(exchange=EXCHANGE, product_code="BTC_JPY", trigger="schedule")
    await runs.finish(run, success=False, error_type="Boom", error_message="x" * 5000)
    assert run.error_message is not None
    assert len(run.error_message) == 2000


# --------------------------------------------------------------------------- #
# cross-backend insert reporting
# --------------------------------------------------------------------------- #
class _StubBind:
    def __init__(self, dialect: Any) -> None:
        self.dialect = dialect


class _StubSession:
    """Just enough of a session for ``_insert_ignore`` to pick a dialect."""

    def __init__(self, dialect: Any) -> None:
        self.bind = _StubBind(dialect)


async def test_stored_flags_come_from_returning_not_rowcount(
    session: AsyncSession, repo: MarketDataRepository
) -> None:
    """A regression guard for the PostgreSQL insert-count bug.

    ``psycopg`` reports ``rowcount == -1`` ("unknown") for a plain ``INSERT``, so
    deriving "was it stored?" from the row count reported every stored tick as a
    duplicate on PostgreSQL while working on SQLite.  Both writes must therefore
    go through the ``RETURNING`` helper.
    """
    captured: list[str] = []

    async def capture(_session: AsyncSession, statement: Any, pk: Any) -> bool:
        captured.append(str(statement.returning(pk).compile()))
        return True

    with mock.patch.object(repository_module, "_insert_returned_a_row", capture):
        assert await repo.insert_tick(_ticker()) is True
        assert await repo.insert_order_book_snapshot(_book(), depth_levels=2) is True

    assert len(captured) == 2
    for sql in captured:
        assert "ON CONFLICT" in sql.upper()
        assert "RETURNING" in sql.upper()


@pytest.mark.parametrize(
    ("dialect_name", "dialect"),
    [("postgresql", postgresql.dialect()), ("sqlite", sqlite.dialect())],
)
def test_insert_ignore_supports_returning_on_both_backends(dialect_name: str, dialect: Any) -> None:
    """Whichever backend is deployed, the idempotent insert can report its id."""
    statement = repository_module._insert_ignore(
        _StubSession(dialect),  # type: ignore[arg-type]
        MarketTick,
        ["exchange", "product_code", "exchange_timestamp"],
    ).values(exchange=EXCHANGE, product_code="BTC_JPY", exchange_timestamp=BASE_TIME)

    sql = str(statement.returning(MarketTick.id).compile(dialect=dialect)).upper()

    assert dialect.name == dialect_name
    assert "ON CONFLICT" in sql
    assert "RETURNING" in sql


def test_an_unsupported_backend_fails_loudly() -> None:
    """Silently falling back to a plain INSERT would resurrect duplicate rows."""
    with pytest.raises(NotImplementedError, match="mysql"):
        repository_module._insert_ignore(
            _StubSession(mysql.dialect()),  # type: ignore[arg-type]
            MarketTick,
            ["exchange"],
        )
