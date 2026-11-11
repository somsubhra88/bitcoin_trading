"""Persistence and idempotency (PROJECT.md §10.4, §12.2, §51.22).

The contract under test: collecting the same observation twice must never create
a second row, and the returned counts must be the number of rows that really
landed — the dashboard's "inserted" figures and the backfill anchor both depend
on that being exact.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any
from unittest import mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.db.models import MarketTick
from btc_trader.exchange.models import Ticker
from btc_trader.market_data import repository as repository_module
from btc_trader.market_data.repository import CollectorRunRepository, MarketDataRepository
from tests.fixtures.fake_exchange import BASE_TIME, make_ticker

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
# collector runs
# --------------------------------------------------------------------------- #
async def test_collector_run_lifecycle(session: AsyncSession) -> None:
    runs = CollectorRunRepository(session)
    run = await runs.start(exchange=EXCHANGE, product_code="BTC_JPY", trigger="manual")
    assert run.id is not None
    assert run.finished_at is None

    await runs.finish(run, success=True, ticks_inserted=1)
    assert run.success is True
    assert run.finished_at is not None
    assert run.duration_ms is not None and run.duration_ms >= 0
    assert run.ticks_inserted == 1

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
    duplicate on PostgreSQL while working on SQLite. The write must therefore go
    through the ``RETURNING`` helper.
    """
    captured: list[str] = []

    async def capture(_session: AsyncSession, statement: Any, pk: Any) -> bool:
        captured.append(str(statement.returning(pk).compile()))
        return True

    with mock.patch.object(repository_module, "_insert_returned_a_row", capture):
        assert await repo.insert_tick(_ticker()) is True

    assert len(captured) == 1
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
