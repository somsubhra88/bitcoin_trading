"""Market-data persistence.

Inserts are idempotent: every write uses ``INSERT ... ON CONFLICT DO NOTHING``
against the table's natural key, so re-running the collector or a backfill can
never duplicate an observation (PROJECT.md §10.4, §12.2, §51.22).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.dml import Insert

from btc_trader.db.models import CollectorRun, MarketExecution, MarketTick, OrderBookSnapshot
from btc_trader.exchange.models import Execution, OrderBook, Ticker
from btc_trader.monitoring.logging import get_logger
from btc_trader.utils.serialization import jsonable
from btc_trader.utils.timeutils import utcnow

logger = get_logger(__name__)


async def _insert_returned_a_row(session: AsyncSession, statement: Insert, pk: Any) -> bool:
    """Execute an insert-ignore and report whether a new row was actually stored.

    ``RETURNING`` rather than ``rowcount``: psycopg's async cursor reports ``-1``
    ("unknown") for a plain ``INSERT``, which made every PostgreSQL insert look
    like a skipped duplicate even though the row landed — the collector then
    recorded ``ticks_inserted=0`` for a successful collection.  ``ON CONFLICT DO
    NOTHING`` returns no row when it skips, so on both supported backends the
    presence of a returned id is the answer, independent of the driver.
    """
    result = await session.execute(statement.returning(pk))
    return result.scalar_one_or_none() is not None


def _insert_ignore(session: AsyncSession, table: Any, index_elements: list[str]) -> Insert:
    """Backend-appropriate ``INSERT ... ON CONFLICT DO NOTHING`` construct."""
    dialect = session.bind.dialect.name if session.bind is not None else "postgresql"
    if dialect == "sqlite":
        return sqlite_insert(table).on_conflict_do_nothing(index_elements=index_elements)
    if dialect == "postgresql":
        return pg_insert(table).on_conflict_do_nothing(index_elements=index_elements)
    raise NotImplementedError(f"idempotent insert is not implemented for dialect {dialect!r}")


class MarketDataRepository:
    """All reads and writes of collected market data."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    async def insert_tick(self, ticker: Ticker) -> bool:
        """Store one ticker observation. Returns ``True`` if a row was inserted."""
        statement = _insert_ignore(
            self._session,
            MarketTick,
            ["exchange", "product_code", "exchange_timestamp"],
        ).values(
            exchange=ticker.exchange,
            product_code=ticker.product_code,
            exchange_timestamp=ticker.exchange_timestamp,
            received_at=ticker.received_at,
            tick_id=ticker.tick_id,
            state=ticker.state,
            ltp=ticker.ltp,
            best_bid=ticker.best_bid,
            best_ask=ticker.best_ask,
            best_bid_size=ticker.best_bid_size,
            best_ask_size=ticker.best_ask_size,
            total_bid_depth=ticker.total_bid_depth,
            total_ask_depth=ticker.total_ask_depth,
            market_bid_size=ticker.market_bid_size,
            market_ask_size=ticker.market_ask_size,
            volume=ticker.volume,
            volume_by_product=ticker.volume_by_product,
            raw_payload=jsonable(ticker.raw_payload),
        )
        return await _insert_returned_a_row(self._session, statement, MarketTick.id)

    async def insert_executions(self, executions: list[Execution]) -> int:
        """Bulk-store public executions, skipping ones already present.

        Existing ids are filtered out up front so the return value is an exact
        count of new rows (``rowcount`` is unreliable for ``executemany``), while
        ``ON CONFLICT DO NOTHING`` still protects against a concurrent writer.
        """
        if not executions:
            return 0
        candidates = _deduplicate_executions(executions)
        known = await self._existing_execution_ids(candidates)
        new_executions = [
            execution for execution in candidates if execution.exchange_execution_id not in known
        ]
        if not new_executions:
            return 0
        rows = [
            {
                "exchange": execution.exchange,
                "product_code": execution.product_code,
                "exchange_execution_id": execution.exchange_execution_id,
                "side": execution.side,
                "price": execution.price,
                "size": execution.size,
                "executed_at": execution.executed_at,
                "received_at": execution.received_at,
            }
            for execution in new_executions
        ]
        statement = _insert_ignore(
            self._session,
            MarketExecution,
            ["exchange", "product_code", "exchange_execution_id"],
        )
        await self._session.execute(statement, rows)
        return len(rows)

    async def _existing_execution_ids(self, executions: list[Execution]) -> set[str]:
        """Ids from ``executions`` that are already stored, grouped per market."""
        known: set[str] = set()
        by_market: dict[tuple[str, str], list[str]] = {}
        for execution in executions:
            by_market.setdefault((execution.exchange, execution.product_code), []).append(
                execution.exchange_execution_id
            )
        for (exchange, product_code), ids in by_market.items():
            statement = select(MarketExecution.exchange_execution_id).where(
                MarketExecution.exchange == exchange,
                MarketExecution.product_code == product_code,
                MarketExecution.exchange_execution_id.in_(ids),
            )
            known.update((await self._session.execute(statement)).scalars())
        return known

    async def insert_order_book_snapshot(self, book: OrderBook, depth_levels: int) -> bool:
        truncated = book.truncated(depth_levels)
        statement = _insert_ignore(
            self._session,
            OrderBookSnapshot,
            ["exchange", "product_code", "received_at"],
        ).values(
            exchange=truncated.exchange,
            product_code=truncated.product_code,
            received_at=truncated.received_at,
            mid_price=truncated.mid_price,
            depth_levels=depth_levels,
            bids=jsonable([level.model_dump() for level in truncated.bids]),
            asks=jsonable([level.model_dump() for level in truncated.asks]),
        )
        return await _insert_returned_a_row(self._session, statement, OrderBookSnapshot.id)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    async def latest_tick(
        self, product_code: str, exchange: str | None = None
    ) -> MarketTick | None:
        statement = (
            select(MarketTick)
            .where(MarketTick.product_code == product_code)
            .order_by(MarketTick.exchange_timestamp.desc())
            .limit(1)
        )
        if exchange:
            statement = statement.where(MarketTick.exchange == exchange)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def ticks_between(
        self,
        product_code: str,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        exchange: str | None = None,
        limit: int = 5000,
        ascending: bool = True,
    ) -> list[MarketTick]:
        statement: Select[tuple[MarketTick]] = select(MarketTick).where(
            MarketTick.product_code == product_code
        )
        if exchange:
            statement = statement.where(MarketTick.exchange == exchange)
        if start is not None:
            statement = statement.where(MarketTick.exchange_timestamp >= start)
        if end is not None:
            statement = statement.where(MarketTick.exchange_timestamp <= end)
        # Take the most recent `limit` rows, then present oldest-first if asked.
        statement = statement.order_by(MarketTick.exchange_timestamp.desc()).limit(limit)
        rows = list((await self._session.execute(statement)).scalars())
        return list(reversed(rows)) if ascending else rows

    async def count_ticks(self, product_code: str) -> int:
        statement = (
            select(func.count())
            .select_from(MarketTick)
            .where(MarketTick.product_code == product_code)
        )
        return int((await self._session.execute(statement)).scalar_one())

    async def tick_time_range(self, product_code: str) -> tuple[datetime | None, datetime | None]:
        statement = select(
            func.min(MarketTick.exchange_timestamp), func.max(MarketTick.exchange_timestamp)
        ).where(MarketTick.product_code == product_code)
        first, last = (await self._session.execute(statement)).one()
        return first, last

    async def executions_between(
        self,
        product_code: str,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        limit: int = 20000,
        ascending: bool = True,
    ) -> list[MarketExecution]:
        statement: Select[tuple[MarketExecution]] = select(MarketExecution).where(
            MarketExecution.product_code == product_code
        )
        if start is not None:
            statement = statement.where(MarketExecution.executed_at >= start)
        if end is not None:
            statement = statement.where(MarketExecution.executed_at <= end)
        statement = statement.order_by(MarketExecution.executed_at.desc()).limit(limit)
        rows = list((await self._session.execute(statement)).scalars())
        return list(reversed(rows)) if ascending else rows

    async def oldest_stored_execution_id(self, product_code: str, exchange: str) -> int | None:
        """Smallest stored numeric execution id — the anchor for backfill paging.

        Ids are stored as text (the column is exchange-agnostic), so sorting by
        length first and then lexicographically reproduces numeric order for the
        non-negative integers bitFlyer uses, on both PostgreSQL and SQLite.
        """
        statement = (
            select(MarketExecution.exchange_execution_id)
            .where(
                MarketExecution.product_code == product_code,
                MarketExecution.exchange == exchange,
            )
            .order_by(
                func.length(MarketExecution.exchange_execution_id),
                MarketExecution.exchange_execution_id,
            )
            .limit(1)
        )
        value = (await self._session.execute(statement)).scalar_one_or_none()
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:  # pragma: no cover - ids are numeric on bitFlyer
            logger.warning("non_numeric_execution_id", execution_id=value)
            return None

    async def count_executions(self, product_code: str) -> int:
        statement = (
            select(func.count())
            .select_from(MarketExecution)
            .where(MarketExecution.product_code == product_code)
        )
        return int((await self._session.execute(statement)).scalar_one())

    async def latest_execution_id(self, product_code: str, exchange: str) -> int | None:
        """Return the newest numeric execution id for an incremental cursor."""
        statement = (
            select(MarketExecution.exchange_execution_id)
            .where(
                MarketExecution.product_code == product_code,
                MarketExecution.exchange == exchange,
            )
            .order_by(
                func.length(MarketExecution.exchange_execution_id).desc(),
                MarketExecution.exchange_execution_id.desc(),
            )
            .limit(1)
        )
        value = (await self._session.execute(statement)).scalar_one_or_none()
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            logger.warning("non_numeric_latest_execution_id", execution_id=value)
            return None

    async def latest_order_book(self, product_code: str) -> OrderBookSnapshot | None:
        statement = (
            select(OrderBookSnapshot)
            .where(OrderBookSnapshot.product_code == product_code)
            .order_by(OrderBookSnapshot.received_at.desc())
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def order_books_between(
        self,
        product_code: str,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        limit: int = 5000,
        ascending: bool = True,
    ) -> list[OrderBookSnapshot]:
        """Return the newest stored order-book snapshots in chronological order."""
        statement: Select[tuple[OrderBookSnapshot]] = select(OrderBookSnapshot).where(
            OrderBookSnapshot.product_code == product_code
        )
        if start is not None:
            statement = statement.where(OrderBookSnapshot.received_at >= start)
        if end is not None:
            statement = statement.where(OrderBookSnapshot.received_at <= end)
        statement = statement.order_by(OrderBookSnapshot.received_at.desc()).limit(limit)
        rows = list((await self._session.execute(statement)).scalars())
        return list(reversed(rows)) if ascending else rows

    async def price_at_or_before(self, product_code: str, moment: datetime) -> Decimal | None:
        """LTP of the newest tick at or before ``moment`` (period-change baseline)."""
        statement = (
            select(MarketTick.ltp)
            .where(
                MarketTick.product_code == product_code,
                MarketTick.exchange_timestamp <= moment,
                MarketTick.ltp.is_not(None),
            )
            .order_by(MarketTick.exchange_timestamp.desc())
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()


class CollectorRunRepository:
    """Bookkeeping for collector runs (dashboard "System" panel)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start(self, *, exchange: str, product_code: str, trigger: str) -> CollectorRun:
        run = CollectorRun(
            started_at=utcnow(), exchange=exchange, product_code=product_code, trigger=trigger
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def finish(
        self,
        run: CollectorRun,
        *,
        success: bool,
        ticks_inserted: int = 0,
        executions_inserted: int = 0,
        order_books_inserted: int = 0,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> CollectorRun:
        run.finished_at = utcnow()
        run.duration_ms = int((run.finished_at - run.started_at).total_seconds() * 1000)
        run.success = success
        run.ticks_inserted = ticks_inserted
        run.executions_inserted = executions_inserted
        run.order_books_inserted = order_books_inserted
        run.error_type = error_type
        run.error_message = error_message[:2000] if error_message else None
        await self._session.flush()
        return run

    async def last_run(self) -> CollectorRun | None:
        statement = select(CollectorRun).order_by(CollectorRun.started_at.desc()).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def last_successful_run(self) -> CollectorRun | None:
        statement = (
            select(CollectorRun)
            .where(CollectorRun.success.is_(True))
            .order_by(CollectorRun.started_at.desc())
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def recent_runs(self, limit: int = 20) -> list[CollectorRun]:
        statement = select(CollectorRun).order_by(CollectorRun.started_at.desc()).limit(limit)
        return list((await self._session.execute(statement)).scalars())


def _deduplicate_executions(executions: list[Execution]) -> list[Execution]:
    """Drop repeats inside a single batch (ON CONFLICT cannot fix intra-batch dupes)."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[Execution] = []
    for execution in executions:
        key = (execution.exchange, execution.product_code, execution.exchange_execution_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(execution)
    return unique
