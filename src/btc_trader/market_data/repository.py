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

from btc_trader.db.models import CollectorRun, MarketTick
from btc_trader.exchange.models import Ticker
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
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> CollectorRun:
        run.finished_at = utcnow()
        run.duration_ms = int((run.finished_at - run.started_at).total_seconds() * 1000)
        run.success = success
        run.ticks_inserted = ticks_inserted
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
