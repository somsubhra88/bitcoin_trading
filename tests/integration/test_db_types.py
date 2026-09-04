"""Column-type invariants on a real database (PROJECT.md §13, §38).

These run against SQLite, which has neither NUMERIC nor timezone support — the
harder of the two backends.  If a ``Decimal`` survives here it survives on
PostgreSQL too.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import MarketExecution, MarketTick, SystemEvent
from btc_trader.db.session import dispose_engine, get_engine
from btc_trader.enums import Severity, Side

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


async def test_file_backed_sqlite_creates_its_parent_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "btc-trader.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_path}")
    await dispose_engine()
    try:
        get_engine(settings)
        assert db_path.parent.is_dir()
    finally:
        await dispose_engine()


def _tick(**overrides: object) -> MarketTick:
    values: dict[str, object] = {
        "exchange": "bitflyer",
        "product_code": "BTC_JPY",
        "exchange_timestamp": NOW,
        "received_at": NOW,
    }
    values.update(overrides)
    return MarketTick(**values)


# --------------------------------------------------------------------------- #
# Decimal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        "12411935.0",
        "0.00171919",
        "1155.80728167",
        "0.1",  # unrepresentable as a binary float
        "99999999999999999.123456789012",  # full 30/12 precision
        "0",
        "-12345.6789",
    ],
)
async def test_decimal_survives_a_round_trip(session: AsyncSession, raw: str) -> None:
    session.add(_tick(ltp=Decimal(raw)))
    await session.flush()
    session.expunge_all()

    stored = (await session.execute(select(MarketTick.ltp))).scalar_one()
    assert isinstance(stored, Decimal)
    assert stored == Decimal(raw)
    assert str(stored) == raw


async def test_decimal_addition_stays_exact_after_reload(session: AsyncSession) -> None:
    """The classic float trap: 0.1 + 0.2 must be 0.3, not 0.30000000000000004."""
    for index, value in enumerate(("0.1", "0.2")):
        session.add(
            MarketExecution(
                exchange="bitflyer",
                product_code="BTC_JPY",
                exchange_execution_id=str(index),
                side=Side.BUY,
                price=Decimal("12000000"),
                size=Decimal(value),
                executed_at=NOW + timedelta(seconds=index),
                received_at=NOW,
            )
        )
    await session.flush()
    session.expunge_all()

    sizes = list((await session.execute(select(MarketExecution.size))).scalars())
    assert sum(sizes) == Decimal("0.3")


async def test_null_decimal_stays_null(session: AsyncSession) -> None:
    session.add(_tick(ltp=None))
    await session.flush()
    session.expunge_all()
    assert (await session.execute(select(MarketTick.ltp))).scalar_one() is None


# --------------------------------------------------------------------------- #
# UtcDateTime
# --------------------------------------------------------------------------- #
async def test_datetime_comes_back_aware_and_utc(session: AsyncSession) -> None:
    session.add(_tick())
    await session.flush()
    session.expunge_all()

    stored = (await session.execute(select(MarketTick.exchange_timestamp))).scalar_one()
    assert stored.tzinfo is not None
    assert stored.utcoffset() == timedelta(0)
    assert stored == NOW


async def test_offset_aware_input_is_normalised_to_utc(session: AsyncSession) -> None:
    tokyo = datetime(2026, 9, 1, 21, 0, tzinfo=timezone(timedelta(hours=9)))
    session.add(_tick(exchange_timestamp=tokyo))
    await session.flush()
    session.expunge_all()

    stored = (await session.execute(select(MarketTick.exchange_timestamp))).scalar_one()
    assert stored == datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    assert stored.utcoffset() == timedelta(0)


async def test_naive_datetime_is_rejected(session: AsyncSession) -> None:
    """§38: a naive datetime must never reach a column."""
    session.add(_tick(exchange_timestamp=datetime(2026, 9, 1, 12, 0)))  # deliberately naive
    with pytest.raises(StatementError, match="naive datetimes are not allowed"):
        await session.flush()
    await session.rollback()  # the fixture commits on exit; the transaction is dead


async def test_non_datetime_is_rejected(session: AsyncSession) -> None:
    session.add(_tick(exchange_timestamp="2026-09-01T12:00:00Z"))
    with pytest.raises(StatementError, match="expected datetime"):
        await session.flush()
    await session.rollback()


async def test_ordering_is_chronological_after_the_utc_normalisation(
    session: AsyncSession,
) -> None:
    """Mixed offsets must not scramble ``ORDER BY``."""
    session.add(_tick(exchange_timestamp=datetime(2026, 9, 1, 12, 0, tzinfo=UTC), ltp=Decimal(2)))
    session.add(
        _tick(
            exchange_timestamp=datetime(2026, 9, 1, 20, 0, tzinfo=timezone(timedelta(hours=9))),
            ltp=Decimal(1),
        )
    )
    await session.flush()
    session.expunge_all()

    statement = select(MarketTick.ltp).order_by(MarketTick.exchange_timestamp)
    assert list((await session.execute(statement)).scalars()) == [Decimal(1), Decimal(2)]


# --------------------------------------------------------------------------- #
# JSON payloads and enums
# --------------------------------------------------------------------------- #
async def test_json_payload_round_trip(session: AsyncSession) -> None:
    payload = {"ltp": "12411935.0", "levels": [{"price": "1", "size": "2"}]}
    session.add(_tick(raw_payload=payload))
    await session.flush()
    session.expunge_all()
    assert (await session.execute(select(MarketTick.raw_payload))).scalar_one() == payload


async def test_enum_columns_round_trip_as_values(session: AsyncSession) -> None:
    session.add(
        SystemEvent(
            timestamp=NOW,
            severity=Severity.WARNING,
            component="test",
            event_type="unit",
            message="hello",
        )
    )
    await session.flush()
    session.expunge_all()
    stored = (await session.execute(select(SystemEvent))).scalar_one()
    assert stored.severity is Severity.WARNING
