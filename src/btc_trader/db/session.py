"""Async engine / session management.

A single lazily-created engine per process.  Callers acquire sessions through
:func:`session_scope` (transactional) or :func:`get_session` (FastAPI dependency)
and never construct engines themselves.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from btc_trader.config import Settings, get_settings
from btc_trader.monitoring.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "echo": settings.db_echo,
        "future": True,
        "pool_pre_ping": True,
    }
    if settings.async_database_url.startswith("sqlite"):
        # SQLite: no server-side pool sizing, and NullPool avoids cross-loop reuse.
        kwargs["pool_pre_ping"] = False
    else:
        kwargs.update(pool_size=5, max_overflow=5, pool_recycle=1800)
    return kwargs


@event.listens_for(Engine, "connect")
def _configure_sqlite_connection(dbapi_connection: Any, _record: Any) -> None:
    """Enable foreign keys and WAL for SQLite; harmless for other backends."""
    module = type(dbapi_connection).__module__
    if "sqlite" not in module:
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return (creating on first call) the process-wide async engine."""
    global _engine, _sessionmaker
    if _engine is None:
        resolved = settings or get_settings()
        url = resolved.async_database_url
        _ensure_sqlite_parent(url)
        _engine = create_async_engine(url, **_engine_kwargs(resolved))
        _sessionmaker = async_sessionmaker(
            _engine, expire_on_commit=False, autoflush=False, class_=AsyncSession
        )
        logger.info("database_engine_created", backend=url.split("://", 1)[0])
    return _engine


def _ensure_sqlite_parent(url: str) -> None:
    """Create a file-backed SQLite parent directory on first local start."""
    parsed = make_url(url)
    if not parsed.drivername.startswith("sqlite") or not parsed.database:
        return
    if parsed.database == ":memory:":
        return
    Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    get_engine(settings)
    assert _sessionmaker is not None  # noqa: S101 - set by get_engine
    return _sessionmaker


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """Transactional session: commits on success, rolls back on failure."""
    factory = get_sessionmaker(settings)
    session = factory()
    try:
        yield session
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    finally:
        await session.close()


async def check_database(settings: Settings | None = None) -> tuple[bool, str | None]:
    """Cheap liveness probe used by ``/health``."""
    try:
        engine = get_engine(settings)
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - health probe reports, never raises
        logger.warning("database_health_check_failed", error=str(exc))
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


async def dispose_engine() -> None:
    """Close pooled connections (application shutdown, test teardown)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        logger.info("database_engine_disposed")
    _engine = None
    _sessionmaker = None
