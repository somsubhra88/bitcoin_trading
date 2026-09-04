"""Shared test fixtures.

Two rules the whole suite depends on:

1. **No test reads the developer's ``.env``.** Environment variables take
   precedence over the ``.env`` file in pydantic-settings, so every setting that
   could change behaviour — credentials above all — is pinned here.
2. **No test contacts a private bitFlyer endpoint.** Public traffic is mocked
   with ``respx``; the handful of tests that really call bitFlyer carry the
   ``live`` marker and are deselected by default (PROJECT.md §46).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, get_settings, reload_settings
from btc_trader.db.base import Base
from btc_trader.db.session import dispose_engine, get_engine, session_scope

#: Deterministic configuration for every test. Credentials are empty on purpose:
#: nothing in the suite may be able to authenticate against bitFlyer.
BASE_ENV: dict[str, str] = {
    "APP_ENV": "development",
    "LOG_LEVEL": "WARNING",
    "LOG_FORMAT": "console",
    "TRADING_MODE": "data_only",
    "EXCHANGE": "bitflyer",
    "PRODUCT_CODE": "BTC_JPY",
    "BITFLYER_PUBLIC_BASE_URL": "https://api.bitflyer.com",
    "BITFLYER_API_KEY": "",
    "BITFLYER_API_SECRET": "",
    "BITFLYER_MAX_RETRIES": "0",
    "MARKET_DATA_INTERVAL_SECONDS": "300",
    "COLLECT_ORDER_BOOK": "true",
    "COLLECT_EXECUTIONS": "true",
    "EXECUTIONS_FETCH_LIMIT": "100",
    "ORDER_BOOK_DEPTH_LEVELS": "10",
    "ENABLE_AUTOMATIC_TRADING": "false",
    "LIVE_TRADING_ACKNOWLEDGED": "false",
    "TRADING_KILL_SWITCH": "false",
    "RUN_SCHEDULER_IN_API": "false",
    "MAX_MARKET_DATA_AGE_SECONDS": "600",
    "DASHBOARD_TIMEZONE": "Asia/Tokyo",
    "DB_ECHO": "false",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Pin the environment and give the test its own SQLite file."""
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    yield db_path
    get_settings.cache_clear()


@pytest.fixture
def settings(env: Path) -> Settings:
    """Settings built from the pinned environment (``env`` is for ordering)."""
    return reload_settings()


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Settings]:
    """Create the schema from the ORM metadata and dispose the engine after."""
    await dispose_engine()
    engine = get_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield settings
    finally:
        await dispose_engine()


@pytest.fixture
async def session(database: Settings) -> AsyncIterator[AsyncSession]:
    async with session_scope(database) as db_session:
        yield db_session
