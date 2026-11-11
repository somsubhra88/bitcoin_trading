"""Alembic migrations (PROJECT.md §12, §51.5).

The initial revision must build the schema the ORM expects, on both supported
backends.  SQLite is the one exercised here (it is the stricter of the two for
primary keys and has no NUMERIC), and the check that matters is that a migrated
database and the ORM metadata do not drift: if they do, ``init_db`` and
``alembic upgrade head`` would produce different databases.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from btc_trader.config import Settings
from btc_trader.db.base import Base

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _alembic(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    """Run alembic in a subprocess so it builds its own engine and settings."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "DATABASE_URL": database_url,
            # Credentials stay empty: a migration must never need them.
            "BITFLYER_API_KEY": "",
            "BITFLYER_API_SECRET": "",
            "TRADING_MODE": "data_only",
            "LOG_LEVEL": "WARNING",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'migrated.db'}"


def test_upgrade_head_builds_the_schema(sqlite_url: str, tmp_path: Path) -> None:
    result = _alembic("upgrade", "head", database_url=sqlite_url)

    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{tmp_path / 'migrated.db'}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert "alembic_version" in tables
    # Every ORM table exists, including the ones later phases will populate.
    assert set(Base.metadata.tables) <= tables


def test_the_migrated_schema_matches_the_orm_metadata(sqlite_url: str) -> None:
    """``alembic check`` is the guard against a model change without a revision."""
    assert _alembic("upgrade", "head", database_url=sqlite_url).returncode == 0

    result = _alembic("check", database_url=sqlite_url)

    assert result.returncode == 0, f"schema drift detected:\n{result.stdout}\n{result.stderr}"


def test_primary_keys_autoincrement_on_sqlite(sqlite_url: str, tmp_path: Path) -> None:
    """A BIGINT primary key is not a SQLite ROWID alias, so inserts would fail."""
    assert _alembic("upgrade", "head", database_url=sqlite_url).returncode == 0

    engine = create_engine(f"sqlite:///{tmp_path / 'migrated.db'}")
    try:
        inspector = inspect(engine)
        for table in ("market_ticks", "orders", "candles", "collector_runs", "system_events"):
            columns = {column["name"]: column for column in inspector.get_columns(table)}
            assert str(columns["id"]["type"]) == "INTEGER", table
    finally:
        engine.dispose()


def test_downgrade_removes_everything(sqlite_url: str, tmp_path: Path) -> None:
    assert _alembic("upgrade", "head", database_url=sqlite_url).returncode == 0

    result = _alembic("downgrade", "base", database_url=sqlite_url)

    assert result.returncode == 0, result.stderr
    engine = create_engine(f"sqlite:///{tmp_path / 'migrated.db'}")
    try:
        tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()
    assert tables == set()


def test_a_migration_needs_no_credentials(sqlite_url: str) -> None:
    """§7: schema work must never require the API key to be present."""
    result = _alembic("upgrade", "head", database_url=sqlite_url)

    assert result.returncode == 0
    assert "credential" not in result.stderr.lower()


def test_settings_derives_a_sync_url_for_alembic(settings: Settings) -> None:
    """Alembic runs on a blocking driver; the async URL must be translated."""
    assert "+aiosqlite" in settings.database_url
    # The blocking driver is named explicitly rather than left to SQLAlchemy's
    # default, so a mis-set DATABASE_URL fails loudly instead of silently using
    # the async driver under a synchronous engine.
    assert settings.sync_database_url.startswith("sqlite+pysqlite:///")
    assert "aiosqlite" not in settings.sync_database_url
