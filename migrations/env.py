"""Alembic environment.

Uses the *blocking* driver: migrations are short, run once at start-up, and a
synchronous engine keeps this file simple.  The URL always comes from Settings so
``alembic.ini`` stays free of credentials.
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any

import sqlalchemy as sa
from alembic import context
from sqlalchemy import engine_from_config, pool

from btc_trader.config import get_settings
from btc_trader.db import models as _models  # noqa: F401 - registers all tables
from btc_trader.db.base import Base
from btc_trader.db.types import DecimalType, UtcDateTime

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

config.set_main_option("sqlalchemy.url", get_settings().sync_database_url)

target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render the project's portable column types instead of their SQLite/PG impls.

    Without this, autogenerate would freeze whatever the *current* dialect happens
    to use (``VARCHAR`` for Decimal on SQLite, ``JSON`` instead of ``JSONB`` on
    PostgreSQL) into the migration, so the same revision would build different
    schemas on different backends.
    """
    if type_ != "type":
        return False
    if isinstance(obj, DecimalType):
        autogen_context.imports.add("import btc_trader.db.types as bt")
        return f"bt.DecimalType({obj.precision}, {obj.scale})"
    if isinstance(obj, UtcDateTime):
        autogen_context.imports.add("import btc_trader.db.types as bt")
        return "bt.UtcDateTime()"
    if isinstance(obj, sa.JSON):
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "sa.JSON().with_variant(postgresql.JSONB(), 'postgresql')"
    if isinstance(obj, sa.BigInteger) and "sqlite" in getattr(obj, "_variant_mapping", {}):
        # The surrogate primary key: BIGINT on PostgreSQL, INTEGER (the ROWID
        # alias, hence autoincrementing) on SQLite.
        return "sa.BigInteger().with_variant(sa.Integer(), 'sqlite')"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_item=render_item,
            # batch mode lets SQLite handle ALTER TABLE in future migrations
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
