"""Portable SQLAlchemy column types.

Two invariants from PROJECT.md must hold on both PostgreSQL and SQLite:

* §13 — money and quantities are ``Decimal``, never binary floats.
* §38 — datetimes are timezone-aware UTC, never naive.

PostgreSQL enforces both natively.  SQLite has neither NUMERIC nor timezone
support, so these ``TypeDecorator``s store text there and convert on the way
back out.  That keeps local development faithful to production semantics.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, Dialect, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, TypeDecorator

from btc_trader.utils.timeutils import ensure_utc

#: Precision wide enough for JPY notionals and 8-decimal BTC sizes alike.
MONEY_PRECISION = 30
MONEY_SCALE = 12


class DecimalType(TypeDecorator[Decimal]):
    """``NUMERIC`` on PostgreSQL, lossless ``TEXT`` on SQLite."""

    impl = Numeric
    cache_ok = True

    def __init__(self, precision: int = MONEY_PRECISION, scale: int = MONEY_SCALE) -> None:
        self.precision = precision
        self.scale = scale
        super().__init__(precision=precision, scale=scale, asdecimal=True)

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(64))
        return dialect.type_descriptor(Numeric(self.precision, self.scale, asdecimal=True))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if dialect.name == "sqlite":
            return format(decimal_value, "f")
        return decimal_value

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetime that survives a SQLite round trip."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"expected datetime, got {type(value)!r}")
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed (PROJECT.md §38)")
        utc_value = ensure_utc(value)
        # SQLite stores naive strings; strip the offset after normalising to UTC
        # so ordering comparisons stay correct.
        return utc_value.replace(tzinfo=None) if dialect.name == "sqlite" else utc_value

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)


#: JSON payloads: JSONB where available, plain JSON elsewhere.
JsonPayload = JSON().with_variant(JSONB, "postgresql")

#: Free-form text (reasons, messages).
LongText = Text()

__all__ = [
    "MONEY_PRECISION",
    "MONEY_SCALE",
    "DecimalType",
    "JsonPayload",
    "LongText",
    "UtcDateTime",
]
