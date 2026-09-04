"""SQLAlchemy 2.x models for the schema described in PROJECT.md §12.

The market-data tables (``market_ticks``, ``executions``, ``order_book_snapshots``)
are written by the collector. Paper strategy evaluation also writes signals,
simulated orders, fills, positions and portfolio snapshots; live execution remains
deliberately disabled.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from btc_trader.db.base import Base
from btc_trader.db.types import DecimalType, JsonPayload, LongText, UtcDateTime
from btc_trader.enums import (
    CandleSource,
    ExitReason,
    OrderStatus,
    OrderType,
    Severity,
    Side,
    TradingMode,
    TradingSignal,
)

#: Surrogate primary key.  SQLite only auto-assigns a value for an ``INTEGER``
#: primary key (the ROWID alias) — a ``BIGINT`` one fails its NOT NULL check on
#: insert — so the variant keeps 64-bit ids on PostgreSQL and working ids on the
#: SQLite option offered in .env.example.
BigIntPk = BigInteger().with_variant(Integer(), "sqlite")


def _enum(python_enum: type, name: str) -> Enum:
    """Portable enum column: VARCHAR + CHECK on every backend.

    ``native_enum=False`` avoids PostgreSQL ``CREATE TYPE`` migrations whenever a
    new state is added to the order state machine.
    """
    return Enum(
        python_enum,
        name=name,
        native_enum=False,
        validate_strings=True,
        values_callable=lambda enum_cls: [member.value for member in enum_cls],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


# --------------------------------------------------------------------------- #
# §12.1 market_ticks
# --------------------------------------------------------------------------- #
class MarketTick(Base):
    """One normalised ticker observation.

    Deduplicated on ``(exchange, product_code, exchange_timestamp)``: repeated
    polling of an unchanged ticker inserts nothing (PROJECT.md §10.4).
    """

    __tablename__ = "market_ticks"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)

    exchange_timestamp: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    received_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    tick_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    state: Mapped[str | None] = mapped_column(String(32), nullable=True)

    ltp: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    best_bid: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    best_ask: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    best_bid_size: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    best_ask_size: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    total_bid_depth: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    total_ask_depth: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    market_bid_size: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    market_ask_size: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    volume: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    volume_by_product: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "product_code",
            "exchange_timestamp",
            name="uq_market_ticks_exchange_product_code_exchange_timestamp",
        ),
        Index("ix_market_ticks_product_code_ts", "product_code", "exchange_timestamp"),
        Index(
            "ix_market_ticks_exchange_product_ts",
            "exchange",
            "product_code",
            "exchange_timestamp",
        ),
    )

    @property
    def spread(self) -> Decimal | None:
        if self.best_ask is None or self.best_bid is None:
            return None
        return self.best_ask - self.best_bid


# --------------------------------------------------------------------------- #
# §12.2 executions (public market executions)
# --------------------------------------------------------------------------- #
class MarketExecution(Base):
    """A single public trade printed by the exchange.

    ``exchange_execution_id`` makes ingestion idempotent, which is what allows
    incremental backfill without duplicates (PROJECT.md §12.2).
    """

    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange_execution_id: Mapped[str] = mapped_column(String(64), nullable=False)

    side: Mapped[Side | None] = mapped_column(_enum(Side, "side_enum"), nullable=True)
    price: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    size: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)

    executed_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    received_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "product_code",
            "exchange_execution_id",
            name="uq_executions_exchange_product_code_exchange_execution_id",
        ),
        Index("ix_executions_product_code_executed_at", "product_code", "executed_at"),
    )


# --------------------------------------------------------------------------- #
# order-book snapshots (optional depth capture, PROJECT.md §9)
# --------------------------------------------------------------------------- #
class OrderBookSnapshot(Base):
    """Truncated order-book depth captured alongside a tick."""

    __tablename__ = "order_book_snapshots"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)
    received_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    mid_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    depth_levels: Mapped[int] = mapped_column(Integer, nullable=False)

    bids: Mapped[list[dict[str, str]]] = mapped_column(JsonPayload, nullable=False)
    asks: Mapped[list[dict[str, str]]] = mapped_column(JsonPayload, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "product_code",
            "received_at",
            name="uq_order_book_snapshots_exchange_product_code_received_at",
        ),
        Index("ix_order_book_snapshots_product_code_received_at", "product_code", "received_at"),
    )


# --------------------------------------------------------------------------- #
# §12.3 strategy_signals
# --------------------------------------------------------------------------- #
class StrategySignal(Base):
    __tablename__ = "strategy_signals"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    signal: Mapped[TradingSignal] = mapped_column(
        _enum(TradingSignal, "trading_signal_enum"), nullable=False
    )
    confidence: Mapped[Decimal | None] = mapped_column(DecimalType(10, 6), nullable=True)

    reference_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    reference_bid: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    reference_ask: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    reason: Mapped[str | None] = mapped_column(LongText, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload, nullable=True)

    __table_args__ = (
        Index("ix_strategy_signals_product_code_timestamp", "product_code", "timestamp"),
        Index("ix_strategy_signals_strategy_name_timestamp", "strategy_name", "timestamp"),
    )


# --------------------------------------------------------------------------- #
# §12.4 orders
# --------------------------------------------------------------------------- #
class Order(Base, TimestampMixin):
    """An intended order and its exchange lifecycle.

    ``client_order_id`` is unique: it is generated before submission and is the
    mechanism that prevents duplicate live orders (PROJECT.md §30).
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)

    side: Mapped[Side] = mapped_column(_enum(Side, "side_enum"), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(
        _enum(OrderType, "order_type_enum"), nullable=False
    )

    requested_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    requested_size: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)

    filled_size: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )
    average_fill_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    status: Mapped[OrderStatus] = mapped_column(
        _enum(OrderStatus, "order_status_enum"), nullable=False, default=OrderStatus.CREATED
    )

    strategy_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signal_id: Mapped[int | None] = mapped_column(
        ForeignKey("strategy_signals.id", ondelete="SET NULL"), nullable=True
    )
    exit_reason: Mapped[ExitReason | None] = mapped_column(
        _enum(ExitReason, "exit_reason_enum"), nullable=True
    )

    submitted_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)

    trading_mode: Mapped[TradingMode] = mapped_column(
        _enum(TradingMode, "trading_mode_enum"), nullable=False
    )

    rejection_reason: Mapped[str | None] = mapped_column(LongText, nullable=True)
    #: Full audit context: market state, risk checks, profitability maths (§36).
    decision_context: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload, nullable=True)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload, nullable=True)

    signal: Mapped[StrategySignal | None] = relationship(lazy="selectin")
    fills: Mapped[list[Fill]] = relationship(
        back_populates="order", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_orders_product_code_created_at", "product_code", "created_at"),
        Index("ix_orders_status", "status"),
        CheckConstraint("requested_size > 0", name="requested_size_positive"),
    )


# --------------------------------------------------------------------------- #
# §12.5 fills
# --------------------------------------------------------------------------- #
class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False
    )
    exchange_execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    price: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    size: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    commission: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False, default=Decimal("0"))

    executed_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    order: Mapped[Order] = relationship(back_populates="fills")

    __table_args__ = (
        UniqueConstraint(
            "order_id",
            "exchange_execution_id",
            name="uq_fills_order_id_exchange_execution_id",
        ),
        Index("ix_fills_executed_at", "executed_at"),
    )


# --------------------------------------------------------------------------- #
# §12.6 positions
# --------------------------------------------------------------------------- #
class Position(Base):
    """Effective spot inventory and weighted-average cost basis (§24)."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    product_code: Mapped[str] = mapped_column(String(32), nullable=False)
    trading_mode: Mapped[TradingMode] = mapped_column(
        _enum(TradingMode, "trading_mode_enum"), nullable=False
    )

    btc_quantity: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )
    average_entry_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    high_watermark_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)
    cost_basis_jpy: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )

    realized_pnl_jpy: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )
    unrealized_pnl_jpy: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )

    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "product_code", "trading_mode", name="uq_positions_product_code_trading_mode"
        ),
    )


# --------------------------------------------------------------------------- #
# §12.7 portfolio_snapshots
# --------------------------------------------------------------------------- #
class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    timestamp: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    trading_mode: Mapped[TradingMode] = mapped_column(
        _enum(TradingMode, "trading_mode_enum"), nullable=False
    )

    jpy_balance: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    btc_balance: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)

    btc_market_price: Mapped[Decimal | None] = mapped_column(DecimalType(), nullable=True)

    btc_value_jpy: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    total_value_jpy: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)

    realized_pnl_jpy: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )
    unrealized_pnl_jpy: Mapped[Decimal] = mapped_column(
        DecimalType(), nullable=False, default=Decimal("0")
    )

    __table_args__ = (Index("ix_portfolio_snapshots_timestamp", "timestamp"),)


# --------------------------------------------------------------------------- #
# §12.8 system_events
# --------------------------------------------------------------------------- #
class SystemEvent(Base):
    """Operational audit log: collector failures, rate limits, risk blocks."""

    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    timestamp: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    severity: Mapped[Severity] = mapped_column(_enum(Severity, "severity_enum"), nullable=False)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(LongText, nullable=False)
    event_metadata: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload, nullable=True)

    __table_args__ = (
        Index("ix_system_events_timestamp", "timestamp"),
        Index("ix_system_events_component_event_type", "component", "event_type"),
        Index("ix_system_events_severity_timestamp", "severity", "timestamp"),
    )


class RuntimeSetting(Base):
    """A dashboard-owned override of an environment setting.

    Values are JSON so Decimal and enum values retain their lossless wire form.
    Secrets and connection settings are intentionally not writable here.
    """

    __tablename__ = "runtime_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JsonPayload, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


# --------------------------------------------------------------------------- #
# collector run bookkeeping (observability for §15 "System" panel)
# --------------------------------------------------------------------------- #
class CollectorRun(Base):
    """One market-data collection attempt, successful or not."""

    __tablename__ = "collector_runs"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    started_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)  # scheduled | manual | cli

    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ticks_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    executions_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    order_books_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(LongText, nullable=True)

    __table_args__ = (
        Index("ix_collector_runs_started_at", "started_at"),
        Index("ix_collector_runs_success_started_at", "success", "started_at"),
    )


# --------------------------------------------------------------------------- #
# candles (§40) — materialised only when explicitly requested
# --------------------------------------------------------------------------- #
class Candle(Base):
    """OHLCV bar with explicit provenance.

    ``source`` distinguishes true execution-derived bars from bars sampled off
    LTP observations, which must never be presented as exchange OHLC (§40).
    """

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True, autoincrement=True)

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    product_code: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    source: Mapped[CandleSource] = mapped_column(
        _enum(CandleSource, "candle_source_enum"), nullable=False
    )

    bucket_start: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    open: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    high: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    low: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    close: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False)
    volume: Mapped[Decimal] = mapped_column(DecimalType(), nullable=False, default=Decimal("0"))
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "product_code",
            "interval",
            "source",
            "bucket_start",
            name="uq_candles_exchange_product_code_interval_source_bucket_start",
        ),
        Index(
            "ix_candles_product_code_interval_bucket_start",
            "product_code",
            "interval",
            "bucket_start",
        ),
    )


__all__ = [
    "Base",
    "Candle",
    "CollectorRun",
    "Fill",
    "MarketExecution",
    "MarketTick",
    "Order",
    "OrderBookSnapshot",
    "PortfolioSnapshot",
    "Position",
    "RuntimeSetting",
    "StrategySignal",
    "SystemEvent",
]
