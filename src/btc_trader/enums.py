"""Domain enumerations shared by configuration, persistence and services.

These are deliberately exchange-agnostic: adapters translate bitFlyer's
vocabulary into these values, never the other way round.
"""

from __future__ import annotations

from enum import StrEnum


class TradingMode(StrEnum):
    """Operating mode of the platform (PROJECT.md §2.3)."""

    DATA_ONLY = "data_only"
    PAPER = "paper"
    LIVE = "live"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    """Explicit order state machine (PROJECT.md §31)."""

    #: Persisted locally; the submission request is about to be sent.
    SUBMITTING = "SUBMITTING"
    #: Accepted by the exchange; final state not yet confirmed.
    SUBMITTED = "SUBMITTED"
    #: The submission's outcome is unknown (e.g. timeout); must be reconciled.
    UNKNOWN = "UNKNOWN"
    FILLED = "FILLED"
    #: Ended without filling completely (an IOC remainder); may hold a partial fill.
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

    @property
    def is_open(self) -> bool:
        """Not final yet: no new order may be sent while one is open."""
        return self in _OPEN_ORDER_STATES


_OPEN_ORDER_STATES = frozenset({OrderStatus.SUBMITTING, OrderStatus.SUBMITTED, OrderStatus.UNKNOWN})


class TradingSignal(StrEnum):
    """Strategy output (PROJECT.md §17)."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ExitReason(StrEnum):
    """Why a sell is made (PROJECT.md §21).

    The minimum-profit rule applies to ``STRATEGY_EXIT`` only; stop and
    volatility exits are risk controls and must be able to sell at a loss.
    """

    STRATEGY_EXIT = "STRATEGY_EXIT"
    RISK_EXIT = "RISK_EXIT"
    TRAILING_STOP = "TRAILING_STOP"


class Severity(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CandleSource(StrEnum):
    """Provenance of a stored daily bar (PROJECT.md §40)."""

    SAMPLED_TICKS = "sampled_ticks"
    #: Bars published by an external exchange (e.g. bitbank daily OHLC), used
    #: to bootstrap long daily history that bitFlyer's public API cannot serve.
    EXCHANGE_OHLC = "exchange_ohlc"


class ExchangeHealthState(StrEnum):
    """Normalised exchange availability (bitFlyer ``getboardstate`` health)."""

    NORMAL = "NORMAL"
    BUSY = "BUSY"
    VERY_BUSY = "VERY_BUSY"
    SUPER_BUSY = "SUPER_BUSY"
    NO_ORDER = "NO_ORDER"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, value: object) -> ExchangeHealthState:
        """bitFlyer spells states with spaces ("VERY BUSY", "NO ORDER")."""
        key = str(value or "").strip().upper().replace(" ", "_")
        return cls.__members__.get(key, cls.UNKNOWN)

    @property
    def accepts_orders(self) -> bool:
        # SUPER BUSY: "orders may fail or be processed after a delay" — not
        # worth the risk for a strategy that can wait five minutes.
        return self in (
            ExchangeHealthState.NORMAL,
            ExchangeHealthState.BUSY,
            ExchangeHealthState.VERY_BUSY,
        )
