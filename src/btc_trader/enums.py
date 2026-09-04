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

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATES

    @property
    def is_open(self) -> bool:
        return self in _OPEN_ORDER_STATES


_TERMINAL_ORDER_STATES = frozenset(
    {
        OrderStatus.REJECTED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.FAILED,
    }
)

_OPEN_ORDER_STATES = frozenset(
    {
        OrderStatus.CREATED,
        OrderStatus.VALIDATED,
        OrderStatus.SUBMITTING,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCEL_PENDING,
    }
)


class TradingSignal(StrEnum):
    """Strategy output (PROJECT.md §17)."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ExitReason(StrEnum):
    """Why an exit is being attempted (PROJECT.md §21).

    The minimum-profit rule applies to ``STRATEGY_EXIT`` only; every other reason
    is a risk control and must be able to override it — including when the live
    strategy signal is still BUY.
    """

    STRATEGY_EXIT = "STRATEGY_EXIT"
    RISK_EXIT = "RISK_EXIT"
    MANUAL_EXIT = "MANUAL_EXIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXIT = "TIME_EXIT"

    @property
    def is_risk_control(self) -> bool:
        """Risk controls ignore ``min_profit_percent`` and ignore the signal."""
        return self is not ExitReason.STRATEGY_EXIT


class Severity(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CandleSource(StrEnum):
    """Provenance of an OHLCV bar (PROJECT.md §40).

    ``SAMPLED_TICKS`` bars are *not* true exchange OHLC and must be labelled as
    such wherever they are displayed.
    """

    EXECUTIONS = "executions"
    SAMPLED_TICKS = "sampled_ticks"


class ExchangeHealthState(StrEnum):
    """Normalised exchange availability."""

    NORMAL = "NORMAL"
    BUSY = "BUSY"
    VERY_BUSY = "VERY_BUSY"
    SUPER_BUSY = "SUPER_BUSY"
    NO_ORDER = "NO_ORDER"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"

    @property
    def accepts_orders(self) -> bool:
        return self not in (
            ExchangeHealthState.NO_ORDER,
            ExchangeHealthState.STOP,
            ExchangeHealthState.UNKNOWN,
        )
