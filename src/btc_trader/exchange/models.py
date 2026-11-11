"""Exchange-neutral market and trading value objects.

Pydantic models sit at the external boundary (PROJECT.md §49).  Every numeric
field is a ``Decimal``: values are parsed from the JSON *text* so no binary
float ever touches a price or size.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from btc_trader.enums import ExchangeHealthState, Side


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _to_decimal(value: Any) -> Any:
    """Coerce ints/floats/strings to ``Decimal`` without float rounding.

    ``json.loads`` yields floats for unquoted JSON numbers; ``repr`` of a float
    is the shortest string that round-trips, so ``Decimal(repr(x))`` recovers the
    value the exchange actually sent rather than its binary expansion.
    """
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, int | str):
        return Decimal(str(value))
    return value


class Ticker(_Frozen):
    """Normalised ticker snapshot."""

    exchange: str
    product_code: str
    exchange_timestamp: datetime
    received_at: datetime

    tick_id: int | None = None
    state: str | None = None

    ltp: Decimal | None = None
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    best_bid_size: Decimal | None = None
    best_ask_size: Decimal | None = None
    total_bid_depth: Decimal | None = None
    total_ask_depth: Decimal | None = None
    market_bid_size: Decimal | None = None
    market_ask_size: Decimal | None = None
    volume: Decimal | None = None
    volume_by_product: Decimal | None = None

    raw_payload: dict[str, Any] = Field(default_factory=dict)

    _decimals = field_validator(
        "ltp",
        "best_bid",
        "best_ask",
        "best_bid_size",
        "best_ask_size",
        "total_bid_depth",
        "total_ask_depth",
        "market_bid_size",
        "market_ask_size",
        "volume",
        "volume_by_product",
        mode="before",
    )(_to_decimal)


class Balance(_Frozen):
    currency_code: str
    amount: Decimal
    available: Decimal

    _decimals = field_validator("amount", "available", mode="before")(_to_decimal)


class ExchangeHealth(_Frozen):
    """Order-book state: orders are only sent while it is RUNNING and healthy."""

    exchange: str
    product_code: str
    state: ExchangeHealthState
    board_state: str | None = None
    checked_at: datetime

    @field_validator("state", mode="before")
    @classmethod
    def _state(cls, value: object) -> ExchangeHealthState:
        return ExchangeHealthState.parse(value)

    @property
    def accepts_orders(self) -> bool:
        return self.state.accepts_orders and self.board_state in (None, "RUNNING")


class OrderRequest(_Frozen):
    """A limit order to submit. ``price`` is the worst acceptable fill price."""

    product_code: str
    side: Side
    size: Decimal
    price: Decimal
    time_in_force: Literal["GTC", "IOC", "FOK"] = "IOC"
    minute_to_expire: int = Field(default=1, ge=1, le=43_200)

    _decimals = field_validator("size", "price", mode="before")(_to_decimal)

    @field_validator("size", "price")
    @classmethod
    def _positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("order size and price must be positive")
        return value


class ExchangeOrder(_Frozen):
    """An order as the exchange reports it (``getchildorders``)."""

    acceptance_id: str
    side: Side
    size: Decimal
    state: Literal["ACTIVE", "COMPLETED", "CANCELED", "EXPIRED", "REJECTED"]
    executed_size: Decimal = Decimal("0")
    average_price: Decimal | None = None
    total_commission: Decimal = Decimal("0")
    ordered_at: datetime
    raw: dict[str, Any] = Field(default_factory=dict)

    _decimals = field_validator(
        "size", "executed_size", "average_price", "total_commission", mode="before"
    )(_to_decimal)

    @property
    def is_final(self) -> bool:
        return self.state != "ACTIVE"


class OwnExecution(_Frozen):
    """One fill of one of our orders (``me/getexecutions``).

    bitFlyer charges BTC/JPY spot commission in BTC, so ``commission`` is BTC.
    """

    execution_id: str
    acceptance_id: str
    side: Side
    price: Decimal
    size: Decimal
    commission: Decimal = Decimal("0")
    executed_at: datetime

    _decimals = field_validator("price", "size", "commission", mode="before")(_to_decimal)


__all__ = [
    "Balance",
    "ExchangeHealth",
    "ExchangeOrder",
    "OrderRequest",
    "OwnExecution",
    "Ticker",
]
