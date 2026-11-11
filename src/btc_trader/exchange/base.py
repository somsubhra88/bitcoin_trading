"""Exchange-agnostic client interface (PROJECT.md §8).

Nothing above this layer may import ``bitflyer``. Adding another exchange means
implementing this interface, not touching strategy, storage or dashboard code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from btc_trader.exchange.models import (
    Balance,
    ExchangeHealth,
    ExchangeOrder,
    OrderRequest,
    OwnExecution,
    Ticker,
)


class ExchangeClient(ABC):
    """Read and trade against a single exchange.

    Implementations route every HTTP request through their own rate-limited
    transport (PROJECT.md §33) and never retry an order submission.
    """

    #: Short identifier persisted in the ``exchange`` column.
    name: str

    # ------------------------------------------------------------------ #
    # public market data
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_ticker(self, product_code: str) -> Ticker: ...

    @abstractmethod
    async def get_health(self, product_code: str) -> ExchangeHealth: ...

    # ------------------------------------------------------------------ #
    # private / authenticated
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_balances(self) -> list[Balance]: ...

    @abstractmethod
    async def get_trading_commission(self, product_code: str) -> Decimal:
        """The account's current commission rate as a fraction (0.0015 = 0.15%)."""

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> str:
        """Submit once and return the exchange's acceptance id. Never retried."""

    @abstractmethod
    async def get_order(self, product_code: str, acceptance_id: str) -> ExchangeOrder | None:
        """The order, or ``None`` if the exchange does not list it (yet)."""

    @abstractmethod
    async def get_recent_orders(self, product_code: str, count: int = 20) -> list[ExchangeOrder]:
        """Newest orders first, including open ones."""

    @abstractmethod
    async def get_executions(self, product_code: str, acceptance_id: str) -> list[OwnExecution]:
        """Fills of one order."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release network resources."""
