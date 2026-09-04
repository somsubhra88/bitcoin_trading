"""Exchange-agnostic client interface (PROJECT.md §8).

Nothing above this layer may import ``bitflyer``.  Adding Kraken or Binance
later means implementing this interface, not touching strategy, storage or
dashboard code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from btc_trader.exchange.models import (
    Balance,
    ExchangeHealth,
    Execution,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatusReport,
    Ticker,
)


class ExchangeClient(ABC):
    """Read and trade against a single exchange.

    Implementations must be safe to use concurrently from one event loop and
    must route *every* HTTP request through their own rate-limited transport
    (PROJECT.md §33).
    """

    #: Short identifier persisted in the ``exchange`` column.
    name: str

    # ------------------------------------------------------------------ #
    # public market data
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_ticker(self, product_code: str) -> Ticker: ...

    @abstractmethod
    async def get_order_book(self, product_code: str) -> OrderBook: ...

    @abstractmethod
    async def get_recent_executions(
        self,
        product_code: str,
        limit: int = 100,
        *,
        before_id: int | None = None,
        after_id: int | None = None,
    ) -> list[Execution]: ...

    @abstractmethod
    async def get_health(self, product_code: str) -> ExchangeHealth: ...

    # ------------------------------------------------------------------ #
    # private / authenticated
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def get_balances(self) -> list[Balance]: ...

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse: ...

    @abstractmethod
    async def cancel_order(self, product_code: str, order_id: str) -> None: ...

    @abstractmethod
    async def get_order_status(self, product_code: str, order_id: str) -> OrderStatusReport: ...

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def aclose(self) -> None:
        """Release network resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
