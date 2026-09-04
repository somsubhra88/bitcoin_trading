"""An in-memory :class:`ExchangeClient` for tests.

Lets collector/scheduler tests exercise the real storage and bookkeeping paths
with zero network traffic, and lets each optional call be made to fail
independently so degradation can be asserted (PROJECT.md §10, §46).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from btc_trader.enums import ExchangeHealthState, Side
from btc_trader.errors import ExchangeError
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.models import (
    Balance,
    ExchangeHealth,
    Execution,
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    OrderResponse,
    OrderStatusReport,
    Ticker,
)
from btc_trader.utils.timeutils import utcnow

BASE_TIME = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def make_ticker(
    *,
    exchange: str = "fake",
    product_code: str = "BTC_JPY",
    timestamp: datetime = BASE_TIME,
    ltp: str = "12000000",
    best_bid: str = "11999000",
    best_ask: str = "12001000",
) -> Ticker:
    return Ticker(
        exchange=exchange,
        product_code=product_code,
        exchange_timestamp=timestamp,
        received_at=timestamp,
        tick_id=1,
        state="RUNNING",
        ltp=Decimal(ltp),
        best_bid=Decimal(best_bid),
        best_ask=Decimal(best_ask),
        volume=Decimal("1234.5678"),
        raw_payload={"ltp": ltp},
    )


def make_order_book(
    *,
    exchange: str = "fake",
    product_code: str = "BTC_JPY",
    received_at: datetime = BASE_TIME,
    levels: int = 3,
) -> OrderBook:
    return OrderBook(
        exchange=exchange,
        product_code=product_code,
        received_at=received_at,
        mid_price=Decimal("12000000"),
        bids=tuple(
            OrderBookLevel(price=Decimal(11999000 - i), size=Decimal("0.5")) for i in range(levels)
        ),
        asks=tuple(
            OrderBookLevel(price=Decimal(12001000 + i), size=Decimal("0.5")) for i in range(levels)
        ),
    )


def make_executions(
    count: int,
    *,
    exchange: str = "fake",
    product_code: str = "BTC_JPY",
    first_id: int = 1000,
    start: datetime = BASE_TIME,
) -> list[Execution]:
    return [
        Execution(
            exchange=exchange,
            product_code=product_code,
            exchange_execution_id=str(first_id + index),
            side=Side.BUY if index % 2 == 0 else Side.SELL,
            price=Decimal(12000000 + index),
            size=Decimal("0.01"),
            executed_at=start + timedelta(seconds=index),
            received_at=start + timedelta(seconds=index),
        )
        for index in range(count)
    ]


class FakeExchangeClient(ExchangeClient):
    """Serves canned market data and counts calls.

    Any of ``fail_ticker`` / ``fail_order_book`` / ``fail_executions`` raises an
    :class:`ExchangeError` from the matching call.
    """

    name = "fake"

    def __init__(
        self,
        *,
        ticker: Ticker | None = None,
        order_book: OrderBook | None = None,
        executions: list[Execution] | None = None,
        fail_ticker: bool = False,
        fail_order_book: bool = False,
        fail_executions: bool = False,
        health_state: ExchangeHealthState = ExchangeHealthState.NORMAL,
    ) -> None:
        self.ticker = ticker or make_ticker()
        self.order_book = order_book or make_order_book()
        self.executions = executions if executions is not None else make_executions(3)
        self.fail_ticker = fail_ticker
        self.fail_order_book = fail_order_book
        self.fail_executions = fail_executions
        self.health_state = health_state
        self.calls: list[str] = []
        self.closed = False

    # -- public ------------------------------------------------------------
    async def get_ticker(self, product_code: str) -> Ticker:
        self.calls.append("get_ticker")
        if self.fail_ticker:
            raise ExchangeError("ticker unavailable")
        return self.ticker

    async def get_order_book(self, product_code: str) -> OrderBook:
        self.calls.append("get_order_book")
        if self.fail_order_book:
            raise ExchangeError("board unavailable")
        return self.order_book

    async def get_recent_executions(
        self,
        product_code: str,
        limit: int = 100,
        *,
        before_id: int | None = None,
        after_id: int | None = None,
    ) -> list[Execution]:
        self.calls.append("get_recent_executions")
        if self.fail_executions:
            raise ExchangeError("executions unavailable")
        return self.executions[:limit]

    async def get_health(self, product_code: str) -> ExchangeHealth:
        self.calls.append("get_health")
        return ExchangeHealth(
            exchange=self.name,
            product_code=product_code,
            state=self.health_state,
            checked_at=utcnow(),
        )

    # -- private -----------------------------------------------------------
    # These must never be reachable from a test: the suite has no credentials
    # and no exchange order may be submitted by the test fixture (PROJECT.md §48).
    async def get_balances(self) -> list[Balance]:
        raise AssertionError("a test reached a private endpoint")

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        raise AssertionError("a test attempted to submit an order")

    async def cancel_order(self, product_code: str, order_id: str) -> None:
        raise AssertionError("a test attempted to cancel an order")

    async def get_order_status(self, product_code: str, order_id: str) -> OrderStatusReport:
        raise AssertionError("a test reached a private endpoint")

    async def aclose(self) -> None:
        self.closed = True
