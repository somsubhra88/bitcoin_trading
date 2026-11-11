"""An in-memory :class:`ExchangeClient` for tests.

Serves canned market data and simulates bitFlyer's private trading API: limit
IOC orders fill immediately at the touch (if the limit allows) with commission
charged in BTC, balances move accordingly, and every failure mode the live
engine must survive can be switched on. No network traffic is possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from btc_trader.enums import ExchangeHealthState, Side
from btc_trader.errors import ExchangeError, ExchangeResponseError, ExchangeTransientError
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.models import (
    Balance,
    ExchangeHealth,
    ExchangeOrder,
    OrderRequest,
    OwnExecution,
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


@dataclass
class _SimOrder:
    acceptance_id: str
    request: OrderRequest
    executions: list[OwnExecution] = field(default_factory=list)
    state: str = "COMPLETED"
    ordered_at: datetime = field(default_factory=utcnow)


class FakeExchangeClient(ExchangeClient):
    """Canned market data plus a simulated spot account.

    Failure switches: ``fail_ticker``; ``reject_orders`` (exchange refuses);
    ``timeout_orders`` (the request is lost *after* the exchange accepted it, so
    the order exists but the caller never learns its id); ``fill_ratio`` < 1
    leaves an IOC remainder unfilled.
    """

    name = "fake"

    def __init__(
        self,
        *,
        ticker: Ticker | None = None,
        fail_ticker: bool = False,
        health_state: ExchangeHealthState = ExchangeHealthState.NORMAL,
        board_state: str = "RUNNING",
        jpy: str = "0",
        btc: str = "0",
        commission_rate: str = "0.0015",
    ) -> None:
        self.ticker = ticker or make_ticker(timestamp=utcnow())
        self.fail_ticker = fail_ticker
        self.health_state = health_state
        self.board_state = board_state
        self.jpy = Decimal(jpy)
        self.btc = Decimal(btc)
        self.commission_rate = Decimal(commission_rate)
        self.reject_orders = False
        self.timeout_orders = False
        self.fill_ratio = Decimal("1")
        self.orders: list[_SimOrder] = []
        self.calls: list[str] = []
        self.closed = False

    # -- public ------------------------------------------------------------
    async def get_ticker(self, product_code: str) -> Ticker:
        self.calls.append("get_ticker")
        if self.fail_ticker:
            raise ExchangeError("ticker unavailable")
        return self.ticker

    async def get_health(self, product_code: str) -> ExchangeHealth:
        self.calls.append("get_health")
        return ExchangeHealth(
            exchange=self.name,
            product_code=product_code,
            state=self.health_state,
            board_state=self.board_state,
            checked_at=utcnow(),
        )

    # -- private -----------------------------------------------------------
    async def get_balances(self) -> list[Balance]:
        self.calls.append("get_balances")
        return [
            Balance(currency_code="JPY", amount=self.jpy, available=self.jpy),
            Balance(currency_code="BTC", amount=self.btc, available=self.btc),
        ]

    async def get_trading_commission(self, product_code: str) -> Decimal:
        self.calls.append("get_trading_commission")
        return self.commission_rate

    async def place_order(self, order: OrderRequest) -> str:
        self.calls.append("place_order")
        if self.reject_orders:
            raise ExchangeResponseError('HTTP 400: {"status":-200,"error_message":"rejected"}')
        acceptance_id = f"JRF-{len(self.orders) + 1:04d}"
        sim = _SimOrder(acceptance_id, order)
        bid, ask = self.ticker.best_bid, self.ticker.best_ask
        assert bid is not None and ask is not None
        price = ask if order.side is Side.BUY else bid
        marketable = price <= order.price if order.side is Side.BUY else price >= order.price
        size = (order.size * self.fill_ratio).quantize(Decimal("1e-8")) if marketable else 0
        if size > 0:
            commission = (size * self.commission_rate).quantize(Decimal("1e-8"))
            if order.side is Side.BUY:
                if size * price > self.jpy:
                    raise ExchangeResponseError("insufficient JPY")
                self.jpy -= size * price
                self.btc += size - commission
            else:
                if size + commission > self.btc:
                    raise ExchangeResponseError("insufficient BTC")
                self.btc -= size + commission
                self.jpy += size * price
            sim.executions.append(
                OwnExecution(
                    execution_id=f"{acceptance_id}-1",
                    acceptance_id=acceptance_id,
                    side=order.side,
                    price=price,
                    size=size,
                    commission=commission,
                    executed_at=utcnow(),
                )
            )
        sim.state = "COMPLETED" if size == order.size else "CANCELED"
        self.orders.append(sim)
        if self.timeout_orders:
            raise ExchangeTransientError("ReadTimeout: response lost")
        return acceptance_id

    async def get_order(self, product_code: str, acceptance_id: str) -> ExchangeOrder | None:
        self.calls.append("get_order")
        sim = next((sim for sim in self.orders if sim.acceptance_id == acceptance_id), None)
        # Like bitFlyer: an order invalidated without any execution is not listed.
        return self._report(sim) if sim is not None and sim.executions else None

    async def get_recent_orders(self, product_code: str, count: int = 20) -> list[ExchangeOrder]:
        self.calls.append("get_recent_orders")
        # Like bitFlyer: orders that never executed are not listed.
        listed = [sim for sim in self.orders if sim.executions]
        return [self._report(sim) for sim in reversed(listed[-count:])]

    async def get_executions(self, product_code: str, acceptance_id: str) -> list[OwnExecution]:
        self.calls.append("get_executions")
        sim = next((sim for sim in self.orders if sim.acceptance_id == acceptance_id), None)
        return list(sim.executions) if sim else []

    async def aclose(self) -> None:
        self.closed = True

    def _report(self, sim: _SimOrder) -> ExchangeOrder:
        executed = sum((item.size for item in sim.executions), Decimal("0"))
        return ExchangeOrder(
            acceptance_id=sim.acceptance_id,
            side=sim.request.side,
            size=sim.request.size,
            state=sim.state,
            executed_size=executed,
            average_price=sim.executions[0].price if sim.executions else None,
            total_commission=sum((item.commission for item in sim.executions), Decimal("0")),
            ordered_at=sim.ordered_at,
        )
