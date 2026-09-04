"""bitFlyer adapter.

Only this module knows bitFlyer's URLs, field names and timestamp format.
Responses are normalised into the value objects in :mod:`btc_trader.exchange.models`
using exactly the fields bitFlyer actually returns — nothing is fabricated
(PROJECT.md §9).

Order *submission* is deliberately not implemented: live trading belongs to a
later phase, and until then ``place_order`` raises rather than offering a code
path that could reach the exchange with real money (PROJECT.md §52, §53.12).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from decimal import Decimal
from typing import Any

import httpx

from btc_trader.config import Settings, get_settings
from btc_trader.enums import ExchangeHealthState
from btc_trader.errors import (
    ExchangeAuthenticationError,
    ExchangeResponseError,
    TradingDisabledError,
)
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.http import RateLimitedHttpClient
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
from btc_trader.monitoring.logging import get_logger
from btc_trader.utils.serialization import jsonable
from btc_trader.utils.timeutils import parse_bitflyer_timestamp, utcnow

logger = get_logger(__name__)

EXCHANGE_NAME = "bitflyer"

_PATH_TICKER = "/v1/ticker"
_PATH_BOARD = "/v1/board"
_PATH_EXECUTIONS = "/v1/executions"
_PATH_HEALTH = "/v1/gethealth"
_PATH_MARKETS = "/v1/markets"
_PATH_BALANCE = "/v1/me/getbalance"
_PATH_CHILD_ORDERS = "/v1/me/getchildorders"

#: bitFlyer caps ``count`` on the public executions endpoint.
MAX_EXECUTIONS_PER_REQUEST = 500

#: bitFlyer child-order state -> the wire value we report upward.
_BITFLYER_ORDER_STATES = {
    "ACTIVE": "SUBMITTED",
    "COMPLETED": "FILLED",
    "CANCELED": "CANCELLED",
    "EXPIRED": "CANCELLED",
    "REJECTED": "REJECTED",
}


class BitflyerClient(ExchangeClient):
    """bitFlyer implementation of :class:`ExchangeClient`."""

    name = EXCHANGE_NAME

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: RateLimitedHttpClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if http_client is not None:
            self._http = http_client
        else:
            underlying = (
                httpx.AsyncClient(
                    base_url=self._settings.bitflyer_public_base_url,
                    transport=transport,
                    timeout=httpx.Timeout(self._settings.bitflyer_timeout_seconds),
                )
                if transport is not None
                else None
            )
            self._http = RateLimitedHttpClient(
                self._settings.bitflyer_public_base_url,
                timeout_seconds=self._settings.bitflyer_timeout_seconds,
                max_retries=self._settings.bitflyer_max_retries,
                rate_limit_requests=self._settings.bitflyer_rate_limit_requests,
                rate_limit_window_seconds=self._settings.bitflyer_rate_limit_window_seconds,
                client=underlying,
            )

    # ------------------------------------------------------------------ #
    # public market data
    # ------------------------------------------------------------------ #
    async def get_ticker(self, product_code: str) -> Ticker:
        payload = await self._http.request_json(
            "GET", _PATH_TICKER, params={"product_code": product_code}
        )
        return self._parse_ticker(payload)

    async def get_order_book(self, product_code: str) -> OrderBook:
        payload = await self._http.request_json(
            "GET", _PATH_BOARD, params={"product_code": product_code}
        )
        return self._parse_order_book(product_code, payload)

    async def get_recent_executions(
        self,
        product_code: str,
        limit: int = 100,
        *,
        before_id: int | None = None,
        after_id: int | None = None,
    ) -> list[Execution]:
        params: dict[str, Any] = {
            "product_code": product_code,
            "count": min(max(limit, 1), MAX_EXECUTIONS_PER_REQUEST),
        }
        if before_id is not None:
            params["before"] = before_id
        if after_id is not None:
            params["after"] = after_id
        payload = await self._http.request_json("GET", _PATH_EXECUTIONS, params=params)
        return self._parse_executions(product_code, payload)

    async def get_health(self, product_code: str) -> ExchangeHealth:
        payload = await self._http.request_json(
            "GET", _PATH_HEALTH, params={"product_code": product_code}
        )
        if not isinstance(payload, dict):
            raise ExchangeResponseError(f"unexpected gethealth payload: {type(payload)!r}")
        return ExchangeHealth(
            exchange=self.name,
            product_code=product_code,
            state=payload.get("status", ExchangeHealthState.UNKNOWN),
            checked_at=utcnow(),
        )

    async def get_markets(self) -> list[dict[str, Any]]:
        """Available products; used to validate PRODUCT_CODE at startup."""
        payload = await self._http.request_json("GET", _PATH_MARKETS)
        if not isinstance(payload, list):
            raise ExchangeResponseError(f"unexpected markets payload: {type(payload)!r}")
        return [item for item in payload if isinstance(item, dict)]

    # ------------------------------------------------------------------ #
    # private (read-only)
    # ------------------------------------------------------------------ #
    async def get_balances(self) -> list[Balance]:
        payload = await self._signed_request("GET", _PATH_BALANCE)
        if not isinstance(payload, list):
            raise ExchangeResponseError(f"unexpected getbalance payload: {type(payload)!r}")
        return [
            Balance(
                currency_code=str(item["currency_code"]),
                amount=item.get("amount", 0),
                available=item.get("available", 0),
            )
            for item in payload
            if isinstance(item, dict) and "currency_code" in item
        ]

    async def get_order_status(self, product_code: str, order_id: str) -> OrderStatusReport:
        payload = await self._signed_request(
            "GET",
            _PATH_CHILD_ORDERS,
            params={"product_code": product_code, "child_order_acceptance_id": order_id},
        )
        if not isinstance(payload, list) or not payload:
            raise ExchangeResponseError(f"no child order found for {order_id}")
        order = payload[0]
        raw_state = str(order.get("child_order_state", "")).upper()
        return OrderStatusReport(
            exchange_order_id=str(order.get("child_order_acceptance_id") or order_id),
            client_order_id=None,
            product_code=str(order.get("product_code", product_code)),
            side=order.get("side") or None,
            order_type=str(order.get("child_order_type", "")).upper() or None,
            status=_BITFLYER_ORDER_STATES.get(raw_state, "UNKNOWN"),
            requested_size=order.get("size"),
            filled_size=order.get("executed_size", 0),
            average_price=order.get("average_price") or None,
            price=order.get("price") or None,
            commission=order.get("total_commission", 0),
            raw_response=jsonable(order),
        )

    # ------------------------------------------------------------------ #
    # private (trading) — intentionally unavailable until a later phase
    # ------------------------------------------------------------------ #
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        logger.error(
            "live_order_submission_blocked",
            client_order_id=order.client_order_id,
            product_code=order.product_code,
            side=order.side.value,
            reason="not_implemented",
        )
        raise TradingDisabledError(
            "bitFlyer order submission is not implemented. Live execution is a later "
            "phase (PROJECT.md §50 Phase 11); use the paper execution adapter."
        )

    async def cancel_order(self, product_code: str, order_id: str) -> None:
        raise TradingDisabledError(
            "bitFlyer order cancellation is not implemented (PROJECT.md §50 Phase 11)."
        )

    # ------------------------------------------------------------------ #
    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def rate_limit_used(self) -> int:
        return self._http.rate_limit_used

    # ------------------------------------------------------------------ #
    # authentication
    # ------------------------------------------------------------------ #
    def _auth_headers(self, method: str, path_with_query: str, body: str) -> dict[str, str]:
        """Build bitFlyer HMAC-SHA256 auth headers.

        Signature = HMAC(secret, timestamp + METHOD + path + body).  Neither the
        key, the secret nor the signature is ever logged (PROJECT.md §35).
        """
        key = self._settings.bitflyer_api_key.get_secret_value()
        secret = self._settings.bitflyer_api_secret.get_secret_value()
        if not key or not secret:
            raise ExchangeAuthenticationError(
                "bitFlyer API credentials are not configured; private endpoints unavailable"
            )
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path_with_query}{body}"
        signature = hmac.new(
            secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return {
            "ACCESS-KEY": key,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-SIGN": signature,
            "Content-Type": "application/json",
        }

    async def _signed_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        idempotent: bool = True,
    ) -> Any:
        query = ""
        if params:
            query = "?" + "&".join(f"{key}={value}" for key, value in params.items())
        body = json.dumps(json_body, separators=(",", ":")) if json_body else ""
        headers = self._auth_headers(method, f"{path}{query}", body)
        return await self._http.request_json(
            method,
            path,
            params=params,
            json_body=json_body,
            headers=headers,
            idempotent=idempotent,
        )

    # ------------------------------------------------------------------ #
    # normalisation
    # ------------------------------------------------------------------ #
    def _parse_ticker(self, payload: Any) -> Ticker:
        if not isinstance(payload, dict) or "product_code" not in payload:
            raise ExchangeResponseError(f"unexpected ticker payload: {payload!r:.200}")
        timestamp_raw = payload.get("timestamp")
        if not isinstance(timestamp_raw, str):
            raise ExchangeResponseError("ticker payload is missing 'timestamp'")
        tick_id = payload.get("tick_id")
        return Ticker(
            exchange=self.name,
            product_code=str(payload["product_code"]),
            exchange_timestamp=parse_bitflyer_timestamp(timestamp_raw),
            received_at=utcnow(),
            tick_id=(
                int(tick_id) if isinstance(tick_id, int | Decimal | str) and tick_id != "" else None
            ),
            state=payload.get("state"),
            ltp=payload.get("ltp"),
            best_bid=payload.get("best_bid"),
            best_ask=payload.get("best_ask"),
            best_bid_size=payload.get("best_bid_size"),
            best_ask_size=payload.get("best_ask_size"),
            total_bid_depth=payload.get("total_bid_depth"),
            total_ask_depth=payload.get("total_ask_depth"),
            market_bid_size=payload.get("market_bid_size"),
            market_ask_size=payload.get("market_ask_size"),
            volume=payload.get("volume"),
            volume_by_product=payload.get("volume_by_product"),
            raw_payload=jsonable(payload),
        )

    def _parse_order_book(self, product_code: str, payload: Any) -> OrderBook:
        if not isinstance(payload, dict) or "bids" not in payload or "asks" not in payload:
            raise ExchangeResponseError(f"unexpected board payload: {payload!r:.200}")
        return OrderBook(
            exchange=self.name,
            product_code=product_code,
            received_at=utcnow(),
            mid_price=payload.get("mid_price"),
            bids=tuple(_parse_levels(payload["bids"], descending=True)),
            asks=tuple(_parse_levels(payload["asks"], descending=False)),
        )

    def _parse_executions(self, product_code: str, payload: Any) -> list[Execution]:
        if not isinstance(payload, list):
            raise ExchangeResponseError(f"unexpected executions payload: {type(payload)!r}")
        received_at = utcnow()
        executions: list[Execution] = []
        for item in payload:
            if not isinstance(item, dict) or "id" not in item:
                logger.warning("execution_payload_skipped", reason="missing_id")
                continue
            exec_date = item.get("exec_date")
            if not isinstance(exec_date, str):
                logger.warning("execution_payload_skipped", reason="missing_exec_date")
                continue
            executions.append(
                Execution(
                    exchange=self.name,
                    product_code=product_code,
                    exchange_execution_id=str(int(item["id"])),
                    side=item.get("side"),
                    price=item["price"],
                    size=item["size"],
                    executed_at=parse_bitflyer_timestamp(exec_date),
                    received_at=received_at,
                )
            )
        return executions


def _parse_levels(raw_levels: Any, *, descending: bool) -> list[OrderBookLevel]:
    if not isinstance(raw_levels, list):
        raise ExchangeResponseError("board side is not a list")
    levels = [
        OrderBookLevel(price=item["price"], size=item["size"])
        for item in raw_levels
        if isinstance(item, dict) and "price" in item and "size" in item
    ]
    levels.sort(key=lambda level: level.price, reverse=descending)
    return levels
