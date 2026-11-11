"""bitFlyer adapter (https://lightning.bitflyer.com/docs).

Only this module knows bitFlyer's URLs, field names and timestamp format.
Responses are normalised into :mod:`btc_trader.exchange.models` using the
fields bitFlyer actually returns (PROJECT.md §9).

Authentication: ``ACCESS-SIGN = HMAC-SHA256(secret, timestamp + METHOD +
path_with_query + body)``. The signed path and body are the exact bytes sent:
the query string is encoded once here and the JSON body is pre-serialised, so
no HTTP library can re-encode them differently. Credentials and signatures are
never logged (PROJECT.md §35).

Order submission (``POST /v1/me/sendchildorder``) is sent at most once per
call: a timeout leaves the outcome unknown, and the caller reconciles instead
of retrying (PROJECT.md §30, §33).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from btc_trader.config import Settings, get_settings
from btc_trader.errors import ExchangeAuthenticationError, ExchangeResponseError
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.http import RateLimitedHttpClient
from btc_trader.exchange.models import (
    Balance,
    ExchangeHealth,
    ExchangeOrder,
    OrderRequest,
    OwnExecution,
    Ticker,
)
from btc_trader.utils.serialization import jsonable
from btc_trader.utils.timeutils import parse_bitflyer_timestamp, utcnow

EXCHANGE_NAME = "bitflyer"

_PATH_TICKER = "/v1/ticker"
_PATH_BOARD_STATE = "/v1/getboardstate"
_PATH_BALANCE = "/v1/me/getbalance"
_PATH_COMMISSION = "/v1/me/gettradingcommission"
_PATH_SEND_ORDER = "/v1/me/sendchildorder"
_PATH_CHILD_ORDERS = "/v1/me/getchildorders"
_PATH_MY_EXECUTIONS = "/v1/me/getexecutions"
_PATH_PERMISSIONS = "/v1/me/getpermissions"


class BitflyerClient(ExchangeClient):
    """bitFlyer implementation of :class:`ExchangeClient`."""

    name = EXCHANGE_NAME

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
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
    # public
    # ------------------------------------------------------------------ #
    async def get_ticker(self, product_code: str) -> Ticker:
        payload = await self._http.request_json(
            "GET", _query(_PATH_TICKER, {"product_code": product_code})
        )
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
            tick_id=int(tick_id)
            if isinstance(tick_id, int | Decimal | str) and tick_id != ""
            else None,
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

    async def get_health(self, product_code: str) -> ExchangeHealth:
        payload = await self._http.request_json(
            "GET", _query(_PATH_BOARD_STATE, {"product_code": product_code})
        )
        if not isinstance(payload, dict):
            raise ExchangeResponseError(f"unexpected getboardstate payload: {type(payload)!r}")
        return ExchangeHealth(
            exchange=self.name,
            product_code=product_code,
            state=payload.get("health"),
            board_state=payload.get("state"),
            checked_at=utcnow(),
        )

    # ------------------------------------------------------------------ #
    # private
    # ------------------------------------------------------------------ #
    async def get_permissions(self) -> list[str]:
        payload = await self._signed("GET", _PATH_PERMISSIONS)
        return [str(item) for item in payload] if isinstance(payload, list) else []

    async def get_balances(self) -> list[Balance]:
        payload = await self._signed("GET", _PATH_BALANCE)
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

    async def get_trading_commission(self, product_code: str) -> Decimal:
        payload = await self._signed("GET", _PATH_COMMISSION, {"product_code": product_code})
        try:
            return Decimal(str(payload["commission_rate"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ExchangeResponseError("unexpected gettradingcommission payload") from exc

    async def place_order(self, order: OrderRequest) -> str:
        body = {
            "product_code": order.product_code,
            "child_order_type": "LIMIT",
            "side": order.side.value,
            # JSON numbers: integer yen, and the size as its exact decimal digits.
            "price": int(order.price),
            "size": _json_number(order.size),
            "minute_to_expire": order.minute_to_expire,
            "time_in_force": order.time_in_force,
        }
        payload = await self._signed("POST", _PATH_SEND_ORDER, body=body, idempotent=False)
        acceptance_id = (
            payload.get("child_order_acceptance_id") if isinstance(payload, dict) else None
        )
        if not acceptance_id:
            raise ExchangeResponseError(f"order not accepted: {payload!r:.300}")
        return str(acceptance_id)

    async def get_order(self, product_code: str, acceptance_id: str) -> ExchangeOrder | None:
        payload = await self._signed(
            "GET",
            _PATH_CHILD_ORDERS,
            {"product_code": product_code, "child_order_acceptance_id": acceptance_id},
        )
        orders = _parse_orders(payload)
        return orders[0] if orders else None

    async def get_recent_orders(self, product_code: str, count: int = 20) -> list[ExchangeOrder]:
        payload = await self._signed(
            "GET", _PATH_CHILD_ORDERS, {"product_code": product_code, "count": count}
        )
        return _parse_orders(payload)

    async def get_executions(self, product_code: str, acceptance_id: str) -> list[OwnExecution]:
        payload = await self._signed(
            "GET",
            _PATH_MY_EXECUTIONS,
            {"product_code": product_code, "child_order_acceptance_id": acceptance_id},
        )
        if not isinstance(payload, list):
            raise ExchangeResponseError(f"unexpected getexecutions payload: {type(payload)!r}")
        return [
            OwnExecution(
                execution_id=str(item["id"]),
                acceptance_id=str(item.get("child_order_acceptance_id") or acceptance_id),
                side=item["side"],
                price=item["price"],
                size=item["size"],
                commission=item.get("commission") or 0,
                executed_at=parse_bitflyer_timestamp(str(item["exec_date"])),
            )
            for item in payload
            if isinstance(item, dict) and "id" in item
        ]

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ #
    # authentication
    # ------------------------------------------------------------------ #
    def auth_headers(self, method: str, path_with_query: str, body: str) -> dict[str, str]:
        key = self._settings.bitflyer_api_key.get_secret_value()
        secret = self._settings.bitflyer_api_secret.get_secret_value()
        if not key or not secret:
            raise ExchangeAuthenticationError(
                "bitFlyer API credentials are not configured; private endpoints unavailable"
            )
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path_with_query}{body}"
        signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        return {
            "ACCESS-KEY": key,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-SIGN": signature,
            "Content-Type": "application/json",
        }

    async def _signed(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        body: dict[str, Any] | None = None,
        idempotent: bool = True,
    ) -> Any:
        path_with_query = _query(path, params)
        content = json.dumps(body, separators=(",", ":")) if body is not None else ""
        return await self._http.request_json(
            method,
            path_with_query,
            content=content or None,
            headers=self.auth_headers(method, path_with_query, content),
            idempotent=idempotent,
        )


def _query(path: str, params: dict[str, Any] | None) -> str:
    return f"{path}?{urlencode(params)}" if params else path


def _json_number(value: Decimal) -> float:
    """Order sizes as JSON numbers.

    ``repr(float)`` is the shortest string that round-trips, so a size with at
    most 8 decimals (bitFlyer's step) serialises with exactly its own digits.
    """
    number = float(value)
    if Decimal(repr(number)) != value:
        raise ValueError(f"order size {value} cannot be sent exactly")
    return number


def _parse_orders(payload: Any) -> list[ExchangeOrder]:
    if not isinstance(payload, list):
        raise ExchangeResponseError(f"unexpected getchildorders payload: {type(payload)!r}")
    return [
        ExchangeOrder(
            acceptance_id=str(item["child_order_acceptance_id"]),
            side=item["side"],
            size=item["size"],
            state=str(item.get("child_order_state", "")).upper(),
            executed_size=item.get("executed_size") or 0,
            average_price=item.get("average_price") or None,
            total_commission=item.get("total_commission") or 0,
            ordered_at=parse_bitflyer_timestamp(str(item["child_order_date"])),
            raw=jsonable(item),
        )
        for item in payload
        if isinstance(item, dict) and "child_order_acceptance_id" in item
    ]
