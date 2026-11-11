"""bitFlyer adapter: parsing, request signing and order submission (§9, §30, §46).

Responses are served through ``respx`` — no socket is opened. Load-bearing rules:

* prices and sizes become ``Decimal`` with the exact digits bitFlyer sent (§13);
* only fields bitFlyer actually returns are populated (§9);
* the HMAC signature covers exactly the path, query and body bytes sent;
* an order submission is sent once and never retried (§30).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
import respx
from pydantic import SecretStr

from btc_trader.config import Settings
from btc_trader.enums import ExchangeHealthState, Side
from btc_trader.errors import (
    ExchangeAuthenticationError,
    ExchangeResponseError,
    ExchangeTransientError,
)
from btc_trader.exchange.bitflyer import BitflyerClient
from btc_trader.exchange.models import OrderRequest
from tests.fixtures.bitflyer_payloads import (
    BALANCE_JSON,
    BOARD_STATE_BUSY_JSON,
    BOARD_STATE_JSON,
    BOARD_STATE_NO_ORDER_JSON,
    CHILD_ORDERS_JSON,
    COMMISSION_JSON,
    EXPECTED_TICKER,
    MY_EXECUTIONS_JSON,
    SEND_ORDER_JSON,
    TICKER_JSON,
)

BASE_URL = "https://api.bitflyer.com"

pytestmark = pytest.mark.usefixtures("env")


KEY, SECRET = "test-key", "test-secret-value"


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[BitflyerClient]:
    """A client whose credentials are empty — a signed call cannot succeed."""
    client = BitflyerClient(settings)
    yield client
    await client.aclose()


@pytest.fixture
async def signed(settings: Settings) -> AsyncIterator[BitflyerClient]:
    """A client with throwaway credentials; retries enabled to prove POSTs skip them."""
    client = BitflyerClient(
        settings.model_copy(
            update={
                "bitflyer_api_key": SecretStr(KEY),
                "bitflyer_api_secret": SecretStr(SECRET),
                "bitflyer_max_retries": 3,
            }
        )
    )
    yield client
    await client.aclose()


def _assert_signed(request: httpx.Request) -> None:
    """The signature must be recomputable from the bytes actually on the wire."""
    timestamp = request.headers["ACCESS-TIMESTAMP"]
    path = request.url.raw_path.decode()
    message = f"{timestamp}{request.method}{path}{request.content.decode()}"
    expected = hmac.new(SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    assert request.headers["ACCESS-KEY"] == KEY
    assert request.headers["ACCESS-SIGN"] == expected


def _json_response(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "application/json"})


# --------------------------------------------------------------------------- #
# ticker
# --------------------------------------------------------------------------- #
@respx.mock(base_url=BASE_URL, assert_all_called=True)
async def test_ticker_is_parsed_into_decimals(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    route = respx_mock.get("/v1/ticker").mock(return_value=_json_response(TICKER_JSON))

    ticker = await client.get_ticker("BTC_JPY")

    assert route.call_count == 1
    assert dict(route.calls[0].request.url.params) == {"product_code": "BTC_JPY"}

    assert ticker.exchange == "bitflyer"
    assert ticker.product_code == "BTC_JPY"
    assert ticker.state == EXPECTED_TICKER["state"]
    assert ticker.tick_id == EXPECTED_TICKER["tick_id"]
    for field, expected in (
        ("ltp", EXPECTED_TICKER["ltp"]),
        ("best_bid", EXPECTED_TICKER["best_bid"]),
        ("best_ask", EXPECTED_TICKER["best_ask"]),
        ("volume", EXPECTED_TICKER["volume"]),
    ):
        value = getattr(ticker, field)
        assert isinstance(value, Decimal)
        # str() of the Decimal keeps bitFlyer's own digits — no float rounding.
        assert str(value) == expected


@respx.mock(base_url=BASE_URL)
async def test_ticker_timestamp_is_utc_aware(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    """bitFlyer sends UTC without an offset; we must attach it, not localise (§38)."""
    respx_mock.get("/v1/ticker").mock(return_value=_json_response(TICKER_JSON))

    ticker = await client.get_ticker("BTC_JPY")

    assert ticker.exchange_timestamp == datetime(2026, 9, 1, 17, 47, 23, 553000, tzinfo=UTC)
    assert ticker.received_at.tzinfo is not None


@respx.mock(base_url=BASE_URL)
async def test_ticker_keeps_the_raw_payload_for_audit(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    respx_mock.get("/v1/ticker").mock(return_value=_json_response(TICKER_JSON))

    ticker = await client.get_ticker("BTC_JPY")

    assert ticker.raw_payload is not None
    # Serialised for JSONB: Decimals become strings, never floats.
    assert ticker.raw_payload["ltp"] == "12411935.0"
    assert isinstance(ticker.raw_payload["tick_id"], int)


@pytest.mark.parametrize(
    "body",
    [
        '{"state": "RUNNING"}',  # no product_code
        '{"product_code": "BTC_JPY"}',  # no timestamp
        "[]",  # wrong shape entirely
    ],
)
async def test_a_malformed_ticker_is_rejected_not_guessed(
    client: BitflyerClient, body: str
) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/ticker").mock(return_value=_json_response(body))

        with pytest.raises(ExchangeResponseError):
            await client.get_ticker("BTC_JPY")


@respx.mock(base_url=BASE_URL)
async def test_missing_optional_ticker_fields_stay_none(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    """§9: an absent field is ``None``, never a fabricated zero."""
    minimal = '{"product_code": "BTC_JPY", "timestamp": "2026-09-01T17:47:23.553"}'
    respx_mock.get("/v1/ticker").mock(return_value=_json_response(minimal))

    ticker = await client.get_ticker("BTC_JPY")

    assert ticker.ltp is None
    assert ticker.best_bid is None
    assert ticker.volume is None
    assert ticker.tick_id is None
    assert ticker.state is None


# --------------------------------------------------------------------------- #
# board state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("body", "state", "accepts"),
    [
        (BOARD_STATE_JSON, ExchangeHealthState.NORMAL, True),
        # bitFlyer spells it with a space; it must not degrade to UNKNOWN.
        (BOARD_STATE_BUSY_JSON, ExchangeHealthState.VERY_BUSY, True),
        (BOARD_STATE_NO_ORDER_JSON, ExchangeHealthState.NO_ORDER, False),
        ('{"health": "SUPER BUSY", "state": "RUNNING"}', ExchangeHealthState.SUPER_BUSY, False),
        ('{"health": "NORMAL", "state": "CIRCUIT BREAK"}', ExchangeHealthState.NORMAL, False),
        ('{"health": "SOMETHING NEW"}', ExchangeHealthState.UNKNOWN, False),
    ],
)
async def test_board_state_decides_whether_orders_may_be_sent(
    client: BitflyerClient, body: str, state: ExchangeHealthState, accepts: bool
) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/getboardstate").mock(return_value=_json_response(body))

        health = await client.get_health("BTC_JPY")

    assert health.state is state
    assert health.accepts_orders is accepts


# --------------------------------------------------------------------------- #
# private reads
# --------------------------------------------------------------------------- #
@respx.mock(base_url=BASE_URL)
async def test_balances_and_commission_are_signed_and_parsed(
    respx_mock: respx.MockRouter, signed: BitflyerClient
) -> None:
    balance = respx_mock.get("/v1/me/getbalance").mock(return_value=_json_response(BALANCE_JSON))
    commission = respx_mock.get("/v1/me/gettradingcommission").mock(
        return_value=_json_response(COMMISSION_JSON)
    )

    balances = {item.currency_code: item for item in await signed.get_balances()}
    rate = await signed.get_trading_commission("BTC_JPY")

    assert balances["BTC"].amount == Decimal("0.01234567")
    assert balances["JPY"].available == Decimal("508000")
    assert rate == Decimal("0.0015")
    _assert_signed(balance.calls[0].request)
    _assert_signed(commission.calls[0].request)
    assert commission.calls[0].request.url.raw_path == (
        b"/v1/me/gettradingcommission?product_code=BTC_JPY"
    )


@respx.mock(base_url=BASE_URL)
async def test_orders_and_fills_are_parsed_with_btc_commission(
    respx_mock: respx.MockRouter, signed: BitflyerClient
) -> None:
    acceptance = "JRF20260901-050237-639234"
    respx_mock.get("/v1/me/getchildorders").mock(return_value=_json_response(CHILD_ORDERS_JSON))
    fills = respx_mock.get("/v1/me/getexecutions").mock(
        return_value=_json_response(MY_EXECUTIONS_JSON)
    )

    order = await signed.get_order("BTC_JPY", acceptance)
    executions = await signed.get_executions("BTC_JPY", acceptance)

    assert order is not None
    assert order.state == "COMPLETED"
    assert order.is_final
    assert order.executed_size == Decimal("0.01")
    assert order.ordered_at == datetime(2026, 9, 1, 8, 45, 53, tzinfo=UTC)
    assert executions[0].commission == Decimal("0.000015")
    assert executions[0].side is Side.BUY
    _assert_signed(fills.calls[0].request)


@respx.mock(base_url=BASE_URL)
async def test_an_unlisted_order_is_none(
    respx_mock: respx.MockRouter, signed: BitflyerClient
) -> None:
    respx_mock.get("/v1/me/getchildorders").mock(return_value=_json_response("[]"))

    assert await signed.get_order("BTC_JPY", "JRF-unknown") is None


# --------------------------------------------------------------------------- #
# order submission
# --------------------------------------------------------------------------- #
def _order() -> OrderRequest:
    return OrderRequest(
        product_code="BTC_JPY", side=Side.BUY, size=Decimal("0.00123456"), price=Decimal("12450000")
    )


@respx.mock(base_url=BASE_URL)
async def test_an_order_is_a_signed_limit_ioc_with_exact_numbers(
    respx_mock: respx.MockRouter, signed: BitflyerClient
) -> None:
    route = respx_mock.post("/v1/me/sendchildorder").mock(
        return_value=_json_response(SEND_ORDER_JSON)
    )

    acceptance = await signed.place_order(_order())

    assert acceptance == "JRF20260901-050237-639234"
    request = route.calls[0].request
    _assert_signed(request)
    assert request.content == (
        b'{"product_code":"BTC_JPY","child_order_type":"LIMIT","side":"BUY",'
        b'"price":12450000,"size":0.00123456,"minute_to_expire":1,"time_in_force":"IOC"}'
    )
    assert json.loads(request.content)["size"] == 0.00123456


@respx.mock(base_url=BASE_URL)
async def test_a_timed_out_order_is_never_resent(
    respx_mock: respx.MockRouter, signed: BitflyerClient
) -> None:
    route = respx_mock.post("/v1/me/sendchildorder").mock(
        side_effect=httpx.ReadTimeout("no response")
    )

    with pytest.raises(ExchangeTransientError):
        await signed.place_order(_order())

    assert route.call_count == 1


@respx.mock(base_url=BASE_URL)
async def test_an_exchange_rejection_is_an_error_not_an_acceptance(
    respx_mock: respx.MockRouter, signed: BitflyerClient
) -> None:
    respx_mock.post("/v1/me/sendchildorder").mock(
        return_value=httpx.Response(
            400, json={"status": -205, "error_message": "Margin amount is insufficient"}
        )
    )

    with pytest.raises(ExchangeResponseError, match="insufficient"):
        await signed.place_order(_order())


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
async def test_a_private_call_without_credentials_never_reaches_the_network(
    client: BitflyerClient,
) -> None:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        route = router.route()

        with pytest.raises(ExchangeAuthenticationError):
            await client.get_balances()
        with pytest.raises(ExchangeAuthenticationError):
            await client.place_order(_order())

        assert route.call_count == 0


async def test_auth_headers_never_expose_the_secret(signed: BitflyerClient) -> None:
    headers = signed.auth_headers("GET", "/v1/me/getbalance", "")

    assert SECRET not in "".join(headers.values())
    assert headers["ACCESS-KEY"] == KEY
