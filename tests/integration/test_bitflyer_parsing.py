"""bitFlyer response normalisation against recorded payloads (PROJECT.md §9, §46).

Every response here is a real captured body served through ``respx`` — no socket
is opened.  Two rules are load-bearing:

* prices and sizes arrive as JSON numbers and must become ``Decimal`` with the
  exact digits bitFlyer sent (§13);
* only fields bitFlyer actually returns are populated — nothing is invented (§9).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
import respx

from btc_trader.config import Settings
from btc_trader.enums import ExchangeHealthState, Side
from btc_trader.errors import ExchangeAuthenticationError, ExchangeResponseError
from btc_trader.exchange.bitflyer import MAX_EXECUTIONS_PER_REQUEST, BitflyerClient
from tests.fixtures.bitflyer_payloads import (
    BOARD_JSON,
    EXECUTIONS_JSON,
    EXPECTED_EXECUTION_IDS,
    EXPECTED_TICKER,
    HEALTH_JSON,
    HEALTH_STOP_JSON,
    TICKER_JSON,
)

BASE_URL = "https://api.bitflyer.com"

pytestmark = pytest.mark.usefixtures("env")


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[BitflyerClient]:
    """A client whose credentials are empty — a signed call cannot succeed."""
    client = BitflyerClient(settings)
    yield client
    await client.aclose()


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
# board
# --------------------------------------------------------------------------- #
@respx.mock(base_url=BASE_URL)
async def test_board_sides_are_sorted_best_price_first(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    """The recorded payload is deliberately unsorted, as bitFlyer's can be."""
    respx_mock.get("/v1/board").mock(return_value=_json_response(BOARD_JSON))

    book = await client.get_order_book("BTC_JPY")

    bid_prices = [level.price for level in book.bids]
    ask_prices = [level.price for level in book.asks]
    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ask_prices == sorted(ask_prices)
    assert bid_prices[0] == Decimal("12412067.0")
    assert ask_prices[0] == Decimal("12420247.0")
    assert book.mid_price == Decimal("12416157.0")
    assert all(isinstance(level.size, Decimal) for level in book.bids)


@respx.mock(base_url=BASE_URL)
async def test_board_best_prices_come_from_the_sorted_sides(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    respx_mock.get("/v1/board").mock(return_value=_json_response(BOARD_JSON))

    book = await client.get_order_book("BTC_JPY")

    assert book.best_bid == Decimal("12412067.0")
    assert book.best_ask == Decimal("12420247.0")
    assert book.best_ask - book.best_bid == Decimal("8180.0")


@respx.mock(base_url=BASE_URL)
async def test_board_truncation_keeps_the_top_of_book(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    respx_mock.get("/v1/board").mock(return_value=_json_response(BOARD_JSON))

    book = (await client.get_order_book("BTC_JPY")).truncated(2)

    assert [level.price for level in book.bids] == [Decimal("12412067.0"), Decimal("12412066.0")]
    assert [level.price for level in book.asks] == [Decimal("12420247.0"), Decimal("12420249.0")]


@respx.mock(base_url=BASE_URL)
async def test_a_malformed_board_is_rejected(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    respx_mock.get("/v1/board").mock(return_value=_json_response('{"mid_price": 1}'))

    with pytest.raises(ExchangeResponseError):
        await client.get_order_book("BTC_JPY")


# --------------------------------------------------------------------------- #
# executions
# --------------------------------------------------------------------------- #
@respx.mock(base_url=BASE_URL)
async def test_executions_are_parsed_in_order(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    respx_mock.get("/v1/executions").mock(return_value=_json_response(EXECUTIONS_JSON))

    executions = await client.get_recent_executions("BTC_JPY")

    assert [item.exchange_execution_id for item in executions] == EXPECTED_EXECUTION_IDS
    first = executions[0]
    assert first.side is Side.BUY
    assert first.price == Decimal("12411935.0")
    assert first.size == Decimal("0.023552")
    assert first.executed_at == datetime(2026, 9, 1, 17, 47, 4, 587000, tzinfo=UTC)


@respx.mock(base_url=BASE_URL)
async def test_a_blank_side_becomes_none_not_a_guess(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    """bitFlyer sends ``"side": ""`` for some prints; inventing BUY would be a lie (§9)."""
    respx_mock.get("/v1/executions").mock(return_value=_json_response(EXECUTIONS_JSON))

    executions = await client.get_recent_executions("BTC_JPY")

    assert executions[-1].exchange_execution_id == "2650513076"
    assert executions[-1].side is None
    # ...and its price/size are still recorded.
    assert executions[-1].price == Decimal("12411057.0")


@respx.mock(base_url=BASE_URL)
async def test_a_short_exec_date_fraction_still_parses(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    """``17:46:45.98`` is two fractional digits, not three."""
    respx_mock.get("/v1/executions").mock(return_value=_json_response(EXECUTIONS_JSON))

    executions = await client.get_recent_executions("BTC_JPY")

    assert executions[-1].executed_at == datetime(2026, 9, 1, 17, 46, 45, 980000, tzinfo=UTC)


@respx.mock(base_url=BASE_URL)
async def test_unusable_execution_rows_are_skipped_not_fabricated(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    body = """
    [
      {"id": 1, "side": "BUY", "price": 1.0, "size": 0.1, "exec_date": "2026-09-01T00:00:00.000"},
      {"side": "BUY", "price": 2.0, "size": 0.1, "exec_date": "2026-09-01T00:00:01.000"},
      {"id": 3, "side": "BUY", "price": 3.0, "size": 0.1}
    ]
    """
    respx_mock.get("/v1/executions").mock(return_value=_json_response(body))

    executions = await client.get_recent_executions("BTC_JPY")

    assert [item.exchange_execution_id for item in executions] == ["1"]


@respx.mock(base_url=BASE_URL)
async def test_execution_paging_parameters_are_forwarded(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    route = respx_mock.get("/v1/executions").mock(return_value=_json_response("[]"))

    await client.get_recent_executions("BTC_JPY", limit=250, before_id=42, after_id=7)

    params = dict(route.calls[0].request.url.params)
    assert params == {"product_code": "BTC_JPY", "count": "250", "before": "42", "after": "7"}


@pytest.mark.parametrize(("requested", "expected"), [(0, "1"), (-5, "1"), (10_000, "500")])
async def test_the_execution_count_is_clamped_to_the_api_limit(
    client: BitflyerClient, requested: int, expected: str
) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/executions").mock(return_value=_json_response("[]"))

        await client.get_recent_executions("BTC_JPY", limit=requested)

    assert dict(route.calls[0].request.url.params)["count"] == expected
    assert int(expected) <= MAX_EXECUTIONS_PER_REQUEST


@respx.mock(base_url=BASE_URL)
async def test_a_non_list_executions_payload_is_rejected(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    respx_mock.get("/v1/executions").mock(return_value=_json_response('{"error": "nope"}'))

    with pytest.raises(ExchangeResponseError):
        await client.get_recent_executions("BTC_JPY")


# --------------------------------------------------------------------------- #
# health and markets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (HEALTH_JSON, ExchangeHealthState.NORMAL),
        (HEALTH_STOP_JSON, ExchangeHealthState.STOP),
        ('{"status": "SOMETHING_NEW"}', ExchangeHealthState.UNKNOWN),
        ("{}", ExchangeHealthState.UNKNOWN),
    ],
)
async def test_health_states_including_unknown_ones(
    client: BitflyerClient,
    body: str,
    expected: ExchangeHealthState,
) -> None:
    """An unrecognised status must degrade to UNKNOWN, not crash the collector."""
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/gethealth").mock(return_value=_json_response(body))

        health = await client.get_health("BTC_JPY")

    assert health.state is expected
    assert health.exchange == "bitflyer"
    assert health.checked_at.tzinfo is not None


@respx.mock(base_url=BASE_URL)
async def test_markets_returns_only_dict_entries(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    body = '[{"product_code": "BTC_JPY"}, "garbage", {"product_code": "ETH_JPY"}]'
    respx_mock.get("/v1/markets").mock(return_value=_json_response(body))

    markets = await client.get_markets()

    assert [market["product_code"] for market in markets] == ["BTC_JPY", "ETH_JPY"]


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
@respx.mock(base_url=BASE_URL, assert_all_called=False)
async def test_a_private_call_without_credentials_never_reaches_the_network(
    respx_mock: respx.MockRouter, client: BitflyerClient
) -> None:
    """§7/§51.8: without a key the client must fail closed, before any request."""
    route = respx_mock.get("/v1/me/getbalance").mock(return_value=_json_response("[]"))

    with pytest.raises(ExchangeAuthenticationError):
        await client.get_balances()

    assert route.call_count == 0


async def test_auth_headers_never_expose_the_secret() -> None:
    """§35: the signature may be sent, but the secret itself must not appear."""
    settings = Settings(BITFLYER_API_KEY="key-123", BITFLYER_API_SECRET="secret-456")
    client = BitflyerClient(settings)

    headers = client._auth_headers("GET", "/v1/me/getbalance", "")

    assert headers["ACCESS-KEY"] == "key-123"
    assert "secret-456" not in "".join(headers.values())
    assert len(headers["ACCESS-SIGN"]) == 64  # hex sha256
    await client.aclose()
