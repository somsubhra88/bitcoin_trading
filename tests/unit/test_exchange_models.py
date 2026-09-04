"""Exchange value objects (PROJECT.md §13, §49)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from btc_trader.enums import ExchangeHealthState, OrderType, Side
from btc_trader.exchange.models import (
    ExchangeHealth,
    Execution,
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    Ticker,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _ticker(**overrides: object) -> Ticker:
    values: dict[str, object] = {
        "exchange": "bitflyer",
        "product_code": "BTC_JPY",
        "exchange_timestamp": NOW,
        "received_at": NOW,
    }
    values.update(overrides)
    return Ticker(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
def test_floats_are_converted_via_repr_not_binary_expansion() -> None:
    """A float from ``json.loads`` must not smear into 0.00171918999...."""
    ticker = _ticker(ltp=12411935.0, best_bid_size=0.00171919)
    assert ticker.ltp == Decimal("12411935.0")
    assert str(ticker.best_bid_size) == "0.00171919"


@pytest.mark.parametrize("value", ["12411935.0", 12411935, Decimal("12411935.0")])
def test_strings_ints_and_decimals_all_land_on_decimal(value: object) -> None:
    assert _ticker(ltp=value).ltp == Decimal("12411935")


def test_missing_fields_stay_none_rather_than_zero() -> None:
    """§9: an absent field is unknown, not zero."""
    ticker = _ticker()
    assert ticker.ltp is None
    assert ticker.volume is None
    assert ticker.spread is None
    assert ticker.mid_price is None


def test_spread_and_mid_price() -> None:
    ticker = _ticker(best_bid="12412066", best_ask="12420247")
    assert ticker.spread == Decimal("8181")
    assert ticker.mid_price == Decimal("12416156.5")


def test_ticker_is_immutable() -> None:
    ticker = _ticker(ltp="1")
    with pytest.raises(ValidationError):
        ticker.ltp = Decimal("2")  # type: ignore[misc]


def test_unexpected_fields_are_rejected() -> None:
    """A new bitFlyer field must surface as an error, not be silently dropped."""
    with pytest.raises(ValidationError):
        _ticker(surprise_field="?")


# --------------------------------------------------------------------------- #
def test_order_book_truncation_keeps_the_top_of_book() -> None:
    book = OrderBook(
        exchange="bitflyer",
        product_code="BTC_JPY",
        received_at=NOW,
        bids=tuple(OrderBookLevel(price=Decimal(100 - i), size=Decimal("1")) for i in range(5)),
        asks=tuple(OrderBookLevel(price=Decimal(200 + i), size=Decimal("1")) for i in range(5)),
    )
    top = book.truncated(2)
    assert [level.price for level in top.bids] == [Decimal("100"), Decimal("99")]
    assert [level.price for level in top.asks] == [Decimal("200"), Decimal("201")]
    assert top.best_bid == Decimal("100")
    assert top.best_ask == Decimal("200")


def test_empty_order_book_has_no_best_prices() -> None:
    book = OrderBook(exchange="bitflyer", product_code="BTC_JPY", received_at=NOW)
    assert book.best_bid is None
    assert book.best_ask is None


# --------------------------------------------------------------------------- #
def test_blank_execution_side_becomes_none() -> None:
    """bitFlyer sends ``""`` for engine-triggered prints; that is not a side."""
    execution = Execution(
        exchange="bitflyer",
        product_code="BTC_JPY",
        exchange_execution_id="1",
        side="",
        price="100",
        size="1",
        executed_at=NOW,
        received_at=NOW,
    )
    assert execution.side is None


def test_known_execution_side_is_parsed() -> None:
    execution = Execution(
        exchange="bitflyer",
        product_code="BTC_JPY",
        exchange_execution_id="1",
        side="SELL",
        price="100",
        size="1",
        executed_at=NOW,
        received_at=NOW,
    )
    assert execution.side is Side.SELL


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", ["0", "-0.01"])
def test_non_positive_order_size_is_rejected(size: str) -> None:
    with pytest.raises(ValidationError, match="order size must be positive"):
        OrderRequest(
            product_code="BTC_JPY",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size=Decimal(size),
            client_order_id="x",
        )


def test_order_request_requires_a_client_order_id() -> None:
    """§30: the idempotency key is not optional."""
    with pytest.raises(ValidationError):
        OrderRequest(  # type: ignore[call-arg]
            product_code="BTC_JPY",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.01"),
        )


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected", "accepts_orders"),
    [
        ("NORMAL", ExchangeHealthState.NORMAL, True),
        ("BUSY", ExchangeHealthState.BUSY, True),
        ("NO ORDER", ExchangeHealthState.UNKNOWN, False),
        ("NO_ORDER", ExchangeHealthState.NO_ORDER, False),
        ("STOP", ExchangeHealthState.STOP, False),
        ("SOMETHING_NEW", ExchangeHealthState.UNKNOWN, False),
    ],
)
def test_health_state_normalisation_fails_closed(
    raw: str, expected: ExchangeHealthState, accepts_orders: bool
) -> None:
    health = ExchangeHealth(exchange="bitflyer", product_code="BTC_JPY", state=raw, checked_at=NOW)
    assert health.state is expected
    assert health.state.accepts_orders is accepts_orders
