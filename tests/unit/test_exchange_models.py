"""Exchange value objects (PROJECT.md §13, §49)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from btc_trader.enums import ExchangeHealthState, Side
from btc_trader.exchange.models import ExchangeHealth, ExchangeOrder, OrderRequest, Ticker

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


def test_ticker_is_immutable() -> None:
    ticker = _ticker(ltp="1")
    with pytest.raises(ValidationError):
        ticker.ltp = Decimal("2")  # type: ignore[misc]


def test_unexpected_fields_are_rejected() -> None:
    """A new bitFlyer field must surface as an error, not be silently dropped."""
    with pytest.raises(ValidationError):
        _ticker(surprise_field="?")


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("size", "price"), [("0", "100"), ("-0.01", "100"), ("0.01", "0")])
def test_non_positive_order_size_or_price_is_rejected(size: str, price: str) -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        OrderRequest(
            product_code="BTC_JPY", side=Side.BUY, size=Decimal(size), price=Decimal(price)
        )


def test_orders_default_to_immediate_or_cancel_with_a_short_expiry() -> None:
    """A bot order must never rest in the book unattended."""
    order = OrderRequest(
        product_code="BTC_JPY", side=Side.SELL, size=Decimal("0.01"), price=Decimal("100")
    )
    assert order.time_in_force == "IOC"
    assert order.minute_to_expire == 1


def test_only_active_exchange_orders_are_non_final() -> None:
    def order(state: str) -> ExchangeOrder:
        return ExchangeOrder(
            acceptance_id="JRF", side=Side.BUY, size="0.01", state=state, ordered_at=NOW
        )

    assert not order("ACTIVE").is_final
    assert all(order(state).is_final for state in ("COMPLETED", "CANCELED", "EXPIRED"))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected", "accepts_orders"),
    [
        ("NORMAL", ExchangeHealthState.NORMAL, True),
        ("BUSY", ExchangeHealthState.BUSY, True),
        ("VERY BUSY", ExchangeHealthState.VERY_BUSY, True),
        ("SUPER BUSY", ExchangeHealthState.SUPER_BUSY, False),
        ("NO ORDER", ExchangeHealthState.NO_ORDER, False),
        ("STOP", ExchangeHealthState.STOP, False),
        ("SOMETHING_NEW", ExchangeHealthState.UNKNOWN, False),
        (None, ExchangeHealthState.UNKNOWN, False),
    ],
)
def test_health_state_normalisation_fails_closed(
    raw: str | None, expected: ExchangeHealthState, accepts_orders: bool
) -> None:
    health = ExchangeHealth(exchange="bitflyer", product_code="BTC_JPY", state=raw, checked_at=NOW)
    assert health.state is expected
    assert health.accepts_orders is accepts_orders
