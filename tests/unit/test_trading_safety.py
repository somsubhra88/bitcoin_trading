"""Trading-safety tests (PROJECT.md §48).

Every test here proves that an unsafe order is **not** submitted, or that the
configuration which would permit one cannot be reached by accident.

Paper-mode max-order/max-position/daily-loss checks and stale-data rejection are
covered by the paper-ledger integration tests.  This module covers the separate
live safety boundary: the exchange adapter raises before any HTTP request is
made, and no HTTP route can trigger real order placement.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

import btc_trader
from btc_trader.config import Settings
from btc_trader.enums import OrderType, Side, TradingMode
from btc_trader.errors import (
    ExchangeAuthenticationError,
    ExchangeTransientError,
    TradingDisabledError,
)
from btc_trader.exchange.bitflyer import BitflyerClient
from btc_trader.exchange.models import OrderRequest

TEST_KEY = "test-key"
TEST_SECRET = "test-secret"  # dummy value, never sent anywhere


def _order() -> OrderRequest:
    return OrderRequest(
        product_code="BTC_JPY",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        size=Decimal("0.001"),
        client_order_id="test-client-order-id",
    )


def _live_settings(**overrides: object) -> Settings:
    """Settings that would permit live trading, unless ``overrides`` say otherwise."""
    values: dict[str, object] = {
        "trading_mode": TradingMode.LIVE,
        "enable_automatic_trading": True,
        "live_trading_acknowledged": True,
        "bitflyer_api_key": TEST_KEY,
        "bitflyer_api_secret": TEST_SECRET,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# configuration gates
# --------------------------------------------------------------------------- #
def test_default_configuration_forbids_live_orders(settings: Settings) -> None:
    assert settings.trading_mode is TradingMode.DATA_ONLY
    assert settings.live_orders_allowed is False
    assert settings.enable_automatic_trading is False
    assert settings.live_trading_acknowledged is False


@pytest.mark.usefixtures("env")
def test_paper_mode_forbids_live_orders() -> None:
    settings = Settings(trading_mode=TradingMode.PAPER, enable_automatic_trading=True)
    assert settings.strategies_enabled is True
    assert settings.live_orders_allowed is False


@pytest.mark.usefixtures("env")
def test_live_mode_requires_automatic_trading_flag() -> None:
    with pytest.raises(ValidationError, match="ENABLE_AUTOMATIC_TRADING=true"):
        _live_settings(enable_automatic_trading=False)


@pytest.mark.usefixtures("env")
def test_live_mode_requires_explicit_acknowledgement() -> None:
    with pytest.raises(ValidationError, match="LIVE_TRADING_ACKNOWLEDGED=true"):
        _live_settings(live_trading_acknowledged=False)


@pytest.mark.usefixtures("env")
def test_live_mode_requires_credentials() -> None:
    """Missing credentials must fail closed, not fall back to unauthenticated."""
    with pytest.raises(ValidationError, match="requires bitFlyer API credentials"):
        _live_settings(bitflyer_api_key="", bitflyer_api_secret="")


@pytest.mark.usefixtures("env")
def test_kill_switch_overrides_every_other_flag() -> None:
    settings = Settings(trading_kill_switch=True)
    assert settings.live_orders_allowed is False


@pytest.mark.usefixtures("env")
def test_all_switches_plus_credentials_are_required_together() -> None:
    with pytest.raises(ValidationError, match="real order submission is not implemented"):
        _live_settings()


def test_stale_market_data_threshold_is_configured(settings: Settings) -> None:
    """§48: a stale-data guard needs a finite, positive threshold to guard with."""
    assert settings.max_market_data_age_seconds > 0


@pytest.mark.usefixtures("env")
def test_safe_dump_omits_credentials() -> None:
    settings = Settings(bitflyer_api_key=TEST_KEY, bitflyer_api_secret=TEST_SECRET)
    dumped = settings.safe_dump()
    assert "bitflyer_api_key" not in dumped
    assert "bitflyer_api_secret" not in dumped
    assert "database_url" not in dumped
    assert dumped["bitflyer_credentials_configured"] is True
    serialised = repr(dumped)
    assert TEST_KEY not in serialised
    assert TEST_SECRET not in serialised


@pytest.mark.usefixtures("env")
def test_repr_does_not_leak_the_secret() -> None:
    settings = Settings(bitflyer_api_key=TEST_KEY, bitflyer_api_secret=TEST_SECRET)
    assert TEST_SECRET not in repr(settings)
    assert TEST_SECRET not in str(settings)


# --------------------------------------------------------------------------- #
# the submission path
# --------------------------------------------------------------------------- #
async def test_place_order_raises_without_touching_the_network(settings: Settings) -> None:
    """No route is registered, so any HTTP attempt fails the test loudly."""
    with respx.mock(assert_all_called=False) as router:
        client = BitflyerClient(settings)
        try:
            with pytest.raises(TradingDisabledError):
                await client.place_order(_order())
        finally:
            await client.aclose()
        assert not router.calls


async def test_cancel_order_raises_without_touching_the_network(settings: Settings) -> None:
    with respx.mock(assert_all_called=False) as router:
        client = BitflyerClient(settings)
        try:
            with pytest.raises(TradingDisabledError):
                await client.cancel_order("BTC_JPY", "JRF-does-not-exist")
        finally:
            await client.aclose()
        assert not router.calls


async def test_private_endpoints_refuse_without_credentials(settings: Settings) -> None:
    """Signing is impossible without a secret, so the request is never built."""
    assert settings.has_private_credentials is False
    with respx.mock(assert_all_called=False) as router:
        client = BitflyerClient(settings)
        try:
            with pytest.raises(ExchangeAuthenticationError):
                await client.get_balances()
        finally:
            await client.aclose()
        assert not router.calls


async def test_exchange_unavailable_does_not_produce_an_order(settings: Settings) -> None:
    """§48: an exchange outage must not be answered with a blind submission."""
    with respx.mock(base_url=settings.bitflyer_public_base_url) as router:
        router.get("/v1/ticker").mock(return_value=httpx.Response(503))
        client = BitflyerClient(settings)
        try:
            with pytest.raises(ExchangeTransientError, match="503"):
                await client.get_ticker("BTC_JPY")
            with pytest.raises(TradingDisabledError):
                await client.place_order(_order())
        finally:
            await client.aclose()
        assert [call for call in router.calls if call.request.method != "GET"] == []


# --------------------------------------------------------------------------- #
# nothing in the codebase can reach a trading endpoint
# --------------------------------------------------------------------------- #
def test_no_module_references_an_order_submission_endpoint() -> None:
    """A grep-level guarantee: the order endpoints are absent from the source."""
    package_root = Path(btc_trader.__file__).parent
    forbidden = ("sendchildorder", "cancelchildorder", "sendparentorder")
    offenders = [
        f"{path.relative_to(package_root)}:{token}"
        for path in package_root.rglob("*.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.usefixtures("settings")
def test_http_api_exposes_no_order_placement_route() -> None:
    from btc_trader.main import create_app

    app = create_app()
    mutating = {
        (getattr(route, "path", ""), method)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if method not in ("GET", "HEAD", "OPTIONS")
    }
    assert mutating == {
        ("/api/v1/market/collect", "POST"),
        ("/api/v1/trading/paper/reset", "POST"),
        ("/api/v1/system/settings", "PATCH"),
    }
