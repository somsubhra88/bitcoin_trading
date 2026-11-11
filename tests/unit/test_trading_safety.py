"""Trading-safety tests (PROJECT.md §48).

Every test here proves that an unsafe order is **not** submitted, or that the
configuration which would permit one cannot be reached by accident.

Paper and live limit checks are covered by the ledger integration tests
(test_paper, test_live). This module covers the configuration gates, the
adapter's refusal to sign without credentials, and the guarantee that only the
exchange adapter can reach an order endpoint and no HTTP route can place one.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import respx
from pydantic import ValidationError

import btc_trader
from btc_trader.config import Settings
from btc_trader.enums import Side, TradingMode
from btc_trader.errors import ExchangeAuthenticationError
from btc_trader.exchange.bitflyer import BitflyerClient
from btc_trader.exchange.models import OrderRequest
from btc_trader.runtime_settings import validate_runtime_patch

TEST_KEY = "test-key"
TEST_SECRET = "test-secret"  # dummy value, never sent anywhere


def _order() -> OrderRequest:
    return OrderRequest(
        product_code="BTC_JPY", side=Side.BUY, size=Decimal("0.001"), price=Decimal("12000000")
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
def test_all_switches_plus_credentials_permit_live_orders() -> None:
    assert _live_settings().live_orders_allowed is True


@pytest.mark.usefixtures("env")
def test_the_dashboard_can_engage_the_kill_switch_while_live() -> None:
    """The emergency stop must never be refused for being 'not live-ready'."""
    stopped = validate_runtime_patch(_live_settings(), {"trading_kill_switch": True})
    assert stopped.trading_kill_switch is True
    assert stopped.live_orders_allowed is False


@pytest.mark.usefixtures("env")
def test_the_dashboard_cannot_enable_live_without_the_live_gates() -> None:
    with pytest.raises(ValidationError, match="ENABLE_AUTOMATIC_TRADING=true"):
        validate_runtime_patch(
            _live_settings(), {"trading_mode": "live", "enable_automatic_trading": False}
        )


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


async def test_without_credentials_an_order_never_reaches_the_network(
    settings: Settings,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        client = BitflyerClient(settings)
        try:
            with pytest.raises(ExchangeAuthenticationError):
                await client.place_order(_order())
        finally:
            await client.aclose()
        assert not router.calls


# --------------------------------------------------------------------------- #
# only the exchange adapter can reach a trading endpoint
# --------------------------------------------------------------------------- #
def test_only_the_adapter_references_order_endpoints() -> None:
    """A grep-level guarantee: one module sends orders, nothing can move funds out."""
    package_root = Path(btc_trader.__file__).parent
    endpoints = (
        "/v1/me/sendchildorder",
        "/v1/me/sendparentorder",
        "/v1/me/cancelallchildorders",
        "/v1/me/withdraw",
    )
    offenders = {
        f"{path.relative_to(package_root)}:{endpoint}"
        for path in package_root.rglob("*.py")
        for endpoint in endpoints
        if endpoint in path.read_text(encoding="utf-8")
    }
    # live_check only lists the path among the permissions it verifies.
    assert offenders == {
        "exchange/bitflyer.py:/v1/me/sendchildorder",
        "scripts/live_check.py:/v1/me/sendchildorder",
    }


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
        # Loads public daily candles into the database; it cannot place orders.
        ("/api/v1/market/daily-history/sync", "POST"),
        # Paper only: returns 409 in live mode (see test_api).
        ("/api/v1/strategy/evaluate", "POST"),
        ("/api/v1/trading/paper/reset", "POST"),
        ("/api/v1/system/settings", "PATCH"),
    }
