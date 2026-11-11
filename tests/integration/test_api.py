"""HTTP surface (PROJECT.md §34, §43, §53.12).

Driven through an in-process ASGI transport against a real SQLite database, so
routing, dependency wiring, serialisation and status codes are the production
ones.  Two properties matter beyond the happy paths:

* money crosses the wire as a JSON *string*, never a float (§13);
* no endpoint can submit a real exchange order, and none leaks a credential (§34, §35).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, reload_settings
from btc_trader.db.models import MarketTick, RuntimeSetting, StrategySignal
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.enums import Severity, TradingMode, TradingSignal
from btc_trader.main import create_app
from btc_trader.market_data.daily import store_external_bars
from btc_trader.market_data.history_provider import ExternalDailyBar
from btc_trader.market_data.repository import CollectorRunRepository, MarketDataRepository
from btc_trader.paper import rebalance_paper
from btc_trader.utils.timeutils import utcnow
from tests.fixtures.daily import up_then_down
from tests.fixtures.fake_exchange import BASE_TIME, make_ticker

BITFLYER_URL = "https://api.bitflyer.com"


@pytest.fixture
async def api(database: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """A client for the app.

    The lifespan is deliberately not run: it would start a scheduler and dispose
    the engine the fixtures own.  ``scheduler`` therefore resolves to ``None``,
    which is exactly the deployment shape of ``RUN_SCHEDULER_IN_API=false``.
    """
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def stored(session: AsyncSession) -> None:
    """A small amount of history: three ticks."""
    repository = MarketDataRepository(session)
    for offset in (0, 60, 120):
        await repository.insert_tick(
            make_ticker(
                exchange="bitflyer",
                timestamp=BASE_TIME + timedelta(seconds=offset),
                ltp=str(12000000 + offset),
            )
        )
    await session.commit()


# --------------------------------------------------------------------------- #
# root and docs
# --------------------------------------------------------------------------- #
async def test_root_describes_the_service(api: httpx.AsyncClient) -> None:
    response = await api.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "btc-trader"
    assert body["product_code"] == "BTC_JPY"
    assert body["trading_mode"] == "data_only"


async def test_the_openapi_schema_is_served(api: httpx.AsyncClient) -> None:
    response = await api.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/market/latest" in response.json()["paths"]
    assert "/api/v1/backtest" in response.json()["paths"]
    assert "/api/v1/strategy/preview" in response.json()["paths"]
    assert "/api/v1/market/daily-history" in response.json()["paths"]
    assert "/api/v1/trading/paper/reset" in response.json()["paths"]


async def test_backtest_explains_missing_daily_history(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/backtest", params={"strategy_name": "trend_ensemble"})

    assert response.status_code == 422
    assert "not enough daily history" in response.json()["detail"]


async def test_strategy_preview_reports_warm_up_without_history(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/strategy/preview")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert "Load daily history" in body["message"]
    assert body["history"]["bars"] == 0
    assert body["history"]["required_bars"] == 360


async def test_daily_history_sync_without_a_provider_is_a_no_op(api: httpx.AsyncClient) -> None:
    response = await api.post("/api/v1/market/daily-history/sync")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "none"
    assert body["external_bars_inserted"] == 0
    assert body["history"]["bars"] == 0


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
async def test_health_is_ok_without_probing_the_exchange(api: httpx.AsyncClient) -> None:
    response = await api.get("/health", params={"probe_exchange": "false"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["exchange_public_api"] == "disabled"
    # This process hosts no scheduler, which is not a failure (§43).
    assert body["scheduler"] == "disabled"


async def test_health_probes_the_exchange_when_asked(api: httpx.AsyncClient) -> None:
    with respx.mock(base_url=BITFLYER_URL) as router:
        router.get("/v1/getboardstate").mock(
            return_value=httpx.Response(200, json={"health": "NORMAL", "state": "RUNNING"})
        )

        response = await api.get("/health", params={"probe_exchange": "true"})

    assert response.status_code == 200
    assert response.json()["exchange_public_api"] == "ok"


async def test_a_broken_exchange_is_degraded_but_still_healthy_enough_to_stay_up(
    api: httpx.AsyncClient,
) -> None:
    """§43: a bitFlyer hiccup must not make the orchestrator restart the container."""
    with respx.mock(base_url=BITFLYER_URL) as router:
        router.get("/v1/getboardstate").mock(return_value=httpx.Response(503))

        response = await api.get("/health", params={"probe_exchange": "true"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "ok"


async def test_health_never_leaks_a_credential(api: httpx.AsyncClient) -> None:
    response = await api.get("/health", params={"probe_exchange": "false"})

    assert "secret" not in response.text.lower()
    assert "access-sign" not in response.text.lower()


# --------------------------------------------------------------------------- #
# market data
# --------------------------------------------------------------------------- #
async def test_latest_is_404_before_any_collection(api: httpx.AsyncClient) -> None:
    """An empty database is a documented state, not a server error (§9)."""
    response = await api.get("/api/v1/market/latest")

    assert response.status_code == 404
    assert "no market data" in response.json()["detail"]


@pytest.mark.usefixtures("stored")
async def test_latest_returns_the_newest_tick_with_string_money(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/market/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["product_code"] == "BTC_JPY"
    # §13: a JSON number here would silently round the price.
    assert isinstance(body["ltp"], str)
    assert body["ltp"] == "12000120"


@pytest.mark.usefixtures("stored")
async def test_overview_reports_stored_counts_and_change(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/market/overview", params={"change_window": "24h"})

    assert response.status_code == 200
    body = response.json()
    assert body["stored_tick_count"] == 3
    assert body["ltp"] == "12000120"
    assert body["change_window"] == "24h"


async def test_an_unknown_period_is_rejected(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/market/overview", params={"change_window": "17y"})

    assert response.status_code == 422
    assert "unknown period" in response.json()["detail"]


@pytest.mark.usefixtures("stored")
async def test_history_is_bounded_and_ordered(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/market/history", params={"period": "all", "limit": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    prices = [tick["ltp"] for tick in body["ticks"]]
    assert prices == ["12000060", "12000120"]  # oldest-first inside the window


@pytest.mark.parametrize("limit", ["0", "50001"])
async def test_history_rejects_an_out_of_range_limit(api: httpx.AsyncClient, limit: str) -> None:
    response = await api.get("/api/v1/market/history", params={"limit": limit})

    assert response.status_code == 422


@pytest.mark.usefixtures("stored")
async def test_candles_are_folded_from_sampled_ticks(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/market/candles", params={"interval": "1h"})

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "1h"
    assert body["count"] == 1
    bar = body["candles"][0]
    assert bar["samples"] == 3
    assert isinstance(bar["close"], str)


async def test_an_unknown_interval_is_rejected(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/market/candles", params={"interval": "3s"})

    assert response.status_code == 422
    assert "unknown interval" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# manual collection
# --------------------------------------------------------------------------- #
async def test_collect_stores_a_tick_through_the_same_service_as_the_scheduler(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    from tests.fixtures.bitflyer_payloads import TICKER_JSON

    json_headers = {"content-type": "application/json"}
    with respx.mock(base_url=BITFLYER_URL) as router:
        router.get("/v1/ticker").mock(
            return_value=httpx.Response(200, text=TICKER_JSON, headers=json_headers)
        )

        response = await api.post("/api/v1/market/collect")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["tick_inserted"] is True
    assert body["ltp"] == "12411935.0"

    assert await MarketDataRepository(session).count_ticks("BTC_JPY") == 1


async def test_collect_reports_502_when_the_exchange_is_down(api: httpx.AsyncClient) -> None:
    with respx.mock(base_url=BITFLYER_URL) as router:
        router.get("/v1/ticker").mock(return_value=httpx.Response(503))

        response = await api.post("/api/v1/market/collect")

    assert response.status_code == 502
    assert response.json()["detail"]


# --------------------------------------------------------------------------- #
# system
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("stored")
async def test_system_status_aggregates_the_operational_picture(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    run = await CollectorRunRepository(session).start(
        exchange="bitflyer", product_code="BTC_JPY", trigger="manual"
    )
    await CollectorRunRepository(session).finish(run, success=True, ticks_inserted=1)
    await session.commit()

    response = await api.get("/api/v1/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["product_code"] == "BTC_JPY"
    assert body["trading_mode"] == "data_only"
    assert body["live_orders_allowed"] is False
    assert body["stored_tick_count"] == 3
    assert body["scheduler_jobs"] == []
    assert body["collector"]["last_run_success"] is True


async def test_system_config_redacts_secrets(api: httpx.AsyncClient) -> None:
    """§35/§51.8: the effective config is useful for support, secrets are not."""
    response = await api.get("/api/v1/system/settings")

    assert response.status_code == 200
    config = response.json()["config"]
    assert config["product_code"] == "BTC_JPY"
    flattened = str(config).lower()
    assert "bitflyer_api_secret" not in flattened or "***" in flattened
    assert "alert_webhook_url" not in config
    for key, value in config.items():
        if "secret" in key or "key" in key:
            assert value in ("***redacted***", "not_set", None, False, True), (key, value)


async def test_runtime_settings_support_partial_updates_without_none_overwrites(
    api: httpx.AsyncClient,
) -> None:
    await api.patch("/api/v1/system/settings", json={"strategy_name": "trend_ensemble"})
    response = await api.patch("/api/v1/system/settings", json={"market_data_interval_seconds": 60})

    assert response.status_code == 200
    config = response.json()["config"]
    assert config["market_data_interval_seconds"] == 60
    assert config["strategy_name"] == "trend_ensemble"

    persisted = await api.get("/api/v1/system/settings")
    assert persisted.status_code == 200
    assert persisted.json()["config"]["market_data_interval_seconds"] == 60


async def test_runtime_settings_can_be_applied_with_a_patch_body(
    api: httpx.AsyncClient,
) -> None:
    response = await api.patch(
        "/api/v1/system/settings",
        json={"strategy_name": "TREND_ENSEMBLE", "trading_mode": "paper"},
    )

    assert response.status_code == 200
    config = response.json()["config"]
    assert config["strategy_name"] == "trend_ensemble"
    assert config["trading_mode"] == "paper"


async def test_runtime_settings_reject_fee_unsafe_profit_target(
    api: httpx.AsyncClient,
) -> None:
    response = await api.patch(
        "/api/v1/system/settings",
        json={"min_profit_percent": "0.1", "taker_fee_percent": "0.15"},
    )

    assert response.status_code == 422
    assert "round-trip taker fees" in response.json()["detail"]


async def test_runtime_settings_reject_live_mode_without_all_safety_gates(
    api: httpx.AsyncClient,
) -> None:
    response = await api.patch(
        "/api/v1/system/settings", json={"trading_mode": "live", "live_confirmation": True}
    )

    assert response.status_code == 422
    assert "TRADING_MODE=live" in response.json()["detail"]


async def test_strategy_selection_is_persisted_for_other_processes(
    api: httpx.AsyncClient,
) -> None:
    response = await api.patch("/api/v1/system/settings", json={"strategy_name": "trend_ensemble"})

    assert response.status_code == 200
    status_response = await api.get("/api/v1/trading/status")
    assert status_response.status_code == 200
    assert status_response.json()["strategy_name"] == "trend_ensemble"


async def test_a_retired_strategy_saved_by_an_older_version_is_ignored(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    session.add(RuntimeSetting(key="strategy_name", value="deeplob"))
    await session.commit()

    response = await api.get("/api/v1/trading/status")

    assert response.status_code == 200
    assert response.json()["strategy_name"] == "none"


async def test_system_events_are_returned_newest_first(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    events = SystemEventRepository(session)
    await events.record(severity=Severity.INFO, component="test", event_type="first", message="one")
    await events.record(
        severity=Severity.ERROR, component="test", event_type="second", message="two"
    )
    await session.commit()

    response = await api.get("/api/v1/system/events", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert [event["event_type"] for event in body] == ["second", "first"]
    assert body[0]["severity"] == "ERROR"


async def test_collector_runs_endpoint(api: httpx.AsyncClient, session: AsyncSession) -> None:
    runs = CollectorRunRepository(session)
    failed = await runs.start(exchange="bitflyer", product_code="BTC_JPY", trigger="scheduled")
    await runs.finish(failed, success=False, error_type="ExchangeError", error_message="boom")
    await session.commit()

    response = await api.get("/api/v1/system/collector")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["success"] is False
    assert body[0]["error_type"] == "ExchangeError"


# --------------------------------------------------------------------------- #
# trading surface: read-only projections; paper mode populates them
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["/api/v1/signals", "/api/v1/orders", "/api/v1/trades"])
async def test_trading_projections_are_empty_not_broken(api: httpx.AsyncClient, path: str) -> None:
    response = await api.get(path)

    assert response.status_code == 200
    assert response.json() == []


async def test_portfolio_is_empty_before_paper_trading_starts(
    api: httpx.AsyncClient,
) -> None:
    response = await api.get("/api/v1/portfolio")

    assert response.status_code == 200
    body = response.json()
    assert body["position"] is None
    assert body["latest_snapshot"] is None


async def test_paper_reset_requires_explicit_confirmation(api: httpx.AsyncClient) -> None:
    response = await api.post("/api/v1/trading/paper/reset", json={"confirm": False})

    assert response.status_code == 400
    assert "confirm=true" in response.json()["detail"]


async def test_paper_reset_endpoint_reinitializes_the_account(
    api: httpx.AsyncClient, session: AsyncSession, settings: Settings
) -> None:
    applied = await api.patch("/api/v1/system/settings", json={"trading_mode": "paper"})
    assert applied.status_code == 200

    moment = utcnow()
    session.add(
        MarketTick(
            exchange="bitflyer",
            product_code="BTC_JPY",
            exchange_timestamp=moment,
            received_at=moment,
            ltp=Decimal("100"),
            best_bid=Decimal("99"),
            best_ask=Decimal("101"),
        )
    )
    signal = StrategySignal(
        strategy_name="trend_ensemble",
        strategy_version="2.0.0",
        product_code="BTC_JPY",
        timestamp=moment,
        signal=TradingSignal.BUY,
        confidence=Decimal("1"),
        reference_price=Decimal("100"),
    )
    session.add(signal)
    await session.flush()
    paper = settings.model_copy(
        update={"trading_mode": TradingMode.PAPER, "enforce_exchange_minimums": False}
    )
    assert await rebalance_paper(session, paper, signal, Decimal("0.5")) is not None
    await session.commit()

    response = await api.post("/api/v1/trading/paper/reset", json={"confirm": True})

    assert response.status_code == 200
    body = response.json()
    assert body["orders_deleted"] == 1
    assert body["fills_deleted"] == 1
    assert body["positions_deleted"] == 1
    assert body["snapshots_deleted"] == 1
    assert body["signals_preserved"] is True

    portfolio = (await api.get("/api/v1/portfolio")).json()
    assert portfolio["position"] is None
    assert portfolio["latest_snapshot"]["btc_balance"] == "0"
    assert portfolio["latest_snapshot"]["jpy_balance"] == str(settings.paper_initial_jpy)
    assert (await api.get("/api/v1/orders")).json() == []
    assert (await api.get("/api/v1/trades")).json() == []
    assert len((await api.get("/api/v1/signals")).json()) == 1
    events = (await api.get("/api/v1/system/events", params={"limit": 1})).json()
    assert events[0]["event_type"] == "paper_reset"


async def test_backtest_endpoint_replays_stored_daily_history_without_trading(
    api: httpx.AsyncClient, session: AsyncSession, settings: Settings
) -> None:
    today = utcnow().date()
    bars = up_then_down()
    shift = today - timedelta(days=1) - bars[-1].day
    await store_external_bars(
        session,
        settings,
        "bitbank",
        [
            ExternalDailyBar(
                bar.day + shift, bar.open, bar.close, bar.close, bar.close, Decimal("1")
            )
            for bar in bars
        ],
    )
    await session.commit()

    response = await api.get(
        "/api/v1/backtest",
        params={"strategy_name": "trend_ensemble", "include_trades": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_name"] == "trend_ensemble"
    assert body["metrics"]["trade_count"] == len(body["trades"]) > 0
    assert body["benchmark"]["name"] == "buy_and_hold"
    assert isinstance(body["metrics"]["final_equity_jpy"], str)
    assert [item["cost_multiplier"] for item in body["cost_stress"]] == [0.5, 1.0, 2.0, 4.0]
    assert body["periods"]
    assert (await api.get("/api/v1/orders")).json() == []

    preview = (await api.get("/api/v1/strategy/preview")).json()
    assert preview["ready"] is True
    assert preview["target_exposure"] == "0"
    assert len(preview["sleeves"]) == 9


async def test_trading_status_advertises_every_safety_switch(api: httpx.AsyncClient) -> None:
    response = await api.get("/api/v1/trading/status")

    assert response.status_code == 200
    body = response.json()
    assert body["trading_mode"] == "data_only"
    assert body["automatic_trading_enabled"] is False
    assert body["live_trading_acknowledged"] is False
    assert body["kill_switch"] is False
    assert body["live_orders_allowed"] is False
    assert body["credentials_configured"] is False
    assert body["live_capital_jpy"] == "0"
    assert body["alerts_configured"] is False
    assert body["min_order_size_btc"] == "0.001"
    assert body["enforce_exchange_minimums"] is True


async def test_no_route_can_submit_an_order(api: httpx.AsyncClient) -> None:
    """§53.12: mutating routes collect data, apply settings, or reset paper state."""
    schema = (await api.get("/openapi.json")).json()

    mutating = {
        (path, method.upper())
        for path, operations in schema["paths"].items()
        for method in operations
        if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
    }
    assert mutating == {
        ("/api/v1/market/collect", "POST"),
        ("/api/v1/market/daily-history/sync", "POST"),
        ("/api/v1/strategy/evaluate", "POST"),
        ("/api/v1/trading/paper/reset", "POST"),
        ("/api/v1/system/settings", "PATCH"),
    }


async def test_http_evaluation_never_trades_live(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    """Only the worker may place live orders; an HTTP call must not."""
    session.add(RuntimeSetting(key="trading_mode", value="live"))
    await session.commit()

    response = await api.post("/api/v1/strategy/evaluate")

    # The deployment does not meet the live gates, so the stored live mode is
    # ignored and evaluation runs in the deployment's own (data-only) mode.
    assert response.status_code == 409
    assert (await api.get("/api/v1/orders")).json() == []


async def test_http_evaluation_is_refused_in_live_mode(
    api: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "TRADING_MODE": "live",
        "ENABLE_AUTOMATIC_TRADING": "true",
        "LIVE_TRADING_ACKNOWLEDGED": "true",
        "BITFLYER_API_KEY": "key",
        "BITFLYER_API_SECRET": "secret",
    }.items():
        monkeypatch.setenv(key, value)
    reload_settings()

    response = await api.post("/api/v1/strategy/evaluate")

    assert response.status_code == 409
    assert "worker" in response.json()["detail"]
