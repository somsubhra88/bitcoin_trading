"""Dashboard rendering (PROJECT.md §15, §16).

The real Streamlit script is executed through ``AppTest`` against a mocked
platform API, so the whole page — sidebar, chart, tables — is built the way a
browser would build it.  The two rules under test beyond "it renders":

* the dashboard reads only the platform API and **never** calls bitFlyer (§16);
* an unreachable API produces an explanation, not a stack trace (§15).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from streamlit.testing.v1 import AppTest

API_URL = "http://dashboard-test-api"

APP_PATH = str(Path(__file__).resolve().parents[2] / "dashboard" / "app.py")

OVERVIEW: dict[str, Any] = {
    "product_code": "BTC_JPY",
    "exchange": "bitflyer",
    "latest_timestamp": "2026-09-01T17:47:23.553000Z",
    "data_age_seconds": 12.5,
    "is_stale": False,
    "ltp": "12411935.0",
    "best_bid": "12412067.0",
    "best_ask": "12420247.0",
    "spread": "8180.0",
    "spread_percent": "0.065",
    "volume_24h": "1161.95655792",
    "change_absolute": "125000.0",
    "change_percent": "1.02",
    "change_window": "24h",
    "stored_tick_count": 3,
    "stored_execution_count": 5,
}

STATUS: dict[str, Any] = {
    "app_env": "development",
    "trading_mode": "data_only",
    "product_code": "BTC_JPY",
    "exchange": "bitflyer",
    "strategy_name": "none",
    "automatic_trading_enabled": False,
    "live_orders_allowed": False,
    "kill_switch": False,
    "market_data_interval_seconds": 300,
    "dashboard_refresh_seconds": 300,
    "stored_tick_count": 3,
    "stored_execution_count": 5,
    "scheduler_jobs": [],
    "database": "ok",
    "collector": {
        "last_run_at": "2026-09-01T17:47:24Z",
        "last_run_success": True,
        "last_success_at": "2026-09-01T17:47:24Z",
        "runs_24h": 12,
        "failures_24h": 0,
    },
}

TRADING: dict[str, Any] = {
    "trading_mode": "data_only",
    "strategy_name": "none",
    "automatic_trading_enabled": False,
    "live_trading_acknowledged": False,
    "kill_switch": False,
    "live_orders_allowed": False,
    "credentials_configured": False,
    "min_profit_percent": "1.0",
    "max_order_jpy": "10000",
    "max_position_jpy": "100000",
    "max_daily_loss_jpy": "5000",
    "max_market_data_age_seconds": 600,
    "execution_engine_implemented": False,
}

HISTORY: dict[str, Any] = {
    "product_code": "BTC_JPY",
    "count": 2,
    "ticks": [
        {"exchange_timestamp": "2026-09-01T17:46:00Z", "ltp": "12400000.0"},
        {"exchange_timestamp": "2026-09-01T17:47:00Z", "ltp": "12411935.0"},
    ],
}

CANDLES: dict[str, Any] = {
    "product_code": "BTC_JPY",
    "interval": "1m",
    "source": "executions",
    "is_true_ohlc": True,
    "count": 1,
    "candles": [
        {
            "bucket_start": "2026-09-01T17:47:00Z",
            "open": "12400000.0",
            "high": "12420000.0",
            "low": "12399000.0",
            "close": "12411935.0",
            "volume": "1.5",
            "trade_count": 42,
        }
    ],
}

EVENTS: list[dict[str, Any]] = [
    {
        "occurred_at": "2026-09-01T17:40:00Z",
        "severity": "WARNING",
        "component": "market_data_collector",
        "event_type": "collection_degraded",
        "message": "degraded endpoints: order_book",
    }
]

RUNS: list[dict[str, Any]] = [
    {
        "started_at": "2026-09-01T17:47:24Z",
        "finished_at": "2026-09-01T17:47:25Z",
        "trigger": "scheduled",
        "success": True,
        "ticks_inserted": 1,
        "executions_inserted": 3,
        "order_books_inserted": 1,
        "duration_ms": 198,
        "error_type": None,
        "error_message": None,
    }
]


@pytest.fixture
def dashboard_env(monkeypatch: pytest.MonkeyPatch, env: Path) -> None:
    """Point the dashboard at the mocked API (``env`` pins everything else)."""
    monkeypatch.setenv("API_BASE_URL", API_URL)
    monkeypatch.setenv("DASHBOARD_REFRESH_SECONDS", "300")


@pytest.fixture
def api(dashboard_env: None) -> Iterator[respx.MockRouter]:
    """A fully-populated platform API, plus a bitFlyer that must stay untouched."""
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API_URL}/api/v1/market/overview").mock(
            return_value=httpx.Response(200, json=OVERVIEW)
        )
        router.get(f"{API_URL}/api/v1/market/history").mock(
            return_value=httpx.Response(200, json=HISTORY)
        )
        router.get(f"{API_URL}/api/v1/market/candles").mock(
            return_value=httpx.Response(200, json=CANDLES)
        )
        router.get(f"{API_URL}/api/v1/system/status").mock(
            return_value=httpx.Response(200, json=STATUS)
        )
        router.get(f"{API_URL}/api/v1/system/settings").mock(
            return_value=httpx.Response(200, json={"config": {}})
        )
        router.get(f"{API_URL}/api/v1/trading/status").mock(
            return_value=httpx.Response(200, json=TRADING)
        )
        router.get(f"{API_URL}/api/v1/portfolio").mock(
            return_value=httpx.Response(
                200,
                json={
                    "product_code": "BTC_JPY",
                    "trading_mode": "data_only",
                    "position": None,
                    "latest_snapshot": None,
                },
            )
        )
        router.get(f"{API_URL}/api/v1/system/events").mock(
            return_value=httpx.Response(200, json=EVENTS)
        )
        router.get(f"{API_URL}/api/v1/system/collector").mock(
            return_value=httpx.Response(200, json=RUNS)
        )
        for path in ("/api/v1/signals", "/api/v1/orders", "/api/v1/trades"):
            router.get(f"{API_URL}{path}").mock(return_value=httpx.Response(200, json=[]))
        # Any bitFlyer traffic is a bug, so make it loud rather than silent.
        router.route(host="api.bitflyer.com").mock(
            side_effect=AssertionError("the dashboard must not call bitFlyer (§16)")
        )
        yield router


def _run(timeout: float = 60.0) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=timeout)
    return app.run()


def _text(app: AppTest) -> str:
    """Everything the page rendered, as one searchable string."""
    parts: list[str] = []
    for element in app.markdown:
        parts.append(str(element.value))
    for element in app.caption:
        parts.append(str(element.value))
    for element in app.subheader:
        parts.append(str(element.value))
    for element in app.title:
        parts.append(str(element.value))
    for element in app.metric:
        parts.append(f"{element.label} {element.value} {element.delta}")
    for element in app.info:
        parts.append(str(element.value))
    for element in app.warning:
        parts.append(str(element.value))
    for element in app.error:
        parts.append(str(element.value))
    for element in app.success:
        parts.append(str(element.value))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# the page renders
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("api")
def test_the_page_renders_without_exceptions() -> None:
    app = _run()

    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.title[0].value == "BTC/JPY Trading Platform"


@pytest.mark.usefixtures("api")
def test_deep_rl_workspace_renders() -> None:
    app = _run()

    app.sidebar.radio[0].set_value("Deep RL Research").run()

    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.title[0].value == "Deep RL Research"
    assert "Phase 1" in {str(button.label) for button in app.button}
    assert "Evaluation" in {str(element.value) for element in app.subheader}


@pytest.mark.usefixtures("api")
def test_every_section_is_present() -> None:
    """§15: the dashboard must show all of the specified panels."""
    app = _run()

    headings = {str(element.value) for element in app.subheader}
    assert headings >= {
        "Market Overview",
        "Price Chart",
        "Portfolio",
        "Strategy",
        "Trading",
        "System",
    }


@pytest.mark.usefixtures("api")
def test_the_price_is_rendered_from_the_api_string() -> None:
    app = _run()

    assert "¥12,411,935" in _text(app)


@pytest.mark.usefixtures("api")
def test_the_chart_is_drawn_from_stored_history() -> None:
    app = _run()

    assert not app.exception
    # One figure: the line chart for the default period.
    assert len(app.get("plotly_chart")) == 1


@pytest.mark.usefixtures("api")
def test_the_sidebar_offers_the_documented_controls() -> None:
    app = _run()

    # ``options`` are the human labels; the selected ``value`` is the API key.
    assert [str(option) for option in app.sidebar.selectbox[0].options] == [
        "1 hour",
        "6 hours",
        "24 hours",
        "7 days",
        "30 days",
        "All",
    ]
    assert app.sidebar.selectbox[0].value == "24h"
    assert list(app.sidebar.selectbox[1].options) == ["1m", "5m", "15m", "1h", "4h", "1d"]
    assert app.sidebar.selectbox[1].value == "15m"  # the default for a 24h window
    assert "DeepLOB-style order book" in app.sidebar.selectbox[2].options
    labels = {str(button.label) for button in app.sidebar.button}
    assert labels == {"Collect now", "Refresh view", "Apply settings"}


@pytest.mark.usefixtures("api")
def test_data_only_mode_is_declared_not_faked() -> None:
    """§9/§53.12: data-only mode must not imply paper or live execution."""
    app = _run()

    page = _text(app).lower()
    assert "data-only mode — no orders can be placed" in page
    assert "paper execution is disabled in data-only mode" in page
    assert "values appear once paper trading is enabled" in page


# --------------------------------------------------------------------------- #
# bitFlyer is off limits
# --------------------------------------------------------------------------- #
def test_the_dashboard_never_calls_bitflyer(api: respx.MockRouter) -> None:
    """§16: per-browser-refresh exchange calls would burn the rate-limit budget."""
    app = _run()

    assert not app.exception
    called = {call.request.url.host for call in api.calls}
    assert "api.bitflyer.com" not in called
    assert called == {"dashboard-test-api"}


# --------------------------------------------------------------------------- #
# degraded modes
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("dashboard_env")
def test_an_unreachable_api_is_explained_not_raised() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.route(host="dashboard-test-api").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        app = _run()

    assert not app.exception
    errors = " ".join(str(element.value) for element in app.error)
    assert "Cannot reach the platform API" in errors


@pytest.mark.usefixtures("dashboard_env")
def test_an_empty_database_invites_a_collection() -> None:
    """A fresh install has no data; that is a documented state, not an error (§9)."""
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API_URL}/api/v1/market/overview").mock(return_value=httpx.Response(404))
        router.get(f"{API_URL}/api/v1/market/history").mock(return_value=httpx.Response(404))
        router.get(f"{API_URL}/api/v1/market/candles").mock(return_value=httpx.Response(404))
        router.get(f"{API_URL}/api/v1/system/status").mock(
            return_value=httpx.Response(200, json=STATUS)
        )
        router.get(f"{API_URL}/api/v1/trading/status").mock(
            return_value=httpx.Response(200, json=TRADING)
        )
        router.route(host="dashboard-test-api").mock(return_value=httpx.Response(200, json=[]))

        app = _run()

    assert not app.exception
    assert "No market data stored yet" in _text(app)


def test_an_unknown_timezone_falls_back_to_utc(
    monkeypatch: pytest.MonkeyPatch, api: respx.MockRouter
) -> None:
    """§38: a bad DASHBOARD_TIMEZONE must not blank the page."""
    monkeypatch.setenv("DASHBOARD_TIMEZONE", "Mars/Olympus_Mons")

    app = _run()

    assert not app.exception
    warnings = " ".join(str(element.value) for element in app.warning)
    assert "Mars/Olympus_Mons" in warnings
    assert "UTC" in warnings
    assert api.calls  # the page still fetched and rendered


@pytest.mark.usefixtures("api")
def test_collect_now_posts_to_the_api(api: respx.MockRouter) -> None:
    """The manual trigger goes through the API, not straight to the exchange (§16)."""
    route = api.post(f"{API_URL}/api/v1/market/collect").mock(
        return_value=httpx.Response(200, json={"success": True, "duration_ms": 123, "degraded": []})
    )
    app = _run()

    collect = next(button for button in app.sidebar.button if button.label == "Collect now")
    collect.click().run()

    assert route.call_count == 1
    assert not app.exception


@pytest.mark.usefixtures("api")
def test_apply_runtime_settings_submits_the_dashboard_controls(api: respx.MockRouter) -> None:
    route = api.patch(f"{API_URL}/api/v1/system/settings").mock(
        return_value=httpx.Response(200, json={"config": {}})
    )
    app = _run()

    apply = next(item for item in app.sidebar.button if item.key == "apply_runtime_settings")
    apply.click().run()

    assert route.call_count == 1
    assert b'"market_data_interval_seconds":300' in route.calls[0].request.content
    assert b'"collect_order_book":true' in route.calls[0].request.content
    assert b'"strategy_momentum_threshold_percent":"0.01"' in route.calls[0].request.content
    assert b'"strategy_deeplob_window":32' in route.calls[0].request.content
    assert not app.exception


@pytest.mark.usefixtures("api")
def test_paper_mode_exposes_a_confirmed_reset_button(api: respx.MockRouter) -> None:
    api.get(f"{API_URL}/api/v1/portfolio").mock(
        return_value=httpx.Response(
            200,
            json={
                "product_code": "BTC_JPY",
                "trading_mode": "paper",
                "position": None,
                "latest_snapshot": {
                    "btc_balance": "0.25",
                    "jpy_balance": "5000",
                    "btc_market_price": "100",
                    "btc_value_jpy": "25",
                    "total_value_jpy": "5025",
                    "realized_pnl_jpy": "0",
                    "unrealized_pnl_jpy": "0",
                },
            },
        )
    )
    reset = api.post(f"{API_URL}/api/v1/trading/paper/reset").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": "Paper trading reset. BTC holdings are zero.",
            },
        )
    )
    app = _run()

    button = next(item for item in app.button if item.key == "reset_paper_trading")
    assert button.disabled is True
    app.checkbox(key="confirm_paper_reset").set_value(True).run()
    button = next(item for item in app.button if item.key == "reset_paper_trading")
    assert button.disabled is False
    button.click().run()

    assert reset.call_count == 1
    assert b'{"confirm":true}' in reset.calls[0].request.content
    assert not app.exception


@pytest.mark.usefixtures("api")
def test_strategy_change_waits_for_apply(api: respx.MockRouter) -> None:
    route = api.patch(f"{API_URL}/api/v1/system/settings").mock(
        return_value=httpx.Response(200, json={"config": {}})
    )
    app = _run()

    app.sidebar.selectbox[2].set_value("ema_crossover").run()

    assert route.call_count == 0
    apply = next(item for item in app.sidebar.button if item.key == "apply_runtime_settings")
    apply.click().run()

    assert route.call_count == 1
    assert b'"strategy_name":"ema_crossover"' in route.calls[0].request.content
    assert not app.exception
