"""Streamlit dashboard (PROJECT.md §15, §16).

Every value shown here comes from the platform's own HTTP API, which reads the
database.  The dashboard never calls bitFlyer: repeating exchange requests per
browser refresh would burn the shared rate-limit budget for data the collector
has already stored (§16).

Timestamps are stored in UTC and rendered in ``DASHBOARD_TIMEZONE`` (§38).
Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_trader.config import Settings, get_settings
from btc_trader.utils.timeutils import ensure_utc, utcnow

REQUEST_TIMEOUT_SECONDS = 20.0

#: Chart periods offered by the API (§15 "Price Chart").
PERIOD_LABELS = {
    "1h": "1 hour",
    "6h": "6 hours",
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
    "all": "All",
}

CANDLE_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

#: Longer periods need coarser bars to stay readable.
DEFAULT_INTERVAL_FOR_PERIOD = {
    "1h": "1m",
    "6h": "5m",
    "24h": "15m",
    "7d": "1h",
    "30d": "4h",
    "all": "1d",
}

_STATUS_ICONS = {"ok": "🟢", "degraded": "🟡", "error": "🔴", "disabled": "⚪", "unknown": "⚪"}


# --------------------------------------------------------------------------- #
# API access
# --------------------------------------------------------------------------- #
class ApiError(RuntimeError):
    """The API could not be reached, or answered with an error status."""


class ApiClient:
    """Thin synchronous wrapper around the platform API."""

    def __init__(self, base_url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def get(self, path: str, **params: Any) -> Any:
        """GET ``path``; ``None`` when the resource does not exist yet (404)."""
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
                response = client.get(
                    path, params={k: v for k, v in params.items() if v is not None}
                )
        except httpx.HTTPError as exc:
            raise ApiError(f"{self._base_url}{path} unreachable: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise ApiError(f"{path} returned HTTP {response.status_code}: {_detail(response)}")
        return response.json()

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
                response = client.post(path, json=json)
        except httpx.HTTPError as exc:
            raise ApiError(f"{self._base_url}{path} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(f"{path} returned HTTP {response.status_code}: {_detail(response)}")
        return response.json()

    def patch(self, path: str, json: dict[str, Any] | None = None) -> Any:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
                response = client.patch(path, json=json)
        except httpx.HTTPError as exc:
            raise ApiError(f"{self._base_url}{path} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(f"{path} returned HTTP {response.status_code}: {_detail(response)}")
        return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)[:200]
    return str(payload)[:200]


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _decimal(value: Any) -> Decimal | None:
    """Parse an API money field. Decimals arrive as JSON strings by design (§13)."""
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _moment(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return ensure_utc(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def _local(value: Any, tz: ZoneInfo, *, with_seconds: bool = True) -> str:
    moment = _moment(value)
    if moment is None:
        return "—"
    pattern = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    return moment.astimezone(tz).strftime(pattern)


def _money(value: Any, unit: str = "¥", places: int = 0) -> str:
    amount = _decimal(value)
    if amount is None:
        return "—"
    quantum = Decimal(1).scaleb(-places)
    return f"{unit}{amount.quantize(quantum):,}"


def _quantity(value: Any, places: int = 8) -> str:
    amount = _decimal(value)
    if amount is None:
        return "—"
    return f"{amount.normalize():,f}" if amount else f"{Decimal(0):.{places}f}"


def _percent(value: Any, places: int = 2) -> str:
    amount = _decimal(value)
    if amount is None:
        return "—"
    return f"{amount.quantize(Decimal(1).scaleb(-places)):+}%"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _icon(state: str | None) -> str:
    return _STATUS_ICONS.get((state or "unknown").lower(), "⚪")


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def render_market_overview(overview: dict[str, Any] | None, tz: ZoneInfo) -> None:
    st.subheader("Market Overview")
    if not overview:
        st.info("No market data stored yet. Use **Collect now** in the sidebar.")
        return

    window = overview.get("change_window", "24h")
    first, second, third, fourth = st.columns(4)
    first.metric(
        "BTC/JPY last traded",
        _money(overview.get("ltp")),
        delta=(
            f"{_money(overview.get('change_absolute'))} "
            f"({_percent(overview.get('change_percent'))})"
            if overview.get("change_absolute") is not None
            else None
        ),
        help=f"Change measured over the {PERIOD_LABELS.get(window, window)} window.",
    )
    second.metric("Best bid", _money(overview.get("best_bid")))
    third.metric("Best ask", _money(overview.get("best_ask")))
    fourth.metric(
        "Spread",
        _money(overview.get("spread")),
        delta=_percent(overview.get("spread_percent"), places=4),
        delta_color="off",
    )

    left, middle, right = st.columns(3)
    left.metric("Latest data timestamp", _local(overview.get("latest_timestamp"), tz))
    middle.metric("Data age", _duration(overview.get("data_age_seconds")))
    right.metric("24h volume (BTC)", _quantity(overview.get("volume_24h")))

    if overview.get("is_stale"):
        st.warning(
            f"Market data is stale (age {_duration(overview.get('data_age_seconds'))}). "
            "The collector may be failing — see the System section."
        )


def render_price_chart(
    client: ApiClient, period: str, interval: str, chart_type: str, tz: ZoneInfo
) -> None:
    st.subheader("Price Chart")
    if chart_type == "Candlestick":
        _render_candles(client, period, interval, tz)
    else:
        _render_line(client, period, tz)


def _render_line(client: ApiClient, period: str, tz: ZoneInfo) -> None:
    payload = client.get("/api/v1/market/history", period=period, limit=20_000)
    ticks = (payload or {}).get("ticks", [])
    if not ticks:
        st.info("Not enough stored ticks to draw a chart yet.")
        return
    frame = pd.DataFrame(
        {
            "time": [_moment(t["exchange_timestamp"]) for t in ticks],
            "ltp": [_as_float(t.get("ltp")) for t in ticks],
            "bid": [_as_float(t.get("best_bid")) for t in ticks],
            "ask": [_as_float(t.get("best_ask")) for t in ticks],
        }
    ).dropna(subset=["time", "ltp"])
    if frame.empty:
        st.info("Stored history has no valid price points for this period yet.")
        return
    frame["time"] = frame["time"].dt.tz_convert(tz)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=frame["time"], y=frame["ltp"], name="LTP", mode="lines", line={"width": 2})
    )
    for column, name in (("bid", "Best bid"), ("ask", "Best ask")):
        if bool(frame[column].notna().any()):
            figure.add_trace(
                go.Scatter(
                    x=frame["time"],
                    y=frame[column],
                    name=name,
                    mode="lines",
                    line={"width": 1, "dash": "dot"},
                    opacity=0.5,
                )
            )
    _add_trend_overlay(figure, frame["time"], frame["ltp"])
    _layout(figure, tz, f"{len(frame):,} stored ticks")
    st.plotly_chart(figure, width="stretch")


def _render_candles(client: ApiClient, period: str, interval: str, tz: ZoneInfo) -> None:
    start = _period_start(period)
    payload = client.get(
        "/api/v1/market/candles",
        interval=interval,
        start=start.isoformat() if start else None,
        # Execution candles are intentionally sparse when no trade occurred in
        # a bucket. Use collected ticker samples for a continuous dashboard
        # view; the API still exposes true execution candles when requested.
        source="sampled_ticks",
    )
    candles = (payload or {}).get("candles", [])
    if not candles:
        st.info("No candles for this period yet.")
        return
    frame = pd.DataFrame(
        {
            "time": [_moment(c["bucket_start"]) for c in candles],
            "open": [_as_float(c["open"]) for c in candles],
            "high": [_as_float(c["high"]) for c in candles],
            "low": [_as_float(c["low"]) for c in candles],
            "close": [_as_float(c["close"]) for c in candles],
        }
    ).dropna()
    if frame.empty:
        st.info("Stored candles have no valid price points for this period yet.")
        return
    frame["time"] = frame["time"].dt.tz_convert(tz)

    figure = go.Figure(
        go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=interval,
        )
    )
    _add_trend_overlay(figure, frame["time"], frame["close"])
    figure.update_layout(xaxis_rangeslider_visible=False)
    _layout(figure, tz, f"{len(frame):,} x {interval} bars")
    st.plotly_chart(figure, width="stretch")

    if not payload.get("is_true_ohlc", False):
        st.caption(
            "⚠️ These bars are aggregated from sampled ticker snapshots, not from "
            "exchange trades: highs, lows and volume are approximations and volume "
            "is reported as zero. Enable `COLLECT_EXECUTIONS` for true OHLCV (§40)."
        )


def _layout(figure: go.Figure, tz: ZoneInfo, subtitle: str) -> None:
    figure.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title=f"Time ({tz.key})",
        yaxis_title="JPY",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        title={"text": subtitle, "font": {"size": 12}, "x": 0.0},
        hovermode="x unified",
    )


def _add_trend_overlay(figure: go.Figure, times: pd.Series, prices: pd.Series) -> None:
    """Add a robust visual guide: regression trend plus a residual channel."""
    if len(prices) < 4:
        return
    x = pd.Series(range(len(prices)), dtype="float64")
    coefficients = x.to_numpy()
    slope, intercept = np.polyfit(coefficients, prices.to_numpy(), 1)
    trend = slope * coefficients + intercept
    residual_std = float((prices.to_numpy() - trend).std())
    figure.add_trace(
        go.Scatter(
            x=times,
            y=trend,
            name="Trend",
            mode="lines",
            line={"color": "#ff9f43", "width": 2, "dash": "dash"},
        )
    )
    if residual_std > 0:
        figure.add_trace(
            go.Scatter(
                x=times,
                y=trend + residual_std,
                name="Trend envelope",
                mode="lines",
                line={"color": "#ff9f43", "width": 0},
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=trend - residual_std,
                mode="lines",
                line={"color": "#ff9f43", "width": 0},
                fill="tonexty",
                fillcolor="rgba(255,159,67,0.10)",
                showlegend=False,
            )
        )
    if len(prices) >= 8:
        ema = prices.ewm(span=8, adjust=False).mean()
        figure.add_trace(
            go.Scatter(
                x=times,
                y=ema,
                name="EMA 8",
                mode="lines",
                line={"color": "#54a0ff", "width": 1.5},
            )
        )


def _reset_paper_trading(client: ApiClient) -> None:
    """Clear simulated trades only after an explicit user confirmation."""
    message = st.session_state.pop("paper_reset_message", None)
    if message:
        st.success(message)

    st.markdown("#### Paper trading controls")
    st.caption(
        "Reset clears simulated orders, fills, holdings, and portfolio snapshots. "
        "Market data and strategy signals are kept."
    )
    confirmed = st.checkbox(
        "I understand this starts the paper account from zero BTC",
        key="confirm_paper_reset",
    )
    if st.button(
        "Reset paper trading",
        key="reset_paper_trading",
        disabled=not confirmed,
        help="Reinitialize paper trading with the configured starting JPY balance.",
    ):
        try:
            result = client.post(
                "/api/v1/trading/paper/reset",
                json={"confirm": True},
            )
        except ApiError as exc:
            st.error(str(exc))
            return
        st.session_state["paper_reset_message"] = result.get(
            "message",
            "Paper trading reset; holdings are now zero BTC.",
        )
        st.session_state.pop("confirm_paper_reset", None)
        st.rerun()


def render_portfolio(
    portfolio: dict[str, Any] | None,
    overview: dict[str, Any] | None,
    client: ApiClient,
) -> None:
    st.subheader("Portfolio")
    if (portfolio or {}).get("trading_mode") == "paper":
        _reset_paper_trading(client)

    position = (portfolio or {}).get("position")
    snapshot = (portfolio or {}).get("latest_snapshot")
    if not position and not snapshot:
        if (portfolio or {}).get("trading_mode") == "paper":
            st.info(
                "No paper portfolio state yet. The next paper mark or strategy "
                "evaluation will initialize it."
            )
        else:
            st.info("No portfolio state yet — values appear once paper trading is enabled.")
        return

    btc = (snapshot or {}).get("btc_balance") or (position or {}).get("btc_quantity")
    price = (snapshot or {}).get("btc_market_price") or (overview or {}).get("ltp")
    btc_value = (snapshot or {}).get("btc_value_jpy")
    if btc_value is None and btc is not None and price is not None:
        btc_amount, btc_price = _decimal(btc), _decimal(price)
        btc_value = btc_amount * btc_price if btc_amount is not None and btc_price else None

    first, second, third = st.columns(3)
    first.metric("BTC balance", _quantity(btc))
    second.metric("JPY balance", _money((snapshot or {}).get("jpy_balance")))
    third.metric("BTC market value", _money(btc_value))

    fourth, fifth, sixth, seventh = st.columns(4)
    fourth.metric("Average cost basis", _money((position or {}).get("average_entry_price")))
    fifth.metric(
        "Realized P&L",
        _money((snapshot or position or {}).get("realized_pnl_jpy")),
    )
    sixth.metric(
        "Unrealized P&L",
        _money((snapshot or position or {}).get("unrealized_pnl_jpy")),
    )
    seventh.metric("Total value", _money((snapshot or {}).get("total_value_jpy")))


def render_strategy(status: dict[str, Any], signals: list[dict[str, Any]], tz: ZoneInfo) -> None:
    st.subheader("Strategy")
    latest = signals[0] if signals else None
    first, second, third = st.columns(3)
    first.metric("Active strategy", status.get("strategy_name") or "none")
    second.metric("Strategy version", (latest or {}).get("strategy_version") or "—")
    third.metric(
        "Automatic trading",
        "Enabled" if status.get("automatic_trading_enabled") else "Disabled",
    )

    fourth, fifth, sixth = st.columns(3)
    fourth.metric("Latest signal", (latest or {}).get("signal") or "—")
    fifth.metric("Signal timestamp", _local((latest or {}).get("timestamp"), tz))
    sixth.metric("Confidence", _decimal((latest or {}).get("confidence")) or "—")

    if latest is None:
        st.caption("No signals recorded. Select an algorithm and press Apply settings.")
    elif latest.get("reason"):
        st.caption(f"Reason: {latest['reason']}")


def render_backtest(client: ApiClient, status: dict[str, Any]) -> None:
    """Run historical validation on demand so normal refreshes stay cheap."""
    st.subheader("Backtest")
    strategy_name = str(status.get("strategy_name") or "none")
    if strategy_name == "none":
        st.info("Select an algorithm, press Apply settings, then run a backtest.")
        return

    first, second = st.columns(2)
    with first:
        run_backtest = st.button("Run backtest", key="run_backtest")
    with second:
        run_walk_forward = st.button("Run walk-forward", key="run_walk_forward")

    if run_backtest:
        try:
            st.session_state["backtest_result"] = client.get(
                "/api/v1/backtest", strategy_name=strategy_name
            )
            st.session_state.pop("backtest_error", None)
        except ApiError as exc:
            st.session_state["backtest_error"] = str(exc)
    if run_walk_forward:
        try:
            st.session_state["walk_forward_result"] = client.get(
                "/api/v1/backtest/walk-forward", strategy_name=strategy_name
            )
            st.session_state.pop("backtest_error", None)
        except ApiError as exc:
            st.session_state["backtest_error"] = str(exc)

    if st.session_state.get("backtest_error"):
        st.error(st.session_state["backtest_error"])
    result = st.session_state.get("backtest_result")
    if isinstance(result, dict) and isinstance(result.get("metrics"), dict):
        st.caption("Historical replay using stored ticks, fees, slippage, and paper risk limits.")
        _render_backtest_metrics(result["metrics"])
    walk_forward = st.session_state.get("walk_forward_result")
    if isinstance(walk_forward, dict) and isinstance(walk_forward.get("aggregate_metrics"), dict):
        st.caption("Walk-forward scores use each train window only to select sensitivity.")
        _render_backtest_metrics(walk_forward["aggregate_metrics"])
        windows = walk_forward.get("windows") or []
        if windows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Train end": window.get("train_end"),
                            "Test end": window.get("test_end"),
                            "Test return": _percent(
                                (window.get("test_metrics") or {}).get("total_return_percent")
                            ),
                            "Test trades": (window.get("test_metrics") or {}).get("trade_count", 0),
                            "Parameters": str(window.get("selected_parameters") or {}),
                        }
                        for window in windows
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def _render_backtest_metrics(metrics: dict[str, Any]) -> None:
    first, second, third, fourth, fifth, sixth = st.columns(6)
    first.metric("Return", _percent(metrics.get("total_return_percent")))
    second.metric("Net P&L", _money(metrics.get("net_profit_jpy")))
    third.metric("Trades", metrics.get("trade_count", 0))
    fourth.metric("Win rate", _percent(metrics.get("win_rate_percent")))
    fifth.metric("Max drawdown", _percent(metrics.get("max_drawdown_percent")))
    sixth.metric("Sharpe", f"{float(metrics.get('sharpe_ratio') or 0):.2f}")


def render_trading(
    trading: dict[str, Any],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    tz: ZoneInfo,
) -> None:
    st.subheader("Trading")
    mode = str(trading.get("trading_mode", "unknown"))
    live_allowed = bool(trading.get("live_orders_allowed"))
    first, second, third, fourth = st.columns(4)
    first.metric("Trade mode", mode.replace("_", " ").title())
    second.metric("Live orders allowed", "YES" if live_allowed else "NO")
    third.metric("Kill switch", "ENGAGED" if trading.get("kill_switch") else "off")
    fourth.metric(
        "Credentials configured", "yes" if trading.get("credentials_configured") else "no"
    )

    if live_allowed:
        st.error(
            "🔴 LIVE MODE CONFIGURED — real exchange execution is still disabled in "
            "this build; the bitFlyer adapter refuses order submission."
        )
    elif mode == "paper":
        st.info("📝 Paper mode — orders are simulated; no funds move.")
        if trading.get("enforce_exchange_minimums"):
            st.caption(
                "Paper entries enforce the exchange minimum of "
                f"{trading.get('min_order_size_btc', '—')} BTC. "
                "Increase Maximum order if signals are blocked by lot size."
            )
    else:
        st.success("🔒 Data-only mode — no orders can be placed.")

    if mode == "data_only":
        st.caption(
            "Paper execution is disabled in data-only mode; the order and trade "
            "tables remain empty."
        )
    elif not trading.get("execution_engine_implemented", False):
        st.caption("Live exchange execution is disabled; only paper orders are simulated.")

    open_orders = [o for o in orders if str(o.get("status")) in _OPEN_STATUSES]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Open orders** ({len(open_orders)})")
        _order_table(open_orders, tz)
    with right:
        st.markdown(f"**Recent orders** ({len(orders)})")
        _order_table(orders, tz)

    st.markdown(f"**Recent executions** ({len(fills)})")
    if not fills:
        st.caption("None.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Executed at": _local(fill.get("executed_at"), tz),
                        "Price": _money(fill.get("price")),
                        "Size": _quantity(fill.get("size")),
                        "Commission": _money(fill.get("commission"), places=4),
                    }
                    for fill in fills
                ]
            ),
            width="stretch",
            hide_index=True,
        )


_OPEN_STATUSES = {
    "CREATED",
    "VALIDATED",
    "SUBMITTING",
    "SUBMITTED",
    "PARTIALLY_FILLED",
    "CANCEL_PENDING",
}


def _order_table(orders: list[dict[str, Any]], tz: ZoneInfo) -> None:
    if not orders:
        st.caption("None.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Created": _local(order.get("created_at"), tz, with_seconds=False),
                    "Side": order.get("side"),
                    "Type": order.get("order_type"),
                    "Size": _quantity(order.get("requested_size")),
                    "Filled": _quantity(order.get("filled_size")),
                    "Price": _money(
                        order.get("average_fill_price") or order.get("requested_price")
                    ),
                    "Status": order.get("status"),
                }
                for order in orders
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render_system(
    status: dict[str, Any],
    runs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    tz: ZoneInfo,
) -> None:
    st.subheader("System")
    health = status.get("health") or {}
    collector = status.get("collector") or {}

    first, second, third, fourth = st.columns(4)
    first.metric("API", f"{_icon(health.get('status'))} {health.get('status', 'unknown')}")
    second.metric(
        "Database", f"{_icon(health.get('database'))} {health.get('database', 'unknown')}"
    )
    third.metric(
        "Scheduler", f"{_icon(health.get('scheduler'))} {health.get('scheduler', 'unknown')}"
    )
    exchange_state = health.get("exchange_public_api", "unknown")
    fourth.metric("bitFlyer public API", f"{_icon(exchange_state)} {exchange_state}")

    fifth, sixth, seventh, eighth = st.columns(4)
    fifth.metric("Collector last run", _local(collector.get("last_run_at"), tz))
    sixth.metric("Last successful run", _local(collector.get("last_successful_run_at"), tz))
    seventh.metric("Stored ticks", f"{status.get('stored_tick_count', 0):,}")
    eighth.metric("Stored executions", f"{status.get('stored_execution_count', 0):,}")

    last_error = status.get("last_error")
    if collector.get("last_run_success") is False:
        st.error(
            f"Last collection failed: {collector.get('last_error_type') or 'error'} — "
            f"{collector.get('last_error_message') or 'no message'}"
        )
    if last_error:
        st.warning(
            f"Last error event ({_local(last_error.get('timestamp'), tz)}) "
            f"[{last_error.get('component')}/{last_error.get('event_type')}]: "
            f"{last_error.get('message')}"
        )
    if not last_error and collector.get("last_run_success") is not False:
        st.success("No errors recorded.")

    jobs = status.get("scheduler_jobs") or []
    if jobs:
        st.markdown("**Scheduled jobs**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Job": job.get("id"),
                        "Name": job.get("name"),
                        "Next run": _local(job.get("next_run_time"), tz),
                        "Trigger": job.get("trigger"),
                    }
                    for job in jobs
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(
            "This API process does not host the scheduler; collection runs in the "
            "worker container (`RUN_SCHEDULER_IN_API=false`)."
        )

    with st.expander(f"Recent collector runs ({len(runs)})"):
        if not runs:
            st.caption("None.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Started": _local(run.get("started_at"), tz),
                            "Trigger": run.get("trigger"),
                            "OK": "✅" if run.get("success") else "❌",
                            "ms": run.get("duration_ms"),
                            "Ticks": run.get("ticks_inserted"),
                            "Execs": run.get("executions_inserted"),
                            "Books": run.get("order_books_inserted"),
                            "Error": run.get("error_type") or "",
                        }
                        for run in runs
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    with st.expander(f"Recent events ({len(events)})"):
        if not events:
            st.caption("None.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time": _local(event.get("timestamp"), tz),
                            "Severity": event.get("severity"),
                            "Component": event.get("component"),
                            "Type": event.get("event_type"),
                            "Message": event.get("message"),
                        }
                        for event in events
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #
def _as_float(value: Any) -> float | None:
    """Chart coordinate. Plotly needs floats; money is never *computed* here (§13)."""
    amount = _decimal(value)
    return float(amount) if amount is not None else None


def _period_start(period: str) -> datetime | None:
    spans = {
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(days=1),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    span = spans.get(period)
    return utcnow() - span if span else None


def _effective_dashboard_settings(settings: Settings, client: ApiClient) -> Settings:
    """Use persisted settings when available, with env settings as a fallback."""
    try:
        payload = client.get("/api/v1/system/settings") or {}
        config = payload.get("config") if isinstance(payload, dict) else None
        if isinstance(config, dict):
            return Settings.model_validate({**settings.model_dump(), **config})
    except Exception:  # noqa: BLE001 - dashboard remains usable during API outages
        return settings
    return settings


def _runtime_settings_patch() -> dict[str, Any]:
    """Build the persisted patch from the reactive sidebar widget values."""
    state = st.session_state
    return {
        "trading_mode": state["runtime_mode"],
        "strategy_name": state["runtime_strategy"],
        "strategy_fast_window": int(state["runtime_fast_window"]),
        "strategy_slow_window": int(state["runtime_slow_window"]),
        "strategy_mean_reversion_window": int(state["runtime_mean_window"]),
        "strategy_momentum_window": int(state["runtime_momentum_window"]),
        "strategy_volatility_window": int(state["runtime_volatility_window"]),
        "strategy_ema_threshold_percent": str(state["runtime_ema_threshold"]),
        "strategy_mean_reversion_zscore_threshold": str(state["runtime_mean_threshold"]),
        "strategy_momentum_threshold_percent": str(state["runtime_momentum_threshold"]),
        "strategy_risk_fraction": str(state["runtime_risk_fraction"]),
        "strategy_target_volatility_percent": str(state["runtime_target_volatility"]),
        "strategy_edge_multiple": str(state["runtime_edge_multiple"]),
        "strategy_regime_filter": state["runtime_regime_filter"],
        "strategy_deeplob_window": int(state["runtime_deeplob_window"]),
        "strategy_deeplob_horizon": int(state["runtime_deeplob_horizon"]),
        "strategy_deeplob_training_window": int(state["runtime_deeplob_training_window"]),
        "strategy_deeplob_levels": int(state["runtime_deeplob_levels"]),
        "strategy_deeplob_threshold_percent": str(state["runtime_deeplob_threshold"]),
        "strategy_deeplob_min_quality_percent": str(state["runtime_deeplob_quality"]),
        "market_data_interval_seconds": int(state["runtime_interval"]),
        "min_profit_percent": str(state["runtime_min_profit"]),
        "taker_fee_percent": str(state["runtime_taker_fee"]),
        "maker_fee_percent": str(state["runtime_maker_fee"]),
        "estimated_slippage_bps": str(state["runtime_slippage"]),
        "paper_slippage_bps": str(state["runtime_paper_slippage"]),
        "paper_initial_jpy": str(state["runtime_paper_initial"]),
        "paper_min_confidence": str(state["runtime_paper_confidence"]),
        "paper_trailing_stop_percent": str(state["runtime_paper_stop"]),
        "paper_stop_loss_percent": str(state["runtime_paper_hard_stop"]),
        "collect_order_book": state["runtime_order_book"],
        "collect_executions": state["runtime_executions"],
        "executions_fetch_limit": int(state["runtime_execution_limit"]),
        "order_book_depth_levels": int(state["runtime_depth"]),
        "max_position_jpy": str(state["runtime_max_position"]),
        "max_order_jpy": str(state["runtime_max_order"]),
        "max_daily_loss_jpy": str(state["runtime_max_loss"]),
        "max_market_data_age_seconds": int(state["runtime_max_age"]),
        "dashboard_refresh_seconds": int(state["runtime_refresh"]),
        "portfolio_snapshot_interval_seconds": int(state["runtime_snapshot"]),
        "enable_automatic_trading": state["runtime_auto"],
        "live_trading_acknowledged": state["runtime_ack"],
        "trading_kill_switch": state["runtime_kill"],
        "live_confirmation": state["runtime_mode"] == "live" and state["runtime_ack"],
    }


def _apply_runtime_settings(client: ApiClient) -> None:
    """Persist one settings snapshot from the explicit Apply action."""
    try:
        client.patch("/api/v1/system/settings", json=_runtime_settings_patch())
        st.session_state["runtime_settings_message"] = (
            "Settings saved; worker will reschedule on its next cycle."
        )
        st.session_state.pop("runtime_settings_error", None)
    except ApiError as exc:
        st.session_state["runtime_settings_error"] = str(exc)
        st.session_state.pop("runtime_settings_message", None)


def _sidebar(settings: Settings, client: ApiClient) -> dict[str, Any]:
    settings = _effective_dashboard_settings(settings, client)
    with st.sidebar:
        st.header("Controls")
        page = st.radio(
            "Workspace",
            ["Trading Platform", "Deep RL Research"],
            key="workspace",
        )
        if page == "Deep RL Research":
            st.caption("Train and replay the local ArchetypeTrader research stack.")
            return {"page": page}
        period = st.selectbox(
            "Chart period",
            options=list(PERIOD_LABELS),
            format_func=lambda key: PERIOD_LABELS[key],
            index=list(PERIOD_LABELS).index("24h"),
            key="period",
        )
        chart_type = st.radio("Chart type", ["Line (LTP)", "Candlestick"], key="chart_type")
        interval = st.selectbox(
            "Candle interval",
            options=CANDLE_INTERVALS,
            index=CANDLE_INTERVALS.index(DEFAULT_INTERVAL_FOR_PERIOD[period]),
            key=f"interval_{period}",
            disabled=chart_type != "Candlestick",
        )
        strategy = st.selectbox(
            "Signal algorithm",
            options=["none", "ema_crossover", "mean_reversion", "momentum_regime", "deeplob"],
            format_func=lambda key: {
                "none": "Disabled",
                "ema_crossover": "EMA crossover",
                "mean_reversion": "Mean reversion",
                "momentum_regime": "Momentum regime",
                "deeplob": "DeepLOB-style order book",
            }[key],
            index=(
                ["none", "ema_crossover", "mean_reversion", "momentum_regime", "deeplob"].index(
                    settings.strategy_name
                )
                if settings.strategy_name
                in {"none", "ema_crossover", "mean_reversion", "momentum_regime", "deeplob"}
                else 0
            ),
            key="runtime_strategy",
        )
        st.caption("Runtime changes take effect after you press **Apply settings**.")
        st.divider()
        with st.expander("Runtime settings"):
            st.caption("Edit the controls, then press **Apply settings** to save them.")
            with st.container():
                st.selectbox(
                    "Trading mode",
                    ["data_only", "paper", "live"],
                    index=["data_only", "paper", "live"].index(settings.trading_mode.value),
                    key="runtime_mode",
                )
                st.number_input(
                    "Collection interval (seconds)",
                    min_value=5,
                    max_value=86400,
                    value=settings.market_data_interval_seconds,
                    step=5,
                    key="runtime_interval",
                )
                st.caption(
                    "Lower signal thresholds create more trades but also more noise. "
                    "Paper-test changes before relying on them."
                )
                st.number_input(
                    "EMA fast window",
                    min_value=2,
                    max_value=200,
                    value=settings.strategy_fast_window,
                    step=1,
                    key="runtime_fast_window",
                )
                st.number_input(
                    "EMA slow window",
                    min_value=3,
                    max_value=500,
                    value=settings.strategy_slow_window,
                    step=1,
                    key="runtime_slow_window",
                )
                st.number_input(
                    "Mean-reversion window",
                    min_value=5,
                    max_value=500,
                    value=settings.strategy_mean_reversion_window,
                    step=1,
                    key="runtime_mean_window",
                )
                st.number_input(
                    "Momentum window",
                    min_value=3,
                    max_value=500,
                    value=settings.strategy_momentum_window,
                    step=1,
                    key="runtime_momentum_window",
                )
                st.number_input(
                    "Volatility window",
                    min_value=5,
                    max_value=500,
                    value=settings.strategy_volatility_window,
                    step=1,
                    key="runtime_volatility_window",
                )
                st.number_input(
                    "EMA signal threshold (%)",
                    min_value=0.0001,
                    max_value=100.0,
                    value=float(settings.strategy_ema_threshold_percent),
                    step=0.01,
                    key="runtime_ema_threshold",
                )
                st.number_input(
                    "Mean-reversion z-score threshold",
                    min_value=0.01,
                    max_value=10.0,
                    value=float(settings.strategy_mean_reversion_zscore_threshold),
                    step=0.1,
                    key="runtime_mean_threshold",
                )
                st.number_input(
                    "Momentum signal threshold (%)",
                    min_value=0.0001,
                    max_value=100.0,
                    value=float(settings.strategy_momentum_threshold_percent),
                    step=0.01,
                    key="runtime_momentum_threshold",
                )
                st.number_input(
                    "Risk fraction",
                    min_value=0.01,
                    max_value=1.0,
                    value=float(settings.strategy_risk_fraction),
                    step=0.05,
                    key="runtime_risk_fraction",
                )
                st.number_input(
                    "Target volatility (%)",
                    min_value=0.01,
                    max_value=100.0,
                    value=float(settings.strategy_target_volatility_percent),
                    step=0.1,
                    key="runtime_target_volatility",
                )
                st.number_input(
                    "Signal noise-edge multiple",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(settings.strategy_edge_multiple),
                    step=0.1,
                    key="runtime_edge_multiple",
                    help=(
                        "Raises EMA/momentum thresholds above recent noise. "
                        "Set to 0 to disable this adaptive floor."
                    ),
                )
                st.checkbox(
                    "Enable regime filter",
                    value=settings.strategy_regime_filter,
                    key="runtime_regime_filter",
                    help=(
                        "Blocks trend entries in mean-reverting tapes and "
                        "mean-reversion entries in trending tapes."
                    ),
                )
                st.caption(
                    "DeepLOB uses recent order-book snapshots. Enable order-book collection "
                    "and wait for enough history before expecting a signal."
                )
                st.number_input(
                    "DeepLOB snapshot window",
                    min_value=8,
                    max_value=500,
                    value=settings.strategy_deeplob_window,
                    step=1,
                    key="runtime_deeplob_window",
                )
                st.number_input(
                    "DeepLOB forecast horizon",
                    min_value=1,
                    max_value=100,
                    value=settings.strategy_deeplob_horizon,
                    step=1,
                    key="runtime_deeplob_horizon",
                )
                st.number_input(
                    "DeepLOB training window",
                    min_value=32,
                    max_value=5000,
                    value=settings.strategy_deeplob_training_window,
                    step=8,
                    key="runtime_deeplob_training_window",
                )
                st.number_input(
                    "DeepLOB order-book levels",
                    min_value=1,
                    max_value=50,
                    value=settings.strategy_deeplob_levels,
                    step=1,
                    key="runtime_deeplob_levels",
                )
                st.number_input(
                    "DeepLOB signal threshold (%)",
                    min_value=0.0001,
                    max_value=100.0,
                    value=float(settings.strategy_deeplob_threshold_percent),
                    step=0.01,
                    key="runtime_deeplob_threshold",
                )
                st.number_input(
                    "DeepLOB minimum book quality (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(settings.strategy_deeplob_min_quality_percent),
                    step=5.0,
                    key="runtime_deeplob_quality",
                )
                st.number_input(
                    "Minimum net profit (%)",
                    min_value=0.0,
                    value=float(settings.min_profit_percent),
                    step=0.05,
                    key="runtime_min_profit",
                )
                st.number_input(
                    "Taker fee (%)",
                    min_value=0.0,
                    value=float(settings.taker_fee_percent),
                    step=0.01,
                    key="runtime_taker_fee",
                )
                st.number_input(
                    "Estimated slippage (bps)",
                    min_value=0.0,
                    value=float(settings.estimated_slippage_bps),
                    step=1.0,
                    key="runtime_slippage",
                )
                st.number_input(
                    "Maker fee (%)",
                    min_value=0.0,
                    value=float(settings.maker_fee_percent),
                    step=0.01,
                    key="runtime_maker_fee",
                )
                st.number_input(
                    "Paper slippage (bps)",
                    min_value=0.0,
                    value=float(settings.paper_slippage_bps),
                    step=1.0,
                    key="runtime_paper_slippage",
                )
                st.number_input(
                    "Paper starting JPY",
                    min_value=0.0,
                    value=float(settings.paper_initial_jpy),
                    step=1000.0,
                    key="runtime_paper_initial",
                )
                st.number_input(
                    "Paper minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(settings.paper_min_confidence),
                    step=0.05,
                    key="runtime_paper_confidence",
                )
                st.number_input(
                    "Paper trailing stop (%)",
                    min_value=0.0,
                    value=float(settings.paper_trailing_stop_percent),
                    step=0.05,
                    key="runtime_paper_stop",
                )
                st.number_input(
                    "Paper hard stop-loss (%)",
                    min_value=0.0,
                    value=float(settings.paper_stop_loss_percent),
                    step=0.05,
                    key="runtime_paper_hard_stop",
                    help=(
                        "Exit at a loss when the bid falls this far below the average entry price."
                    ),
                )
                st.checkbox(
                    "Collect order book",
                    value=settings.collect_order_book,
                    key="runtime_order_book",
                )
                st.checkbox(
                    "Collect executions",
                    value=settings.collect_executions,
                    key="runtime_executions",
                )
                st.number_input(
                    "Execution fetch limit",
                    min_value=1,
                    max_value=500,
                    value=settings.executions_fetch_limit,
                    step=1,
                    key="runtime_execution_limit",
                )
                st.number_input(
                    "Order-book depth levels",
                    min_value=1,
                    max_value=200,
                    value=settings.order_book_depth_levels,
                    step=1,
                    key="runtime_depth",
                )
                st.number_input(
                    "Maximum position (JPY)",
                    min_value=0.0,
                    value=float(settings.max_position_jpy),
                    step=1000.0,
                    key="runtime_max_position",
                )
                st.number_input(
                    "Maximum order (JPY)",
                    min_value=0.0,
                    value=float(settings.max_order_jpy),
                    step=1000.0,
                    key="runtime_max_order",
                )
                st.number_input(
                    "Maximum daily loss (JPY)",
                    min_value=0.0,
                    value=float(settings.max_daily_loss_jpy),
                    step=1000.0,
                    key="runtime_max_loss",
                )
                st.number_input(
                    "Maximum market-data age (seconds)",
                    min_value=1,
                    value=settings.max_market_data_age_seconds,
                    step=1,
                    key="runtime_max_age",
                )
                st.number_input(
                    "Dashboard refresh (seconds)",
                    min_value=5,
                    max_value=86400,
                    value=settings.dashboard_refresh_seconds,
                    step=5,
                    key="runtime_refresh",
                )
                st.number_input(
                    "Portfolio snapshot interval (seconds)",
                    min_value=30,
                    max_value=86400,
                    value=settings.portfolio_snapshot_interval_seconds,
                    step=30,
                    key="runtime_snapshot",
                )
                st.checkbox(
                    "Enable automatic trading",
                    value=settings.enable_automatic_trading,
                    key="runtime_auto",
                )
                st.checkbox(
                    "I acknowledge live trading",
                    value=settings.live_trading_acknowledged,
                    key="runtime_ack",
                )
                st.checkbox(
                    "Trading kill switch",
                    value=settings.trading_kill_switch,
                    key="runtime_kill",
                )
                st.button(
                    "Apply settings",
                    key="apply_runtime_settings",
                    on_click=_apply_runtime_settings,
                    args=(client,),
                    width="stretch",
                )
            if st.session_state.get("runtime_settings_message"):
                st.success(st.session_state["runtime_settings_message"])
            if st.session_state.get("runtime_settings_error"):
                st.error(st.session_state["runtime_settings_error"])
        st.caption(f"Auto-refresh every {settings.dashboard_refresh_seconds}s")
        if st.button("Collect now", width="stretch", help="One immediate collection"):
            _collect_now(client)
        if st.button("Refresh view", width="stretch"):
            st.rerun()
        st.divider()
        st.caption(f"API: {settings.api_base_url}")
        st.caption(f"Timezone: {settings.dashboard_timezone}")
    return {
        "page": page,
        "period": period,
        "interval": interval,
        "chart_type": chart_type,
        "strategy": strategy,
    }


def _collect_now(client: ApiClient) -> None:
    try:
        result = client.post("/api/v1/market/collect")
    except ApiError as exc:
        st.sidebar.error(str(exc))
        return
    if result.get("success"):
        degraded = result.get("degraded") or []
        message = f"Collected in {result.get('duration_ms')} ms."
        if degraded:
            message += f" Degraded: {', '.join(degraded)}."
        st.sidebar.success(message)
    else:
        st.sidebar.error("Collection failed — see the System section.")


def _fetch_all(client: ApiClient, period: str) -> dict[str, Any]:
    """One coherent read of everything the page needs."""
    return {
        "overview": client.get("/api/v1/market/overview", change_window=period),
        "status": client.get("/api/v1/system/status") or {},
        "trading": client.get("/api/v1/trading/status") or {},
        "portfolio": client.get("/api/v1/portfolio"),
        "signals": client.get("/api/v1/signals", limit=10) or [],
        "orders": client.get("/api/v1/orders", limit=25) or [],
        "fills": client.get("/api/v1/trades", limit=25) or [],
        "runs": client.get("/api/v1/system/collector", limit=20) or [],
        "events": client.get("/api/v1/system/events", limit=25) or [],
    }


def render_page(settings: Settings, client: ApiClient, controls: dict[str, Any]) -> None:
    tz = _timezone(settings.dashboard_timezone)
    period = controls["period"]
    try:
        data = _fetch_all(client, period)
    except ApiError as exc:
        st.error(f"Cannot reach the platform API.\n\n{exc}")
        st.caption(
            "Check that the API container is running and that `API_BASE_URL` points "
            "at it. The dashboard shows only stored data and never calls bitFlyer."
        )
        return

    rendered_at = utcnow()
    overview = data["overview"]
    status = data["status"]

    header, timing = st.columns([3, 2])
    with header:
        st.title("BTC/JPY Trading Platform")
        st.caption(
            f"{status.get('product_code', settings.product_code)} on "
            f"{status.get('exchange', settings.exchange)} · "
            f"environment `{status.get('app_env', settings.app_env)}`"
        )
    with timing:
        latest = _local((overview or {}).get("latest_timestamp"), tz)
        next_refresh = (
            rendered_at + timedelta(seconds=settings.dashboard_refresh_seconds)
        ).astimezone(tz)
        st.markdown(
            f"**Last market update:** {latest}  \n"
            f"**Page rendered:** {rendered_at.astimezone(tz):%Y-%m-%d %H:%M:%S}  \n"
            f"**Next refresh:** {next_refresh:%Y-%m-%d %H:%M:%S} ({tz.key})"
        )

    render_market_overview(overview, tz)
    st.divider()
    render_price_chart(client, period, controls["interval"], controls["chart_type"], tz)
    st.divider()
    render_portfolio(data["portfolio"], overview, client)
    st.divider()
    render_strategy(status or data["trading"], data["signals"], tz)
    st.divider()
    render_backtest(client, status or data["trading"])
    st.divider()
    render_trading(data["trading"], data["orders"], data["fills"], tz)
    st.divider()
    render_system(status, data["runs"], data["events"], tz)


def _rl_config_paths() -> list[Path]:
    return sorted(Path("conf").glob("*.yaml"))


def _rl_result(key: str, value: Any) -> None:
    st.session_state[key] = value


def render_rl_page() -> None:
    """Run the local ArchetypeTrader pipeline from a dedicated UI workspace."""
    st.title("Deep RL Research")
    st.caption(
        "Phase 1 learns archetypes, Phase 2 selects them with recurrent PPO/DQN, "
        "and Phase 3 adapts them. All training and replay remain paper-only."
    )

    config_paths = _rl_config_paths()
    if not config_paths:
        st.error("No RL configs found. Expected `conf/smoke.yaml` or `conf/btc_full.yaml`.")
        return

    selected_path = st.selectbox(
        "Experiment configuration",
        config_paths,
        format_func=str,
        key="rl_config_path",
    )
    try:
        from archetype_trader.config import load_config

        cfg = load_config(selected_path)
    except ModuleNotFoundError as exc:
        st.error(
            "The RL dependencies are not installed. Run "
            "`uv sync --locked --extra research --extra dashboard`."
        )
        st.caption(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - config errors belong in the page
        st.error(f"Invalid RL configuration: {exc}")
        return

    fold = st.number_input("Walk-forward fold", min_value=0, step=1, value=0, key="rl_fold")
    seed = st.number_input("Random seed", min_value=0, step=1, value=cfg.seed, key="rl_seed")
    st.info(
        f"Output: `{cfg.run_dir()}` · selector: `{cfg.phase2.algo}` · "
        f"device: `{cfg.device}` · live execution is unavailable"
    )
    if cfg.data.source != "synthetic" and cfg.data.path:
        data_path = Path(cfg.data.path)
        if not data_path.exists():
            st.warning(
                f"Input data is missing: `{data_path}`. Place the Parquet/CSV file under "
                "the host `data/` directory, then refresh the page."
            )

    try:
        from archetype_trader.pipeline import RunLayout

        layout = RunLayout.for_config(cfg)
    except ModuleNotFoundError as exc:
        st.error("Install the research extra before using the RL workspace.")
        st.caption(str(exc))
        return

    st.subheader("Pipeline")
    st.caption("Run stages in order. Training may take time and runs in the Streamlit process.")
    stage_columns = st.columns(5)
    stages = (
        ("Ingest", "rl_ingest"),
        ("Features", "rl_features"),
        ("Phase 1", "rl_phase1"),
        ("Phase 2", "rl_phase2"),
        ("Phase 3", "rl_phase3"),
    )
    for column, (label, key) in zip(stage_columns, stages, strict=True):
        if column.button(label, key=f"button_{key}", width="stretch"):
            try:
                from archetype_trader import pipeline

                with st.spinner(f"Running {label}…"):
                    if key == "rl_ingest":
                        result = pipeline.run_ingest(cfg)
                    elif key == "rl_features":
                        result = pipeline.run_features(cfg)
                    elif key == "rl_phase1":
                        result = pipeline.run_phase1(cfg, fold_index=int(fold), seed=int(seed))
                    elif key == "rl_phase2":
                        result = pipeline.run_phase2(cfg, fold_index=int(fold), seed=int(seed))
                    else:
                        result = pipeline.run_phase3(cfg, fold_index=int(fold), seed=int(seed))
                _rl_result("rl_last_message", f"{label} completed.")
                _rl_result("rl_last_result", result)
            except Exception as exc:  # noqa: BLE001 - show failed stage in UI
                _rl_result("rl_last_message", None)
                if isinstance(exc, FileNotFoundError):
                    st.error(
                        f"{label} failed because the configured input is missing: {exc}. "
                        "Use `conf/smoke.yaml` for synthetic data or add the configured "
                        "file under the host `data/` directory."
                    )
                else:
                    st.error(f"{label} failed: {exc}")
    if st.session_state.get("rl_last_message"):
        st.success(st.session_state["rl_last_message"])

    st.subheader("Checkpoints")
    checkpoint_rows = []
    for phase in (1, 2, 3):
        path = layout.checkpoint(phase, int(fold), int(seed))
        checkpoint_rows.append(
            {
                "Phase": phase,
                "Path": str(path),
                "Available": "yes" if path.exists() else "no",
            }
        )
    st.dataframe(pd.DataFrame(checkpoint_rows), width="stretch", hide_index=True)

    st.subheader("Evaluation")
    evaluate, paper = st.columns(2)
    if evaluate.button("Run walk-forward evaluation", key="rl_evaluate", width="stretch"):
        try:
            from archetype_trader.pipeline import run_evaluate

            with st.spinner("Running evaluation matrix…"):
                result = run_evaluate(cfg, make_plots=False)
            _rl_result("rl_evaluation", result)
            st.success("Evaluation completed.")
        except Exception as exc:  # noqa: BLE001 - show failed evaluation in UI
            st.error(f"Evaluation failed: {exc}")
    if paper.button("Replay paper test", key="rl_paper", width="stretch"):
        try:
            from archetype_trader.pipeline import run_paper

            with st.spinner("Replaying held-out test segment…"):
                result = run_paper(cfg, fold_index=int(fold), seed=int(seed), split="test")
            _rl_result("rl_paper_result", result)
            st.success("Paper replay completed.")
        except Exception as exc:  # noqa: BLE001 - show failed replay in UI
            st.error(f"Paper replay failed: {exc}")

    evaluation = st.session_state.get("rl_evaluation")
    if isinstance(evaluation, dict) and evaluation.get("report") is not None:
        report = evaluation["report"]
        st.markdown("#### Evaluation summary")
        selection = report.selection
        st.write(
            f"Selected on validation by `{selection.get('metric')}`: "
            f"`{selection.get('chosen') or 'none'}`"
        )
        if report.cells:
            st.dataframe(
                pd.DataFrame(report.table("test")), width="stretch", hide_index=True
            )
        verdict = report.verdict
        st.write(
            f"Robustness verdict: **{'accepted' if verdict.get('accepted') else 'rejected'}**"
        )

    paper_result = st.session_state.get("rl_paper_result")
    if isinstance(paper_result, dict) and paper_result.get("result") is not None:
        result = paper_result["result"]
        st.markdown("#### Paper replay summary")
        first, second, third = st.columns(3)
        first.metric("Bars", result.bars)
        second.metric("Final equity", f"{result.final_equity:,.2f}")
        third.metric("Orders", result.metrics.orders_submitted)
        st.json(result.agent_stats)


def _timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - any tz database problem must not blank the page
        st.warning(f"Unknown DASHBOARD_TIMEZONE {name!r}; falling back to UTC.")
        return ZoneInfo("UTC")


def main() -> None:
    settings = get_settings()
    st.set_page_config(
        page_title="BTC/JPY Trading Platform",
        page_icon="₿",
        layout="wide",
    )
    client = ApiClient(settings.api_base_url)
    effective_settings = _effective_dashboard_settings(settings, client)
    controls = _sidebar(effective_settings, client)

    if controls["page"] == "Deep RL Research":
        render_rl_page()
        return

    # A fragment reruns on its own timer, so the page refreshes without the user
    # touching anything and without restarting the whole script (§16).
    @st.fragment(run_every=effective_settings.dashboard_refresh_seconds)
    def _body() -> None:
        render_page(effective_settings, client, controls)

    _body()


if __name__ == "__main__":  # Streamlit executes the script as __main__.
    main()
