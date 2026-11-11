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

STRATEGY_LABELS = {
    "none": "Disabled",
    "trend_ensemble": "Trend ensemble (daily, volatility-targeted)",
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
    st.caption(
        "Bars are folded from sampled ticker snapshots, not exchange trades: highs and "
        "lows are approximations (§40)."
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


def render_strategy(
    status: dict[str, Any],
    signals: list[dict[str, Any]],
    preview: dict[str, Any] | None,
    client: ApiClient,
    tz: ZoneInfo,
) -> None:
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
    sixth.metric("Sleeves long", _fraction((latest or {}).get("confidence")))

    if latest is None:
        st.caption("No signals recorded. Select an algorithm and press Apply settings.")
    elif latest.get("reason"):
        st.caption(f"Reason: {latest['reason']}")
        execution = (latest.get("metadata_json") or {}).get("execution") or {}
        if execution:
            st.caption(f"Execution: {execution.get('status')} — {execution.get('detail')}")

    _render_trend_preview(preview, client)


def _render_trend_preview(preview: dict[str, Any] | None, client: ApiClient) -> None:
    """Daily trend state; available in every mode so it can be inspected first."""
    st.markdown("#### Trend ensemble (daily)")
    history = (preview or {}).get("history") or {}
    message = st.session_state.pop("daily_history_message", None)
    if message:
        st.success(message)
    left, right = st.columns([3, 1])
    left.caption(
        f"Daily bars: {history.get('bars', 0):,} "
        f"({history.get('first_day') or '—'} → {history.get('last_day') or '—'}); "
        f"{history.get('own_bitflyer_days', 0):,} from bitFlyer ticks, "
        f"{history.get('external_days', 0):,} from {history.get('provider', 'provider')}. "
        f"The strategy needs {history.get('required_bars', 360)} for every sleeve."
    )
    if right.button("Load daily history", key="sync_daily_history", width="stretch"):
        try:
            result = client.post("/api/v1/market/daily-history/sync")
            st.session_state["daily_history_message"] = (
                f"Stored {result.get('external_bars_inserted', 0)} provider bars and "
                f"{result.get('own_bars_inserted', 0)} bitFlyer bars."
            )
        except ApiError as exc:
            st.error(str(exc))
            return
        st.rerun()

    if not preview:
        st.info("Trend preview unavailable.")
        return
    if not preview.get("ready"):
        st.info(preview.get("message") or "Not enough daily history yet.")
        return
    if preview.get("stale"):
        st.warning(preview.get("message"))

    first, second, third, fourth = st.columns(4)
    first.metric(
        "Target BTC exposure",
        _fraction(preview.get("target_exposure")),
        delta=_exposure_delta(preview),
        delta_color="off",
    )
    second.metric(
        "Paper exposure",
        _fraction(preview.get("current_exposure"))
        if preview.get("current_exposure") is not None
        else "—",
    )
    third.metric(
        "Realised volatility",
        f"{float(preview['realized_volatility']):.0%}"
        if preview.get("realized_volatility") is not None
        else "—",
        help="Annualised, from the last daily returns of the volatility window.",
    )
    fourth.metric("As of (UTC close)", str(preview.get("as_of") or "—"))
    st.caption(preview.get("reason") or "")
    sleeves = preview.get("sleeves") or []
    if sleeves:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Horizon": f"{sleeve['horizon']}d",
                        "State": (
                            "warming up"
                            if sleeve.get("warming_up")
                            else "LONG"
                            if sleeve.get("active")
                            else "flat"
                        ),
                        "Event": (
                            "entered"
                            if sleeve.get("entered")
                            else "stopped out"
                            if sleeve.get("exited")
                            else ""
                        ),
                        "Trailing stop": _money(sleeve.get("stop")),
                        "Channel high": _money(sleeve.get("channel_high")),
                        "Channel low": _money(sleeve.get("channel_low")),
                    }
                    for sleeve in sleeves
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _fraction(value: Any) -> str:
    amount = _decimal(value)
    return "—" if amount is None else f"{amount:.1%}"


def _exposure_delta(preview: dict[str, Any]) -> str | None:
    target = _decimal(preview.get("target_exposure"))
    previous = _decimal(preview.get("previous_target_exposure"))
    if target is None or previous is None or target == previous:
        return None
    return f"{target - previous:+.1%} vs previous day"


def render_backtest(client: ApiClient, status: dict[str, Any]) -> None:
    """Run historical validation on demand so normal refreshes stay cheap."""
    st.subheader("Backtest")
    st.caption(
        "Daily replay of the trend ensemble through the paper ledger's sizing, fees and "
        "slippage, compared with buy-and-hold over the same days. Decisions use a day's "
        "close and fill at the next day's open."
    )
    first, second, third = st.columns(3)
    cost = first.selectbox(
        "Cost assumption",
        options=[0.5, 1.0, 2.0, 4.0],
        index=1,
        format_func=lambda value: f"{value:g}x fees and slippage",
        key="backtest_cost",
    )
    limits = second.checkbox(
        "Apply JPY order/position limits",
        value=False,
        key="backtest_limits",
        help="Off measures the strategy; on also applies MAX_ORDER_JPY / MAX_POSITION_JPY.",
    )
    if third.button("Run backtest", key="run_backtest", width="stretch"):
        try:
            st.session_state["backtest_result"] = client.get(
                "/api/v1/backtest",
                strategy_name="trend_ensemble",
                cost_multiplier=cost,
                apply_jpy_limits=str(limits).lower(),
            )
            st.session_state.pop("backtest_error", None)
        except ApiError as exc:
            st.session_state["backtest_error"] = str(exc)

    if st.session_state.get("backtest_error"):
        st.error(st.session_state["backtest_error"])
    result = st.session_state.get("backtest_result")
    if not (isinstance(result, dict) and isinstance(result.get("metrics"), dict)):
        return
    metrics, benchmark = result["metrics"], result.get("benchmark") or {}
    st.caption(f"{metrics.get('start')} → {metrics.get('end')} ({metrics.get('days', 0):,} days)")
    st.dataframe(
        pd.DataFrame(
            [_metric_row("Trend ensemble", metrics), _metric_row("Buy & hold", benchmark)]
        ),
        width="stretch",
        hide_index=True,
    )
    curve = result.get("equity_curve") or []
    if curve:
        frame = pd.DataFrame(
            {
                "date": [point["date"] for point in curve],
                "Trend ensemble": [_as_float(point["equity_jpy"]) for point in curve],
                "Buy & hold": [_as_float(point["benchmark_equity_jpy"]) for point in curve],
            }
        )
        figure = go.Figure()
        for column in ("Trend ensemble", "Buy & hold"):
            figure.add_trace(go.Scatter(x=frame["date"], y=frame[column], name=column))
        figure.update_layout(
            height=360,
            yaxis_type="log",
            yaxis_title="Equity (JPY, log scale)",
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        st.plotly_chart(figure, width="stretch")
    periods = result.get("periods") or []
    if periods:
        st.markdown("**Calendar years**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Year": period["year"],
                        "Strategy": f"{period['strategy_return_percent']:+.1f}%",
                        "Buy & hold": f"{period['benchmark_return_percent']:+.1f}%",
                        "Strategy max DD": f"{period['strategy_max_drawdown_percent']:.1f}%",
                        "Buy & hold max DD": f"{period['benchmark_max_drawdown_percent']:.1f}%",
                    }
                    for period in periods
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    stress = result.get("cost_stress") or []
    if stress:
        st.markdown("**Cost stress** — the edge must survive higher friction")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Costs": f"{item['cost_multiplier']:g}x",
                        "CAGR": f"{item['cagr_percent']:+.1f}%",
                        "Sharpe": f"{item['sharpe_ratio']:.2f}",
                        "Max drawdown": f"{item['max_drawdown_percent']:.1f}%",
                    }
                    for item in stress
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Past performance is a simulation, not a forecast. Treat results as a check that the "
        "rules behave as intended, not as an expected return."
    )


def _metric_row(label: str, metrics: dict[str, Any]) -> dict[str, str]:
    def number(key: str, pattern: str) -> str:
        value = metrics.get(key)
        return "—" if value is None else pattern.format(float(value))

    return {
        "": label,
        "Total return": number("total_return_percent", "{:+,.1f}%"),
        "CAGR": number("cagr_percent", "{:+.1f}%"),
        "Volatility": number("annual_volatility_percent", "{:.1f}%"),
        "Sharpe": number("sharpe_ratio", "{:.2f}"),
        "Sortino": number("sortino_ratio", "{:.2f}"),
        "Max drawdown": number("max_drawdown_percent", "{:.1f}%"),
        "Calmar": number("calmar_ratio", "{:.2f}"),
        "Avg exposure": number("average_exposure_percent", "{:.0f}%"),
        "Trades": str(metrics.get("trade_count", 0)),
        "Fees": _money(metrics.get("fees_paid_jpy")),
    }


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
            "🔴 LIVE TRADING — the worker places real bitFlyer orders with up to "
            f"¥{float(_decimal(trading.get('live_capital_jpy')) or 0):,.0f} of capital. "
            "Engage the kill switch to stop new orders."
        )
    elif mode == "live":
        st.warning(
            "Live mode is selected but a safety gate is closed (automatic trading, "
            "acknowledgement, credentials or kill switch); no orders are sent."
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


_OPEN_STATUSES = {"SUBMITTING", "SUBMITTED", "UNKNOWN"}


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

    fifth, sixth, seventh = st.columns(3)
    fifth.metric("Collector last run", _local(collector.get("last_run_at"), tz))
    sixth.metric("Last successful run", _local(collector.get("last_successful_run_at"), tz))
    seventh.metric("Stored ticks", f"{status.get('stored_tick_count', 0):,}")

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
        "strategy_trend_horizons": str(state["runtime_trend_horizons"]),
        "strategy_volatility_window_days": int(state["runtime_volatility_window"]),
        "strategy_target_annual_volatility_percent": str(state["runtime_target_volatility"]),
        "strategy_max_exposure_percent": str(state["runtime_max_exposure"]),
        "strategy_rebalance_band_percent": str(state["runtime_rebalance_band"]),
        "market_data_interval_seconds": int(state["runtime_interval"]),
        "min_profit_percent": str(state["runtime_min_profit"]),
        "taker_fee_percent": str(state["runtime_taker_fee"]),
        "estimated_slippage_bps": str(state["runtime_slippage"]),
        "paper_slippage_bps": str(state["runtime_paper_slippage"]),
        "paper_initial_jpy": str(state["runtime_paper_initial"]),
        "live_capital_jpy": str(state["runtime_live_capital"]),
        "live_max_slippage_bps": str(state["runtime_live_slippage"]),
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
            options=list(STRATEGY_LABELS),
            format_func=lambda key: STRATEGY_LABELS[key],
            index=(
                list(STRATEGY_LABELS).index(settings.strategy_name)
                if settings.strategy_name in STRATEGY_LABELS
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
                    "Trend ensemble: one sleeve per Donchian lookback (daily bars). A sleeve "
                    "is long after a new N-day closing high until a close below its "
                    "trailing midpoint stop."
                )
                st.text_input(
                    "Trend lookbacks (days)",
                    value=settings.strategy_trend_horizons,
                    key="runtime_trend_horizons",
                )
                st.number_input(
                    "Volatility window (days)",
                    min_value=10,
                    max_value=365,
                    value=settings.strategy_volatility_window_days,
                    step=5,
                    key="runtime_volatility_window",
                )
                st.number_input(
                    "Target annual volatility (%)",
                    min_value=1.0,
                    max_value=200.0,
                    value=float(settings.strategy_target_annual_volatility_percent),
                    step=5.0,
                    key="runtime_target_volatility",
                    help=(
                        "Each long sleeve holds target / realised volatility of its share. "
                        "Higher means more BTC exposure and deeper drawdowns."
                    ),
                )
                st.number_input(
                    "Maximum BTC exposure (% of equity)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(settings.strategy_max_exposure_percent),
                    step=5.0,
                    key="runtime_max_exposure",
                )
                st.number_input(
                    "Rebalance band (% of holding)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(settings.strategy_rebalance_band_percent),
                    step=5.0,
                    key="runtime_rebalance_band",
                    help="Volatility-only resizes smaller than this are skipped to save fees.",
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
                    "Live capital (JPY)",
                    min_value=0.0,
                    value=float(settings.live_capital_jpy),
                    step=10000.0,
                    key="runtime_live_capital",
                    help=(
                        "JPY the bot may manage in live mode; 0 blocks live orders. "
                        "All BTC in the account counts as the bot's."
                    ),
                )
                st.number_input(
                    "Live max slippage (bps)",
                    min_value=1.0,
                    max_value=500.0,
                    value=float(settings.live_max_slippage_bps),
                    step=5.0,
                    key="runtime_live_slippage",
                    help="Limit price distance through the touch for live IOC orders.",
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
        "preview": _optional(client, "/api/v1/strategy/preview"),
    }


def _optional(client: ApiClient, path: str) -> Any:
    """A panel-level read: its failure must not blank the rest of the page."""
    try:
        return client.get(path)
    except ApiError:
        return None


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
    render_strategy(status or data["trading"], data["signals"], data["preview"], client, tz)
    st.divider()
    render_backtest(client, status or data["trading"])
    st.divider()
    render_trading(data["trading"], data["orders"], data["fills"], tz)
    st.divider()
    render_system(status, data["runs"], data["events"], tz)


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

    # A fragment reruns on its own timer, so the page refreshes without the user
    # touching anything and without restarting the whole script (§16).
    @st.fragment(run_every=effective_settings.dashboard_refresh_seconds)
    def _body() -> None:
        render_page(effective_settings, client, controls)

    _body()


if __name__ == "__main__":  # Streamlit executes the script as __main__.
    main()
