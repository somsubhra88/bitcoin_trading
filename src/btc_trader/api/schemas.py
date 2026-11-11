"""Pydantic response models for the HTTP API.

``Decimal`` fields serialise to JSON *strings*, which is deliberate: a JSON
number would be read back by most clients as a binary float and silently lose
precision on JPY prices (PROJECT.md §13).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class _Response(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @field_serializer("*", when_used="json")
    def _decimals_as_strings(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return format(value, "f")
        return value


class TickResponse(_Response):
    exchange: str
    product_code: str
    exchange_timestamp: datetime
    received_at: datetime
    tick_id: int | None = None
    state: str | None = None
    ltp: Decimal | None = None
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    best_bid_size: Decimal | None = None
    best_ask_size: Decimal | None = None
    total_bid_depth: Decimal | None = None
    total_ask_depth: Decimal | None = None
    volume: Decimal | None = None
    volume_by_product: Decimal | None = None


class MarketOverviewResponse(_Response):
    product_code: str
    exchange: str | None
    latest_timestamp: datetime | None
    data_age_seconds: float | None
    is_stale: bool
    ltp: Decimal | None
    best_bid: Decimal | None
    best_ask: Decimal | None
    spread: Decimal | None
    spread_percent: Decimal | None
    volume_24h: Decimal | None
    change_absolute: Decimal | None
    change_percent: Decimal | None
    change_window: str
    stored_tick_count: int
    first_stored_timestamp: datetime | None


class HistoryResponse(_Response):
    product_code: str
    period: str
    count: int
    ticks: list[TickResponse]


class CandleResponse(_Response):
    bucket_start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    samples: int


class CandlesResponse(_Response):
    """Bars folded from ticker samples: display only, not exchange OHLC (§40)."""

    product_code: str
    interval: str
    count: int
    candles: list[CandleResponse]


class CollectionResponse(_Response):
    success: bool
    trigger: str
    product_code: str
    exchange: str
    duration_ms: int
    tick_inserted: bool
    tick_duplicate: bool
    ltp: Decimal | None
    exchange_timestamp: datetime | None


class CollectorStatusResponse(_Response):
    last_run_at: datetime | None
    last_run_success: bool | None
    last_successful_run_at: datetime | None
    last_run_duration_ms: int | None
    last_error_type: str | None
    last_error_message: str | None
    interval_seconds: int
    seconds_since_last_success: float | None


class CollectorRunResponse(_Response):
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    exchange: str
    product_code: str
    trigger: str
    success: bool
    ticks_inserted: int
    error_type: str | None
    error_message: str | None


class SystemEventResponse(_Response):
    timestamp: datetime
    severity: str
    component: str
    event_type: str
    message: str
    event_metadata: dict[str, Any] | None = None


class HealthResponse(_Response):
    status: str
    database: str
    scheduler: str
    exchange_public_api: str
    details: dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(_Response):
    app_env: str
    trading_mode: str
    product_code: str
    exchange: str
    strategy_name: str
    automatic_trading_enabled: bool
    live_orders_allowed: bool
    kill_switch: bool
    market_data_interval_seconds: int
    dashboard_refresh_seconds: int
    stored_tick_count: int
    collector: CollectorStatusResponse
    health: HealthResponse
    scheduler_jobs: list[dict[str, Any]]
    last_error: SystemEventResponse | None


class SignalResponse(_Response):
    strategy_name: str
    strategy_version: str
    product_code: str
    timestamp: datetime
    signal: str
    confidence: Decimal | None
    reference_price: Decimal | None
    reason: str | None
    metadata_json: dict[str, Any] | None


class FillResponse(_Response):
    order_id: int
    exchange_execution_id: str | None
    price: Decimal
    size: Decimal
    commission: Decimal
    executed_at: datetime


class OrderResponse(_Response):
    client_order_id: str
    exchange_order_id: str | None
    exchange: str
    product_code: str
    side: str
    order_type: str
    requested_price: Decimal | None
    requested_size: Decimal
    filled_size: Decimal
    average_fill_price: Decimal | None
    status: str
    strategy_name: str | None
    exit_reason: str | None
    trading_mode: str
    created_at: datetime
    submitted_at: datetime | None
    completed_at: datetime | None
    rejection_reason: str | None


class PositionResponse(_Response):
    product_code: str
    trading_mode: str
    btc_quantity: Decimal
    average_entry_price: Decimal | None
    cost_basis_jpy: Decimal
    realized_pnl_jpy: Decimal
    unrealized_pnl_jpy: Decimal
    updated_at: datetime


class PortfolioSnapshotResponse(_Response):
    timestamp: datetime
    trading_mode: str
    jpy_balance: Decimal
    btc_balance: Decimal
    btc_market_price: Decimal | None
    btc_value_jpy: Decimal
    total_value_jpy: Decimal
    realized_pnl_jpy: Decimal
    unrealized_pnl_jpy: Decimal


class PortfolioResponse(_Response):
    product_code: str
    trading_mode: str
    position: PositionResponse | None
    latest_snapshot: PortfolioSnapshotResponse | None


class PaperResetRequest(BaseModel):
    """Explicit confirmation required before clearing simulated execution state."""

    confirm: bool = False


class PaperResetResponse(_Response):
    product_code: str
    trading_mode: str
    orders_deleted: int
    fills_deleted: int
    positions_deleted: int
    snapshots_deleted: int
    signals_preserved: bool
    message: str


class BacktestTradeResponse(_Response):
    day: date
    side: str
    price: Decimal
    size: Decimal
    fee: Decimal
    exposure_before: Decimal
    target_exposure: Decimal
    exit_reason: str | None


class BacktestMetricsResponse(_Response):
    name: str
    start: date
    end: date
    days: int
    initial_equity_jpy: Decimal
    final_equity_jpy: Decimal
    net_profit_jpy: Decimal
    total_return_percent: float
    cagr_percent: float
    annual_volatility_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_percent: float
    calmar_ratio: float
    trade_count: int
    turnover_per_year: float
    fees_paid_jpy: Decimal
    average_exposure_percent: float
    time_in_market_percent: float


class PeriodReturnResponse(_Response):
    year: int
    strategy_return_percent: float
    benchmark_return_percent: float
    strategy_max_drawdown_percent: float
    benchmark_max_drawdown_percent: float


class CostStressResponse(_Response):
    cost_multiplier: float
    cagr_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    fees_paid_jpy: Decimal


class BacktestResponse(_Response):
    strategy_name: str
    metrics: BacktestMetricsResponse
    benchmark: BacktestMetricsResponse = Field(
        description="Buy-and-hold over the same days, through the same cost model."
    )
    periods: list[PeriodReturnResponse]
    cost_stress: list[CostStressResponse]
    parameters: dict[str, Any]
    trades: list[BacktestTradeResponse]
    equity_curve: list[dict[str, str]]


class SleeveResponse(_Response):
    horizon: int
    active: bool
    warming_up: bool
    stop: float | None
    channel_high: float | None
    channel_low: float | None
    entered: bool
    exited: bool


class DailyHistoryResponse(_Response):
    product_code: str
    provider: str
    bars: int
    first_day: date | None
    last_day: date | None
    own_bitflyer_days: int
    external_days: int
    basis_ratio: Decimal | None
    required_bars: int = Field(description="Bars the strategy needs before every sleeve is live.")


class DailyHistorySyncResponse(_Response):
    provider: str
    external_bars_inserted: int
    own_bars_inserted: int
    history: DailyHistoryResponse


class StrategyPreviewResponse(_Response):
    strategy_name: str
    ready: bool
    message: str
    as_of: date | None = None
    bar_age_days: int | None = None
    stale: bool = False
    close: Decimal | None = None
    target_exposure: Decimal | None = None
    previous_target_exposure: Decimal | None = None
    current_exposure: Decimal | None = Field(
        default=None, description="Paper holding as a share of paper equity, at the bid."
    )
    active_fraction: Decimal | None = None
    realized_volatility: float | None = None
    target_volatility: float | None = None
    volatility_scale: float | None = None
    signal: str | None = None
    reason: str | None = None
    sleeves: list[SleeveResponse] = Field(default_factory=list)
    history: DailyHistoryResponse


class TradingStatusResponse(_Response):
    trading_mode: str
    strategy_name: str
    automatic_trading_enabled: bool
    live_trading_acknowledged: bool
    kill_switch: bool
    live_orders_allowed: bool
    credentials_configured: bool
    min_profit_percent: Decimal
    max_order_jpy: Decimal
    max_position_jpy: Decimal
    max_daily_loss_jpy: Decimal
    max_market_data_age_seconds: int
    min_order_size_btc: Decimal
    order_size_step_btc: Decimal
    enforce_exchange_minimums: bool
    live_capital_jpy: Decimal
    live_max_slippage_bps: Decimal
    alerts_configured: bool


class ConfigResponse(_Response):
    """Effective configuration with secrets removed."""

    config: dict[str, Any]


class SettingsUpdateRequest(BaseModel):
    """Dashboard-editable settings. Omitted fields retain their current value."""

    trading_mode: str | None = None
    market_data_interval_seconds: int | None = Field(default=None, ge=5, le=86_400)
    strategy_name: str | None = None
    strategy_trend_horizons: str | None = None
    strategy_volatility_window_days: int | None = Field(default=None, ge=10, le=365)
    strategy_target_annual_volatility_percent: Decimal | None = Field(default=None, gt=0, le=200)
    strategy_max_exposure_percent: Decimal | None = Field(default=None, gt=0, le=100)
    strategy_rebalance_band_percent: Decimal | None = Field(default=None, ge=0, le=100)
    strategy_max_daily_bar_age_days: int | None = Field(default=None, ge=1, le=30)
    daily_history_provider: str | None = None
    daily_history_auto_sync: bool | None = None
    min_profit_percent: Decimal | None = Field(default=None, ge=0)
    taker_fee_percent: Decimal | None = Field(default=None, ge=0)
    estimated_slippage_bps: Decimal | None = Field(default=None, ge=0)
    paper_slippage_bps: Decimal | None = Field(default=None, ge=0)
    paper_initial_jpy: Decimal | None = Field(default=None, ge=0)
    max_position_jpy: Decimal | None = Field(default=None, ge=0)
    max_order_jpy: Decimal | None = Field(default=None, ge=0)
    max_daily_loss_jpy: Decimal | None = Field(default=None, ge=0)
    max_market_data_age_seconds: int | None = Field(default=None, ge=1)
    enable_automatic_trading: bool | None = None
    live_trading_acknowledged: bool | None = None
    trading_kill_switch: bool | None = None
    live_capital_jpy: Decimal | None = Field(default=None, ge=0)
    live_max_slippage_bps: Decimal | None = Field(default=None, gt=0, le=500)
    dashboard_refresh_seconds: int | None = Field(default=None, ge=5, le=86_400)
    portfolio_snapshot_interval_seconds: int | None = Field(default=None, ge=30, le=86_400)
    live_confirmation: bool = False

    def patch(self) -> dict[str, Any]:
        # With ``Depends()`` FastAPI supplies every optional query parameter;
        # ``None`` therefore means omitted and must not overwrite a setting.
        values = {
            key: value
            for key, value in self.model_dump(
                exclude_unset=True, exclude={"live_confirmation"}, mode="json"
            ).items()
            if value is not None
        }
        if self.live_confirmation:
            values["live_trading_acknowledged"] = True
        return values
