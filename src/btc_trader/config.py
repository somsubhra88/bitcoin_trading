"""Application configuration.

Every configurable behaviour of the platform is declared here and sourced from
environment variables (optionally via a local ``.env``).  Nothing in the code
base may hardcode an operational constant that belongs in this module.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from btc_trader.enums import TradingMode

__all__ = ["Settings", "TradingMode", "get_settings", "reload_settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --- trading mode ------------------------------------------------------
    trading_mode: TradingMode = TradingMode.DATA_ONLY

    # --- exchange ----------------------------------------------------------
    exchange: Literal["bitflyer"] = "bitflyer"
    product_code: str = "BTC_JPY"
    bitflyer_public_base_url: str = "https://api.bitflyer.com"
    bitflyer_timeout_seconds: float = Field(default=10.0, gt=0)
    bitflyer_max_retries: int = Field(default=4, ge=0, le=10)
    bitflyer_rate_limit_requests: int = Field(default=400, gt=0)
    bitflyer_rate_limit_window_seconds: float = Field(default=300.0, gt=0)

    bitflyer_api_key: SecretStr = SecretStr("")
    bitflyer_api_secret: SecretStr = SecretStr("")

    # --- market data collection -------------------------------------------
    market_data_interval_seconds: int = Field(default=300, ge=5, le=86_400)
    collect_order_book: bool = True
    collect_executions: bool = True
    executions_fetch_limit: int = Field(default=100, ge=1, le=500)
    order_book_depth_levels: int = Field(default=10, ge=1, le=200)

    # --- database ----------------------------------------------------------
    # Compose injects PostgreSQL explicitly; SQLite is a credential-free local
    # fallback when the application is run without an environment file.
    database_url: str = "sqlite+aiosqlite:///./data/btc_trader.db"
    db_echo: bool = False

    # --- strategy ----------------------------------------------------------
    strategy_name: str = "none"
    strategy_fast_window: int = Field(default=8, ge=2, le=200)
    strategy_slow_window: int = Field(default=21, ge=3, le=500)
    strategy_mean_reversion_window: int = Field(default=20, ge=5, le=500)
    strategy_momentum_window: int = Field(default=12, ge=3, le=500)
    strategy_volatility_window: int = Field(default=20, ge=5, le=500)
    strategy_ema_threshold_percent: Decimal = Field(default=Decimal("0.05"), gt=0, le=100)
    strategy_mean_reversion_zscore_threshold: Decimal = Field(default=Decimal("1.0"), gt=0, le=10)
    strategy_momentum_threshold_percent: Decimal = Field(default=Decimal("0.01"), gt=0, le=100)
    strategy_risk_fraction: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)
    #: Target *daily* return volatility. Observed per-bar volatility is scaled by
    #: ``sqrt(bars_per_day)`` before comparison, so this number is in the units a
    #: human reasons about ("I want ~2% daily swings") rather than per-bar units.
    strategy_target_volatility_percent: Decimal = Field(default=Decimal("2.0"), gt=0, le=100)
    #: Required signal edge as a multiple of recent baseline volatility. 0
    #: disables the volatility floor and uses the configured signal thresholds.
    strategy_edge_multiple: Decimal = Field(default=Decimal("1.5"), ge=0, le=100)
    #: Gate trend strategies in mean-reverting tapes and mean-reversion strategies
    #: in trending tapes, measured over the volatility window.
    strategy_regime_filter: bool = True
    strategy_deeplob_window: int = Field(default=32, ge=8, le=500)
    strategy_deeplob_horizon: int = Field(default=3, ge=1, le=100)
    strategy_deeplob_training_window: int = Field(default=256, ge=32, le=5000)
    strategy_deeplob_levels: int = Field(default=10, ge=1, le=50)
    strategy_deeplob_threshold_percent: Decimal = Field(default=Decimal("0.01"), gt=0, le=100)
    strategy_deeplob_min_quality_percent: Decimal = Field(default=Decimal("80"), gt=0, le=100)
    strategy_deeplob_ridge_lambda: Decimal = Field(default=Decimal("0"), ge=0)
    strategy_order_flow_window: int = Field(default=20, ge=3, le=500)
    strategy_order_flow_threshold: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)
    strategy_breakout_window: int = Field(default=24, ge=3, le=500)
    strategy_microprice_threshold_bps: Decimal = Field(default=Decimal("1.5"), gt=0, le=1000)

    # --- profitability -----------------------------------------------------
    min_profit_percent: Decimal = Field(default=Decimal("1.0"), ge=0)
    taker_fee_percent: Decimal = Field(default=Decimal("0.15"), ge=0)
    maker_fee_percent: Decimal = Field(default=Decimal("0.15"), ge=0)
    estimated_slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    paper_slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    #: Starting balance for a fresh paper account. The default is sized so that a
    #: single order clears bitFlyer's 0.001 BTC minimum at realistic BTC/JPY
    #: prices — at ¥10,000 every order would be rejected as sub-minimum.
    paper_initial_jpy: Decimal = Field(default=Decimal("1000000"), ge=0)
    paper_min_confidence: Decimal = Field(default=Decimal("0.35"), ge=0, le=1)
    paper_trailing_stop_percent: Decimal = Field(default=Decimal("0.8"), ge=0)
    #: Hard stop measured from the entry price, independent of the trailing stop.
    #: Without it a zero trailing stop leaves a position with no downside exit.
    paper_stop_loss_percent: Decimal = Field(default=Decimal("1.5"), ge=0)
    #: Bars to wait after an exit before the same strategy may re-enter.
    paper_cooldown_bars: int = Field(default=1, ge=0, le=10_000)
    #: Force an exit after this many bars in a position; 0 means "no time limit".
    paper_max_holding_bars: int = Field(default=0, ge=0, le=100_000)

    # --- execution model ---------------------------------------------------
    #: ``taker`` crosses the spread and pays ``taker_fee_percent`` plus slippage.
    #: ``maker`` rests at the touch and pays ``maker_fee_percent`` with no
    #: slippage, but only fills when the market later comes to the resting price
    #: within ``maker_timeout_bars`` — which is also how adverse selection enters
    #: the simulation instead of being assumed away.
    execution_style: Literal["taker", "maker"] = "taker"
    maker_timeout_bars: int = Field(default=2, ge=1, le=1000)
    #: Exchange lot constraints. bitFlyer BTC_JPY trades in 0.001 BTC minimum with
    #: a 0.00000001 increment and integer JPY prices; a simulated fill that ignores
    #: these is not reproducible live.
    min_order_size_btc: Decimal = Field(default=Decimal("0.001"), ge=0)
    order_size_step_btc: Decimal = Field(default=Decimal("0.00000001"), gt=0)
    price_tick_jpy: Decimal = Field(default=Decimal("1"), gt=0)
    min_order_notional_jpy: Decimal = Field(default=Decimal("0"), ge=0)
    #: When true, an order smaller than the exchange minimum is rejected rather
    #: than filled. Turning it off makes small test accounts trade, at the cost of
    #: results that could not be reproduced on the real exchange.
    enforce_exchange_minimums: bool = True

    # --- strategy ensemble (adaptive allocation) ---------------------------
    #: Evaluate every strategy on every tick as an independent shadow ledger and
    #: allocate real (paper) capital to the winner via Thompson sampling.
    enable_strategy_ensemble: bool = False
    ensemble_prior_strength: Decimal = Field(default=Decimal("4"), gt=0, le=1000)
    ensemble_half_life_trades: Decimal = Field(default=Decimal("25"), gt=0, le=10_000)
    ensemble_min_trades: int = Field(default=3, ge=0, le=1000)

    # --- risk limits -------------------------------------------------------
    max_position_jpy: Decimal = Field(default=Decimal("500000"), ge=0)
    max_order_jpy: Decimal = Field(default=Decimal("300000"), ge=0)
    #: Realised loss that halts new entries for the rest of the trading day.
    #: ``0`` disables the limit — use ``TRADING_KILL_SWITCH`` to stop trading.
    max_daily_loss_jpy: Decimal = Field(default=Decimal("30000"), ge=0)
    max_market_data_age_seconds: int = Field(default=600, ge=1)
    #: Timezone whose midnight starts a new risk day for the daily loss limit.
    risk_day_timezone: str = "Asia/Tokyo"

    # --- trading safety ----------------------------------------------------
    enable_automatic_trading: bool = False
    live_trading_acknowledged: bool = False
    trading_kill_switch: bool = False

    # --- api / dashboard ---------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - bound inside container network
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://api:8000"
    dashboard_port: int = Field(default=8501, ge=1, le=65535)
    dashboard_refresh_seconds: int = Field(default=300, ge=5, le=86_400)
    dashboard_timezone: str = "Asia/Tokyo"

    # --- scheduler ---------------------------------------------------------
    run_scheduler_in_api: bool = False
    portfolio_snapshot_interval_seconds: int = Field(default=900, ge=30, le=86_400)

    # ------------------------------------------------------------------ #
    # validators
    # ------------------------------------------------------------------ #
    @field_validator("product_code")
    @classmethod
    def _upper_product_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("bitflyer_public_base_url", "api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("database_url")
    @classmethod
    def _require_database_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("DATABASE_URL must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_live_trading_preconditions(self) -> Settings:
        """Live mode must be configured deliberately (PROJECT.md §26)."""
        if self.strategy_fast_window >= self.strategy_slow_window:
            raise ValueError("STRATEGY_FAST_WINDOW must be smaller than STRATEGY_SLOW_WINDOW")
        if self.trading_mode is TradingMode.LIVE:
            missing = [
                name
                for name, enabled in (
                    ("ENABLE_AUTOMATIC_TRADING", self.enable_automatic_trading),
                    ("LIVE_TRADING_ACKNOWLEDGED", self.live_trading_acknowledged),
                )
                if not enabled
            ]
            if missing:
                raise ValueError(
                    "TRADING_MODE=live requires " + " and ".join(f"{m}=true" for m in missing)
                )
            if not (
                self.bitflyer_api_key.get_secret_value()
                and self.bitflyer_api_secret.get_secret_value()
            ):
                raise ValueError("TRADING_MODE=live requires bitFlyer API credentials")
            if not self.live_execution_implemented:
                raise ValueError(
                    "TRADING_MODE=live is unavailable: real order submission is not implemented"
                )
        if self.min_profit_percent < self.round_trip_cost_percent:
            raise ValueError(
                "MIN_PROFIT_PERCENT must cover round-trip taker fees and estimated slippage "
                f"({self.round_trip_cost_percent}% at EXECUTION_STYLE={self.execution_style})"
            )
        if self.min_order_size_btc > 0 and self.order_size_step_btc > self.min_order_size_btc:
            raise ValueError("ORDER_SIZE_STEP_BTC must not exceed MIN_ORDER_SIZE_BTC")
        _validate_timezone(self.risk_day_timezone, "RISK_DAY_TIMEZONE")
        return self

    # ------------------------------------------------------------------ #
    # derived properties
    # ------------------------------------------------------------------ #
    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_private_credentials(self) -> bool:
        return bool(
            self.bitflyer_api_key.get_secret_value() and self.bitflyer_api_secret.get_secret_value()
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def live_orders_allowed(self) -> bool:
        """The *only* place that answers "may a real order be sent?"."""
        return (
            self.live_execution_implemented
            and
            self.trading_mode is TradingMode.LIVE
            and self.enable_automatic_trading
            and self.live_trading_acknowledged
            and not self.trading_kill_switch
            and self.has_private_credentials
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def live_execution_implemented(self) -> bool:
        """Whether the configured exchange adapter can submit real orders."""
        return False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def strategies_enabled(self) -> bool:
        return self.trading_mode in (TradingMode.PAPER, TradingMode.LIVE)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def entry_fee_percent(self) -> Decimal:
        """Fee actually charged on a fill, given the configured execution style."""
        return self.maker_fee_percent if self.execution_style == "maker" else self.taker_fee_percent

    @computed_field  # type: ignore[prop-decorator]
    @property
    def slippage_bps(self) -> Decimal:
        """Price concession per fill. A resting maker order concedes none."""
        return Decimal("0") if self.execution_style == "maker" else self.paper_slippage_bps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def round_trip_cost_percent(self) -> Decimal:
        """Total cost of one entry plus one exit, as a percent of notional.

        This is the number every profitability decision is measured against: a
        strategy whose average winner is smaller than this cannot make money no
        matter how good its hit rate looks.
        """
        fee = self.entry_fee_percent * 2
        slippage = (
            Decimal("0")
            if self.execution_style == "maker"
            else self.estimated_slippage_bps / Decimal("100")
        )
        return fee + slippage

    @property
    def bars_per_day(self) -> Decimal:
        """Observations per day at the configured collection interval."""
        return Decimal(86_400) / Decimal(self.market_data_interval_seconds)

    @property
    def bars_per_year(self) -> Decimal:
        return self.bars_per_day * Decimal(365)

    @property
    def async_database_url(self) -> str:
        """DATABASE_URL normalised to an asyncio-capable driver."""
        return _normalise_database_url(self.database_url, async_driver=True)

    @property
    def sync_database_url(self) -> str:
        """DATABASE_URL normalised to a blocking driver (Alembic, scripts)."""
        return _normalise_database_url(self.database_url, async_driver=False)

    def safe_dump(self) -> dict[str, object]:
        """Configuration snapshot with secrets removed (safe to log/serve)."""
        data = self.model_dump(mode="json", exclude={"bitflyer_api_key", "bitflyer_api_secret"})
        data["bitflyer_credentials_configured"] = self.has_private_credentials
        data["database_backend"] = self.database_url.split(":", 1)[0]
        data.pop("database_url", None)
        return data


_ASYNC_DRIVERS = {"postgresql": "psycopg", "sqlite": "aiosqlite"}
_SYNC_DRIVERS = {"postgresql": "psycopg", "sqlite": "pysqlite"}


def _validate_timezone(name: str, label: str) -> None:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"{label} is not a known IANA timezone: {name!r}") from exc


def _normalise_database_url(url: str, *, async_driver: bool) -> str:
    """Swap the SQLAlchemy driver so the same URL works sync and async.

    ``postgresql+psycopg`` (psycopg3) supports both modes, so only the SQLite
    driver actually changes — but normalising both keeps callers honest.
    """
    scheme, _, remainder = url.partition("://")
    backend = scheme.partition("+")[0]
    table = _ASYNC_DRIVERS if async_driver else _SYNC_DRIVERS
    if backend not in table:
        # Unknown backend: hand the URL back untouched rather than guessing.
        return url
    return f"{backend}+{table[backend]}://{remainder}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache and re-read the environment (tests, SIGHUP)."""
    get_settings.cache_clear()
    return get_settings()
