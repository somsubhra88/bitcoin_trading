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

    # --- database ----------------------------------------------------------
    # Compose injects PostgreSQL explicitly; SQLite is a credential-free local
    # fallback when the application is run without an environment file.
    database_url: str = "sqlite+aiosqlite:///./data/btc_trader.db"
    db_echo: bool = False

    # --- strategy ----------------------------------------------------------
    #: ``none`` or ``trend_ensemble`` (see ``btc_trader.strategy``).
    strategy_name: str = "none"
    #: Donchian lookbacks in daily bars. Each horizon is one equally weighted
    #: sleeve; the defaults follow Zarattini, Pagani & Barbon (2025).
    strategy_trend_horizons: str = "5,10,20,30,60,90,150,250,360"
    #: Daily bars used to estimate realised volatility for position sizing.
    strategy_volatility_window_days: int = Field(default=90, ge=10, le=365)
    #: Annualised volatility the BTC holding is scaled to. BTC's own realised
    #: volatility is typically 40-80%, so 25% means a sleeve holds roughly
    #: 30-60% of its capital while its trend is up.
    strategy_target_annual_volatility_percent: Decimal = Field(default=Decimal("25"), gt=0, le=200)
    #: Spot account: the BTC holding can never exceed 100% of equity.
    strategy_max_exposure_percent: Decimal = Field(default=Decimal("100"), gt=0, le=100)
    #: Volatility-only resizes are skipped unless they change the holding by at
    #: least this percentage of the current holding. Trend entries and exits are
    #: never delayed by the band.
    strategy_rebalance_band_percent: Decimal = Field(default=Decimal("20"), ge=0, le=100)
    #: Refuse to act when the newest completed daily close is older than this.
    strategy_max_daily_bar_age_days: int = Field(default=2, ge=1, le=30)

    # --- daily history ----------------------------------------------------
    #: External provider for long daily BTC/JPY history (bitFlyer's public API
    #: has no candle endpoint). ``none`` relies on the collector's own ticks.
    daily_history_provider: Literal["bitbank", "none"] = "bitbank"
    daily_history_base_url: str = "https://public.bitbank.cc"
    daily_history_pair: str = "btc_jpy"
    #: Calendar years fetched on the first sync (the ensemble needs >= 360 days).
    daily_history_years: int = Field(default=5, ge=1, le=15)
    #: Let the worker refresh daily history and materialise its own daily bars.
    daily_history_auto_sync: bool = True

    # --- profitability -----------------------------------------------------
    min_profit_percent: Decimal = Field(default=Decimal("1.0"), ge=0)
    taker_fee_percent: Decimal = Field(default=Decimal("0.15"), ge=0)
    estimated_slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    paper_slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    #: Starting balance for a fresh paper account. The default is sized so that a
    #: single order clears bitFlyer's 0.001 BTC minimum at realistic BTC/JPY
    #: prices — at ¥10,000 every order would be rejected as sub-minimum.
    paper_initial_jpy: Decimal = Field(default=Decimal("1000000"), ge=0)

    # --- exchange lot constraints -------------------------------------------
    #: bitFlyer BTC_JPY trades in 0.001 BTC minimum with a 0.00000001 increment;
    #: a simulated fill that ignores these is not reproducible live.
    min_order_size_btc: Decimal = Field(default=Decimal("0.001"), ge=0)
    order_size_step_btc: Decimal = Field(default=Decimal("0.00000001"), gt=0)
    min_order_notional_jpy: Decimal = Field(default=Decimal("0"), ge=0)
    #: When true, an order smaller than the exchange minimum is rejected rather
    #: than filled. Turning it off makes small test accounts trade, at the cost of
    #: results that could not be reproduced on the real exchange.
    enforce_exchange_minimums: bool = True

    # --- risk limits -------------------------------------------------------
    #: Largest BTC holding, valued at the bid. ``0`` disables the cap.
    max_position_jpy: Decimal = Field(default=Decimal("500000"), ge=0)
    #: Largest single buy. Sells reduce risk and are never blocked by it.
    max_order_jpy: Decimal = Field(default=Decimal("300000"), ge=0)
    #: Realised loss that halts new buys for the rest of the trading day.
    #: ``0`` disables the limit — use ``TRADING_KILL_SWITCH`` to stop trading.
    max_daily_loss_jpy: Decimal = Field(default=Decimal("30000"), ge=0)
    max_market_data_age_seconds: int = Field(default=600, ge=1)
    #: Timezone whose midnight starts a new risk day for the daily loss limit.
    risk_day_timezone: str = "Asia/Tokyo"

    # --- trading safety ----------------------------------------------------
    enable_automatic_trading: bool = False
    live_trading_acknowledged: bool = False
    trading_kill_switch: bool = False

    # --- live execution ----------------------------------------------------
    #: Equity (JPY) the bot may manage. ``0`` blocks live orders until it is
    #: set. JPY above the cap is never spent; **all BTC in the account counts
    #: as the bot's**, so keep unrelated BTC in another account.
    live_capital_jpy: Decimal = Field(default=Decimal("0"), ge=0)
    #: Orders are marketable limit IOC orders priced this far through the touch,
    #: so a thin or crashing book cannot fill them at an arbitrary price.
    live_max_slippage_bps: Decimal = Field(default=Decimal("30"), gt=0, le=500)
    #: How long to wait for an accepted order to reach a final state.
    live_order_poll_seconds: int = Field(default=20, ge=1, le=300)
    #: Failed/rejected orders within 24 hours that halt live trading.
    live_max_failed_orders_per_day: int = Field(default=3, ge=1, le=100)

    # --- alerts --------------------------------------------------------------
    #: Slack- or Discord-compatible incoming webhook; empty disables alerts.
    alert_webhook_url: SecretStr = SecretStr("")

    # --- api / dashboard ---------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - bound inside container network
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://api:8000"
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

    @field_validator("bitflyer_public_base_url", "api_base_url", "daily_history_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("strategy_name")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        name = value.strip().lower()
        if name not in STRATEGY_NAMES:
            raise ValueError(
                f"STRATEGY_NAME must be one of {sorted(STRATEGY_NAMES)}, got {value!r}"
            )
        return name

    @field_validator("strategy_trend_horizons")
    @classmethod
    def _valid_horizons(cls, value: str) -> str:
        parse_horizons(value)
        return value

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
        if self.min_profit_percent < self.round_trip_cost_percent:
            raise ValueError(
                "MIN_PROFIT_PERCENT must cover round-trip taker fees and estimated slippage "
                f"({self.round_trip_cost_percent}%)"
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
            self.trading_mode is TradingMode.LIVE
            and self.enable_automatic_trading
            and self.live_trading_acknowledged
            and not self.trading_kill_switch
            and self.has_private_credentials
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def strategies_enabled(self) -> bool:
        return self.trading_mode in (TradingMode.PAPER, TradingMode.LIVE)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def round_trip_cost_percent(self) -> Decimal:
        """Total cost of one entry plus one exit, as a percent of notional.

        This is the number every profitability decision is measured against: a
        strategy whose average winner is smaller than this cannot make money no
        matter how good its hit rate looks.
        """
        return self.taker_fee_percent * 2 + self.estimated_slippage_bps / Decimal("100")

    @property
    def trend_horizons(self) -> tuple[int, ...]:
        return parse_horizons(self.strategy_trend_horizons)

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
        data = self.model_dump(
            mode="json",
            exclude={"bitflyer_api_key", "bitflyer_api_secret", "alert_webhook_url"},
        )
        data["bitflyer_credentials_configured"] = self.has_private_credentials
        data["alerts_configured"] = bool(self.alert_webhook_url.get_secret_value())
        data["database_backend"] = self.database_url.split(":", 1)[0]
        data.pop("database_url", None)
        return data


_ASYNC_DRIVERS = {"postgresql": "psycopg", "sqlite": "aiosqlite"}
_SYNC_DRIVERS = {"postgresql": "psycopg", "sqlite": "pysqlite"}

#: Selectable strategies. Kept here (not in ``strategy``) so configuration can
#: validate names without importing the numerical code.
STRATEGY_NAMES = frozenset({"none", "trend_ensemble"})


def parse_horizons(value: str) -> tuple[int, ...]:
    """``"5,10,20"`` -> ``(5, 10, 20)``: sorted, unique, each between 2 and 1000 days."""
    try:
        horizons = sorted({int(part) for part in value.replace(" ", "").split(",") if part})
    except ValueError as exc:
        raise ValueError(f"trend horizons must be comma-separated integers: {value!r}") from exc
    if not horizons:
        raise ValueError("at least one trend horizon is required")
    if horizons[0] < 2 or horizons[-1] > 1000:
        raise ValueError("trend horizons must be between 2 and 1000 daily bars")
    return tuple(horizons)


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
