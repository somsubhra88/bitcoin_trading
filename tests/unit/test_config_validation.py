"""Configuration validation (PROJECT.md §7, §46 "configuration validation")."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from btc_trader.config import Settings


@pytest.mark.usefixtures("env")
def test_product_code_is_normalised() -> None:
    assert Settings(product_code=" btc_jpy ").product_code == "BTC_JPY"


@pytest.mark.usefixtures("env")
def test_base_urls_lose_their_trailing_slash() -> None:
    settings = Settings(
        bitflyer_public_base_url="https://api.bitflyer.com/",
        api_base_url="http://api:8000/",
    )
    assert settings.bitflyer_public_base_url == "https://api.bitflyer.com"
    assert settings.api_base_url == "http://api:8000"


@pytest.mark.usefixtures("env")
def test_empty_database_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL must not be empty"):
        Settings(database_url="   ")


@pytest.mark.usefixtures("env")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("market_data_interval_seconds", 1),  # below the 5s floor
        ("market_data_interval_seconds", 100_000),  # above the 1-day ceiling
        ("live_max_slippage_bps", 0),
        ("live_capital_jpy", -1),
        ("bitflyer_timeout_seconds", 0),
        ("bitflyer_max_retries", -1),
        ("bitflyer_max_retries", 11),
        ("max_market_data_age_seconds", 0),
        ("dashboard_refresh_seconds", 1),
        ("api_port", 0),
        ("api_port", 70_000),
        ("min_profit_percent", -1),
        ("max_order_jpy", -1),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})  # type: ignore[arg-type]


@pytest.mark.usefixtures("env")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trading_mode", "casino"),
        ("log_level", "TRACE"),
        ("log_format", "xml"),
        ("exchange", "kraken"),  # only bitflyer is implemented
        ("app_env", "qa"),
    ],
)
def test_unknown_enumerated_values_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})  # type: ignore[arg-type]


@pytest.mark.usefixtures("env")
def test_money_settings_are_decimals_not_floats() -> None:
    """§13: a configured fee must not arrive as a binary float."""
    settings = Settings(taker_fee_percent="0.15", min_profit_percent="1.05")
    assert str(settings.taker_fee_percent) == "0.15"
    assert str(settings.min_profit_percent) == "1.05"


@pytest.mark.usefixtures("env")
@pytest.mark.parametrize(
    ("url", "expected_async", "expected_sync"),
    [
        (
            "postgresql://u:p@db:5432/x",
            "postgresql+psycopg://u:p@db:5432/x",
            "postgresql+psycopg://u:p@db:5432/x",
        ),
        (
            "sqlite:///./data/x.db",
            "sqlite+aiosqlite:///./data/x.db",
            "sqlite+pysqlite:///./data/x.db",
        ),
        (
            "sqlite+aiosqlite:///./data/x.db",
            "sqlite+aiosqlite:///./data/x.db",
            "sqlite+pysqlite:///./data/x.db",
        ),
    ],
)
def test_database_url_driver_is_normalised(
    url: str, expected_async: str, expected_sync: str
) -> None:
    settings = Settings(database_url=url)
    assert settings.async_database_url == expected_async
    assert settings.sync_database_url == expected_sync


@pytest.mark.usefixtures("env")
def test_unknown_database_backend_is_passed_through_untouched() -> None:
    settings = Settings(database_url="duckdb:///./x.duckdb")
    assert settings.async_database_url == "duckdb:///./x.duckdb"


def test_environment_overrides_the_dotenv_file(settings: Settings) -> None:
    """The suite's pinned environment must win over the developer's ``.env``.

    If this ever fails, tests could pick up real credentials (PROJECT.md §51.7).
    """
    assert settings.has_private_credentials is False
    assert settings.database_url.startswith("sqlite+aiosqlite:///")
