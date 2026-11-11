"""PostgreSQL-backed runtime configuration shared by API, dashboard and worker."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import RuntimeSetting
from btc_trader.monitoring.logging import get_logger
from btc_trader.strategy import STRATEGIES

logger = get_logger(__name__)

# Operational controls exposed in the dashboard. Credentials, URLs and logging
# configuration remain deployment-owned environment settings.
EDITABLE_SETTINGS = frozenset(
    {
        "trading_mode",
        "market_data_interval_seconds",
        "strategy_name",
        "strategy_trend_horizons",
        "strategy_volatility_window_days",
        "strategy_target_annual_volatility_percent",
        "strategy_max_exposure_percent",
        "strategy_rebalance_band_percent",
        "strategy_max_daily_bar_age_days",
        "daily_history_provider",
        "daily_history_auto_sync",
        "min_profit_percent",
        "taker_fee_percent",
        "estimated_slippage_bps",
        "paper_slippage_bps",
        "paper_initial_jpy",
        "max_position_jpy",
        "max_order_jpy",
        "max_daily_loss_jpy",
        "max_market_data_age_seconds",
        "enable_automatic_trading",
        "live_trading_acknowledged",
        "trading_kill_switch",
        "live_capital_jpy",
        "live_max_slippage_bps",
        "dashboard_refresh_seconds",
        "portfolio_snapshot_interval_seconds",
    }
)


async def get_runtime_settings(session: AsyncSession, base: Settings) -> Settings:
    rows = (await session.execute(select(RuntimeSetting))).scalars()
    overrides = {row.key: row.value for row in rows if row.key in EDITABLE_SETTINGS}
    if base.trading_kill_switch:
        # The deployment's kill switch is an emergency stop: a value saved from
        # the dashboard earlier must never switch it back off.
        overrides.pop("trading_kill_switch", None)
    stored_strategy = overrides.get("strategy_name")
    if stored_strategy is not None and str(stored_strategy).strip().lower() not in STRATEGIES:
        # A strategy saved by an older version no longer exists; fall back to
        # the deployment's configured strategy instead of failing every job.
        logger.warning("runtime_strategy_unknown", stored=stored_strategy)
        overrides.pop("strategy_name")
    if not overrides:
        return base
    try:
        return Settings.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        # E.g. live mode stored while the deployment no longer meets the live
        # gates: fall back to the deployment's own mode, never crash every job.
        logger.error("runtime_settings_invalid", error=str(exc).splitlines()[0])
        overrides.pop("trading_mode", None)
        try:
            return Settings.model_validate({**base.model_dump(), **overrides})
        except ValidationError:
            return base


def validate_runtime_patch(base: Settings, patch: dict[str, Any]) -> Settings:
    unknown = set(patch) - EDITABLE_SETTINGS
    if unknown:
        raise ValueError(f"settings are not dashboard-editable: {', '.join(sorted(unknown))}")
    if "strategy_name" in patch:
        strategy_name = str(patch["strategy_name"]).strip().lower()
        if strategy_name not in STRATEGIES:
            raise ValueError(
                f"unknown strategy {patch['strategy_name']!r}; expected one of {sorted(STRATEGIES)}"
            )
        patch = {**patch, "strategy_name": strategy_name}
    # Settings itself enforces the live preconditions (automatic trading,
    # acknowledgement, credentials). The kill switch is deliberately not one of
    # them: engaging it while live is the dashboard's emergency stop.
    return Settings.model_validate({**base.model_dump(), **patch})


async def update_runtime_settings(
    session: AsyncSession, base: Settings, patch: dict[str, Any]
) -> Settings:
    candidate = validate_runtime_patch(base, patch)
    serialised = candidate.model_dump(mode="json")
    for key, value in patch.items():
        value = serialised[key]
        row = await session.get(RuntimeSetting, key)
        if row is None:
            row = RuntimeSetting(key=key, value=value)
            session.add(row)
        else:
            row.value = value
    await session.flush()
    return candidate
