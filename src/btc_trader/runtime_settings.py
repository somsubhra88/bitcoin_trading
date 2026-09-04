"""PostgreSQL-backed runtime configuration shared by API, dashboard and worker."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import RuntimeSetting
from btc_trader.enums import TradingMode
from btc_trader.strategy import STRATEGIES

# Operational controls exposed in the dashboard. Credentials, URLs and logging
# configuration remain deployment-owned environment settings.
EDITABLE_SETTINGS = frozenset(
    {
        "trading_mode",
        "market_data_interval_seconds",
        "collect_order_book",
        "collect_executions",
        "executions_fetch_limit",
        "order_book_depth_levels",
        "strategy_name",
        "strategy_fast_window",
        "strategy_slow_window",
        "strategy_mean_reversion_window",
        "strategy_momentum_window",
        "strategy_volatility_window",
        "strategy_ema_threshold_percent",
        "strategy_mean_reversion_zscore_threshold",
        "strategy_momentum_threshold_percent",
        "strategy_risk_fraction",
        "strategy_target_volatility_percent",
        "strategy_edge_multiple",
        "strategy_regime_filter",
        "strategy_deeplob_window",
        "strategy_deeplob_horizon",
        "strategy_deeplob_training_window",
        "strategy_deeplob_levels",
        "strategy_deeplob_threshold_percent",
        "strategy_deeplob_min_quality_percent",
        "min_profit_percent",
        "taker_fee_percent",
        "maker_fee_percent",
        "estimated_slippage_bps",
        "paper_slippage_bps",
        "paper_initial_jpy",
        "paper_min_confidence",
        "paper_trailing_stop_percent",
        "paper_stop_loss_percent",
        "max_position_jpy",
        "max_order_jpy",
        "max_daily_loss_jpy",
        "max_market_data_age_seconds",
        "enable_automatic_trading",
        "live_trading_acknowledged",
        "trading_kill_switch",
        "dashboard_refresh_seconds",
        "portfolio_snapshot_interval_seconds",
    }
)


async def get_runtime_settings(session: AsyncSession, base: Settings) -> Settings:
    rows = (await session.execute(select(RuntimeSetting))).scalars()
    overrides = {row.key: row.value for row in rows if row.key in EDITABLE_SETTINGS}
    return Settings.model_validate({**base.model_dump(), **overrides}) if overrides else base


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
    candidate = Settings.model_validate({**base.model_dump(), **patch})
    if candidate.trading_mode is TradingMode.LIVE and not candidate.live_orders_allowed:
        raise ValueError(
            "live mode requires automatic trading, acknowledgement, kill switch off, "
            "and credentials"
        )
    # A round trip pays two taker fees; estimated slippage is expressed in bps.
    minimum_net_cost = candidate.taker_fee_percent * 2 + candidate.estimated_slippage_bps / 100
    if candidate.min_profit_percent < minimum_net_cost:
        raise ValueError("min_profit_percent must cover round-trip taker fees and slippage")
    return candidate


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
