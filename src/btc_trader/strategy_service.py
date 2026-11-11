"""Strategy evaluation without granting any order authority.

The strategy decides a *target exposure* from completed daily bars; this module
records that decision as a signal and hands it to the paper ledger or to live
execution, which apply risk limits and costs. One signal row is written per strategy, daily
bar and parameter set; later evaluations of the same day reuse it, so the
audit trail shows one decision per day plus what the ledger did about it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import Order, StrategySignal
from btc_trader.enums import ExitReason, TradingMode
from btc_trader.exchange.base import ExchangeClient
from btc_trader.live import rebalance_live
from btc_trader.market_data.daily import DailySeries, load_daily_series, sync_own_daily_bars
from btc_trader.paper import rebalance_paper
from btc_trader.strategy import (
    STRATEGY_VERSION,
    TrendDecision,
    TrendParameters,
    evaluate_trend,
    parameters_from_settings,
)
from btc_trader.utils.timeutils import utcnow


@dataclass(frozen=True, slots=True)
class StrategyPreview:
    series: DailySeries
    parameters: TrendParameters
    decision: TrendDecision | None
    bar_age_days: int | None
    stale: bool


async def preview_strategy(session: AsyncSession, settings: Settings) -> StrategyPreview:
    """Current trend decision from stored data; writes only derived daily bars."""
    await sync_own_daily_bars(session, settings)
    series = await load_daily_series(session, settings)
    parameters = parameters_from_settings(settings)
    decision = evaluate_trend(series.bars, parameters)
    age = (utcnow().date() - decision.as_of).days if decision else None
    return StrategyPreview(
        series=series,
        parameters=parameters,
        decision=decision,
        bar_age_days=age,
        stale=age is not None and age > settings.strategy_max_daily_bar_age_days,
    )


async def evaluate_and_store(
    session: AsyncSession,
    settings: Settings,
    *,
    exchange_client: ExchangeClient | None = None,
) -> StrategySignal | None:
    """Evaluate the strategy and hand the decision to the paper or live ledger.

    Live mode needs ``exchange_client``; the worker passes its long-lived one.
    """
    # The database-backed effective settings are the cross-process source of
    # truth across API and worker processes.
    if settings.strategy_name == "none" or not settings.strategies_enabled:
        return None
    preview = await preview_strategy(session, settings)
    decision = preview.decision
    if decision is None:
        return None

    fingerprint = _fingerprint(preview.parameters)
    signal = await _signal_for_day(session, settings, decision, fingerprint)
    if signal is None:
        signal = StrategySignal(
            strategy_name=settings.strategy_name,
            strategy_version=STRATEGY_VERSION,
            product_code=settings.product_code,
            timestamp=utcnow(),
            signal=decision.signal,
            confidence=decision.confidence,
            reference_price=decision.close,
            reason=decision.reason,
            metadata_json={
                **decision.metadata,
                "parameters_fingerprint": fingerprint,
                "history": {
                    "bars": len(preview.series.bars),
                    "own_bitflyer_days": preview.series.own_days,
                    "external_days": preview.series.external_days,
                    "basis_ratio": (
                        float(preview.series.basis_ratio)
                        if preview.series.basis_ratio is not None
                        else None
                    ),
                },
            },
        )
        session.add(signal)
        await session.flush()

    if preview.stale:
        signal.metadata_json = {
            **(signal.metadata_json or {}),
            "execution": {
                "status": "blocked_stale_daily_bar",
                "detail": (
                    f"newest daily close is {preview.bar_age_days} days old "
                    f"(limit {settings.strategy_max_daily_bar_age_days})"
                ),
            },
        }
        return signal

    # Bypass the band only until the ledger has acted on a trend change; a
    # blocked or unfilled attempt (stale tick, IOC with no fill) retries with it.
    previous_outcome = ((signal.metadata_json or {}).get("execution") or {}).get("status")
    filled_orders = (
        await session.execute(
            select(func.count())
            .select_from(Order)
            .where(Order.signal_id == signal.id, Order.filled_size > 0)
        )
    ).scalar_one()
    force = decision.sleeves_changed and not (filled_orders or previous_outcome == "no_trade")
    reason = (
        ExitReason.TRAILING_STOP
        if any(sleeve.exited for sleeve in decision.sleeves)
        else ExitReason.RISK_EXIT
    )
    if settings.trading_mode is TradingMode.LIVE:
        if exchange_client is None:
            raise ValueError("live evaluation needs an exchange client")
        await rebalance_live(
            session,
            settings,
            signal,
            decision.target_exposure,
            client=exchange_client,
            force=force,
            reduction_reason=reason,
        )
    else:
        await rebalance_paper(
            session,
            settings,
            signal,
            decision.target_exposure,
            force=force,
            reduction_reason=reason,
        )
    return signal


async def _signal_for_day(
    session: AsyncSession, settings: Settings, decision: TrendDecision, fingerprint: str
) -> StrategySignal | None:
    latest = (
        await session.execute(
            select(StrategySignal)
            .where(
                StrategySignal.strategy_name == settings.strategy_name,
                StrategySignal.product_code == settings.product_code,
            )
            .order_by(StrategySignal.timestamp.desc(), StrategySignal.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None:
        return None
    metadata = latest.metadata_json or {}
    # Same day, same parameters and same answer: a backfill that changes the
    # history (and therefore the decision) gets a fresh, auditable row.
    if (
        metadata.get("as_of") == decision.as_of.isoformat()
        and metadata.get("parameters_fingerprint") == fingerprint
        and metadata.get("target_exposure") == float(decision.target_exposure)
    ):
        return latest
    return None


def _fingerprint(parameters: TrendParameters) -> str:
    payload = json.dumps(
        {
            "horizons": parameters.horizons,
            "volatility_window": parameters.volatility_window,
            "target_volatility": parameters.target_volatility,
            "max_exposure": parameters.max_exposure,
            "version": STRATEGY_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


__all__ = ["StrategyPreview", "evaluate_and_store", "preview_strategy"]
