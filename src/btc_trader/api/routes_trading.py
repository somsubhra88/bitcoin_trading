"""Trading signals and safety-gated strategy controls.

These are projections of the strategy and portfolio tables. Paper evaluation
also records simulated fills and portfolio state; no endpoint can submit a real
exchange order.

There is deliberately **no** order-placement endpoint: live orders are sent only
by the worker's strategy job, so nothing exposed over HTTP can move money
(PROJECT.md §34, §56).
"""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from btc_trader.api.deps import SessionDep, SettingsDep
from btc_trader.api.routes_market import daily_history_summary
from btc_trader.api.schemas import (
    BacktestMetricsResponse,
    BacktestResponse,
    BacktestTradeResponse,
    CostStressResponse,
    FillResponse,
    OrderResponse,
    PaperResetRequest,
    PaperResetResponse,
    PeriodReturnResponse,
    PortfolioResponse,
    PortfolioSnapshotResponse,
    PositionResponse,
    SignalResponse,
    SleeveResponse,
    StrategyPreviewResponse,
    TradingStatusResponse,
)
from btc_trader.backtesting import run_backtest
from btc_trader.db.models import (
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    StrategySignal,
)
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.enums import OrderStatus, Severity, TradingMode
from btc_trader.market_data.daily import load_daily_series, sync_own_daily_bars
from btc_trader.paper import current_exposure, reset_paper_trading
from btc_trader.runtime_settings import get_runtime_settings
from btc_trader.strategy_service import evaluate_and_store, preview_strategy

OPEN_ORDER_STATES = [state for state in OrderStatus if state.is_open]

router = APIRouter(prefix="/api/v1", tags=["trading"])


@router.get("/signals", response_model=list[SignalResponse], summary="Recent strategy signals")
async def signals(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[SignalResponse]:
    statement = select(StrategySignal).order_by(StrategySignal.timestamp.desc()).limit(limit)
    rows = (await session.execute(statement)).scalars()
    return [SignalResponse.model_validate(row) for row in rows]


@router.get("/orders", response_model=list[OrderResponse], summary="Recent orders")
async def orders(
    session: SessionDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    open_only: bool = False,
) -> list[OrderResponse]:
    settings = await get_runtime_settings(session, settings)
    statement = select(Order).order_by(Order.created_at.desc()).limit(limit)
    statement = statement.where(
        Order.product_code == settings.product_code,
        Order.trading_mode == settings.trading_mode,
    )
    if open_only:
        statement = statement.where(Order.status.in_(OPEN_ORDER_STATES))
    rows = (await session.execute(statement)).scalars()
    return [OrderResponse.model_validate(row) for row in rows]


@router.get("/trades", response_model=list[FillResponse], summary="Recent fills")
async def trades(
    session: SessionDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[FillResponse]:
    settings = await get_runtime_settings(session, settings)
    statement = (
        select(Fill)
        .join(Fill.order)
        .where(
            Order.product_code == settings.product_code,
            Order.trading_mode == settings.trading_mode,
        )
        .order_by(Fill.executed_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(statement)).scalars()
    return [FillResponse.model_validate(row) for row in rows]


@router.get("/portfolio", response_model=PortfolioResponse, summary="Portfolio state")
async def portfolio(session: SessionDep, settings: SettingsDep) -> PortfolioResponse:
    settings = await get_runtime_settings(session, settings)
    position_statement = select(Position).where(
        Position.product_code == settings.product_code,
        Position.trading_mode == settings.trading_mode,
    )
    position = (await session.execute(position_statement)).scalar_one_or_none()

    snapshot_statement = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.trading_mode == settings.trading_mode)
        .order_by(PortfolioSnapshot.timestamp.desc())
        .limit(1)
    )
    snapshot = (await session.execute(snapshot_statement)).scalar_one_or_none()

    return PortfolioResponse(
        product_code=settings.product_code,
        trading_mode=settings.trading_mode.value,
        position=PositionResponse.model_validate(position) if position else None,
        latest_snapshot=(PortfolioSnapshotResponse.model_validate(snapshot) if snapshot else None),
    )


@router.post(
    "/trading/paper/reset",
    response_model=PaperResetResponse,
    summary="Reset the paper-trading ledger",
)
async def reset_paper_ledger(
    request: PaperResetRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> PaperResetResponse:
    """Start a fresh simulated account while retaining market history/signals."""
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to reset the paper-trading ledger.",
        )

    settings = await get_runtime_settings(session, settings)
    result = await reset_paper_trading(session, settings)
    await SystemEventRepository(session).record(
        severity=Severity.INFO,
        component="paper",
        event_type="paper_reset",
        message=(
            f"Reset paper ledger for {result.product_code}: "
            f"{result.orders_deleted} orders, {result.fills_deleted} fills, "
            f"{result.positions_deleted} positions, {result.snapshots_deleted} snapshots deleted."
        ),
        metadata={
            "product_code": result.product_code,
            "orders_deleted": result.orders_deleted,
            "fills_deleted": result.fills_deleted,
            "positions_deleted": result.positions_deleted,
            "snapshots_deleted": result.snapshots_deleted,
            "signals_preserved": True,
        },
    )
    return PaperResetResponse(
        product_code=result.product_code,
        trading_mode="paper",
        orders_deleted=result.orders_deleted,
        fills_deleted=result.fills_deleted,
        positions_deleted=result.positions_deleted,
        snapshots_deleted=result.snapshots_deleted,
        signals_preserved=True,
        message=(
            "Paper trading reset. BTC holdings are zero and the account has been "
            "reinitialized with the configured starting JPY balance."
        ),
    )


@router.get("/backtest", response_model=BacktestResponse, summary="Replay the trend strategy")
async def backtest(
    session: SessionDep,
    settings: SettingsDep,
    strategy_name: str | None = None,
    start: date | None = None,
    end: date | None = None,
    cost_multiplier: Annotated[Decimal, Query(gt=0, le=10)] = Decimal("1"),
    apply_jpy_limits: bool = False,
    include_trades: bool = False,
) -> BacktestResponse:
    """Daily replay through the paper ledger's sizing and cost rules.

    ``apply_jpy_limits=false`` (default) evaluates the strategy itself; with
    ``true`` the absolute MAX_ORDER_JPY/MAX_POSITION_JPY caps also apply, which
    on long histories mostly measures how quickly equity outgrows them.
    """
    settings = await get_runtime_settings(session, settings)
    name = (strategy_name or settings.strategy_name).strip().lower()
    if name == "none":
        name = "trend_ensemble"
    await sync_own_daily_bars(session, settings)
    series = await load_daily_series(session, settings)
    try:
        result = await asyncio.to_thread(
            run_backtest,
            series.bars,
            settings,
            strategy_name=name,
            start=start,
            end=end,
            cost_multiplier=cost_multiplier,
            apply_jpy_limits=apply_jpy_limits,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BacktestResponse(
        strategy_name=result.strategy_name,
        metrics=BacktestMetricsResponse.model_validate(result.metrics),
        benchmark=BacktestMetricsResponse.model_validate(result.benchmark),
        periods=[PeriodReturnResponse.model_validate(period) for period in result.periods],
        cost_stress=[CostStressResponse.model_validate(item) for item in result.cost_stress],
        parameters=result.parameters,
        trades=(
            [BacktestTradeResponse.model_validate(trade) for trade in result.trades]
            if include_trades
            else []
        ),
        equity_curve=result.equity_curve,
    )


@router.get(
    "/strategy/preview",
    response_model=StrategyPreviewResponse,
    summary="Current trend decision without storing a signal or trading",
)
async def strategy_preview(session: SessionDep, settings: SettingsDep) -> StrategyPreviewResponse:
    settings = await get_runtime_settings(session, settings)
    preview = await preview_strategy(session, settings)
    history = daily_history_summary(settings, preview.series)
    decision = preview.decision
    if decision is None:
        return StrategyPreviewResponse(
            strategy_name=settings.strategy_name,
            ready=False,
            message=(
                f"{len(preview.series.bars)} completed daily bars stored; the strategy needs "
                f"{preview.parameters.warmup_bars} before it can size a position. "
                "Load daily history to bootstrap it."
            ),
            history=history,
        )
    return StrategyPreviewResponse(
        strategy_name=settings.strategy_name,
        ready=True,
        message=(
            f"Daily close {decision.as_of} is {preview.bar_age_days} days old; no orders until "
            "fresh daily data arrives."
            if preview.stale
            else decision.reason
        ),
        as_of=decision.as_of,
        bar_age_days=preview.bar_age_days,
        stale=preview.stale,
        close=decision.close,
        target_exposure=decision.target_exposure,
        previous_target_exposure=decision.previous_target_exposure,
        current_exposure=(
            await current_exposure(session, settings) if settings.strategies_enabled else None
        ),
        active_fraction=decision.active_fraction,
        realized_volatility=decision.realized_volatility,
        target_volatility=preview.parameters.target_volatility,
        volatility_scale=decision.volatility_scale,
        signal=decision.signal.value,
        reason=decision.reason,
        sleeves=[SleeveResponse.model_validate(sleeve) for sleeve in decision.sleeves],
        history=history,
    )


@router.get(
    "/trading/status",
    response_model=TradingStatusResponse,
    summary="Trading mode and safety switches",
)
async def trading_status(session: SessionDep, settings: SettingsDep) -> TradingStatusResponse:
    settings = await get_runtime_settings(session, settings)
    return TradingStatusResponse(
        trading_mode=settings.trading_mode.value,
        strategy_name=settings.strategy_name,
        automatic_trading_enabled=settings.enable_automatic_trading,
        live_trading_acknowledged=settings.live_trading_acknowledged,
        kill_switch=settings.trading_kill_switch,
        live_orders_allowed=settings.live_orders_allowed,
        credentials_configured=settings.has_private_credentials,
        min_profit_percent=settings.min_profit_percent,
        max_order_jpy=settings.max_order_jpy,
        max_position_jpy=settings.max_position_jpy,
        max_daily_loss_jpy=settings.max_daily_loss_jpy,
        max_market_data_age_seconds=settings.max_market_data_age_seconds,
        min_order_size_btc=settings.min_order_size_btc,
        order_size_step_btc=settings.order_size_step_btc,
        enforce_exchange_minimums=settings.enforce_exchange_minimums,
        live_capital_jpy=settings.live_capital_jpy,
        live_max_slippage_bps=settings.live_max_slippage_bps,
        alerts_configured=bool(settings.alert_webhook_url.get_secret_value()),
    )


@router.post(
    "/strategy/evaluate",
    response_model=SignalResponse,
    summary="Run one paper evaluation now",
)
async def strategy_evaluate(session: SessionDep, settings: SettingsDep) -> SignalResponse:
    """Paper only: live orders are placed by the worker, never by an HTTP call."""
    settings = await get_runtime_settings(session, settings)
    if settings.trading_mode is TradingMode.LIVE:
        raise HTTPException(
            status_code=409, detail="live evaluation runs only in the worker process"
        )
    signal = await evaluate_and_store(session, settings)
    if signal is None:
        raise HTTPException(
            status_code=409,
            detail="strategy is disabled, trading mode is not paper, or daily history is short",
        )
    return SignalResponse.model_validate(signal)
