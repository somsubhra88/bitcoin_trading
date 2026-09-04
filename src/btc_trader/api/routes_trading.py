"""Trading signals and safety-gated strategy controls.

These are projections of the strategy and portfolio tables. Paper evaluation
also records simulated fills and portfolio state; no endpoint can submit a real
exchange order.

There is deliberately **no** order-placement endpoint: nothing exposed over HTTP
can move money in this version (PROJECT.md §34, §53.12, §56).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from btc_trader.api.deps import SessionDep, SettingsDep
from btc_trader.api.schemas import (
    BacktestMetricsResponse,
    BacktestResponse,
    BacktestTradeResponse,
    FillResponse,
    OrderResponse,
    PaperResetRequest,
    PaperResetResponse,
    PortfolioResponse,
    PortfolioSnapshotResponse,
    PositionResponse,
    SignalResponse,
    StrategySelectionResponse,
    TradingStatusResponse,
    WalkForwardResponse,
    WalkForwardWindowResponse,
)
from btc_trader.backtesting import run_backtest, run_walk_forward
from btc_trader.db.models import (
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    StrategySignal,
)
from btc_trader.db.repositories.system_events import SystemEventRepository
from btc_trader.enums import OrderStatus, Severity
from btc_trader.market_data.repository import MarketDataRepository
from btc_trader.paper import reset_paper_trading
from btc_trader.runtime_settings import get_runtime_settings, update_runtime_settings
from btc_trader.strategy import STRATEGIES
from btc_trader.strategy_service import evaluate_and_store

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


@router.get("/backtest", response_model=BacktestResponse, summary="Replay paper strategy history")
async def backtest(
    session: SessionDep,
    settings: SettingsDep,
    strategy_name: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 5_000,
) -> BacktestResponse:
    settings = await get_runtime_settings(session, settings)
    repository = MarketDataRepository(session)
    name = (strategy_name or settings.strategy_name).strip().lower()
    ticks = await repository.ticks_between(settings.product_code, limit=limit)
    order_books = (
        await repository.order_books_between(settings.product_code, limit=limit)
        if name == "deeplob"
        else None
    )
    try:
        result = await asyncio.to_thread(
            run_backtest,
            ticks,
            settings,
            strategy_name=strategy_name,
            order_books=order_books,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BacktestResponse(
        metrics=BacktestMetricsResponse.model_validate(result.metrics),
        trades=[BacktestTradeResponse.model_validate(trade) for trade in result.trades],
        equity_curve=result.equity_curve,
    )


@router.get(
    "/backtest/walk-forward",
    response_model=WalkForwardResponse,
    summary="Walk-forward validate a paper strategy",
)
async def backtest_walk_forward(
    session: SessionDep,
    settings: SettingsDep,
    strategy_name: str | None = None,
    limit: Annotated[int, Query(ge=1, le=20_000)] = 10_000,
    train_bars: Annotated[int, Query(ge=1, le=10_000)] = 500,
    test_bars: Annotated[int, Query(ge=1, le=5_000)] = 100,
    step_bars: Annotated[int | None, Query(ge=1, le=5_000)] = None,
) -> WalkForwardResponse:
    settings = await get_runtime_settings(session, settings)
    repository = MarketDataRepository(session)
    name = (strategy_name or settings.strategy_name).strip().lower()
    ticks = await repository.ticks_between(settings.product_code, limit=limit)
    order_books = (
        await repository.order_books_between(settings.product_code, limit=limit)
        if name == "deeplob"
        else None
    )
    try:
        result = await asyncio.to_thread(
            run_walk_forward,
            ticks,
            settings,
            strategy_name=strategy_name,
            order_books=order_books,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return WalkForwardResponse(
        strategy_name=result.strategy_name,
        train_bars=result.train_bars,
        test_bars=result.test_bars,
        windows=[
            WalkForwardWindowResponse(
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                selected_parameters=window.selected_parameters,
                train_metrics=BacktestMetricsResponse.model_validate(window.train_metrics),
                test_metrics=BacktestMetricsResponse.model_validate(window.test_metrics),
            )
            for window in result.windows
        ],
        aggregate_metrics=BacktestMetricsResponse.model_validate(result.aggregate_metrics),
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
        execution_engine_implemented=settings.trading_mode.value == "paper",
    )


@router.get("/strategy/options", response_model=StrategySelectionResponse)
async def strategy_options(session: SessionDep, settings: SettingsDep) -> StrategySelectionResponse:
    settings = await get_runtime_settings(session, settings)
    return StrategySelectionResponse(
        strategy_name=settings.strategy_name, available_strategies=STRATEGIES
    )


@router.post("/strategy/select", response_model=StrategySelectionResponse)
async def strategy_select(
    strategy_name: str, session: SessionDep, settings: SettingsDep
) -> StrategySelectionResponse:
    name = strategy_name.strip().lower()
    if name not in STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}",
        )
    current = await get_runtime_settings(session, settings)
    await update_runtime_settings(session, current, {"strategy_name": name})
    return StrategySelectionResponse(strategy_name=name, available_strategies=STRATEGIES)


@router.post("/strategy/evaluate", response_model=SignalResponse)
async def strategy_evaluate(session: SessionDep, settings: SettingsDep) -> SignalResponse:
    settings = await get_runtime_settings(session, settings)
    signal = await evaluate_and_store(session, settings)
    if signal is None:
        raise HTTPException(
            status_code=409, detail="strategy is disabled or needs more market data"
        )
    return SignalResponse.model_validate(signal)
