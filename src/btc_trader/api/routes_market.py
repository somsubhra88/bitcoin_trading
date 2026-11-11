"""Market-data endpoints (PROJECT.md §34).

Reads come from the database. Two endpoints reach the network: the manual
collection trigger (same service as the scheduler, §11) and the daily-history
sync (the configured public history provider).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from btc_trader.api.deps import MarketDataServiceDep, SessionDep, SettingsDep
from btc_trader.api.schemas import (
    CandleResponse,
    CandlesResponse,
    CollectionResponse,
    DailyHistoryResponse,
    DailyHistorySyncResponse,
    HistoryResponse,
    MarketOverviewResponse,
    TickResponse,
)
from btc_trader.config import Settings
from btc_trader.errors import ExchangeError, MarketDataError
from btc_trader.market_data.candles import INTERVALS, candles_from_ticks
from btc_trader.market_data.collector import collect_market_data
from btc_trader.market_data.daily import (
    DailySeries,
    load_daily_series,
    sync_external_history,
    sync_own_daily_bars,
)
from btc_trader.market_data.repository import MarketDataRepository
from btc_trader.market_data.service import PERIODS
from btc_trader.monitoring.logging import get_logger
from btc_trader.runtime_settings import get_runtime_settings

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/market", tags=["market"])

PeriodQuery = Annotated[str, Query(description=f"One of: {', '.join(PERIODS)}")]
IntervalQuery = Annotated[str, Query(description=f"One of: {', '.join(INTERVALS)}")]


@router.get("/latest", response_model=TickResponse, summary="Latest stored ticker")
async def latest(service: MarketDataServiceDep) -> TickResponse:
    tick = await service.latest_tick()
    if tick is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no market data stored yet; trigger a collection first",
        )
    return TickResponse.model_validate(tick)


@router.get("/overview", response_model=MarketOverviewResponse, summary="Market overview")
async def overview(
    service: MarketDataServiceDep,
    change_window: PeriodQuery = "24h",
) -> MarketOverviewResponse:
    _validate_period(change_window)
    return MarketOverviewResponse.model_validate(
        await service.overview(change_window=change_window)
    )


@router.get("/history", response_model=HistoryResponse, summary="Stored price history")
async def history(
    service: MarketDataServiceDep,
    settings: SettingsDep,
    period: PeriodQuery = "24h",
    limit: Annotated[int, Query(ge=1, le=50_000)] = 5000,
) -> HistoryResponse:
    _validate_period(period)
    ticks = await service.history(period=period, limit=limit)
    return HistoryResponse(
        product_code=settings.product_code,
        period=period,
        count=len(ticks),
        ticks=[TickResponse.model_validate(tick) for tick in ticks],
    )


@router.get("/candles", response_model=CandlesResponse, summary="OHLC from sampled ticks")
async def candles(
    session: SessionDep,
    settings: SettingsDep,
    interval: IntervalQuery = "5m",
    start: datetime | None = None,
    end: datetime | None = None,
) -> CandlesResponse:
    """Bars folded from stored ticker samples, for charting only (PROJECT.md §40)."""
    if interval not in INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown interval {interval!r}; expected one of {sorted(INTERVALS)}",
        )
    ticks = await MarketDataRepository(session).ticks_between(
        settings.product_code, start=start, end=end, limit=50_000
    )
    bars = candles_from_ticks(ticks, interval)
    return CandlesResponse(
        product_code=settings.product_code,
        interval=interval,
        count=len(bars),
        candles=[CandleResponse.model_validate(bar) for bar in bars],
    )


@router.post(
    "/collect",
    response_model=CollectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger one immediate market-data collection",
)
async def collect(session: SessionDep, settings: SettingsDep) -> CollectionResponse:
    """Manual collection (PROJECT.md §11) — same service as the scheduled job."""
    try:
        result = await collect_market_data(
            trigger="manual", settings=await get_runtime_settings(session, settings)
        )
    except MarketDataError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return CollectionResponse.model_validate(result)


def daily_history_summary(settings: Settings, series: DailySeries) -> DailyHistoryResponse:
    return DailyHistoryResponse(
        product_code=settings.product_code,
        provider=settings.daily_history_provider,
        bars=len(series.bars),
        first_day=series.bars[0].day if series.bars else None,
        last_day=series.last_day,
        own_bitflyer_days=series.own_days,
        external_days=series.external_days,
        basis_ratio=series.basis_ratio,
        required_bars=max(
            settings.trend_horizons[-1], settings.strategy_volatility_window_days + 1
        ),
    )


@router.get(
    "/daily-history",
    response_model=DailyHistoryResponse,
    summary="Coverage of the daily bars the strategy uses",
)
async def daily_history(session: SessionDep, settings: SettingsDep) -> DailyHistoryResponse:
    settings = await get_runtime_settings(session, settings)
    return daily_history_summary(settings, await load_daily_series(session, settings))


@router.post(
    "/daily-history/sync",
    response_model=DailyHistorySyncResponse,
    summary="Load daily history from the configured provider and own ticks",
)
async def sync_daily_history(
    session: SessionDep,
    settings: SettingsDep,
    years: Annotated[int | None, Query(ge=1, le=15)] = None,
) -> DailyHistorySyncResponse:
    """Idempotent: re-running only adds days that are not stored yet."""
    settings = await get_runtime_settings(session, settings)
    try:
        external = await sync_external_history(session, settings, years=years)
    except (ExchangeError, MarketDataError) as exc:
        logger.warning("daily_history_sync_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    own = await sync_own_daily_bars(session, settings)
    return DailyHistorySyncResponse(
        provider=settings.daily_history_provider,
        external_bars_inserted=external,
        own_bars_inserted=own,
        history=daily_history_summary(settings, await load_daily_series(session, settings)),
    )


def _validate_period(period: str) -> None:
    if period not in PERIODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown period {period!r}; expected one of {sorted(PERIODS)}",
        )
