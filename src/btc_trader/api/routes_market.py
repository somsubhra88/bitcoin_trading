"""Market-data endpoints (PROJECT.md §34).

Reads come from the database; the only endpoint that touches bitFlyer is the
manual collection trigger, which delegates to the same collection service the
scheduler uses (§11).
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
    HistoryResponse,
    MarketOverviewResponse,
    OrderBookResponse,
    TickResponse,
)
from btc_trader.enums import CandleSource
from btc_trader.errors import MarketDataError
from btc_trader.market_data.candles import INTERVALS, CandleService
from btc_trader.market_data.collector import collect_market_data
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


@router.get("/order-book", response_model=OrderBookResponse, summary="Latest order book")
async def order_book(service: MarketDataServiceDep) -> OrderBookResponse:
    snapshot = await service.latest_order_book()
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no order-book snapshot stored (is COLLECT_ORDER_BOOK enabled?)",
        )
    return OrderBookResponse.model_validate(snapshot)


@router.get("/candles", response_model=CandlesResponse, summary="Aggregated OHLCV")
async def candles(
    session: SessionDep,
    settings: SettingsDep,
    interval: IntervalQuery = "5m",
    start: datetime | None = None,
    end: datetime | None = None,
    source: CandleSource | None = None,
) -> CandlesResponse:
    if interval not in INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown interval {interval!r}; expected one of {sorted(INTERVALS)}",
        )
    service = CandleService(session, settings)
    bars = await service.get_candles(interval=interval, start=start, end=end, prefer_source=source)
    resolved_source = bars[0].source if bars else (source or CandleSource.EXECUTIONS)
    return CandlesResponse(
        product_code=settings.product_code,
        interval=interval,
        source=resolved_source,
        is_true_ohlc=resolved_source is CandleSource.EXECUTIONS,
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


def _validate_period(period: str) -> None:
    if period not in PERIODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown period {period!r}; expected one of {sorted(PERIODS)}",
        )
