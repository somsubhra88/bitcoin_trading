"""FastAPI application entry point.

Run with::

    uvicorn btc_trader.main:app --host 0.0.0.0 --port 8000

Set ``RUN_SCHEDULER_IN_API=true`` to host the collector in this process
(single-container deployment); the default keeps API and worker separate so a
restart of one does not interrupt the other.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from btc_trader.api import routes_market, routes_system, routes_trading
from btc_trader.config import get_settings
from btc_trader.db.session import dispose_engine
from btc_trader.errors import BtcTraderError, ExchangeError, MarketDataError
from btc_trader.monitoring.logging import configure_logging, get_logger
from btc_trader.scheduler.jobs import SchedulerService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "application_starting",
        app_env=settings.app_env,
        trading_mode=settings.trading_mode.value,
        product_code=settings.product_code,
        market_data_interval_seconds=settings.market_data_interval_seconds,
        scheduler_in_api=settings.run_scheduler_in_api,
        live_orders_allowed=settings.live_orders_allowed,
    )

    scheduler: SchedulerService | None = None
    if settings.run_scheduler_in_api:
        scheduler = SchedulerService(settings)
        scheduler.start()
    app.state.scheduler = scheduler

    try:
        yield
    finally:
        if scheduler is not None:
            await scheduler.shutdown()
        await dispose_engine()
        logger.info("application_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="btc-trader",
        version="0.1.0",
        summary="BTC/JPY market-data platform (bitFlyer)",
        description=(
            "Market-data collection, storage and visualisation for BTC/JPY. "
            "No endpoint in this version can submit an exchange order."
        ),
        lifespan=lifespan,
    )

    app.include_router(routes_system.router)
    app.include_router(routes_market.router)
    app.include_router(routes_trading.router)

    @app.exception_handler(MarketDataError)
    async def _market_data_error(_request: Request, exc: MarketDataError) -> JSONResponse:
        return JSONResponse(
            status_code=502, content={"error": "market_data_error", "detail": str(exc)}
        )

    @app.exception_handler(ExchangeError)
    async def _exchange_error(_request: Request, exc: ExchangeError) -> JSONResponse:
        return JSONResponse(
            status_code=502, content={"error": "exchange_error", "detail": str(exc)}
        )

    @app.exception_handler(BtcTraderError)
    async def _domain_error(_request: Request, exc: BtcTraderError) -> JSONResponse:
        logger.error("unhandled_domain_error", error_type=type(exc).__name__, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "btc-trader",
            "product_code": settings.product_code,
            "trading_mode": settings.trading_mode.value,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


def run() -> None:  # pragma: no cover - process entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "btc_trader.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
