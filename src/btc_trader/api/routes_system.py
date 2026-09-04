"""Health, status and configuration endpoints (PROJECT.md §34, §43)."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from btc_trader.api.deps import MarketDataServiceDep, SchedulerDep, SessionDep, SettingsDep
from btc_trader.api.schemas import (
    CollectorRunResponse,
    ConfigResponse,
    HealthResponse,
    SettingsUpdateRequest,
    SystemEventResponse,
    SystemStatusResponse,
)
from btc_trader.monitoring.health import build_health_report
from btc_trader.monitoring.logging import get_logger
from btc_trader.runtime_settings import get_runtime_settings, update_runtime_settings

logger = get_logger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness / readiness")
async def health(
    response: Response,
    settings: SettingsDep,
    scheduler: SchedulerDep,
    probe_exchange: Annotated[bool, Query(description="Also probe the public API")] = True,
) -> HealthResponse:
    """Container health check.

    Private credentials are not required to be healthy (§43); a degraded exchange
    still returns HTTP 200 so a transient bitFlyer hiccup does not restart the
    container, while a database failure returns 503.
    """
    report = await build_health_report(
        settings=settings,
        scheduler_running=scheduler.running if scheduler is not None else None,
        probe_exchange=probe_exchange,
    )
    if report.status == "error":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse.model_validate(report.to_dict())


@router.get(
    "/api/v1/system/status",
    response_model=SystemStatusResponse,
    summary="Aggregated operational status",
)
async def system_status(
    service: MarketDataServiceDep,
    settings: SettingsDep,
    scheduler: SchedulerDep,
    session: SessionDep,
) -> SystemStatusResponse:
    settings = await get_runtime_settings(session, settings)
    collector = await service.collector_status()
    report = await build_health_report(
        settings=settings,
        scheduler_running=scheduler.running if scheduler is not None else None,
        probe_exchange=False,
    )
    last_error = await service.last_error_event()
    overview = await service.overview()
    collector = replace(collector, interval_seconds=settings.market_data_interval_seconds)
    return SystemStatusResponse(
        app_env=settings.app_env,
        trading_mode=settings.trading_mode.value,
        product_code=settings.product_code,
        exchange=settings.exchange,
        strategy_name=settings.strategy_name,
        automatic_trading_enabled=settings.enable_automatic_trading,
        live_orders_allowed=settings.live_orders_allowed,
        kill_switch=settings.trading_kill_switch,
        market_data_interval_seconds=settings.market_data_interval_seconds,
        dashboard_refresh_seconds=settings.dashboard_refresh_seconds,
        stored_tick_count=overview.stored_tick_count,
        stored_execution_count=await service.stored_execution_count(),
        collector=collector,
        health=HealthResponse.model_validate(report.to_dict()),
        scheduler_jobs=scheduler.describe() if scheduler is not None else [],
        last_error=(
            SystemEventResponse.model_validate(last_error) if last_error is not None else None
        ),
    )


@router.get(
    "/api/v1/system/events",
    response_model=list[SystemEventResponse],
    summary="Recent operational events",
)
async def system_events(
    service: MarketDataServiceDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[SystemEventResponse]:
    events = await service.recent_events(limit=limit)
    return [SystemEventResponse.model_validate(event) for event in events]


@router.get(
    "/api/v1/system/config",
    response_model=ConfigResponse,
    summary="Effective configuration (secrets redacted)",
)
async def system_config(session: SessionDep, settings: SettingsDep) -> ConfigResponse:
    settings = await get_runtime_settings(session, settings)
    return ConfigResponse(config=settings.safe_dump())


@router.get("/api/v1/system/settings", response_model=ConfigResponse, summary="Runtime settings")
async def runtime_settings(session: SessionDep, settings: SettingsDep) -> ConfigResponse:
    current = await get_runtime_settings(session, settings)
    return ConfigResponse(config=current.safe_dump())


@router.get(
    "/api/v1/system/settings/update", response_model=ConfigResponse, include_in_schema=False
)
async def update_settings(
    session: SessionDep, settings: SettingsDep, request: SettingsUpdateRequest = Depends()
) -> ConfigResponse:
    return await _save_settings(session, settings, request)


@router.patch(
    "/api/v1/system/settings", response_model=ConfigResponse, summary="Apply runtime settings"
)
async def patch_settings(
    session: SessionDep, settings: SettingsDep, request: SettingsUpdateRequest
) -> ConfigResponse:
    """Persist one explicit dashboard settings submission."""
    return await _save_settings(session, settings, request)


async def _save_settings(
    session: SessionDep, settings: SettingsDep, request: SettingsUpdateRequest
) -> ConfigResponse:
    try:
        current = await get_runtime_settings(session, settings)
        updated = await update_runtime_settings(session, current, request.patch())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ConfigResponse(config=updated.safe_dump())


@router.get(
    "/api/v1/system/collector",
    response_model=list[CollectorRunResponse],
    summary="Recent collector runs",
)
async def collector_runs(
    service: MarketDataServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[CollectorRunResponse]:
    runs = await service.recent_collector_runs(limit=limit)
    return [CollectorRunResponse.model_validate(run) for run in runs]
