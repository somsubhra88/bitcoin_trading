"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings, get_settings
from btc_trader.db.session import session_scope
from btc_trader.market_data.service import MarketDataService
from btc_trader.runtime_settings import get_runtime_settings
from btc_trader.scheduler.jobs import SchedulerService


async def db_session() -> AsyncIterator[AsyncSession]:
    async with session_scope() as session:
        yield session


def settings_dep() -> Settings:
    return get_settings()


async def market_data_service(
    session: Annotated[AsyncSession, Depends(db_session)],
    settings: Annotated[Settings, Depends(settings_dep)],
) -> MarketDataService:
    effective = await get_runtime_settings(session, settings)
    return MarketDataService(session, effective)


def scheduler(request: Request) -> SchedulerService | None:
    """The scheduler instance if this process hosts one, else ``None``."""
    return getattr(request.app.state, "scheduler", None)


SessionDep = Annotated[AsyncSession, Depends(db_session)]
SettingsDep = Annotated[Settings, Depends(settings_dep)]
MarketDataServiceDep = Annotated[MarketDataService, Depends(market_data_service)]
# NB: not a string annotation.  FastAPI resolves parameter annotations through
# pydantic, and an unevaluated ForwardRef here makes it treat the parameter as a
# *query* field instead of a dependency — which breaks /health, the status
# endpoint and OpenAPI generation.  ``SchedulerService`` is imported above, so
# the real type is available.
SchedulerDep = Annotated[SchedulerService | None, Depends(scheduler)]
