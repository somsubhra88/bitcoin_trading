"""Operational event log (PROJECT.md §12.8).

Every categorised failure or notable state change is recorded here in addition
to being logged, so the dashboard and post-mortems have a queryable history.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.db.models import SystemEvent
from btc_trader.enums import Severity
from btc_trader.utils.serialization import jsonable
from btc_trader.utils.timeutils import utcnow


class SystemEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        severity: Severity,
        component: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> SystemEvent:
        event = SystemEvent(
            timestamp=utcnow(),
            severity=severity,
            component=component,
            event_type=event_type,
            message=message[:4000],
            event_metadata=jsonable(metadata) if metadata else None,
        )
        self._session.add(event)
        await self._session.flush()
        return event

    async def recent(
        self,
        *,
        limit: int = 50,
        min_severity: Severity | None = None,
        component: str | None = None,
    ) -> list[SystemEvent]:
        statement = select(SystemEvent).order_by(SystemEvent.timestamp.desc()).limit(limit)
        if component:
            statement = statement.where(SystemEvent.component == component)
        if min_severity is not None:
            allowed = _at_least(min_severity)
            statement = statement.where(SystemEvent.severity.in_(allowed))
        return list((await self._session.execute(statement)).scalars())

    async def last_error(self) -> SystemEvent | None:
        statement = (
            select(SystemEvent)
            .where(SystemEvent.severity.in_([Severity.ERROR, Severity.CRITICAL]))
            .order_by(SystemEvent.timestamp.desc())
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none()


_SEVERITY_ORDER = [
    Severity.DEBUG,
    Severity.INFO,
    Severity.WARNING,
    Severity.ERROR,
    Severity.CRITICAL,
]


def _at_least(minimum: Severity) -> list[Severity]:
    return _SEVERITY_ORDER[_SEVERITY_ORDER.index(minimum) :]
