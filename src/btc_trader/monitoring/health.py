"""Health aggregation for ``GET /health`` (PROJECT.md §43).

Private API credentials are deliberately *not* required for the service to be
considered healthy — a data-collection deployment has none.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from btc_trader.config import Settings, get_settings
from btc_trader.db.session import check_database
from btc_trader.exchange.registry import create_exchange_client
from btc_trader.monitoring.logging import get_logger

logger = get_logger(__name__)

Status = Literal["ok", "degraded", "error", "disabled"]

#: Cap on the exchange probe so /health stays fast for container health checks.
EXCHANGE_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class HealthReport:
    status: Status
    database: Status
    scheduler: Status
    exchange_public_api: Status
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "database": self.database,
            "scheduler": self.scheduler,
            "exchange_public_api": self.exchange_public_api,
            "details": self.details,
        }


async def _probe_exchange(settings: Settings) -> tuple[Status, dict[str, Any]]:
    client = create_exchange_client(settings)
    try:
        health = await asyncio.wait_for(
            client.get_health(settings.product_code), timeout=EXCHANGE_PROBE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        return "error", {"exchange_error": "timeout"}
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        logger.warning("exchange_health_probe_failed", error=str(exc))
        return "error", {"exchange_error": f"{type(exc).__name__}: {exc}"}
    else:
        return (
            "ok" if health.state.accepts_orders else "degraded",
            {"exchange_state": health.state.value},
        )
    finally:
        await client.aclose()


async def build_health_report(
    *,
    settings: Settings | None = None,
    scheduler_running: bool | None = None,
    probe_exchange: bool = True,
) -> HealthReport:
    """Assemble the health report.

    ``scheduler_running=None`` means this process does not host the scheduler, so
    its status is reported as ``disabled`` rather than failing the check.
    """
    resolved = settings or get_settings()
    details: dict[str, Any] = {
        "trading_mode": resolved.trading_mode.value,
        "product_code": resolved.product_code,
        "live_orders_allowed": resolved.live_orders_allowed,
        "kill_switch": resolved.trading_kill_switch,
    }

    database_ok, database_error = await check_database(resolved)
    database: Status = "ok" if database_ok else "error"
    if database_error:
        details["database_error"] = database_error

    if scheduler_running is None:
        scheduler: Status = "disabled"
    else:
        scheduler = "ok" if scheduler_running else "error"

    if probe_exchange:
        exchange_status, exchange_details = await _probe_exchange(resolved)
        details.update(exchange_details)
    else:
        exchange_status = "disabled"

    if database == "error" or scheduler == "error":
        overall: Status = "error"
    elif exchange_status in ("error", "degraded"):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthReport(
        status=overall,
        database=database,
        scheduler=scheduler,
        exchange_public_api=exchange_status,
        details=details,
    )
