"""Optional push notifications for unattended operation.

``ALERT_WEBHOOK_URL`` may point at a Slack or Discord incoming webhook; the
payload carries both ``text`` (Slack) and ``content`` (Discord). Delivery is
best effort: an alerting failure is logged and never interrupts trading.
"""

from __future__ import annotations

import httpx

from btc_trader.config import Settings
from btc_trader.enums import Severity
from btc_trader.monitoring.logging import get_logger

logger = get_logger(__name__)

TIMEOUT_SECONDS = 5.0
_ICONS = {
    Severity.INFO: "🟢",
    Severity.WARNING: "⚠️",
    Severity.ERROR: "🔴",
    Severity.CRITICAL: "🚨",
}


async def send_alert(settings: Settings, severity: Severity, title: str, detail: str) -> bool:
    """Post one alert; returns whether it was delivered."""
    url = settings.alert_webhook_url.get_secret_value()
    if not url:
        return False
    text = (
        f"{_ICONS.get(severity, '')} [{settings.trading_mode.value} {settings.product_code}] "
        f"{title}\n{detail}"
    )
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(url, json={"text": text, "content": text[:1900]})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("alert_delivery_failed", error_type=type(exc).__name__, title=title)
        return False
    return True
