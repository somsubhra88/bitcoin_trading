"""Webhook alerts are best effort and never interrupt trading."""

from __future__ import annotations

import json

import httpx
import respx
from pydantic import SecretStr

from btc_trader.config import Settings
from btc_trader.enums import Severity
from btc_trader.monitoring.alerts import send_alert

URL = "https://hooks.example.test/T000/B000"


async def test_no_url_means_no_request(settings: Settings) -> None:
    with respx.mock(assert_all_called=False) as router:
        delivered = await send_alert(settings, Severity.ERROR, "title", "detail")

    assert delivered is False
    assert not router.calls


async def test_the_payload_suits_slack_and_discord(settings: Settings) -> None:
    configured = settings.model_copy(update={"alert_webhook_url": SecretStr(URL)})
    with respx.mock() as router:
        route = router.post(URL).mock(return_value=httpx.Response(204))
        delivered = await send_alert(configured, Severity.CRITICAL, "Live trading halted", "why")

    assert delivered is True
    body = json.loads(route.calls[0].request.content)
    assert body["text"] == body["content"]
    assert "Live trading halted" in body["text"]
    assert "BTC_JPY" in body["text"]


async def test_a_failing_webhook_is_swallowed(settings: Settings) -> None:
    configured = settings.model_copy(update={"alert_webhook_url": SecretStr(URL)})
    with respx.mock() as router:
        router.post(URL).mock(return_value=httpx.Response(500))
        assert await send_alert(configured, Severity.INFO, "t", "d") is False
        router.post(URL).mock(side_effect=httpx.ConnectError("down"))
        assert await send_alert(configured, Severity.INFO, "t", "d") is False
