"""Structured logging (PROJECT.md §35).

Regression guard: the JSON/console pipeline must actually be able to emit an
event.  A processor that assumes a stdlib logger (``add_logger_name``) combined
with a ``PrintLogger`` factory raises ``AttributeError`` on *every* log call, so
the first assertion here is simply "logging does not explode".
"""

from __future__ import annotations

import json

import pytest

from btc_trader.monitoring.logging import configure_logging, get_logger


def _emit(capsys: pytest.CaptureFixture[str], **fields: object) -> dict[str, object]:
    configure_logging("DEBUG", "json")
    get_logger("test.module").info("an_event", **fields)
    captured = capsys.readouterr().err.strip().splitlines()[-1]
    payload: dict[str, object] = json.loads(captured)
    return payload


def test_an_event_can_be_emitted(capsys: pytest.CaptureFixture[str]) -> None:
    payload = _emit(capsys, product_code="BTC_JPY")
    assert payload["event"] == "an_event"
    assert payload["level"] == "info"
    assert payload["logger"] == "test.module"
    assert payload["product_code"] == "BTC_JPY"
    assert "timestamp" in payload


@pytest.mark.parametrize(
    "key",
    [
        "bitflyer_api_key",
        "bitflyer_api_secret",
        "api_secret",
        "password",
        "access-sign",
        "signature",
        "authorization",
        "token",
        "ACCESS-SIGN",
    ],
)
def test_credential_shaped_keys_are_redacted(capsys: pytest.CaptureFixture[str], key: str) -> None:
    payload = _emit(capsys, **{key: "super-secret-value"})
    assert payload[key] == "***redacted***"
    assert "super-secret-value" not in json.dumps(payload)


def test_ordinary_keys_are_not_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    payload = _emit(capsys, ltp="12411935.0", degraded=["daily_history"])
    assert payload["ltp"] == "12411935.0"
    assert payload["degraded"] == ["daily_history"]


def test_console_format_also_renders(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", "console")
    get_logger("test.module").warning("console_event", api_secret="nope")
    err = capsys.readouterr().err
    assert "console_event" in err
    assert "nope" not in err
    assert "***redacted***" in err
