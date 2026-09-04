"""Structured logging (PROJECT.md §35).

``console`` renderer for development, JSON for deployed environments.  Secrets
are scrubbed defensively: any key that looks like a credential is redacted even
if a caller passes it by mistake.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "api_secret",
    "apikey",
    "secret",
    "password",
    "token",
    "authorization",
    "access-sign",
    "signature",
)

_configured = False


def _redact_secrets(
    _logger: Any, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    for key in list(event_dict):
        lowered = key.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS):
            event_dict[key] = "***redacted***"
    return event_dict


def configure_logging(level: str = "INFO", log_format: str = "console") -> None:
    """Idempotently configure stdlib logging + structlog."""
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # NB: not ``structlog.stdlib.add_logger_name`` — that reads ``.name`` off
        # a stdlib logger, which a PrintLogger does not have.  The name is bound
        # in :func:`get_logger` instead.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _redact_secrets,
    ]

    renderer: structlog.types.Processor
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers (uvicorn, sqlalchemy, apscheduler) through stderr at
    # the same level so nothing is silently lost.
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(numeric_level)
    logging.getLogger("uvicorn.access").setLevel(max(numeric_level, logging.INFO))

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _configured:
        from btc_trader.config import get_settings

        settings = get_settings()
        configure_logging(settings.log_level, settings.log_format)
    # ``logger`` is reserved by ``get_logger``, so bind it afterwards.
    return structlog.get_logger().bind(logger=name)
