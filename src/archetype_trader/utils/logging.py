"""Structured logging for research runs and the paper-trading adapter.

Uses ``structlog`` when available (it is a dependency of the sibling
``btc_trader`` package) and degrades to stdlib logging otherwise, so this
package stays importable in a minimal research environment.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False


def configure_logging(level: str = "INFO", *, json_lines: bool = False) -> None:
    """Install a single stderr handler. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    if json_lines:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    _CONFIGURED = True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """Emit one structured event; fields land in the JSON payload."""
    logger.info(event, extra={"context": fields})
