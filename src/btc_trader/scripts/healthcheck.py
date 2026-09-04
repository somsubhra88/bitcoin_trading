"""Container/CLI health check.

    python -m btc_trader.scripts.healthcheck

Exits 0 when the database is reachable and, unless ``--no-exchange`` is passed,
the public bitFlyer API responds.  Private credentials are never required
(PROJECT.md §43).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from btc_trader.config import get_settings
from btc_trader.db.session import dispose_engine
from btc_trader.monitoring.health import build_health_report
from btc_trader.monitoring.logging import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="btc-trader health check")
    parser.add_argument("--no-exchange", action="store_true", help="skip the public exchange probe")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args = parse_args(argv)
    try:
        report = await build_health_report(
            settings=settings, scheduler_running=None, probe_exchange=not args.no_exchange
        )
    finally:
        await dispose_engine()
    print(json.dumps(report.to_dict(), default=str))
    return 0 if report.status in ("ok", "degraded") else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(asyncio.run(main()))
