"""Daily-history backfill for the trend strategy.

    python -m btc_trader.scripts.backfill_daily --years 5

Loads completed daily BTC/JPY bars from the configured provider
(``DAILY_HISTORY_PROVIDER``, default bitbank) and materialises daily bars from
this platform's own bitFlyer ticks. Idempotent: re-running only adds new days.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from btc_trader.config import get_settings
from btc_trader.db.session import dispose_engine, session_scope
from btc_trader.errors import ExchangeError, MarketDataError
from btc_trader.market_data.daily import (
    load_daily_series,
    sync_external_history,
    sync_own_daily_bars,
)
from btc_trader.monitoring.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill daily BTC/JPY history")
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="calendar years to fetch (default: DAILY_HISTORY_YEARS)",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args = parse_args(argv)
    try:
        async with session_scope(settings) as session:
            external = await sync_external_history(session, settings, years=args.years)
            own = await sync_own_daily_bars(session, settings)
            series = await load_daily_series(session, settings)
    except (ExchangeError, MarketDataError) as exc:
        logger.error("daily_backfill_failed", error_type=type(exc).__name__, error=str(exc))
        print(f"daily backfill failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await dispose_engine()
    span = f"{series.bars[0].day} .. {series.last_day}" if series.bars else "none"
    print(
        f"daily backfill complete: {external} external and {own} own bars added; "
        f"{len(series.bars)} daily bars available ({span})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(asyncio.run(main()))
