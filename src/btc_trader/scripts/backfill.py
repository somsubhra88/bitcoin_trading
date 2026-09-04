"""Execution-history backfill (PROJECT.md §39).

    python -m btc_trader.scripts.backfill --pages 20

Walks bitFlyer's public ``/v1/executions`` endpoint backwards using its ``before``
cursor, storing each page idempotently.  This uses **only** data bitFlyer actually
serves: no candle history is fabricated.  bitFlyer's public execution history is
shallow, so the reachable depth is limited by the exchange, not by this script.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from btc_trader.config import get_settings
from btc_trader.db.session import dispose_engine, session_scope
from btc_trader.errors import ExchangeError
from btc_trader.exchange.bitflyer import MAX_EXECUTIONS_PER_REQUEST
from btc_trader.exchange.registry import create_exchange_client
from btc_trader.market_data.repository import MarketDataRepository
from btc_trader.monitoring.logging import configure_logging, get_logger

logger = get_logger(__name__)

DEFAULT_PAGES = 10
#: Courtesy pause between pages; the client's rate limiter is the hard guard.
PAGE_PAUSE_SECONDS = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill public BTC/JPY executions")
    parser.add_argument(
        "--pages", type=int, default=DEFAULT_PAGES, help="maximum pages to walk backwards"
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=MAX_EXECUTIONS_PER_REQUEST,
        help=f"executions per request (max {MAX_EXECUTIONS_PER_REQUEST})",
    )
    parser.add_argument(
        "--before",
        type=int,
        default=None,
        help="start cursor; defaults to the oldest execution id already stored",
    )
    return parser.parse_args(argv)


async def backfill(pages: int, page_size: int, before: int | None) -> int:
    """Store up to ``pages`` pages of older executions. Returns rows inserted."""
    settings = get_settings()
    client = create_exchange_client(settings)
    total_inserted = 0

    try:
        if before is None:
            async with session_scope(settings) as session:
                before = await MarketDataRepository(session).oldest_stored_execution_id(
                    settings.product_code, client.name
                )
            if before is None:
                logger.info(
                    "backfill_seeding",
                    reason="no stored executions; starting from the newest available",
                )

        for page in range(pages):
            try:
                executions = await client.get_recent_executions(
                    settings.product_code, limit=page_size, before_id=before
                )
            except ExchangeError as exc:
                logger.error(
                    "backfill_page_failed",
                    page=page,
                    before=before,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                break

            if not executions:
                logger.info("backfill_exhausted", page=page, before=before)
                break

            async with session_scope(settings) as session:
                inserted = await MarketDataRepository(session).insert_executions(executions)
            total_inserted += inserted

            oldest = min(int(item.exchange_execution_id) for item in executions)
            logger.info(
                "backfill_page_stored",
                page=page,
                fetched=len(executions),
                inserted=inserted,
                oldest_id=oldest,
            )

            if before is not None and oldest >= before:
                # No progress backwards: the exchange has nothing older to give.
                logger.info("backfill_no_progress", oldest_id=oldest, cursor=before)
                break
            before = oldest
            await asyncio.sleep(PAGE_PAUSE_SECONDS)
    finally:
        await client.aclose()
        await dispose_engine()

    return total_inserted


async def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args = parse_args(argv)
    page_size = min(max(args.page_size, 1), MAX_EXECUTIONS_PER_REQUEST)
    inserted = await backfill(max(args.pages, 1), page_size, args.before)
    print(f"backfill complete: {inserted} new executions stored")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(asyncio.run(main()))
