"""Manual single collection (PROJECT.md §11).

    python -m btc_trader.scripts.collect_once

Uses exactly the same collection service as the scheduler and the HTTP endpoint.
Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import asyncio
import sys

from btc_trader.config import get_settings
from btc_trader.db.session import dispose_engine
from btc_trader.errors import MarketDataError
from btc_trader.market_data.collector import collect_market_data
from btc_trader.monitoring.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    try:
        result = await collect_market_data(trigger="cli")
    except MarketDataError as exc:
        logger.error("collect_once_failed", error=str(exc))
        return 1
    finally:
        await dispose_engine()

    print(
        f"{result.product_code} ltp={result.ltp} bid={result.best_bid} ask={result.best_ask} "
        f"tick_inserted={result.tick_inserted} executions_inserted={result.executions_inserted} "
        f"duration_ms={result.duration_ms}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(asyncio.run(main()))
