"""Scheduler worker entry point.

Run with::

    python -m btc_trader.worker

Hosts the APScheduler jobs (market-data collection today) in its own process, so
the API can be restarted without interrupting data collection.
"""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

from btc_trader.config import get_settings
from btc_trader.db.session import dispose_engine
from btc_trader.monitoring.logging import configure_logging, get_logger
from btc_trader.scheduler.jobs import SchedulerService

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "worker_starting",
        trading_mode=settings.trading_mode.value,
        product_code=settings.product_code,
        market_data_interval_seconds=settings.market_data_interval_seconds,
    )

    scheduler = SchedulerService(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    for signal_name in ("SIGINT", "SIGTERM"):
        with suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(getattr(signal, signal_name), stop.set)

    scheduler.start()
    try:
        await stop.wait()
    finally:
        logger.info("worker_stopping")
        await scheduler.shutdown()
        await dispose_engine()


if __name__ == "__main__":  # pragma: no cover - process entry point
    asyncio.run(main())
