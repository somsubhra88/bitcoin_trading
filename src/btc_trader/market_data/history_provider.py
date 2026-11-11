"""External daily OHLC history (PROJECT.md §39: "explicitly introduce another
data provider").

bitFlyer's public API has no candle endpoint and serves only a few weeks of
executions, but the trend ensemble needs at least a year of daily closes. The
bitbank public API publishes complete BTC/JPY daily candles by calendar year,
without credentials, from a liquid Japanese spot market. Those bars are stored
in ``candles`` with ``source=exchange_ohlc`` so their provenance is never
confused with bitFlyer data, and only *completed* UTC days are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from btc_trader.config import Settings
from btc_trader.errors import MarketDataError
from btc_trader.exchange.http import RateLimitedHttpClient
from btc_trader.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExternalDailyBar:
    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class BitbankDailyHistory:
    """Read-only client for ``GET /{pair}/candlestick/1day/{YYYY}``."""

    name = "bitbank"

    def __init__(self, settings: Settings, *, http: RateLimitedHttpClient | None = None) -> None:
        self._pair = settings.daily_history_pair
        self._owns_http = http is None
        self._http = http or RateLimitedHttpClient(
            settings.daily_history_base_url,
            timeout_seconds=settings.bitflyer_timeout_seconds,
            max_retries=settings.bitflyer_max_retries,
            rate_limit_requests=60,
            rate_limit_window_seconds=60.0,
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def fetch_year(self, year: int, *, today: date | None = None) -> list[ExternalDailyBar]:
        """Completed daily bars of ``year``; the still-forming current day is dropped."""
        payload = await self._http.request_json("GET", f"/{self._pair}/candlestick/1day/{year}")
        cutoff = today or datetime.now(UTC).date()
        return [bar for bar in _parse(payload) if bar.day < cutoff]


def _parse(payload: Any) -> list[ExternalDailyBar]:
    if not isinstance(payload, dict) or payload.get("success") != 1:
        code = (payload or {}).get("data", {}).get("code") if isinstance(payload, dict) else None
        raise MarketDataError(f"bitbank candlestick request failed (code {code})")
    try:
        rows = payload["data"]["candlestick"][0]["ohlcv"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MarketDataError("bitbank candlestick payload has an unexpected shape") from exc
    bars: list[ExternalDailyBar] = []
    for row in rows:
        try:
            open_, high, low, close, volume = (Decimal(str(value)) for value in row[:5])
            day = datetime.fromtimestamp(int(row[5]) / 1000, UTC).date()
        except (InvalidOperation, TypeError, ValueError, IndexError):
            logger.warning("daily_history_row_skipped", row=str(row)[:120])
            continue
        if min(open_, high, low, close) <= 0 or high < low:
            logger.warning("daily_history_row_rejected", day=str(day))
            continue
        bars.append(ExternalDailyBar(day, open_, high, low, close, volume))
    return bars


def create_daily_history_provider(settings: Settings) -> BitbankDailyHistory | None:
    if settings.daily_history_provider == "bitbank":
        return BitbankDailyHistory(settings)
    return None


__all__ = ["BitbankDailyHistory", "ExternalDailyBar", "create_daily_history_provider"]
