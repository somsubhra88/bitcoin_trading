"""Daily bar materialisation, merging and the external history provider."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import Candle, MarketTick
from btc_trader.enums import CandleSource
from btc_trader.errors import MarketDataError
from btc_trader.market_data.daily import (
    load_daily_series,
    store_external_bars,
    sync_external_history,
    sync_own_daily_bars,
)
from btc_trader.market_data.history_provider import BitbankDailyHistory, ExternalDailyBar
from tests.fixtures.daily import day_start

TODAY = date(2026, 9, 10)
BITBANK = "https://public.bitbank.cc"


def _external(day: date, close: str) -> ExternalDailyBar:
    price = Decimal(close)
    return ExternalDailyBar(day, price, price, price, price, Decimal("1"))


async def _ticks(session: AsyncSession, day: date, prices: list[str], step_minutes: int) -> None:
    for index, price in enumerate(prices):
        moment = day_start(day) + timedelta(minutes=step_minutes * index)
        session.add(
            MarketTick(
                exchange="bitflyer",
                product_code="BTC_JPY",
                exchange_timestamp=moment,
                received_at=moment,
                ltp=Decimal(price),
                best_bid=Decimal(price),
                best_ask=Decimal(price),
            )
        )
    await session.flush()


async def test_own_daily_bars_cover_completed_days_only(
    session: AsyncSession, settings: Settings
) -> None:
    yesterday = TODAY - timedelta(days=1)
    await _ticks(session, yesterday, ["100", "120", "90", "110"], step_minutes=300)
    await _ticks(session, TODAY, ["111"], step_minutes=5)

    inserted = await sync_own_daily_bars(session, settings, today=TODAY)
    again = await sync_own_daily_bars(session, settings, today=TODAY)

    assert inserted == 1
    assert again == 0
    candle = (await session.execute(select(Candle))).scalar_one()
    assert candle.source is CandleSource.SAMPLED_TICKS
    assert candle.exchange == "bitflyer"
    assert (candle.open, candle.high, candle.low, candle.close) == (
        Decimal("100"),
        Decimal("120"),
        Decimal("90"),
        Decimal("110"),
    )
    assert candle.trade_count == 4


async def test_external_bars_are_stored_idempotently(
    session: AsyncSession, settings: Settings
) -> None:
    bars = [_external(TODAY - timedelta(days=i), "100") for i in range(1, 4)]

    assert await store_external_bars(session, settings, "bitbank", bars) == 3
    assert await store_external_bars(session, settings, "bitbank", bars) == 0
    count = (await session.execute(select(func.count()).select_from(Candle))).scalar_one()
    assert count == 3


async def test_series_prefers_well_sampled_own_bars_and_rescales_the_rest(
    session: AsyncSession, settings: Settings
) -> None:
    hourly = settings.model_copy(update={"market_data_interval_seconds": 3600})
    days = [TODAY - timedelta(days=i) for i in (4, 3, 2, 1)]
    await store_external_bars(session, hourly, "bitbank", [_external(day, "100") for day in days])
    # Two fully sampled own days at a 2% premium, and one sparse day.
    for day in days[2:]:
        await _ticks(session, day, ["102"] * 24, step_minutes=60)
    await _ticks(session, days[1], ["150"], step_minutes=60)
    await sync_own_daily_bars(session, hourly, today=TODAY)

    series = await load_daily_series(session, hourly, today=TODAY)

    assert [bar.day for bar in series.bars] == days
    assert [bar.source for bar in series.bars] == ["bitbank", "bitbank", "bitflyer", "bitflyer"]
    assert series.basis_ratio == Decimal("1.02")
    assert series.bars[0].close == Decimal("102.00")
    assert series.bars[1].close == Decimal("102.00")  # the sparse own day is ignored
    assert series.own_days == 2
    assert series.external_days == 2


async def test_series_excludes_the_still_forming_day(
    session: AsyncSession, settings: Settings
) -> None:
    await store_external_bars(
        session, settings, "bitbank", [_external(TODAY - timedelta(days=1), "100")]
    )
    session.add(
        Candle(
            exchange="bitbank",
            product_code="BTC_JPY",
            interval="1d",
            source=CandleSource.EXCHANGE_OHLC,
            bucket_start=day_start(TODAY),
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            volume=Decimal("0"),
            trade_count=0,
        )
    )
    await session.flush()

    series = await load_daily_series(session, settings, today=TODAY)

    assert series.last_day == TODAY - timedelta(days=1)


def _payload(rows: list[list[object]]) -> dict[str, object]:
    return {"success": 1, "data": {"candlestick": [{"type": "1day", "ohlcv": rows}]}}


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


@respx.mock
async def test_bitbank_provider_parses_completed_days(settings: Settings) -> None:
    respx.get(f"{BITBANK}/btc_jpy/candlestick/1day/2026").mock(
        return_value=httpx.Response(
            200,
            json=_payload(
                [
                    ["100", "110", "90", "105", "12.5", _ms(TODAY - timedelta(days=1))],
                    ["105", "100", "120", "101", "1", _ms(TODAY - timedelta(days=2))],
                    ["106", "107", "105", "106", "3", _ms(TODAY)],
                ]
            ),
        )
    )
    provider = BitbankDailyHistory(settings)
    try:
        bars = await provider.fetch_year(2026, today=TODAY)
    finally:
        await provider.aclose()

    # The forming day is dropped and the high<low row is rejected.
    assert bars == [_external_full(TODAY - timedelta(days=1))]


def _external_full(day: date) -> ExternalDailyBar:
    return ExternalDailyBar(
        day, Decimal("100"), Decimal("110"), Decimal("90"), Decimal("105"), Decimal("12.5")
    )


@respx.mock
async def test_bitbank_error_payload_raises(settings: Settings) -> None:
    respx.get(f"{BITBANK}/btc_jpy/candlestick/1day/2026").mock(
        return_value=httpx.Response(200, json={"success": 0, "data": {"code": 10000}})
    )
    provider = BitbankDailyHistory(settings)
    try:
        with pytest.raises(MarketDataError, match="10000"):
            await provider.fetch_year(2026, today=TODAY)
    finally:
        await provider.aclose()


@respx.mock
async def test_sync_external_history_fetches_each_year_once(
    session: AsyncSession, settings: Settings
) -> None:
    configured = settings.model_copy(update={"daily_history_provider": "bitbank"})
    routes = [
        respx.get(f"{BITBANK}/btc_jpy/candlestick/1day/{year}").mock(
            return_value=httpx.Response(
                200,
                json=_payload([["1", "1", "1", "1", "1", _ms(date(year, 1, 2))]]),
            )
        )
        for year in range(2020, 2031)
    ]

    inserted = await sync_external_history(session, configured, years=2)

    assert inserted == 2
    assert sum(route.call_count for route in routes) == 2


async def test_sync_is_a_no_op_without_a_provider(
    session: AsyncSession, settings: Settings
) -> None:
    assert settings.daily_history_provider == "none"
    assert await sync_external_history(session, settings) == 0
