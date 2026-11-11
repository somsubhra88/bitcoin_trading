"""Rate limiting, retry policy and JSON decoding (PROJECT.md §33)."""

from __future__ import annotations

import asyncio
import random
from decimal import Decimal
from typing import Any

import httpx
import pytest
import respx

from btc_trader.errors import (
    ExchangeAuthenticationError,
    ExchangeRateLimitError,
    ExchangeResponseError,
    ExchangeTransientError,
)
from btc_trader.exchange.http import (
    BASE_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    RateLimitedHttpClient,
    SlidingWindowRateLimiter,
    backoff_delay,
)

BASE_URL = "https://exchange.test"


class FakeClock:
    """Monotonic clock advanced only by the test (or by a patched ``sleep``)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``asyncio.sleep`` with a recorder so tests never really wait."""
    recorded: list[float] = []

    async def _sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return recorded


# --------------------------------------------------------------------------- #
# sliding window
# --------------------------------------------------------------------------- #
def test_limiter_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="max_requests must be positive"):
        SlidingWindowRateLimiter(0, 60)
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        SlidingWindowRateLimiter(1, 0)


async def test_limiter_allows_up_to_the_budget_without_waiting() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(3, 10.0, time_source=clock)
    for _ in range(3):
        assert limiter.delay_for_next() == 0.0
        await limiter.acquire()
    assert limiter.used == 3
    assert limiter.delay_for_next() == pytest.approx(10.0)


async def test_limiter_window_slides(slept: list[float]) -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(2, 10.0, time_source=clock)
    await limiter.acquire()  # t=1000
    clock.advance(4)
    await limiter.acquire()  # t=1004
    assert limiter.delay_for_next() == pytest.approx(6.0)

    # Once the oldest request leaves the window a slot frees up.
    clock.advance(6)  # t=1010, first request is now 10s old
    assert limiter.delay_for_next() == 0.0
    await limiter.acquire()
    assert limiter.used == 2
    assert slept == []


async def test_limiter_acquire_waits_for_the_oldest_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(1, 30.0, time_source=clock)
    waits: list[float] = []

    async def _sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        waits.append(delay)
        clock.advance(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    await limiter.acquire()
    clock.advance(10)
    await limiter.acquire()

    assert waits == [pytest.approx(20.0)]
    assert limiter.used == 1  # the expired timestamp was evicted


# --------------------------------------------------------------------------- #
# backoff
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attempt", range(8))
def test_backoff_is_bounded_and_jittered(attempt: int) -> None:
    rng = random.Random(attempt)  # noqa: S311 - deterministic test jitter
    delay = backoff_delay(attempt, rng=rng)
    ceiling = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2**attempt))
    assert 0.0 <= delay <= ceiling


def test_backoff_ceiling_grows_then_saturates() -> None:
    class _AlwaysCeiling(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return b

    always_max = _AlwaysCeiling()
    assert backoff_delay(0, rng=always_max) == pytest.approx(BASE_BACKOFF_SECONDS)
    assert backoff_delay(3, rng=always_max) == pytest.approx(BASE_BACKOFF_SECONDS * 8)
    assert backoff_delay(20, rng=always_max) == pytest.approx(MAX_BACKOFF_SECONDS)


# --------------------------------------------------------------------------- #
# request handling
# --------------------------------------------------------------------------- #
async def test_json_numbers_are_decoded_as_decimal() -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/x").mock(
            return_value=httpx.Response(200, text='{"price": 12411935.0, "size": 0.00171919}')
        )
        client = RateLimitedHttpClient(BASE_URL)
        try:
            payload = await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()
    assert payload["price"] == Decimal("12411935.0")
    assert isinstance(payload["size"], Decimal)
    # A float round-trip would have produced 0.0017191899999...
    assert str(payload["size"]) == "0.00171919"


async def test_empty_body_decodes_to_none() -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/x").mock(return_value=httpx.Response(204))
        client = RateLimitedHttpClient(BASE_URL)
        try:
            assert await client.request_json("GET", "/v1/x") is None
        finally:
            await client.aclose()


async def test_non_json_body_is_reported_as_a_response_error() -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/x").mock(return_value=httpx.Response(200, text="<html>nope</html>"))
        client = RateLimitedHttpClient(BASE_URL)
        try:
            with pytest.raises(ExchangeResponseError, match="non-JSON"):
                await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()


async def test_transient_error_is_retried_then_succeeds(slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/x").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, text='{"ok": true}'),
            ]
        )
        client = RateLimitedHttpClient(BASE_URL, max_retries=3)
        try:
            assert await client.request_json("GET", "/v1/x") == {"ok": True}
        finally:
            await client.aclose()
    assert route.call_count == 2
    assert len(slept) == 1


async def test_retries_are_exhausted_and_the_last_error_is_raised(slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/x").mock(return_value=httpx.Response(502))
        client = RateLimitedHttpClient(BASE_URL, max_retries=2)
        try:
            with pytest.raises(ExchangeTransientError, match="502"):
                await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()
    assert route.call_count == 3  # 1 attempt + 2 retries
    assert len(slept) == 2


async def test_timeout_is_transient_and_retried(slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/x").mock(side_effect=httpx.ConnectTimeout("timed out"))
        client = RateLimitedHttpClient(BASE_URL, max_retries=1)
        try:
            with pytest.raises(ExchangeTransientError, match="ConnectTimeout"):
                await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()
    assert route.call_count == 2
    assert len(slept) == 1


async def test_rate_limit_response_honours_retry_after(slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/x").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, text="{}"),
            ]
        )
        client = RateLimitedHttpClient(BASE_URL, max_retries=2)
        try:
            assert await client.request_json("GET", "/v1/x") == {}
        finally:
            await client.aclose()
    assert route.call_count == 2
    assert slept == [pytest.approx(7.0)]


async def test_rate_limit_without_retry_after_still_raises(slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/x").mock(return_value=httpx.Response(429))
        client = RateLimitedHttpClient(BASE_URL, max_retries=1)
        try:
            with pytest.raises(ExchangeRateLimitError):
                await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()
    assert len(slept) == 1  # backoff, not Retry-After


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_are_not_retried(status: int, slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/x").mock(return_value=httpx.Response(status))
        client = RateLimitedHttpClient(BASE_URL, max_retries=4)
        try:
            with pytest.raises(ExchangeAuthenticationError) as caught:
                await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()
    assert route.call_count == 1
    assert slept == []
    # §35: the failure message must not carry the response body upward.
    assert "ACCESS-SIGN" not in str(caught.value)


async def test_client_error_is_not_retried(slept: list[float]) -> None:
    with respx.mock(base_url=BASE_URL) as router:
        route = router.get("/v1/x").mock(return_value=httpx.Response(400, text="bad param"))
        client = RateLimitedHttpClient(BASE_URL, max_retries=4)
        try:
            with pytest.raises(ExchangeResponseError, match="400"):
                await client.request_json("GET", "/v1/x")
        finally:
            await client.aclose()
    assert route.call_count == 1
    assert slept == []


async def test_non_idempotent_requests_are_never_retried(slept: list[float]) -> None:
    """§26/§30: a submission must not be silently repeated."""
    with respx.mock(base_url=BASE_URL) as router:
        route = router.post("/v1/x").mock(return_value=httpx.Response(500))
        client = RateLimitedHttpClient(BASE_URL, max_retries=5)
        try:
            with pytest.raises(ExchangeTransientError):
                await client.request_json("POST", "/v1/x", content='{"a":1}', idempotent=False)
        finally:
            await client.aclose()
    assert route.call_count == 1
    assert slept == []


async def test_every_request_consumes_rate_budget() -> None:
    with respx.mock(base_url=BASE_URL) as router:
        router.get("/v1/x").mock(return_value=httpx.Response(200, text="{}"))
        client = RateLimitedHttpClient(BASE_URL, rate_limit_requests=10)
        try:
            assert client.rate_limit_used == 0
            await client.request_json("GET", "/v1/x")
            await client.request_json("GET", "/v1/x")
            assert client.rate_limit_used == 2
        finally:
            await client.aclose()
