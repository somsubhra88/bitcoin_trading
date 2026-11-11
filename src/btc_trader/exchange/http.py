"""Centralised, rate-limited, retrying HTTP transport (PROJECT.md §33).

Every exchange request goes through :class:`RateLimitedHttpClient`.  No module
may create its own ``httpx`` client — that is what makes the rate budget and the
retry policy actually enforceable.

Retry policy:

* read requests (``idempotent=True``) — retried on timeout, connection error,
  429 and 5xx with exponential backoff plus jitter;
* trading requests (``idempotent=False``) — **never** retried automatically, so a
  submission is never silently duplicated (PROJECT.md §26/§30/§51.26).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from decimal import Decimal
from typing import Any

import httpx

from btc_trader.errors import (
    ExchangeAuthenticationError,
    ExchangeRateLimitError,
    ExchangeResponseError,
    ExchangeTransientError,
)
from btc_trader.monitoring.logging import get_logger

logger = get_logger(__name__)

#: Cap on a single backoff sleep, seconds.
MAX_BACKOFF_SECONDS = 20.0
#: Base of the exponential backoff, seconds.
BASE_BACKOFF_SECONDS = 0.5


class SlidingWindowRateLimiter:
    """Allow at most ``max_requests`` per ``window_seconds``, sliding.

    bitFlyer documents a per-IP budget over a 5-minute window, which a sliding
    window models directly.  Independent of the event loop clock so it can be
    unit-tested with an injected ``time_source``.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        *,
        time_source: Any = time.monotonic,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._time = time_source
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def delay_for_next(self) -> float:
        """Seconds a caller must wait before the next request is permitted."""
        now = self._time()
        self._evict(now)
        if len(self._timestamps) < self._max_requests:
            return 0.0
        return max(0.0, self._timestamps[0] + self._window_seconds - now)

    async def acquire(self) -> None:
        """Block until a request slot is available, then consume it."""
        async with self._lock:
            while True:
                delay = self.delay_for_next()
                if delay <= 0:
                    self._timestamps.append(self._time())
                    return
                logger.debug("rate_limit_wait", delay_seconds=round(delay, 3))
                await asyncio.sleep(delay)

    @property
    def used(self) -> int:
        self._evict(self._time())
        return len(self._timestamps)


def backoff_delay(attempt: int, *, rng: random.Random | None = None) -> float:
    """Exponential backoff with full jitter for retry ``attempt`` (0-based)."""
    generator = rng or random
    ceiling = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2**attempt))
    return generator.uniform(0.0, ceiling)


class RateLimitedHttpClient:
    """Thin ``httpx`` wrapper adding rate limiting, retries and error mapping."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 4,
        rate_limit_requests: int = 400,
        rate_limit_window_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._rng = rng
        self._limiter = SlidingWindowRateLimiter(rate_limit_requests, rate_limit_window_seconds)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "btc-trader/0.1 (+https://github.com/)"},
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    @property
    def rate_limit_used(self) -> int:
        return self._limiter.used

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        content: str | None = None,
        headers: dict[str, str] | None = None,
        idempotent: bool = True,
    ) -> Any:
        """Perform a request and decode its JSON body with ``Decimal`` numbers.

        ``content`` is sent byte-for-byte (signed requests need exactly that).
        ``idempotent=False`` disables all retries.
        """
        attempts = self._max_retries + 1 if idempotent else 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            await self._limiter.acquire()
            started = time.perf_counter()
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    content=content,
                    headers=headers,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ExchangeTransientError(f"{type(exc).__name__}: {exc}")
                logger.warning(
                    "exchange_request_transport_error",
                    method=method,
                    path=path,
                    attempt=attempt + 1,
                    error=str(exc),
                )
            else:
                duration_ms = int((time.perf_counter() - started) * 1000)
                try:
                    return self._handle_response(response, method, path, duration_ms)
                except ExchangeRateLimitError as exc:
                    last_error = exc
                    if exc.retry_after_seconds and attempt + 1 < attempts:
                        await asyncio.sleep(exc.retry_after_seconds)
                        continue
                except ExchangeTransientError as exc:
                    last_error = exc

            if attempt + 1 < attempts:
                delay = backoff_delay(attempt, rng=self._rng)
                logger.info(
                    "exchange_request_retry",
                    method=method,
                    path=path,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 3),
                )
                await asyncio.sleep(delay)

        assert last_error is not None  # noqa: S101 - loop always sets it before exit
        raise last_error

    # ------------------------------------------------------------------ #
    def _handle_response(
        self, response: httpx.Response, method: str, path: str, duration_ms: int
    ) -> Any:
        status = response.status_code

        if status == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            logger.warning(
                "exchange_rate_limited", method=method, path=path, retry_after=retry_after
            )
            raise ExchangeRateLimitError(
                f"rate limited by exchange on {method} {path}",
                retry_after_seconds=retry_after,
            )

        if status in (401, 403):
            # Body may echo request context; log status only, never credentials.
            raise ExchangeAuthenticationError(
                f"exchange rejected credentials on {method} {path} (HTTP {status})"
            )

        if status >= 500:
            raise ExchangeTransientError(f"exchange returned HTTP {status} on {method} {path}")

        if status >= 400:
            raise ExchangeResponseError(
                f"exchange returned HTTP {status} on {method} {path}: {response.text[:400]}"
            )

        logger.debug(
            "exchange_request_ok",
            method=method,
            path=path,
            status=status,
            duration_ms=duration_ms,
        )
        return _decode_json(response)


def _decode_json(response: httpx.Response) -> Any:
    """Decode JSON keeping full numeric precision (``parse_float=Decimal``)."""
    if not response.content:
        return None
    try:
        return json.loads(response.text, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise ExchangeResponseError(f"exchange returned non-JSON body: {exc}") from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
