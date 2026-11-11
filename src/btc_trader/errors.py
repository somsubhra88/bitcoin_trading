"""Domain exception hierarchy (PROJECT.md §37).

Errors are never swallowed: they are raised as one of these types so callers can
categorise them, and logged with context at the boundary that handles them.
"""

from __future__ import annotations


class BtcTraderError(Exception):
    """Base class for every error raised by this application."""


# --------------------------------------------------------------------------- #
# exchange
# --------------------------------------------------------------------------- #
class ExchangeError(BtcTraderError):
    """Generic failure while talking to an exchange."""


class ExchangeTransientError(ExchangeError):
    """Retryable failure: timeout, connection reset, 5xx."""


class ExchangeRateLimitError(ExchangeTransientError):
    """The exchange responded 429 / signalled a rate-limit breach."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ExchangeAuthenticationError(ExchangeError):
    """Credentials are missing, malformed or rejected."""


class ExchangeResponseError(ExchangeError):
    """The exchange replied with a payload we cannot parse or trust."""


# --------------------------------------------------------------------------- #
# market data
# --------------------------------------------------------------------------- #
class MarketDataError(BtcTraderError):
    """Collection, normalisation or persistence of market data failed."""


class ConfigurationError(BtcTraderError):
    """The runtime configuration is internally inconsistent."""
