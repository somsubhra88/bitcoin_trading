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


class ExchangeUnavailableError(ExchangeTransientError):
    """The exchange reports itself as down / halted."""


# --------------------------------------------------------------------------- #
# market data
# --------------------------------------------------------------------------- #
class MarketDataError(BtcTraderError):
    """Collection, normalisation or persistence of market data failed."""


class StaleMarketDataError(MarketDataError):
    """Latest stored observation is older than MAX_MARKET_DATA_AGE_SECONDS."""


# --------------------------------------------------------------------------- #
# trading (raised by later phases; defined here so callers can rely on them)
# --------------------------------------------------------------------------- #
class RiskViolation(BtcTraderError):
    """A risk rule rejected a proposed order."""


class ProfitabilityViolation(BtcTraderError):
    """A strategy exit failed the configured minimum-net-profit requirement."""


class OrderSubmissionError(BtcTraderError):
    """An order could not be submitted."""


class DuplicateOrderError(OrderSubmissionError):
    """An identical pending order already exists."""


class TradingDisabledError(BtcTraderError):
    """Configuration forbids sending orders (mode, flags or kill switch)."""


class ConfigurationError(BtcTraderError):
    """The runtime configuration is internally inconsistent."""
