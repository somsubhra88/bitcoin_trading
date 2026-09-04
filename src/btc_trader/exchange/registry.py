"""Exchange factory.

``EXCHANGE=bitflyer`` selects the adapter.  Adding an exchange means adding one
entry here and one module implementing :class:`ExchangeClient`.
"""

from __future__ import annotations

from collections.abc import Callable

from btc_trader.config import Settings, get_settings
from btc_trader.errors import ConfigurationError
from btc_trader.exchange.base import ExchangeClient
from btc_trader.exchange.bitflyer import BitflyerClient

EXCHANGE_CLIENTS: dict[str, Callable[[Settings], ExchangeClient]] = {
    "bitflyer": lambda settings: BitflyerClient(settings),
}


def create_exchange_client(settings: Settings | None = None) -> ExchangeClient:
    resolved = settings or get_settings()
    try:
        factory = EXCHANGE_CLIENTS[resolved.exchange]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown EXCHANGE={resolved.exchange!r}; available: {sorted(EXCHANGE_CLIENTS)}"
        ) from exc
    return factory(resolved)
