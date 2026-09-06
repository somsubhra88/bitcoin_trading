"""Paper-trading adapter: the deployment-shaped path through the trained stack.

Three things live here and nothing else:

* :mod:`~archetype_trader.live.broker` — the order/account seam, plus a paper
  broker that replays historical bars through the same ledger and execution
  simulator the backtest uses.
* :mod:`~archetype_trader.live.agent` — the trained stack driven one bar at a
  time, holding recurrent state and no arrays.
* :mod:`~archetype_trader.live.runner` — the loop that wires agent, risk shield,
  operator limits and broker together, and journals every gate separately.
* :mod:`~archetype_trader.live.safety` — the pre-flight gate. Its default answer
  is no, and for live execution it is currently *always* no: no live exchange
  adapter has been written or tested, and
  :data:`~archetype_trader.live.safety.SUPPORTED_LIVE_EXCHANGES` is empty.

Nothing in this package imports :mod:`archetype_trader.planner` or
:mod:`archetype_trader.phase3.hindsight`. That is enforced by a test over the
whole import closure, not by convention: those two modules are the only ones that
index the future, and the deployed path must not be able to reach them.
"""

from __future__ import annotations

from archetype_trader.live.agent import (
    AgentStats,
    BarDecision,
    HorizonAgent,
    LoadedStack,
    build_agent,
    load_stack,
)
from archetype_trader.live.broker import (
    AccountState,
    BarUpdate,
    Broker,
    BrokerError,
    BrokerRejection,
    Fill,
    FlakyTransport,
    Order,
    OrderAck,
    OrderState,
    PaperBroker,
    Reconciliation,
    client_order_id,
)
from archetype_trader.live.runner import (
    PaperRunner,
    PaperRunResult,
    RunnerMetrics,
    action_sequence,
)
from archetype_trader.live.safety import (
    API_KEY_VAR,
    API_PASSPHRASE_VAR,
    API_SECRET_VAR,
    LIVE_WARNING,
    SUPPORTED_LIVE_EXCHANGES,
    Credentials,
    LiveLimits,
    PreflightReport,
    SafetyError,
    credentials_from_env,
    preflight,
    scan_for_secrets,
)

__all__ = [
    "API_KEY_VAR",
    "API_PASSPHRASE_VAR",
    "API_SECRET_VAR",
    "LIVE_WARNING",
    "SUPPORTED_LIVE_EXCHANGES",
    "AccountState",
    "AgentStats",
    "BarDecision",
    "BarUpdate",
    "Broker",
    "BrokerError",
    "BrokerRejection",
    "Credentials",
    "Fill",
    "FlakyTransport",
    "HorizonAgent",
    "LiveLimits",
    "LoadedStack",
    "Order",
    "OrderAck",
    "OrderState",
    "PaperBroker",
    "PaperRunResult",
    "PaperRunner",
    "PreflightReport",
    "Reconciliation",
    "RunnerMetrics",
    "SafetyError",
    "action_sequence",
    "build_agent",
    "client_order_id",
    "credentials_from_env",
    "load_stack",
    "preflight",
    "scan_for_secrets",
]
