"""Pre-flight gate between a paper run and real money.

This module exists so that "go live" is a decision someone has to make
explicitly, in more than one place, with the reasons written down. Its default
answer is no.

The rules, in the order they are checked:

1. **Live is off unless ``--enable-live`` is passed.** Not a config key, not an
   environment variable — a command-line flag, because a flag cannot be inherited
   from a stale YAML file or a shell someone else exported.
2. **Credentials come from the environment only.** :func:`credentials_from_env`
   is the single reader. :func:`scan_for_secrets` walks the config tree and
   *fails* if anything that looks like a key is in it, so a credential committed
   to a config file stops the run instead of leaking.
3. **The exchange must be on an allowlist and testnet must be explicit.** An
   unknown venue is refused rather than attempted.
4. **Notional, leverage and symbol caps are hard.** They are checked here against
   the config, and again per-order by the runner, because the two failure modes
   (a wrong config, a runaway agent) are different.

Nothing in this module can be satisfied by a default value. Every gate needs a
positive, specific act from the operator.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from archetype_trader.config import ExperimentConfig

#: Printed verbatim before any live session. Deliberately unmissable.
LIVE_WARNING = """
================================================================================
                        !!!  LIVE TRADING REQUESTED  !!!
================================================================================
This will place REAL orders with REAL money on a REAL exchange.

  * Everything in this repository is a research artifact. The backtests are
    simulations. Simulated results are NOT evidence of future performance, and
    nothing here promises, projects or implies any return.
  * The strategy has known negative findings recorded in the evaluation report.
    Read runs/<name>/report.md, including the "Verdict" and "Known limitations"
    sections, before continuing.
  * Fills, spread, slippage, funding and latency were all *modelled*. A real
    book can be worse than every model at once, simultaneously, at the worst
    possible moment.
  * You are responsible for the exchange account, the margin, the tax and the
    outcome. There is no undo.

Trade only capital you can afford to lose entirely.
================================================================================
"""

#: Venues an adapter has actually been written and tested for. Empty on purpose:
#: this repository ships a paper broker and no live adapter, so there is nothing
#: to allow yet. Adding a name here is the commit that must carry the adapter,
#: its tests and its exchange-specific checks.
SUPPORTED_LIVE_EXCHANGES: frozenset[str] = frozenset()

#: Environment variables the live path reads. Nothing else is consulted.
API_KEY_VAR = "ARCHETYPE_LIVE_API_KEY"
API_SECRET_VAR = "ARCHETYPE_LIVE_API_SECRET"  # noqa: S105
API_PASSPHRASE_VAR = "ARCHETYPE_LIVE_API_PASSPHRASE"  # noqa: S105

_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret|passphrase|password|token|private[_-]?key)",
    re.IGNORECASE,
)
# Long opaque strings are the other tell: a 32+ character run of base64/hex with
# no spaces is not a symbol name or a file path.
_SECRET_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")


class SafetyError(RuntimeError):
    """A pre-flight gate refused. Never caught internally — the run must stop."""


@dataclass(frozen=True)
class Credentials:
    """Opaque credential holder. ``repr`` is redacted so it cannot be logged."""

    key: str
    secret: str
    passphrase: str = ""

    def __repr__(self) -> str:  # pragma: no cover - trivial, but the point of the class
        return f"Credentials(key={_redact(self.key)!r}, secret='***', passphrase='***')"

    def __str__(self) -> str:  # pragma: no cover
        return self.__repr__()


@dataclass(frozen=True)
class LiveLimits:
    """Hard caps applied to a live session, independent of the risk shield.

    The risk shield is part of the strategy and is tuned along with it. These are
    not: they are the operator's ceiling, and the runner refuses to breach them
    even when the shield is satisfied.
    """

    max_order_notional: float = 0.0
    max_position_notional: float = 0.0
    max_leverage: float = 1.0
    max_orders_per_hour: int = 0
    allowed_symbols: tuple[str, ...] = ()
    require_testnet: bool = True

    def check_order(self, *, notional: float, position_notional: float, symbol: str) -> list[str]:
        """Per-order violations. Empty list means the order may be sent."""
        bad: list[str] = []
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            bad.append(f"symbol {symbol!r} not in allowed_symbols {self.allowed_symbols}")
        if self.max_order_notional > 0 and notional > self.max_order_notional:
            bad.append(f"order notional {notional:.2f} > cap {self.max_order_notional:.2f}")
        if self.max_position_notional > 0 and position_notional > self.max_position_notional:
            bad.append(
                f"resulting position notional {position_notional:.2f} > cap "
                f"{self.max_position_notional:.2f}"
            )
        return bad


@dataclass(frozen=True)
class PreflightReport:
    """Outcome of the gate. ``approved`` is only ever true for a paper run here."""

    mode: str
    approved: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def raise_if_blocked(self) -> None:
        if not self.approved:
            joined = "\n".join(f"  - {b}" for b in self.blockers)
            raise SafetyError(f"pre-flight refused a {self.mode} run:\n{joined}")


def _redact(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-2:]}"


def credentials_from_env(*, require_passphrase: bool = False) -> Credentials:
    """Read credentials from the environment. The only reader in the repository.

    Raises rather than returning empty strings: a live run that authenticates as
    nobody would fail later, at a less convenient moment, possibly after the
    strategy has already computed a position it thinks it holds.
    """
    key = os.environ.get(API_KEY_VAR, "").strip()
    secret = os.environ.get(API_SECRET_VAR, "").strip()
    passphrase = os.environ.get(API_PASSPHRASE_VAR, "").strip()
    missing = [name for name, v in ((API_KEY_VAR, key), (API_SECRET_VAR, secret)) if not v]
    if require_passphrase and not passphrase:
        missing.append(API_PASSPHRASE_VAR)
    if missing:
        raise SafetyError(
            "live credentials must come from the environment; missing: "
            + ", ".join(missing)
            + ". Export them in the shell that launches the run and nowhere else."
        )
    return Credentials(key=key, secret=secret, passphrase=passphrase)


def scan_for_secrets(tree: Any, *, path: str = "") -> list[str]:
    """Return the config paths that look like they contain a credential.

    Two independent tests, because either alone misses a real case: a *key* that
    names a secret, and a *value* shaped like one. A false positive here costs a
    renamed config field; a false negative costs a leaked key.
    """
    found: list[str] = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            here = f"{path}.{k}" if path else str(k)
            if _SECRET_KEY_PATTERN.search(str(k)) and v not in (None, "", 0, False):
                found.append(here)
            found.extend(scan_for_secrets(v, path=here))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            found.extend(scan_for_secrets(v, path=f"{path}[{i}]"))
    elif isinstance(tree, str) and _SECRET_VALUE_PATTERN.match(tree):
        found.append(path)
    return found


def preflight(
    cfg: ExperimentConfig,
    *,
    mode: str,
    enable_live: bool = False,
    exchange: str = "paper",
    testnet: bool = True,
    limits: LiveLimits | None = None,
    confirm_phrase: str | None = None,
) -> PreflightReport:
    """Decide whether a session may start. ``mode`` is ``"paper"`` or ``"live"``.

    A paper run is approved unless its config is self-contradictory. A live run is
    refused by this repository in every case, because no live adapter has been
    written or tested — :data:`SUPPORTED_LIVE_EXCHANGES` is empty and the refusal
    names that explicitly rather than pretending the gate is about credentials.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    lim = limits or LiveLimits()

    leaked = scan_for_secrets(cfg.model_dump(mode="json"))
    if leaked:
        blockers.append(
            "config contains fields that look like credentials: "
            + ", ".join(leaked)
            + f"; move them to {API_KEY_VAR}/{API_SECRET_VAR} in the environment"
        )

    if mode not in ("paper", "live"):
        blockers.append(f"unknown mode {mode!r}; expected 'paper' or 'live'")

    if cfg.env.mode == "perp" and cfg.env.leverage > lim.max_leverage:
        # A blocker for live, a warning for paper. The operator cap is a
        # *live-session* limit; refusing a paper run over it would push people to
        # relax the cap globally just to get a simulation to start, which is
        # exactly the habit this gate exists to prevent.
        msg = (
            f"env.leverage={cfg.env.leverage} exceeds the operator cap "
            f"{lim.max_leverage}; raise LiveLimits.max_leverage deliberately or "
            "lower the config"
        )
        (blockers if mode == "live" else warnings).append(msg)
    if not cfg.shield.enabled:
        msg = "risk shield is disabled in the config"
        if mode == "live":
            blockers.append(msg + "; a live run without the deterministic shield is refused")
        else:
            warnings.append(msg + "; paper results will not reflect a shielded deployment")

    if mode == "live":
        if not enable_live:
            blockers.append(
                "live execution is disabled by default; pass --enable-live to request it"
            )
        venue = exchange.strip().lower()
        if venue not in SUPPORTED_LIVE_EXCHANGES:
            blockers.append(
                f"no live adapter exists for exchange {exchange!r} "
                f"(supported: {sorted(SUPPORTED_LIVE_EXCHANGES) or 'none'}). This "
                "repository ships a paper broker only; a live adapter must be "
                "written, tested and added to SUPPORTED_LIVE_EXCHANGES first."
            )
        if lim.require_testnet and not testnet:
            blockers.append("mainnet requested but LiveLimits.require_testnet is set")
        if lim.max_order_notional <= 0 or lim.max_position_notional <= 0:
            blockers.append(
                "live runs require explicit max_order_notional and "
                "max_position_notional caps; there is no safe default"
            )
        if confirm_phrase != "I ACCEPT THE RISK":
            blockers.append("live runs require --confirm 'I ACCEPT THE RISK' typed out in full")
        try:
            credentials_from_env()
        except SafetyError as exc:
            blockers.append(str(exc))

    return PreflightReport(
        mode=mode,
        approved=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        notes={
            "exchange": exchange,
            "testnet": testnet,
            "enable_live": enable_live,
            "shield_enabled": cfg.shield.enabled,
            "env_mode": cfg.env.mode,
            "leverage": cfg.env.leverage,
            "limits": {
                "max_order_notional": lim.max_order_notional,
                "max_position_notional": lim.max_position_notional,
                "max_leverage": lim.max_leverage,
                "max_orders_per_hour": lim.max_orders_per_hour,
                "allowed_symbols": list(lim.allowed_symbols),
                "require_testnet": lim.require_testnet,
            },
        },
    )
