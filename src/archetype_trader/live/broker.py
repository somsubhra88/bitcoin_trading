"""Broker abstraction and a paper broker that replays historical bars.

Why a broker at all, when :class:`~archetype_trader.env.trading_env.TradingEnv`
already simulates fills: the environment is a *training* interface. It hands the
agent an observation and takes an action index, and it owns the clock. A live or
paper deployment inverts that — the exchange owns the clock, the agent is handed
an account snapshot it did not compute, and every instruction it issues can be
rejected, partially filled, duplicated by a retry, or acknowledged and then lost.
The :class:`Broker` protocol is the seam where those failure modes live, so the
same seam can be filled by :class:`PaperBroker` here or by a real exchange
adapter later without the agent noticing.

Three properties are deliberate:

* **Orders carry a target position, not a quantity.** The delta is computed at
  execution time from the account state and the price then observable. A quantity
  computed at decision time would be stale by exactly the latency the rest of the
  repo is careful to model, and would silently drift from the target after a
  partial fill. A real adapter should do the same translation against the live
  book at submit time.
* **Submission is an acknowledgement, not a fill.** ``submit`` accepts an order
  and returns immediately; the fill materialises on the next
  :meth:`PaperBroker.advance`, which prices it at the *next* bar's open. This is
  the same next-bar rule the environment enforces, expressed as a queue instead
  of a loop.
* **``client_order_id`` is the idempotency key.** Re-submitting an id that the
  broker has already seen returns the existing record and touches nothing. That is
  what makes a blind retry after an ambiguous network failure safe: at most one
  order per id can ever exist, so a retry cannot double the position.

Accounting reuses :class:`~archetype_trader.env.ledger.Ledger` and
:class:`~archetype_trader.env.execution.ExecutionSimulator` rather than
re-deriving them, so a paper run and a backtest of the same decisions agree to
floating-point noise. ``tests/archetype/test_live.py`` asserts that.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from archetype_trader.config import EnvConfig
from archetype_trader.data.types import MarketData
from archetype_trader.env.execution import ExecutionResult, ExecutionSimulator, FillStatus
from archetype_trader.env.ledger import Ledger

_EPS = 1e-12


class BrokerError(RuntimeError):
    """Transient broker/transport failure. The caller may retry the same order id."""


class BrokerRejection(RuntimeError):
    """Permanent rejection. Retrying the same order will fail again."""


class OrderState(StrEnum):
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Order:
    """A target-position instruction.

    ``target_position`` is a signed fraction of equity, matching
    :attr:`archetype_trader.config.EnvConfig.positions`. ``qty`` is intentionally
    absent: see the module docstring.
    """

    client_order_id: str
    symbol: str
    target_position: float
    #: Bar index (paper) or wall-clock epoch (live) at which the decision was made.
    decided_at: int
    reduce_only: bool = False
    tag: str = ""


@dataclass(frozen=True)
class Fill:
    """Terminal execution outcome for one order."""

    client_order_id: str
    symbol: str
    state: OrderState
    filled_qty: float
    avg_price: float
    fee: float
    slippage_bps: float
    participation: float
    #: Bar index (paper) or epoch (live) at which the fill was booked.
    filled_at: int
    reason: str = ""

    @property
    def side(self) -> str:
        if self.filled_qty > 0:
            return "buy"
        if self.filled_qty < 0:
            return "sell"
        return "none"

    @property
    def notional(self) -> float:
        return abs(self.filled_qty) * self.avg_price


@dataclass(frozen=True)
class OrderAck:
    """What ``submit`` returns. ``duplicate`` is the idempotency signal."""

    client_order_id: str
    state: OrderState
    duplicate: bool
    reason: str = ""


@dataclass(frozen=True)
class AccountState:
    """Everything the agent is allowed to know about its own position.

    Sourced from the broker, never from the agent's own bookkeeping: an agent that
    trusts its intended position over the reported one will silently diverge the
    first time a fill is partial or rejected.
    """

    index: int
    timestamp: np.datetime64
    cash: float
    qty: float
    mark: float
    equity: float
    position: float
    unrealized_pnl: float
    drawdown: float
    n_trades: int
    cum_fees: float
    cum_funding: float
    cum_execution_cost: float
    cum_turnover_notional: float
    is_stale: bool
    spread_bps: float
    bar_volume: float
    realized_vol_per_bar: float
    day_index: int
    liquidated: bool = False
    bankrupt: bool = False


@dataclass(frozen=True)
class BarUpdate:
    """Result of advancing the paper clock one bar."""

    index: int
    account: AccountState
    fills: tuple[Fill, ...]
    funding_cost: float
    #: Net log change in equity over the bar, all costs included.
    net_log_return: float
    #: Public mark-to-mark log return over the same bar, position-independent.
    mark_log_return: float
    #: Execution cost paid on this bar's fills as a fraction of post-fill equity,
    #: both at the execution mark. The exact reconciliation overshoot per unit of
    #: target position; see :meth:`PaperBroker.reconcile`.
    execution_cost_frac: float
    liquidated: bool
    bankrupt: bool
    done: bool


@dataclass(frozen=True)
class Reconciliation:
    """Comparison of what the agent intended against what the broker reports."""

    expected_position: float
    actual_position: float
    drift: float
    tolerance: float
    ok: bool
    note: str = ""


@runtime_checkable
class Broker(Protocol):
    """The seam a live adapter must fill.

    Kept minimal on purpose. Anything an exchange offers beyond this (order books,
    conditional orders, sub-accounts) is out of scope for a target-position
    strategy and would only widen the surface that has to be tested.
    """

    @property
    def symbol(self) -> str: ...

    def account(self) -> AccountState: ...

    def submit(self, order: Order) -> OrderAck: ...

    def fill_of(self, client_order_id: str) -> Fill | None: ...

    def open_orders(self) -> tuple[Order, ...]: ...

    def cancel_all(self) -> int: ...

    def flatten(self, *, reason: str = "manual") -> Fill | None: ...

    def reconcile(self, expected_position: float, *, tolerance: float = 1e-6) -> Reconciliation: ...


def client_order_id(
    *,
    run_id: str,
    index: int,
    target_position: float,
    attempt: int = 0,
) -> str:
    """Deterministic, collision-resistant idempotency key.

    Deterministic in ``(run_id, index, target_position)`` and **not** in
    ``attempt``: a retry of the same decision must reuse the same id or the
    idempotency guarantee is worthless. ``attempt`` is accepted so a caller can
    deliberately mint a *fresh* id after a permanent rejection, which is a
    different decision, and is recorded in the hash so the two cannot be confused.
    """
    payload = f"{run_id}|{index}|{target_position:.10f}|{attempt}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"at-{index}-{digest}"


class PaperBroker:
    """Historical-replay broker with a real order lifecycle.

    Mirrors :class:`~archetype_trader.env.trading_env.TradingEnv`'s bar mechanics
    exactly — mark to the next open, execute, mark to the next close, charge
    funding, then check liquidation — because a paper run whose economics differ
    from the backtest's would make the backtest untestable.
    """

    def __init__(
        self,
        data: MarketData,
        cfg: EnvConfig,
        *,
        start_index: int,
        end_index: int,
        symbol: str | None = None,
        bars_per_year: int = 52_560,
        vol_lookback: int = 36,
        reconcile_every_bar: bool = True,
        fault_injector: Callable[[Order], None] | None = None,
        ack_fault_injector: Callable[[Order], None] | None = None,
    ) -> None:
        if end_index <= start_index or end_index > len(data):
            raise ValueError(f"invalid replay range [{start_index}:{end_index})")
        self.data = data
        self.cfg = cfg
        self._symbol = symbol or "PAPER"
        self.start_index = int(start_index)
        self.end_index = int(end_index)
        self.bars_per_year = bars_per_year
        self.reconcile_every_bar = reconcile_every_bar
        # Injected transport faults. The runner's retry path is otherwise
        # unreachable in a paper run, and an untested retry path is a liability.
        # The two hooks model the two genuinely different failure windows:
        # ``fault_injector`` fires before the order is recorded (the broker never
        # saw it), ``ack_fault_injector`` after (the broker has it but the caller
        # never learned that). Only the second can produce a double fill, and only
        # the ``client_order_id`` check prevents it.
        self.fault_injector = fault_injector
        self.ack_fault_injector = ack_fault_injector

        self.sim = ExecutionSimulator(data, cfg.costs, mode=cfg.mode)
        self.ledger = Ledger(initial_cash=cfg.initial_cash, allow_negative_cash=cfg.mode == "perp")
        self.ledger.initialise_mark(float(data.close[self.start_index]))

        from archetype_trader.features import rolling as rr

        log_ret = data.log_returns()
        self._realized_vol = np.nan_to_num(rr.rolling_std(log_ret, vol_lookback), nan=0.0)
        self._spread_bps = data.half_spread_frac(cfg.costs.fallback_half_spread_bps) * 2e4
        self._day_index = data.ts.astype("datetime64[D]").astype("int64").astype(np.int64)

        self._t = self.start_index
        self._records: dict[str, dict[str, Any]] = {}
        self._queue: list[Order] = []
        self._liquidated = False
        self._bankrupt = False
        self._post_exec_position: float | None = None
        self._post_exec_cost_frac: float = 0.0
        self.n_partial_fills = 0
        self.n_rejected = 0
        self.n_duplicate_submissions = 0
        self.n_liquidations = 0
        self.equity_curve: list[float] = [self.ledger.equity]
        self.net_log_returns: list[float] = []
        self.positions: list[float] = [0.0]

    # -- introspection ------------------------------------------------------ #
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def index(self) -> int:
        return self._t

    @property
    def done(self) -> bool:
        return self._t >= self.end_index - 1 or self._bankrupt or self._liquidated

    def account(self) -> AccountState:
        t = self._t
        return AccountState(
            index=t,
            timestamp=self.data.ts[t],
            cash=self.ledger.cash,
            qty=self.ledger.qty,
            mark=self.ledger.mark,
            equity=self.ledger.equity,
            position=self.ledger.position_frac,
            unrealized_pnl=self.ledger.unrealized_pnl,
            drawdown=self.ledger.drawdown,
            n_trades=self.ledger.n_trades,
            cum_fees=self.ledger.cum_fees,
            cum_funding=self.ledger.cum_funding,
            cum_execution_cost=self.ledger.cum_execution_cost,
            cum_turnover_notional=self.ledger.cum_turnover_notional,
            is_stale=self.sim.is_stale(t),
            spread_bps=float(self._spread_bps[t]),
            bar_volume=float(self.data.volume[t]),
            realized_vol_per_bar=float(self._realized_vol[t]),
            day_index=int(self._day_index[t]),
            liquidated=self._liquidated,
            bankrupt=self._bankrupt,
        )

    # -- order lifecycle ---------------------------------------------------- #
    def submit(self, order: Order) -> OrderAck:
        """Accept an order for execution on the next bar.

        Idempotent in ``order.client_order_id``. A duplicate is *not* an error: it
        is the expected outcome of a retry, and returning the original ack is what
        lets the caller retry blindly.
        """
        existing = self._records.get(order.client_order_id)
        if existing is not None:
            self.n_duplicate_submissions += 1
            return OrderAck(
                client_order_id=order.client_order_id,
                state=existing["state"],
                duplicate=True,
                reason="already known to the broker; not re-queued",
            )
        if order.symbol != self._symbol:
            raise BrokerRejection(f"order for {order.symbol!r}, broker trades {self._symbol!r}")
        if not np.isfinite(order.target_position):
            raise BrokerRejection(f"non-finite target position {order.target_position!r}")
        if self._bankrupt or self._liquidated:
            raise BrokerRejection("account is bankrupt or liquidated; no new orders")

        # Fault injection happens *after* the idempotency check and *before* the
        # order is recorded, which is the genuinely dangerous window: the caller
        # cannot know whether the broker saw it. Recording it here anyway would
        # make the test vacuous.
        if self.fault_injector is not None:
            self.fault_injector(order)

        self._records[order.client_order_id] = {
            "order": order,
            "state": OrderState.ACCEPTED,
            "fill": None,
        }
        self._queue.append(order)
        if self.ack_fault_injector is not None:
            # The order is live at the broker at this point. Whatever this raises,
            # the position will move on the next bar.
            self.ack_fault_injector(order)
        return OrderAck(order.client_order_id, OrderState.ACCEPTED, duplicate=False)

    def fill_of(self, client_order_id: str) -> Fill | None:
        rec = self._records.get(client_order_id)
        return None if rec is None else rec["fill"]

    def state_of(self, client_order_id: str) -> OrderState | None:
        rec = self._records.get(client_order_id)
        return None if rec is None else rec["state"]

    def open_orders(self) -> tuple[Order, ...]:
        return tuple(self._queue)

    def cancel_all(self) -> int:
        n = len(self._queue)
        for order in self._queue:
            self._records[order.client_order_id]["state"] = OrderState.CANCELLED
        self._queue.clear()
        return n

    # -- clock -------------------------------------------------------------- #
    def advance(self) -> BarUpdate:
        """Advance one bar: execute the queue, mark, fund, check liquidation."""
        if self.done:
            raise RuntimeError("paper replay is finished; check .done before advancing")
        t = self._t
        next_t = t + 1
        equity_before = self.ledger.equity

        # Holding PnL first: the inventory carried into the bar is marked to the
        # execution reference before anything trades, so price movement is never
        # booked as an execution cost.
        self.ledger.set_mark(float(self.data.open[next_t]))

        equity_at_exec = self.ledger.equity

        fills: list[Fill] = []
        queued, self._queue = self._queue, []
        for order in queued:
            fills.append(self._execute(order, next_t))
        # Snapshot the position *at the execution mark*. Reconciliation compares
        # against this, not against the close-marked position: a target is a
        # fraction of equity, so re-marking to the close moves the fraction even
        # when nothing traded. Charging that to reconciliation would drown the
        # real signal (partial fills, rejections, caps) in arithmetic.
        self._post_exec_position = self.ledger.position_frac
        # Cost paid on this bar's fills, as a fraction of the equity that survived
        # them, both measured at the execution mark. This is the exact size of the
        # unavoidable reconciliation overshoot: the order was sized from equity
        # *before* the fee and slippage were paid, so the realised fraction ends up
        # larger than the target by ``|target| * this``. Both numerator and
        # denominator have to be taken here — the close-marked equity in
        # :meth:`account` is a different number, and using it underestimates the
        # overshoot by roughly a factor of two.
        self._post_exec_cost_frac = max(0.0, equity_at_exec - self.ledger.equity) / max(
            self.ledger.equity, _EPS
        )

        self.ledger.set_mark(float(self.data.close[next_t]))
        funding_cost = 0.0
        if self.sim.funding_due(next_t):
            funding_cost = self.ledger.apply_funding(self.sim.funding_rate_per_event)

        liquidated = False
        if self.cfg.mode == "perp" and self.ledger.qty != 0.0:
            maintenance = self.cfg.maintenance_margin * self.ledger.notional
            if self.ledger.equity <= maintenance:
                liquidated = True
                self._force_flatten(next_t, reason="liquidation")
                self.n_liquidations += 1
        self._liquidated = self._liquidated or liquidated
        self._bankrupt = self.ledger.equity <= 0.0

        self._t = next_t
        self.ledger.step = next_t - self.start_index
        if self.reconcile_every_bar:
            self.ledger.reconcile()

        net_log = float(np.log(max(self.ledger.equity, _EPS) / max(equity_before, _EPS)))
        mark_log = float(np.log(float(self.data.close[next_t]) / float(self.data.close[t])))
        self.equity_curve.append(self.ledger.equity)
        self.net_log_returns.append(net_log)
        self.positions.append(self.ledger.position_frac)
        return BarUpdate(
            index=next_t,
            account=self.account(),
            fills=tuple(fills),
            funding_cost=funding_cost,
            net_log_return=net_log,
            mark_log_return=mark_log,
            execution_cost_frac=self._post_exec_cost_frac,
            liquidated=liquidated,
            bankrupt=self._bankrupt,
            done=self.done,
        )

    def _execute(self, order: Order, index: int) -> Fill:
        delta = self.sim.requested_delta(
            index=index,
            target_position=order.target_position,
            equity=self.ledger.equity,
            cash=self.ledger.cash,
            current_qty=self.ledger.qty,
            mark=self.ledger.mark,
            allow_negative_cash=self.cfg.mode == "perp",
        )
        if order.reduce_only and delta * self.ledger.qty > 0.0:
            delta = 0.0
        if self.cfg.mode == "perp":
            result = self.sim.execute(index=index, requested_qty=delta)
        else:
            result = self.sim.execute_within_cash(
                index=index, requested_qty=delta, cash=self.ledger.cash
            )
        if result.filled_qty != 0.0:
            self.ledger.apply_trade(result.filled_qty, result.exec_price, result.fee)
        if result.status is FillStatus.PARTIAL:
            self.n_partial_fills += 1
        if result.is_rejected:
            self.n_rejected += 1
        fill = _fill_from_result(order, result, index)
        rec = self._records[order.client_order_id]
        rec["state"] = fill.state
        rec["fill"] = fill
        return fill

    def _force_flatten(self, index: int, *, reason: str) -> Fill | None:
        """Close the position at the bar close paying a full taker cost."""
        if self.ledger.qty == 0.0:
            return None
        qty = -self.ledger.qty
        price = float(self.data.close[index])
        side = 1.0 if qty > 0 else -1.0
        exec_price = price * (1.0 + side * float(self.sim.half_spread[index]))
        fee = abs(qty) * exec_price * self.sim.blended_fee_rate
        self.ledger.apply_trade(qty, exec_price, fee)
        cid = client_order_id(run_id=f"flatten-{reason}", index=index, target_position=0.0)
        fill = Fill(
            client_order_id=cid,
            symbol=self._symbol,
            state=OrderState.FILLED,
            filled_qty=qty,
            avg_price=exec_price,
            fee=fee,
            slippage_bps=float(self.sim.half_spread[index]) * 1e4,
            participation=0.0,
            filled_at=index,
            reason=reason,
        )
        self._records[cid] = {"order": None, "state": OrderState.FILLED, "fill": fill}
        return fill

    def flatten(self, *, reason: str = "manual") -> Fill | None:
        """Immediate forced close at the current bar. The kill switch's hands."""
        self.cancel_all()
        return self._force_flatten(self._t, reason=reason)

    # -- reconciliation ----------------------------------------------------- #
    def reconcile(
        self,
        expected_position: float,
        *,
        tolerance: float = 1e-6,
        at_execution: bool = True,
    ) -> Reconciliation:
        """Compare the agent's intended position with the broker's reported one.

        ``at_execution`` compares against the position snapshotted at the
        execution mark rather than the current close mark; see :meth:`advance`.
        Pass ``False`` to ask the different question "what do I hold right now",
        which is the right one for a start-of-session or post-restart check.

        The tolerance is not cosmetic, and the caller is expected to set it from the
        situation rather than accept the default. Four things make a target
        legitimately unreachable: minimum quantity, minimum notional, the
        participation cap, and — the largest of them at full size — the trade's own
        cost, which shrinks the equity the target is a fraction *of* and so makes the
        realised fraction overshoot outward by roughly ``|target| * cost / equity``.
        :class:`~archetype_trader.live.runner.PaperRunner` derives a floor from all of
        these. A drift above that floor is information — "the position you think you
        have is not the one you have" — rather than automatically a bug. The runner
        logs every non-``ok`` result and re-derives its next order from
        :meth:`account`, never from the intent.
        """
        actual = self.ledger.position_frac
        if at_execution and self._post_exec_position is not None:
            actual = self._post_exec_position
        drift = actual - expected_position
        ok = abs(drift) <= tolerance
        note = ""
        if not ok:
            note = (
                "position drift beyond tolerance; likely a partial fill, a "
                "min-notional rejection or a participation cap"
            )
        return Reconciliation(
            expected_position=float(expected_position),
            actual_position=float(actual),
            drift=float(drift),
            tolerance=float(tolerance),
            ok=ok,
            note=note,
        )

    # -- reporting ---------------------------------------------------------- #
    def summary(self) -> dict[str, Any]:
        """Same shape as ``TradingEnv.episode_summary`` where the keys overlap."""
        self.ledger.reconcile()
        eq = np.asarray(self.equity_curve, dtype=np.float64)
        return {
            "n_steps": len(self.net_log_returns),
            "net_log_returns": np.asarray(self.net_log_returns, dtype=np.float64),
            "equity_curve": eq,
            "positions": np.asarray(self.positions, dtype=np.float64),
            "final_equity": float(eq[-1]) if eq.size else self.cfg.initial_cash,
            "fees": self.ledger.cum_fees,
            "funding": self.ledger.cum_funding,
            "execution_cost": self.ledger.cum_execution_cost,
            "turnover_notional": self.ledger.cum_turnover_notional,
            "n_trades": self.ledger.n_trades,
            "n_partial_fills": self.n_partial_fills,
            "n_rejected": self.n_rejected,
            "n_liquidations": self.n_liquidations,
            "n_duplicate_submissions": self.n_duplicate_submissions,
            "bankrupt": self._bankrupt,
            "realized_pnl": self.ledger.cum_realized_pnl,
            "gross_pnl": self.ledger.cum_market_pnl,
        }


def _fill_from_result(order: Order, result: ExecutionResult, index: int) -> Fill:
    if result.is_rejected:
        state = OrderState.REJECTED
    elif result.status is FillStatus.PARTIAL:
        state = OrderState.PARTIAL
    else:
        state = OrderState.FILLED
    return Fill(
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        state=state,
        filled_qty=result.filled_qty,
        avg_price=result.exec_price,
        fee=result.fee,
        slippage_bps=result.slippage_bps,
        participation=result.participation,
        filled_at=index,
        reason=str(result.status),
    )


@dataclass
class FlakyTransport:
    """Fault injector that fails the first ``n_failures`` submissions of each id.

    Exists so the runner's retry path is exercised by a test rather than by
    production. Raises :class:`BrokerError` — the retryable kind — because the
    interesting case is the one where the caller cannot tell whether the broker
    received the order.
    """

    n_failures: int = 1
    seen: dict[str, int] = field(default_factory=dict)

    def __call__(self, order: Order) -> None:
        count = self.seen.get(order.client_order_id, 0)
        self.seen[order.client_order_id] = count + 1
        if count < self.n_failures:
            raise BrokerError(
                f"transport failure {count + 1}/{self.n_failures} for "
                f"{order.client_order_id} (ambiguous: the order may or may not exist)"
            )
