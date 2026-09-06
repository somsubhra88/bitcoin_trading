"""Paper-trading loop: agent → risk shield → operator limits → broker.

This is the deployment-shaped counterpart to
:func:`~archetype_trader.phase2.train.run_selector`. Same models, same shield,
same accounting — different control flow, because a venue does not hand you an
observation and wait for an action index.

The order of the four gates is not arbitrary:

1. the **agent** proposes a target position from causal inputs only,
2. the **risk shield** clamps it — the same deterministic
   :class:`~archetype_trader.risk.shield.RiskShield` used everywhere else, so a
   paper run and the backtest intervene identically,
3. the **operator limits** (:class:`~archetype_trader.live.safety.LiveLimits`) can
   still refuse; they are outside the strategy and are not tuned with it,
4. the **latency queue** delays the instruction by ``costs.latency_bars`` before
   the broker ever sees it.

Every one of those four is logged separately, per bar, in
``paper_journal.jsonl``. Proposed, shielded, limited and executed positions are
distinct fields: collapsing them would make the shield's intervention rate — a
number the evaluation report prints — unmeasurable in production, which is where
it actually matters.

Rebalancing: the loop submits an instruction on **every** bar, including bars
where the target has not changed, because holding a constant *fraction* of equity
requires trading as the mark moves, and that is exactly what the evaluated
backtest does. ``skip_unchanged_targets=True`` suppresses those no-op
instructions to cut order rate, and is documented as a deviation from the
evaluated strategy rather than an optimisation.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from archetype_trader.config import EnvConfig, ShieldConfig
from archetype_trader.data.types import MarketData
from archetype_trader.features.portfolio import portfolio_features
from archetype_trader.live.agent import HorizonAgent
from archetype_trader.live.broker import (
    BrokerError,
    BrokerRejection,
    Order,
    OrderState,
    PaperBroker,
    client_order_id,
)
from archetype_trader.live.safety import LiveLimits
from archetype_trader.risk.shield import RiskShield

_EPS = 1e-12
log = logging.getLogger("archetype_trader.live")


@dataclass
class RunnerMetrics:
    """Counters an operator watches. Every one of these is a failure mode."""

    bars: int = 0
    orders_submitted: int = 0
    orders_skipped: int = 0
    duplicate_acks: int = 0
    submit_retries: int = 0
    submit_failures: int = 0
    rejections: int = 0
    partial_fills: int = 0
    shield_interventions: int = 0
    limit_blocks: int = 0
    reconciliation_breaches: int = 0
    max_abs_drift: float = 0.0
    forced_flats: int = 0
    liquidations: int = 0
    stale_bars: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class PaperRunResult:
    """Everything a paper session produced."""

    run_id: str
    bars: int
    summary: dict[str, Any]
    agent_stats: dict[str, Any]
    metrics: RunnerMetrics
    journal_path: Path | None
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return float(self.summary.get("final_equity", float("nan")))

    @property
    def net_log_returns(self) -> np.ndarray:
        return np.asarray(self.summary.get("net_log_returns", []), dtype=np.float64)


class _Journal:
    """Append-only JSONL writer. One line per bar, one per lifecycle event.

    JSONL rather than a formatted log: the point of a production journal is that
    somebody can reconstruct exactly what the strategy saw and did after the fact,
    and a human-readable log line that has to be re-parsed is a worse version of
    the same thing.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "wall_clock": time.time(), **payload}
        line = json.dumps(record, default=_json_default, sort_keys=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
        log.debug("%s", line)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.datetime64):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class PaperRunner:
    """Drive a :class:`HorizonAgent` against a :class:`PaperBroker`, bar by bar."""

    def __init__(
        self,
        agent: HorizonAgent,
        broker: PaperBroker,
        data: MarketData,
        scaled: np.ndarray,
        env_cfg: EnvConfig,
        shield_cfg: ShieldConfig,
        *,
        run_id: str = "paper",
        bars_per_year: int = 52_560,
        horizon_hint: int | None = None,
        max_episode_bars: int | None = None,
        limits: LiveLimits | None = None,
        journal_path: str | Path | None = None,
        max_submit_attempts: int = 3,
        backoff_base_s: float = 0.05,
        sleep: Callable[[float], None] | None = None,
        reconcile_tolerance: float = 1e-6,
        skip_unchanged_targets: bool = False,
        stop_for_whole_horizons: bool = True,
        record_decisions: bool = True,
    ) -> None:
        if scaled.shape[0] != len(data):
            raise ValueError("scaled features must cover every bar in the market data")
        if scaled.shape[1] != agent.model.n_features:
            raise ValueError(
                f"decoder expects {agent.model.n_features} features, got {scaled.shape[1]}"
            )
        self.agent = agent
        self.broker = broker
        self.data = data
        self.scaled = scaled
        self.env_cfg = env_cfg
        self.shield_cfg = shield_cfg
        self.run_id = run_id
        self.bars_per_year = bars_per_year
        self.horizon_hint = horizon_hint or agent.horizon
        self.max_episode_bars = int(
            max_episode_bars
            if max_episode_bars is not None
            else max(2, broker.end_index - broker.start_index)
        )
        self.limits = limits or LiveLimits()
        self.max_submit_attempts = max(1, int(max_submit_attempts))
        self.backoff_base_s = float(backoff_base_s)
        self.sleep = sleep or time.sleep
        self.reconcile_tolerance = float(reconcile_tolerance)
        self.skip_unchanged_targets = skip_unchanged_targets
        self.stop_for_whole_horizons = stop_for_whole_horizons
        self.record_decisions = record_decisions

        self.shield = RiskShield(
            shield_cfg,
            mode=env_cfg.mode,
            bars_per_year=bars_per_year,
            maintenance_margin=env_cfg.maintenance_margin,
        )
        self.metrics = RunnerMetrics()
        self._journal = _Journal(Path(journal_path) if journal_path is not None else None)
        self._records: list[dict[str, Any]] = []

    # -- main loop ---------------------------------------------------------- #
    def run(self, *, max_bars: int | None = None) -> PaperRunResult:
        """Trade the replay range and return the session result."""
        broker = self.broker
        agent = self.agent
        agent.reset()
        shield_state = self.shield.reset(self.env_cfg.initial_cash)
        # Pre-filled with "stay flat" so the first fill can only happen
        # ``latency_bars`` after the first decision, exactly as in TradingEnv.
        pending: deque[float] = deque([0.0] * (self.env_cfg.costs.latency_bars - 1))
        turnover_window: deque[float] = deque(maxlen=self.env_cfg.costs.funding_interval_bars)
        target_position = 0.0
        holding_bars = 0
        last_step_return = 0.0
        last_executed_target = 0.0
        episode_start = broker.start_index
        # Mirrors HorizonEnv.latest_start(): a horizon that cannot complete inside
        # the range would be evaluated over fewer bars than the others.
        latest_start = broker.end_index - agent.horizon - 2

        self._journal.write(
            "session_start",
            {
                "run_id": self.run_id,
                "symbol": broker.symbol,
                "start_index": broker.start_index,
                "end_index": broker.end_index,
                "mode": self.env_cfg.mode,
                "leverage": self.env_cfg.leverage,
                "shield_enabled": self.shield_cfg.enabled,
                "horizon": agent.horizon,
                "n_archetypes": agent.n_archetypes,
                "latency_bars": self.env_cfg.costs.latency_bars,
                "skip_unchanged_targets": self.skip_unchanged_targets,
            },
        )

        try:
            while not broker.done:
                t = broker.index
                if max_bars is not None and self.metrics.bars >= max_bars:
                    break
                if self.stop_for_whole_horizons and agent.at_horizon_start and t > latest_start:
                    break

                account = broker.account()
                if account.is_stale:
                    self.metrics.stale_bars += 1
                pf = self._portfolio_vector(
                    account,
                    target_position=target_position,
                    holding_bars=holding_bars,
                    turnover_recent=float(np.sum(turnover_window)) if turnover_window else 0.0,
                    episode_start=episode_start,
                )
                lo = t - agent.context_bars + 1
                if lo < 0:
                    raise IndexError(
                        f"context window needs {agent.context_bars} bars before bar {t}"
                    )
                decision = agent.decide(
                    context=self.scaled[lo : t + 1],
                    market_row=self.scaled[t],
                    portfolio=pf,
                    position=account.position,
                    bar=t,
                )

                # --- gate 2: the deterministic risk shield --------------------- #
                desired_qty = broker.sim.target_qty(
                    target_position=decision.target_position,
                    equity=account.equity,
                    mark=account.mark,
                )
                est_slip = broker.sim.estimate_slippage_bps(index=t, qty=desired_qty - account.qty)
                shielded = self.shield.apply(
                    shield_state,
                    proposed_position=decision.target_position,
                    current_position=account.position,
                    equity=account.equity,
                    drawdown=account.drawdown,
                    realized_vol_per_bar=account.realized_vol_per_bar,
                    spread_bps=account.spread_bps,
                    bar_volume=account.bar_volume,
                    est_slippage_bps=est_slip,
                    is_stale=account.is_stale,
                    day_index=account.day_index,
                    last_step_return=last_step_return,
                    max_positions=agent.positions,
                )
                if shielded.intervened:
                    self.metrics.shield_interventions += 1

                # --- gate 3: operator limits ---------------------------------- #
                limited, limit_reasons = self._apply_limits(
                    shielded.executed_position, account=account
                )
                if limit_reasons:
                    self.metrics.limit_blocks += 1

                # --- gate 4: latency queue ------------------------------------ #
                pending.append(limited)
                executable = pending.popleft() if pending else limited

                submitted, ack, order = self._maybe_submit(
                    executable, bar=t, last_executed_target=last_executed_target
                )

                equity_before = account.equity
                update = broker.advance()
                if update.liquidated:
                    self.metrics.liquidations += 1

                filled_notional = sum(f.notional for f in update.fills)
                filled_qty = sum(f.filled_qty for f in update.fills)
                for f in update.fills:
                    if f.state is OrderState.REJECTED:
                        self.metrics.rejections += 1
                    elif f.state is OrderState.PARTIAL:
                        self.metrics.partial_fills += 1
                turnover_window.append(filled_notional / max(equity_before, _EPS))

                # Two things bound how precisely any target can be hit. The first is
                # a *floor* on the tolerance; the second is an *allowance added to*
                # it, and the distinction matters — see below.
                #
                # (1) The venue's own minimums: an order smaller than min_qty or
                #     min_notional cannot be sent at all.
                # (2) The trade's own cost. A target is a fraction of equity, and
                #     the order is sized from equity *before* the fee and slippage
                #     are paid. Paying them shrinks the denominator, so the realised
                #     fraction overshoots by |target| * cost / equity — always
                #     outward, never inward. At a full-size target on the test
                #     fixture that is ~1.2e-3, a thousand times the 1e-6 default, so
                #     without this term every entry into a full-size position logs a
                #     breach and the signal the tolerance exists to preserve
                #     (partial fills, rejections, participation caps) is buried.
                #
                #     ``execution_cost_frac`` is taken from the broker rather than
                #     recomputed from ``account`` deltas, because both the cost and
                #     the equity have to be measured at the *execution* mark. The
                #     close-marked equity in ``account`` is a different number and
                #     underestimates the overshoot by roughly a factor of two.
                #
                #     With it, ``|drift| == |target| * execution_cost_frac`` holds to
                #     full double precision. That is why the term is *added* rather
                #     than folded into the ``max``: the drift the broker reports is
                #     ``actual - expected``, a subtraction of two numbers near 1.0,
                #     so it carries ~1e-16 of cancellation noise that an exactly
                #     equal tolerance loses to on a coin flip. Adding lets the
                #     ``reconcile_tolerance`` floor (1e-6, ten orders of magnitude
                #     above the noise) absorb it, instead of three of 480 bars
                #     reporting a breach for a last-bit tie.
                costs = self.env_cfg.costs
                unreachable = max(
                    costs.min_notional, costs.min_qty * max(account.mark, _EPS)
                ) / max(account.equity, _EPS)
                cost_overshoot = abs(executable) * update.execution_cost_frac
                recon = broker.reconcile(
                    executable,
                    tolerance=max(self.reconcile_tolerance, unreachable) + cost_overshoot,
                )
                if not recon.ok:
                    self.metrics.reconciliation_breaches += 1
                    self.metrics.max_abs_drift = max(self.metrics.max_abs_drift, abs(recon.drift))

                agent.observe(
                    net_log_return=update.net_log_return,
                    mark_log_return=update.mark_log_return,
                )

                last_step_return = update.net_log_return
                if abs(executable) > _EPS and abs(executable - target_position) < _EPS:
                    holding_bars += 1
                else:
                    holding_bars = 0 if abs(executable) <= _EPS else 1
                target_position = executable
                if submitted:
                    last_executed_target = executable
                self.metrics.bars += 1

                self._log_bar(
                    t=t,
                    decision=decision,
                    shielded=shielded,
                    limited=limited,
                    limit_reasons=limit_reasons,
                    executable=executable,
                    order_id=None if order is None else order.client_order_id,
                    ack=ack,
                    update=update,
                    recon=recon,
                    filled_qty=filled_qty,
                    filled_notional=filled_notional,
                    est_slippage_bps=est_slip,
                )
        finally:
            summary = broker.summary()
            summary["shield_intervention_rate"] = shield_state.intervention_rate
            summary["shield_reason_counts"] = dict(shield_state.reason_counts)
            self._journal.write(
                "session_end",
                {
                    "run_id": self.run_id,
                    "bars": self.metrics.bars,
                    "final_equity": summary.get("final_equity"),
                    "metrics": self.metrics.as_dict(),
                    "agent": self.agent.stats_dict(),
                    "shield_reason_counts": summary["shield_reason_counts"],
                },
            )
            self._journal.close()

        return PaperRunResult(
            run_id=self.run_id,
            bars=self.metrics.bars,
            summary=summary,
            agent_stats=self.agent.stats_dict(),
            metrics=self.metrics,
            journal_path=self._journal.path,
            decisions=self._records,
        )

    # -- gates -------------------------------------------------------------- #
    def _portfolio_vector(
        self,
        account: Any,
        *,
        target_position: float,
        holding_bars: int,
        turnover_recent: float,
        episode_start: int,
    ) -> np.ndarray:
        """The same vector ``TradingEnv._observation`` builds, from broker state.

        Identical by construction: it calls the same
        :func:`~archetype_trader.features.portfolio.portfolio_features`. What
        differs is the *source* — every input comes from the broker's reported
        account, not from the runner's own idea of its position.
        """
        steps_left = max(
            0,
            min(self.broker.end_index - 1, episode_start + self.max_episode_bars) - account.index,
        )
        risk_left = max(0.0, 1.0 - account.drawdown / max(self.shield_cfg.max_drawdown, _EPS))
        return portfolio_features(
            target_position=target_position,
            filled_position=account.position,
            cash=account.cash,
            equity=account.equity,
            initial_equity=self.env_cfg.initial_cash,
            unrealized_pnl=account.unrealized_pnl,
            drawdown=account.drawdown,
            turnover_recent=turnover_recent,
            holding_bars=holding_bars,
            horizon=self.horizon_hint,
            steps_left=steps_left,
            risk_budget_left=risk_left,
            max_leverage=self.env_cfg.leverage,
        ).astype(np.float32)

    def _apply_limits(self, position: float, *, account: Any) -> tuple[float, tuple[str, ...]]:
        """Clamp a shielded position to the operator's hard caps.

        A breach is clamped, not raised: refusing to trade at all would leave an
        existing position unmanaged, which is usually worse than holding a smaller
        one. The clamp and its reason are journalled.
        """
        lim = self.limits
        reasons: list[str] = []
        out = float(position)
        if lim.max_leverage > 0 and abs(out) > lim.max_leverage:
            reasons.append(f"operator_max_leverage:{lim.max_leverage}")
            out = float(np.clip(out, -lim.max_leverage, lim.max_leverage))
        if lim.max_position_notional > 0:
            cap_frac = lim.max_position_notional / max(account.equity, _EPS)
            if abs(out) > cap_frac:
                reasons.append(f"operator_max_position_notional:{lim.max_position_notional}")
                out = float(np.clip(out, -cap_frac, cap_frac))
        if lim.max_order_notional > 0:
            delta_notional = abs(out - account.position) * max(account.equity, _EPS)
            if delta_notional > lim.max_order_notional:
                step = lim.max_order_notional / max(account.equity, _EPS)
                reasons.append(f"operator_max_order_notional:{lim.max_order_notional}")
                out = account.position + np.sign(out - account.position) * step
        if lim.allowed_symbols and self.broker.symbol not in lim.allowed_symbols:
            reasons.append(f"symbol_not_allowed:{self.broker.symbol}")
            out = 0.0
        return out, tuple(reasons)

    def _maybe_submit(
        self, executable: float, *, bar: int, last_executed_target: float
    ) -> tuple[bool, Any, Any]:
        if self.skip_unchanged_targets and abs(executable - last_executed_target) < _EPS:
            self.metrics.orders_skipped += 1
            return False, None, None
        order = Order(
            client_order_id=client_order_id(
                run_id=self.run_id, index=bar, target_position=executable
            ),
            symbol=self.broker.symbol,
            target_position=executable,
            decided_at=bar,
            tag=self.run_id,
        )
        ack = self._submit_with_retry(order)
        return True, ack, order

    def _submit_with_retry(self, order: Any) -> Any:
        """Submit with bounded exponential backoff, reusing the same order id.

        Reusing the id is the whole point: after a transport failure the caller
        cannot know whether the broker received the order, and a retry that minted
        a fresh id could double the position. With the id fixed, the worst case is
        a duplicate acknowledgement, which is recorded and ignored.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_submit_attempts):
            try:
                ack = self.broker.submit(order)
            except BrokerRejection as exc:
                # Permanent. Retrying is pointless and would only delay the loop.
                self.metrics.submit_failures += 1
                self._journal.write(
                    "order_rejected",
                    {
                        "client_order_id": order.client_order_id,
                        "target_position": order.target_position,
                        "error": str(exc),
                        "retryable": False,
                    },
                )
                return None
            except BrokerError as exc:
                last_exc = exc
                self.metrics.submit_retries += 1
                self._journal.write(
                    "order_retry",
                    {
                        "client_order_id": order.client_order_id,
                        "attempt": attempt + 1,
                        "max_attempts": self.max_submit_attempts,
                        "error": str(exc),
                        "retryable": True,
                    },
                )
                if attempt + 1 < self.max_submit_attempts:
                    self.sleep(self.backoff_base_s * (2.0**attempt))
                continue
            self.metrics.orders_submitted += 1
            if ack.duplicate:
                self.metrics.duplicate_acks += 1
                self._journal.write(
                    "order_duplicate",
                    {
                        "client_order_id": order.client_order_id,
                        "state": str(ack.state),
                        "note": "idempotency key matched; nothing re-queued",
                    },
                )
            return ack
        self.metrics.submit_failures += 1
        self._journal.write(
            "order_failed",
            {
                "client_order_id": order.client_order_id,
                "attempts": self.max_submit_attempts,
                "error": str(last_exc),
            },
        )
        return None

    # -- logging ------------------------------------------------------------ #
    def _log_bar(self, **kw: Any) -> None:
        decision = kw["decision"]
        shielded = kw["shielded"]
        update = kw["update"]
        recon = kw["recon"]
        ack = kw["ack"]
        record = {
            "bar": kw["t"],
            "timestamp": str(self.data.ts[kw["t"]]),
            "archetype": decision.archetype,
            "new_horizon": decision.new_horizon,
            "step_in_horizon": decision.step_in_horizon,
            "base_position": decision.base_position,
            "proposed_position": decision.target_position,
            "overrode": decision.overrode,
            "shielded_position": shielded.executed_position,
            "shield_reasons": list(shielded.reasons),
            "shield_intervened": shielded.intervened,
            "limited_position": kw["limited"],
            "limit_reasons": list(kw["limit_reasons"]),
            "executed_target": kw["executable"],
            "client_order_id": kw["order_id"],
            "ack_state": None if ack is None else str(ack.state),
            "ack_duplicate": None if ack is None else bool(ack.duplicate),
            "filled_qty": kw["filled_qty"],
            "filled_notional": kw["filled_notional"],
            "est_slippage_bps": kw["est_slippage_bps"],
            "realized_position": update.account.position,
            "equity": update.account.equity,
            "net_log_return": update.net_log_return,
            "mark_log_return": update.mark_log_return,
            "funding_cost": update.funding_cost,
            "drawdown": update.account.drawdown,
            "reconciled": recon.ok,
            "position_drift": recon.drift,
            "liquidated": update.liquidated,
            "bankrupt": update.bankrupt,
        }
        self._journal.write("bar", record)
        if self.record_decisions:
            self._records.append(record)


def action_sequence(result: PaperRunResult) -> list[float]:
    """Executed target positions, in order. The parity test's comparison key."""
    return [float(r["executed_target"]) for r in result.decisions]
