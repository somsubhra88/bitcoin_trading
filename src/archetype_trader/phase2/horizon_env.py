"""Horizon-level MDP: one decision = one archetype for the next ``h`` bars.

This is the environment the Phase-2 selector sees. A single macro step

1. reads the pre-horizon context (market lookback + portfolio state) — nothing
   from inside the horizon,
2. hands the chosen archetype's code to the **frozen** Phase-1 decoder,
3. rolls that decoder one bar at a time through :class:`TradingEnv`, feeding it
   only the observation of the bar it is acting on, and
4. returns the summed reward and the unshaped net log return of the horizon.

The decoder is fed *its own* previous action rather than the shielded/executed
one. That matches how it will be deployed and keeps the archetype's behaviour
interpretable; the risk shield's overrides are reported in ``info`` instead of
being laundered back into the decoder's memory.

The optional cash archetype is not a learned code — it is a hard "stay flat for
the whole horizon" action, so the selector can abstain rather than being forced
to pick the least-bad trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from archetype_trader.env.trading_env import TradingEnv
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.features.portfolio import N_PORTFOLIO_FEATURES
from archetype_trader.phase1.models import ArchetypeVQ


def build_scaled_features(features: FeatureMatrix, scaler: FeatureScaler) -> np.ndarray:
    """Scaled market matrix with warm-up rows zeroed.

    Identical to what :func:`build_demonstrations` fed the decoder in Phase 1 —
    sharing this one function is what guarantees the decoder sees the same
    representation at train and at select time.
    """
    scaled = scaler.transform(features.values).astype(np.float32)
    scaled[: features.valid_from] = 0.0
    return scaled


@runtime_checkable
class StepOverride(Protocol):
    """Structural contract for a Phase-3 step-level adapter.

    Declared here as a Protocol rather than importing
    :class:`~archetype_trader.phase3.adapter.AdapterController` so that Phase 2
    keeps no dependency on Phase 3: the horizon roll-out is one implementation
    used by Phase-1-only, Phase-1+2 and full Phase-1+2+3 evaluation alike.
    """

    def reset(self, archetype: int) -> None: ...

    def decide(
        self,
        market_row: np.ndarray,
        *,
        base_action: int,
        position: float,
        realized_cum: float,
        base_cum: float,
    ) -> Any: ...


@dataclass
class HorizonOutcome:
    """Result of rolling one archetype for one horizon."""

    archetype: int
    reward: float
    net_log_return: float
    bars: int
    positions: list[float]
    proposed_positions: list[float]
    n_trades: int
    fees: float
    funding: float
    shield_interventions: int
    terminated: bool
    truncated: bool
    equity_start: float
    equity_end: float
    #: The archetype's own plan, before any Phase-3 override.
    base_actions: list[int] = field(default_factory=list)
    #: What was actually sent to the environment.
    actions: list[int] = field(default_factory=list)
    #: Contiguous override runs used this horizon (0 without an adapter).
    n_overrides: int = 0
    #: Counterfactual net log return of the unmodified archetype, priced from
    #: observed bar returns. Equals ``net_log_return`` when nothing overrode.
    base_net_log_return: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Realised minus the archetype's counterfactual return, *within* this horizon.

        Exactly ``0.0`` when nothing was overridden. It does not capture the
        knock-on effect of starting the next horizon from a different position.
        """
        return self.net_log_return - self.base_net_log_return


class HorizonEnv:
    """Macro-step wrapper around :class:`TradingEnv`.

    Not a ``gym.Env``: the action space is archetypes, the step length is a whole
    horizon, and pretending otherwise would invite a step-level RL algorithm to be
    pointed at it by mistake.
    """

    def __init__(
        self,
        env: TradingEnv,
        model: ArchetypeVQ,
        scaled_features: np.ndarray,
        *,
        horizon: int,
        context_bars: int,
        include_cash_archetype: bool = True,
        device: torch.device | None = None,
        greedy_decoder: bool = True,
        override: StepOverride | None = None,
        shadow_cost_frac: float = 0.0,
    ) -> None:
        if scaled_features.shape[0] != len(env.data):
            raise ValueError("scaled_features must cover every bar in the market data")
        if scaled_features.shape[1] != model.n_features:
            raise ValueError(
                f"decoder expects {model.n_features} features, got {scaled_features.shape[1]}"
            )
        if model.n_actions != len(env.positions):
            raise ValueError(
                f"decoder action grid ({model.n_actions}) does not match the environment "
                f"position grid ({len(env.positions)})"
            )
        self.env = env
        model.eval()
        self.model = model
        self.scaled = scaled_features
        self.horizon = horizon
        self.context_bars = context_bars
        self.include_cash = include_cash_archetype
        self.device = device or torch.device("cpu")
        self.greedy_decoder = greedy_decoder
        self.override = override
        # Turnover charge used to price the *difference* between the base plan and
        # what was actually sent, in the planner's log-space convention. Only the
        # difference is shadowed: the counterfactual is anchored to the realised
        # net return, so with no override the base return equals the realised one
        # exactly and no accounting bias can masquerade as improvement.
        self.shadow_cost_frac = float(shadow_cost_frac)

        self.codebook_size = model.codebook_size
        self.n_archetypes = self.codebook_size + (1 if include_cash_archetype else 0)
        self.cash_archetype = self.codebook_size if include_cash_archetype else -1
        self.flat_action = env.action_index(0.0)

        self.min_start = max(
            env.features.valid_from, env.features.lookback - 1, context_bars - 1, env.start_index
        )
        self._obs_portfolio = np.zeros(N_PORTFOLIO_FEATURES, dtype=np.float32)
        self._done = True
        self.history: list[HorizonOutcome] = []

    # -- spaces ------------------------------------------------------------ #
    @property
    def n_features(self) -> int:
        return int(self.scaled.shape[1])

    @property
    def n_portfolio(self) -> int:
        return int(self._obs_portfolio.shape[0])

    def latest_start(self) -> int:
        """Last bar at which a full horizon still fits inside the episode range."""
        return self.env.end_index - self.horizon - 2

    # -- api --------------------------------------------------------------- #
    def reset(
        self, *, start_index: int | None = None, seed: int | None = None
    ) -> dict[str, np.ndarray]:
        start = self.min_start if start_index is None else int(start_index)
        if start < self.min_start:
            raise ValueError(
                f"start_index={start} needs {self.context_bars} bars of context and the "
                f"feature warm-up; earliest admissible is {self.min_start}"
            )
        obs, _ = self.env.reset(seed=seed, options={"start_index": start})
        self._obs_portfolio = np.asarray(obs["portfolio"], dtype=np.float32)
        self._done = False
        self.history = []
        return self.observation()

    def observation(self) -> dict[str, np.ndarray]:
        """Pre-horizon context only: bars ``<= t`` and the portfolio state at ``t``."""
        t = self.env._t
        lo = t - self.context_bars + 1
        if lo < 0:
            raise IndexError(f"context window needs {self.context_bars} bars before bar {t}")
        return {
            "context": self.scaled[lo : t + 1],
            "portfolio": self._obs_portfolio,
        }

    def step(
        self, archetype: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, HorizonOutcome]:
        """Run ``archetype`` for one horizon. Returns ``(obs, reward, term, trunc, outcome)``."""
        if self._done:
            raise RuntimeError("HorizonEnv.step() after episode end; call reset() first")
        if not 0 <= archetype < self.n_archetypes:
            raise ValueError(f"archetype {archetype} outside [0, {self.n_archetypes})")

        is_cash = self.include_cash and archetype == self.cash_archetype
        code = None
        dstate: Any = None
        prev: torch.Tensor | None = None
        if not is_cash:
            idx = torch.tensor([archetype], dtype=torch.long, device=self.device)
            code = self.model.code_embedding(idx)
            dstate = self.model.decoder.init_state(1, self.device, code.dtype)

        if self.override is not None:
            # Budget and recurrent memory are per horizon: the adapter is a
            # within-archetype device and must not carry intent across a
            # selector decision it did not make.
            self.override.reset(archetype)

        equity_start = self.env.ledger.equity
        reward_sum = 0.0
        net_sum = 0.0
        base_net = 0.0
        # Both counterfactuals start from the position actually held, so the first
        # step's turnover charges cancel unless the adapter overrides.
        prev_plan_pos = base_prev_pos = self.env.ledger.position_frac
        base_actions: list[int] = []
        taken_actions: list[int] = []
        n_overrides = 0
        in_override = False
        positions: list[float] = []
        proposed: list[float] = []
        interventions = 0
        fees0, funding0 = self.env.ledger.cum_fees, self.env.ledger.cum_funding
        trades0 = self.env.ledger.n_trades
        terminated = truncated = False
        bars = 0
        last_info: dict[str, Any] = {}

        for _ in range(self.horizon):
            row_np = self.scaled[self.env._t]
            if is_cash:
                base_action = self.flat_action
            else:
                if code is None:
                    raise RuntimeError("non-cash archetype is missing its code embedding")
                row = torch.from_numpy(row_np).to(self.device)
                with torch.no_grad():
                    logits, dstate = self.model.decoder.step(row.unsqueeze(0), code, prev, dstate)
                    pred = (
                        logits.argmax(dim=-1)
                        if self.greedy_decoder
                        else torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
                    )
                # The decoder is fed its *own* previous action, not the overridden
                # one, so the archetype's plan stays intact and the counterfactual
                # stays meaningful.
                prev = pred
                base_action = int(pred.item())

            action = base_action
            if self.override is not None:
                decision = self.override.decide(
                    row_np,
                    base_action=base_action,
                    position=self.env.ledger.position_frac,
                    realized_cum=net_sum,
                    base_cum=base_net,
                )
                action = int(decision.action)
                if action != base_action:
                    n_overrides += int(not in_override)
                    in_override = True
                else:
                    in_override = False
            base_actions.append(base_action)
            taken_actions.append(action)

            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_portfolio = np.asarray(obs["portfolio"], dtype=np.float32)
            reward_sum += float(reward)
            net_sum += float(info["net_log_return"])
            # Counterfactual base return: the realised return, corrected for the
            # position difference at public prices and for the turnover-cost
            # difference. Identically zero correction when nothing was overridden.
            base_pos = self.env.positions[base_action]
            plan_pos = self.env.positions[action]
            c = self.shadow_cost_frac
            # The correction is summed on its own, then added, so that "no override"
            # gives a *bit-exact* zero. Folding it into one expression alongside
            # ``net_log_return`` instead computes ``(r + L) - L``, which is not ``r``
            # in IEEE arithmetic: it leaves a rounding residue of order
            # ``eps * |L|``. That residue is ~1e-19 per bar and utterly harmless in
            # magnitude, but it means ``improvement`` is not exactly 0.0 when the
            # adapter retained everywhere, and "exactly 0.0" is the property the
            # difference-based accounting exists to provide — it is what lets a
            # Phase-3 result be read as a gain over the base plan rather than as a
            # gain over the base plan plus an unknown accounting drift.
            correction = (
                (base_pos - plan_pos) * float(info["mark_log_return"])
                + float(np.log1p(-c * abs(base_pos - base_prev_pos)))
                - float(np.log1p(-c * abs(plan_pos - prev_plan_pos)))
            )
            base_net += float(info["net_log_return"]) + correction
            base_prev_pos = base_pos
            prev_plan_pos = plan_pos
            positions.append(float(info["executed_position"]))
            proposed.append(float(info["proposed_position"]))
            interventions += int(bool(info["shield_intervened"]))
            last_info = info
            bars += 1
            terminated, truncated = bool(term), bool(trunc)
            if terminated or truncated:
                break

        # Running out of room for a *whole* horizon ends the macro episode. It is
        # reported as truncation, not termination, so bootstrapping stays correct.
        if self.env._t > self.latest_start():
            truncated = True
        self._done = terminated or truncated
        outcome = HorizonOutcome(
            archetype=archetype,
            reward=reward_sum,
            net_log_return=net_sum,
            bars=bars,
            positions=positions,
            proposed_positions=proposed,
            n_trades=self.env.ledger.n_trades - trades0,
            fees=self.env.ledger.cum_fees - fees0,
            funding=self.env.ledger.cum_funding - funding0,
            shield_interventions=interventions,
            terminated=terminated,
            truncated=truncated,
            equity_start=equity_start,
            equity_end=self.env.ledger.equity,
            base_actions=base_actions,
            actions=taken_actions,
            n_overrides=n_overrides,
            base_net_log_return=base_net,
            info=last_info,
        )
        self.history.append(outcome)
        # After termination the env refuses further steps, so hand back the last
        # legal observation rather than indexing past the episode.
        obs_out = self.observation() if not (terminated or truncated) else self._terminal_obs()
        return obs_out, reward_sum, terminated, truncated, outcome

    def _terminal_obs(self) -> dict[str, np.ndarray]:
        t = min(self.env._t, len(self.scaled) - 1)
        lo = max(0, t - self.context_bars + 1)
        block = self.scaled[lo : t + 1]
        if block.shape[0] < self.context_bars:  # pad at the front, never with future rows
            pad = np.zeros((self.context_bars - block.shape[0], block.shape[1]), dtype=block.dtype)
            block = np.concatenate([pad, block], axis=0)
        return {"context": block, "portfolio": self._obs_portfolio}

    # -- reporting --------------------------------------------------------- #
    def episode_stats(self) -> dict[str, Any]:
        """Archetype usage, switching frequency and the underlying env economics."""
        summary = self.env.episode_summary()
        chosen = np.asarray([o.archetype for o in self.history], dtype=np.int64)
        counts = (
            np.bincount(chosen, minlength=self.n_archetypes)
            if chosen.size
            else np.zeros(self.n_archetypes, dtype=np.int64)
        )
        switches = int((np.diff(chosen) != 0).sum()) if chosen.size > 1 else 0
        summary.update(
            {
                "n_horizons": int(chosen.size),
                "archetype_sequence": chosen.tolist(),
                "archetype_counts": counts.tolist(),
                "archetype_usage": (counts / max(chosen.size, 1)).tolist(),
                "switch_rate": switches / max(chosen.size - 1, 1),
                "cash_fraction": (
                    float(counts[self.cash_archetype] / max(chosen.size, 1))
                    if self.include_cash and chosen.size
                    else 0.0
                ),
                "horizon_net_log_returns": [o.net_log_return for o in self.history],
                "horizon_rewards": [o.reward for o in self.history],
                "n_overrides": int(sum(o.n_overrides for o in self.history)),
                "override_rate": float(
                    np.mean([o.n_overrides > 0 for o in self.history]) if self.history else 0.0
                ),
                # Sum of *within-horizon* improvements. Deliberately not called a
                # total: once an override changes the position, every later
                # horizon starts from a different state, and that carry-over is
                # not priced here. The only trustworthy end-to-end comparison is
                # two full runs with and without the adapter, which is what the
                # evaluation harness reports.
                "within_horizon_override_improvement": float(
                    sum(o.improvement for o in self.history)
                ),
            }
        )
        return summary
