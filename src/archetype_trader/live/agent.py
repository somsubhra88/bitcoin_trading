"""The deployed decision path: one bar in, one target position out.

:class:`~archetype_trader.phase2.horizon_env.HorizonEnv` expresses the same three
phases as a loop that owns the clock and the ledger. A deployment cannot: the
exchange owns the clock, and the account state arrives from outside. So the
decision logic is extracted here into an object that is *pushed* one bar at a
time and holds nothing but recurrent state.

That is a second expression of logic which already exists, and duplicated logic
drifts. Two things keep it honest:

* :class:`HorizonAgent` calls the *same model methods* the training path calls —
  ``model.code_embedding``, ``decoder.init_state``/``decoder.step``,
  ``policy.act``, ``controller.decide``. Nothing is re-implemented in numpy.
* ``tests/archetype/test_live.py`` runs the agent over a historical segment and
  asserts its action sequence is byte-identical to
  :func:`~archetype_trader.phase2.train.run_selector`'s on that same segment. If
  the two ever diverge, that test fails rather than a paper run quietly trading a
  different strategy from the one that was evaluated.

Nothing in this module imports :mod:`archetype_trader.planner` or
:mod:`archetype_trader.phase3.hindsight`, and the leakage test asserts the whole
import closure stays clean: the deployed path must not be able to reach a
future-indexed array even by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from archetype_trader.config import ExperimentConfig
from archetype_trader.phase1.models import ArchetypeVQ
from archetype_trader.phase2.selector import RecurrentActorCritic, RecurrentQNetwork
from archetype_trader.phase3.adapter import AdapterController, OverrideAdapter

type SelectorNet = RecurrentActorCritic | RecurrentQNetwork


@dataclass(frozen=True)
class BarDecision:
    """Everything the agent decided on one bar, and why.

    ``action`` is the policy's proposal. It is *not* necessarily what gets sent:
    the risk shield sits downstream and may clamp or flatten it. Both are logged
    separately so the shield's intervention rate stays measurable in production,
    exactly as it is in the backtest.
    """

    bar: int
    horizon_index: int
    step_in_horizon: int
    archetype: int
    is_cash_archetype: bool
    base_action: int
    action: int
    base_position: float
    target_position: float
    overrode: bool
    new_horizon: bool
    realized_cum: float
    base_cum: float
    archetype_probs: tuple[float, ...] = ()
    retain_prob: float = 1.0
    budget_left: int = 0


@dataclass
class AgentStats:
    """Counters a paper run reports and a live run alerts on."""

    bars: int = 0
    horizons: int = 0
    overrides: int = 0
    override_runs: int = 0
    archetype_counts: list[int] = field(default_factory=list)
    switches: int = 0

    def usage(self) -> list[float]:
        total = max(sum(self.archetype_counts), 1)
        return [c / total for c in self.archetype_counts]


class HorizonAgent:
    """Stateful, single-trajectory driver for the Phase-1/2/3 stack.

    Usage is a strict alternation, and the order matters:

    1. :meth:`decide` with the causal window, the current bar's feature row and
       the portfolio vector as of that bar. Returns the target position.
    2. execute it however the venue allows,
    3. :meth:`observe` with the realised net log return of the bar and the public
       mark log return, so the Phase-3 adapter's realised-versus-base signal
       advances.

    Skipping step 3 leaves the adapter's counterfactual frozen, which silently
    changes its behaviour. :meth:`decide` raises if called twice in a row.
    """

    def __init__(
        self,
        model: ArchetypeVQ,
        *,
        positions: tuple[float, ...],
        horizon: int,
        context_bars: int,
        policy: SelectorNet | None = None,
        controller: AdapterController | None = None,
        include_cash_archetype: bool = True,
        fixed_archetype: int | None = None,
        device: torch.device | None = None,
        greedy: bool = True,
        shadow_cost_frac: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        if model.n_actions != len(positions):
            raise ValueError(
                f"decoder action grid ({model.n_actions}) does not match the position "
                f"grid ({len(positions)})"
            )
        if policy is None and fixed_archetype is None:
            raise ValueError(
                "give either a Phase-2 policy or fixed_archetype; without one of them "
                "there is nothing to choose an archetype with"
            )
        model.eval()
        self.model = model
        self.policy = policy
        self.controller = controller
        self.positions = tuple(positions)
        self.horizon = int(horizon)
        self.context_bars = int(context_bars)
        self.include_cash = include_cash_archetype
        self.device = device or torch.device("cpu")
        self.greedy = greedy
        self.shadow_cost_frac = float(shadow_cost_frac)
        self.generator = generator

        self.codebook_size = model.codebook_size
        self.n_archetypes = self.codebook_size + (1 if include_cash_archetype else 0)
        self.cash_archetype = self.codebook_size if include_cash_archetype else -1
        self.flat_action = int(np.argmin(np.abs(np.asarray(self.positions))))
        if fixed_archetype is not None and not 0 <= fixed_archetype < self.n_archetypes:
            raise ValueError(f"fixed_archetype {fixed_archetype} outside [0, {self.n_archetypes})")
        self.fixed_archetype = fixed_archetype

        self.stats = AgentStats(archetype_counts=[0] * self.n_archetypes)
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Clear all recurrent state. Call once before the first bar, never mid-run."""
        self._policy_state = (
            self.policy.init_state(1, self.device) if self.policy is not None else None
        )
        self._step_in_horizon = self.horizon  # forces archetype selection on bar 1
        self._horizon_index = -1
        self._archetype = -1
        self._code: torch.Tensor | None = None
        self._dstate: Any = None
        self._prev_action: torch.Tensor | None = None
        self._realized_cum = 0.0
        self._base_cum = 0.0
        self._prev_plan_pos = 0.0
        self._base_prev_pos = 0.0
        self._awaiting_observe = False
        self._last_base_pos = 0.0
        self._last_plan_pos = 0.0
        self.stats = AgentStats(archetype_counts=[0] * self.n_archetypes)
        self.decisions: list[BarDecision] = []

    @property
    def awaiting_observe(self) -> bool:
        return self._awaiting_observe

    @property
    def at_horizon_start(self) -> bool:
        """True when the next :meth:`decide` will select a fresh archetype.

        The runner needs this to reproduce ``HorizonEnv``'s rule that a horizon
        which cannot complete inside the segment is not started at all.
        """
        return not self._awaiting_observe and self._step_in_horizon >= self.horizon

    @property
    def archetype(self) -> int:
        return self._archetype

    # -- decision ----------------------------------------------------------- #
    @torch.no_grad()
    def decide(
        self,
        *,
        context: np.ndarray,
        market_row: np.ndarray,
        portfolio: np.ndarray,
        position: float,
        bar: int = -1,
    ) -> BarDecision:
        """Choose a target position for the bar whose close has just been observed."""
        if self._awaiting_observe:
            raise RuntimeError(
                "decide() called twice without observe(); the adapter's "
                "realised-versus-base signal would be stale"
            )
        if context.shape[0] != self.context_bars:
            raise ValueError(
                f"context has {context.shape[0]} bars, selector expects {self.context_bars}"
            )
        if market_row.shape[-1] != self.model.n_features:
            raise ValueError(
                f"feature row has {market_row.shape[-1]} columns, decoder expects "
                f"{self.model.n_features}"
            )

        new_horizon = self._step_in_horizon >= self.horizon
        probs: tuple[float, ...] = ()
        if new_horizon:
            self._archetype, probs = self._choose_archetype(context, portfolio)
            self._begin_horizon(position)
            self.stats.horizons += 1
            self.stats.archetype_counts[self._archetype] += 1
            if len(self.decisions) and self.decisions[-1].archetype != self._archetype:
                self.stats.switches += 1

        is_cash = self.include_cash and self._archetype == self.cash_archetype
        base_action = self.flat_action if is_cash else self._decoder_step(market_row)

        action = base_action
        retain_prob = 1.0
        budget_left = 0
        if self.controller is not None:
            decision = self.controller.decide(
                np.asarray(market_row, dtype=np.float32),
                base_action=base_action,
                position=float(position),
                realized_cum=self._realized_cum,
                base_cum=self._base_cum,
            )
            action = int(decision.action)
            retain_prob = float(decision.retain_prob)
            budget_left = int(decision.budget_left)
            if action != base_action:
                self.stats.overrides += 1
                self.stats.override_runs += int(decision.started_override)

        out = BarDecision(
            bar=bar,
            horizon_index=self._horizon_index,
            step_in_horizon=self._step_in_horizon,
            archetype=self._archetype,
            is_cash_archetype=is_cash,
            base_action=base_action,
            action=action,
            base_position=self.positions[base_action],
            target_position=self.positions[action],
            overrode=action != base_action,
            new_horizon=new_horizon,
            realized_cum=self._realized_cum,
            base_cum=self._base_cum,
            archetype_probs=probs,
            retain_prob=retain_prob,
            budget_left=budget_left,
        )
        self.decisions.append(out)
        self._last_base_pos = out.base_position
        self._last_plan_pos = out.target_position
        self._awaiting_observe = True
        return out

    def observe(self, *, net_log_return: float, mark_log_return: float) -> None:
        """Advance the horizon clock and the realised-versus-base counterfactual.

        The counterfactual is the same difference-based accounting the training
        environment uses: the base plan's return is *anchored* to the realised one
        and only the position and turnover differences are shadowed. With no
        override the two are equal to the last bit, so the adapter can never be
        rewarded by an accounting artefact.
        """
        if not self._awaiting_observe:
            raise RuntimeError("observe() without a preceding decide()")
        self._realized_cum += float(net_log_return)
        c = self.shadow_cost_frac
        self._base_cum += (
            float(net_log_return)
            + (self._last_base_pos - self._last_plan_pos) * float(mark_log_return)
            + float(np.log1p(-c * abs(self._last_base_pos - self._base_prev_pos)))
            - float(np.log1p(-c * abs(self._last_plan_pos - self._prev_plan_pos)))
        )
        self._base_prev_pos = self._last_base_pos
        self._prev_plan_pos = self._last_plan_pos
        self._step_in_horizon += 1
        self.stats.bars += 1
        self._awaiting_observe = False

    # -- internals ---------------------------------------------------------- #
    def _choose_archetype(
        self, context: np.ndarray, portfolio: np.ndarray
    ) -> tuple[int, tuple[float, ...]]:
        if self.policy is None:
            return int(self.fixed_archetype or 0), ()
        ctx = torch.from_numpy(np.ascontiguousarray(context)).unsqueeze(0).to(self.device).float()
        pf = torch.from_numpy(np.ascontiguousarray(portfolio)).unsqueeze(0).to(self.device).float()
        probs = self.policy.action_probs(ctx, pf, self._policy_state).squeeze(0)
        action, _, self._policy_state = self.policy.act(
            ctx, pf, self._policy_state, greedy=self.greedy, generator=self.generator
        )
        return int(action.item()), tuple(float(p) for p in probs.cpu().tolist())

    def _begin_horizon(self, position: float) -> None:
        self._horizon_index += 1
        self._step_in_horizon = 0
        self._realized_cum = 0.0
        self._base_cum = 0.0
        # Both counterfactuals start from the position actually held, so the first
        # bar's turnover charges cancel unless the adapter overrides.
        self._prev_plan_pos = self._base_prev_pos = float(position)
        self._prev_action = None
        if self.include_cash and self._archetype == self.cash_archetype:
            self._code, self._dstate = None, None
        else:
            idx = torch.tensor([self._archetype], dtype=torch.long, device=self.device)
            code = self.model.code_embedding(idx)
            self._code = code
            self._dstate = self.model.decoder.init_state(1, self.device, code.dtype)
        if self.controller is not None:
            # Budget and recurrent memory are per horizon: the adapter is a
            # within-archetype device and must not carry intent across a selector
            # decision it did not make.
            self.controller.reset(self._archetype)

    def _decoder_step(self, market_row: np.ndarray) -> int:
        code = self._code
        if code is None:  # pragma: no cover - guarded by the cash branch upstream
            raise RuntimeError("decoder used without a code; _begin_horizon was skipped")
        row = torch.from_numpy(np.ascontiguousarray(market_row)).to(self.device).float()
        logits, self._dstate = self.model.decoder.step(
            row.unsqueeze(0), code, self._prev_action, self._dstate
        )
        if self.greedy:
            pred = logits.argmax(dim=-1)
        else:
            pred = torch.multinomial(
                torch.softmax(logits, dim=-1), 1, generator=self.generator
            ).squeeze(-1)
        # The decoder is fed its *own* previous action, never the overridden or
        # shielded one, so the archetype's plan stays intact and the
        # counterfactual it is compared against stays meaningful.
        self._prev_action = pred
        return int(pred.item())

    # -- reporting ---------------------------------------------------------- #
    def stats_dict(self) -> dict[str, Any]:
        s = self.stats
        return {
            "bars": s.bars,
            "n_horizons": s.horizons,
            "archetype_counts": list(s.archetype_counts),
            "archetype_usage": s.usage(),
            "switch_rate": s.switches / max(s.horizons - 1, 1),
            "cash_fraction": (
                s.archetype_counts[self.cash_archetype] / max(s.horizons, 1)
                if self.include_cash
                else 0.0
            ),
            "n_override_bars": s.overrides,
            "n_override_runs": s.override_runs,
            "override_bar_rate": s.overrides / max(s.bars, 1),
        }


@dataclass(frozen=True)
class LoadedStack:
    """Models plus the metadata needed to wire them together consistently."""

    model: ArchetypeVQ
    policy: SelectorNet | None
    adapter: OverrideAdapter | None
    positions: tuple[float, ...]
    horizon: int
    context_bars: int
    include_cash_archetype: bool
    n_archetypes: int
    override_budget: int
    shadow_cost_frac: float
    meta: dict[str, Any] = field(default_factory=dict)


def load_stack(
    *,
    phase1_path: str | Path,
    phase2_path: str | Path | None = None,
    phase3_path: str | Path | None = None,
    device: torch.device,
    cfg: ExperimentConfig | None = None,
) -> LoadedStack:
    """Rebuild the deployed stack from checkpoints.

    Checkpoint metadata wins over ``cfg`` for anything that changes tensor shapes
    or the meaning of an index — position grid, horizon, codebook size, whether a
    cash archetype exists. A config edited after training would otherwise wire a
    decoder to the wrong action grid and produce plausible-looking nonsense.
    """
    from archetype_trader.phase1.train import load_checkpoint
    from archetype_trader.phase2.train import load_selector
    from archetype_trader.phase3.adapter import load_adapter

    model, p1_blob = load_checkpoint(phase1_path, device=device)
    positions = tuple(float(x) for x in p1_blob.get("grid") or ())
    if not positions:
        if cfg is None:
            raise ValueError(
                "the Phase-1 checkpoint carries no position grid and no config was "
                "given; refusing to guess the action grid"
            )
        positions = cfg.env.positions
    horizon = int(p1_blob.get("horizon") or (cfg.phase1.horizon if cfg else 72))

    policy: SelectorNet | None = None
    include_cash = bool(cfg.phase2.include_cash_archetype) if cfg else True
    context_bars = int(cfg.phase2.context_bars) if cfg else 48
    n_archetypes = model.codebook_size + (1 if include_cash else 0)
    meta: dict[str, Any] = {"phase1": p1_blob.get("meta", {})}
    if phase2_path is not None:
        policy, p2_blob = load_selector(phase2_path, device=device)
        n_archetypes = int(p2_blob["n_archetypes"])
        include_cash = n_archetypes > model.codebook_size
        context_bars = int((p2_blob.get("phase2") or {}).get("context_bars", context_bars))
        meta["phase2"] = p2_blob.get("meta", {})

    adapter: OverrideAdapter | None = None
    budget = int(cfg.phase3.override_budget) if cfg else 1
    shadow_cost = 0.0
    if phase3_path is not None:
        adapter, p3_blob = load_adapter(phase3_path, device=device)
        budget = int(p3_blob.get("budget", budget))
        shadow_cost = float(p3_blob.get("cost_frac", 0.0))
        grid = tuple(float(x) for x in p3_blob.get("grid", positions))
        if len(grid) != len(positions):
            raise ValueError(
                f"adapter was trained on a {len(grid)}-level grid, Phase 1 on "
                f"{len(positions)}; these are different strategies"
            )
        meta["phase3"] = p3_blob.get("meta", {})

    return LoadedStack(
        model=model,
        policy=policy,
        adapter=adapter,
        positions=positions,
        horizon=horizon,
        context_bars=context_bars,
        include_cash_archetype=include_cash,
        n_archetypes=n_archetypes,
        override_budget=budget,
        shadow_cost_frac=shadow_cost,
        meta=meta,
    )


def build_agent(
    stack: LoadedStack,
    *,
    device: torch.device,
    fixed_archetype: int | None = None,
    greedy: bool = True,
) -> HorizonAgent:
    """Assemble a :class:`HorizonAgent` from a loaded stack.

    With no Phase-2 policy this needs ``fixed_archetype`` — the Phase-1-only
    deployment, which holds one archetype for the whole run.
    """
    controller = None
    if stack.adapter is not None:
        controller = AdapterController(
            stack.adapter,
            grid=stack.positions,
            n_archetypes=stack.n_archetypes,
            budget=stack.override_budget,
            horizon=stack.horizon,
            device=device,
            greedy=greedy,
        )
    return HorizonAgent(
        stack.model,
        positions=stack.positions,
        horizon=stack.horizon,
        context_bars=stack.context_bars,
        policy=stack.policy,
        controller=controller,
        include_cash_archetype=stack.include_cash_archetype,
        fixed_archetype=fixed_archetype,
        device=device,
        greedy=greedy,
        shadow_cost_frac=stack.shadow_cost_frac,
    )
