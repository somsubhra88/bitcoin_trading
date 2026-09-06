"""Phase-3 step-level override adapter.

The adapter watches the archetype execute and may, a bounded number of times per
horizon, replace its target position. Its head has ``A + 1`` classes: the
positions, plus an explicit **retain** class meaning "do what the archetype
said". Retention being its own class is what makes the budget enforceable by
masking one logit — no post-hoc filtering, no silent divergence between the
budget the loss assumed and the budget the deployment applies.

Observation at step ``t`` (all of it available at ``t``):

* the current causal market feature row,
* a one-hot of the selected archetype,
* a one-hot of the archetype's proposed action and of the position actually held,
* realised cumulative return so far, the shadow *base* cumulative return so far,
  and their difference,
* fraction of the horizon remaining and fraction of the override budget left.

Nothing here reads a future return. :class:`AdapterController` is the deployed
form: it is fed one step at a time and holds no array it could index forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

#: realised cum, base cum, difference, steps-left fraction, budget-left fraction
N_ADAPTER_SCALARS = 5


class OverrideAdapter(nn.Module):
    """Recurrent policy + value head over ``A + 1`` classes."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        n_archetypes: int,
        *,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_archetypes = n_archetypes
        self.hidden_dim = hidden_dim
        self.retain_index = n_actions
        self.input_dim = n_features + n_archetypes + 2 * n_actions + N_ADAPTER_SCALARS
        self.encode = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, n_actions + 1)
        self.value = nn.Linear(hidden_dim, 1)
        # Bias retention at initialisation: an untrained adapter that overrides at
        # random is strictly worse than the archetype it is wrapping, and the
        # override budget would be spent before any signal exists.
        nn.init.zeros_(self.policy.weight)
        with torch.no_grad():
            self.policy.bias.zero_()
            self.policy.bias[self.retain_index] = 2.0

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def forward(self, features: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One step. ``features`` ``(B, input_dim)`` -> ``(logits, value, state)``."""
        new_state = self.cell(self.encode(features), state)
        return self.policy(new_state), self.value(new_state).squeeze(-1), new_state


def build_step_features(
    market_row: Tensor,
    archetype_onehot: Tensor,
    base_action_onehot: Tensor,
    position_onehot: Tensor,
    scalars: Tensor,
) -> Tensor:
    """Concatenate the adapter's step inputs in the one canonical order."""
    return torch.cat(
        [market_row, archetype_onehot, base_action_onehot, position_onehot, scalars], dim=-1
    )


def budget_mask(
    logits: Tensor,
    base_action: Tensor,
    budget_left: Tensor,
    in_override: Tensor,
    *,
    retain_index: int,
) -> Tensor:
    """Mask actions that would start an override the budget cannot pay for.

    Continuing an existing run is free by construction, so the mask only bites
    when ``budget_left == 0`` *and* the previous step was not an override: then
    only retention, or an action equal to the base action, stays legal.
    """
    blocked = (budget_left <= 0) & (~in_override)
    if not bool(blocked.any()):
        return logits
    n_actions = logits.shape[-1] - 1
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[:, retain_index] = True
    allowed.scatter_(1, base_action.unsqueeze(1).clamp(max=n_actions - 1), True)
    mask = blocked.unsqueeze(1) & ~allowed
    return logits.masked_fill(mask, float("-inf"))


@dataclass
class AdapterDecision:
    """What the adapter did on one step, for logging and shield comparison."""

    action: int
    base_action: int
    overrode: bool
    started_override: bool
    budget_left: int
    retain_prob: float


class AdapterController:
    """Stateful, single-trajectory driver for :class:`OverrideAdapter`.

    The deployed Phase-3 path. It is handed one observation at a time and keeps no
    reference to any future-indexed array, which is what the leakage test checks.
    """

    def __init__(
        self,
        model: OverrideAdapter,
        *,
        grid: tuple[float, ...],
        n_archetypes: int,
        budget: int,
        horizon: int,
        device: torch.device | None = None,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.grid = np.asarray(grid, dtype=np.float64)
        self.n_archetypes = n_archetypes
        self.budget = budget
        self.horizon = horizon
        self.device = device or torch.device("cpu")
        self.greedy = greedy
        self.generator = generator
        self.reset(0)

    def reset(self, archetype: int) -> None:
        """Start a new horizon. Budget and recurrent state are per horizon."""
        self.archetype = archetype
        self.state = self.model.init_state(1, self.device)
        self.budget_left = self.budget
        self.in_override = False
        self.step_index = 0
        self.realized_cum = 0.0
        self.base_cum = 0.0

    @torch.no_grad()
    def decide(
        self,
        market_row: np.ndarray,
        *,
        base_action: int,
        position: float,
        realized_cum: float,
        base_cum: float,
    ) -> AdapterDecision:
        """Return the action to send to the environment for this step."""
        self.realized_cum = realized_cum
        self.base_cum = base_cum
        dev = self.device
        n_a = len(self.grid)
        arche = torch.zeros(1, self.n_archetypes, device=dev)
        arche[0, min(self.archetype, self.n_archetypes - 1)] = 1.0
        base_oh = torch.zeros(1, n_a, device=dev)
        base_oh[0, base_action] = 1.0
        pos_oh = torch.zeros(1, n_a, device=dev)
        pos_oh[0, int(np.argmin(np.abs(self.grid - position)))] = 1.0
        steps_left = (self.horizon - self.step_index) / max(self.horizon, 1)
        scalars = torch.tensor(
            [
                [
                    realized_cum,
                    base_cum,
                    realized_cum - base_cum,
                    steps_left,
                    self.budget_left / max(self.budget, 1),
                ]
            ],
            dtype=torch.float32,
            device=dev,
        )
        row = torch.from_numpy(np.ascontiguousarray(market_row)).to(dev).float().unsqueeze(0)
        feats = build_step_features(row, arche, base_oh, pos_oh, scalars)
        logits, _, self.state = self.model(feats, self.state)
        logits = budget_mask(
            logits,
            torch.tensor([base_action], device=dev),
            torch.tensor([self.budget_left], device=dev),
            torch.tensor([self.in_override], device=dev),
            retain_index=self.model.retain_index,
        )
        probs = torch.softmax(logits, dim=-1)
        if self.greedy:
            cls = int(logits.argmax(dim=-1).item())
        else:
            cls = int(torch.multinomial(probs, 1, generator=self.generator).item())

        action = base_action if cls == self.model.retain_index else cls
        overrode = action != base_action
        started = overrode and not self.in_override
        if started:
            self.budget_left -= 1
        self.in_override = overrode
        self.step_index += 1
        return AdapterDecision(
            action=action,
            base_action=base_action,
            overrode=overrode,
            started_override=started,
            budget_left=self.budget_left,
            retain_prob=float(probs[0, self.model.retain_index]),
        )


def load_adapter(
    path: str | Path, *, device: torch.device
) -> tuple[OverrideAdapter, dict[str, Any]]:
    """Rebuild an adapter for inference.

    Deliberately lives here rather than in :mod:`archetype_trader.phase3.train`:
    the deployed path must be able to load an adapter without importing anything
    that can see the future, and the leakage test asserts exactly that.
    """
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    p3 = blob.get("phase3") or {}
    model = OverrideAdapter(
        blob["n_features"],
        blob["n_actions"],
        blob["n_archetypes"],
        hidden_dim=p3.get("hidden_dim", 64),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def make_controller(
    model: OverrideAdapter,
    *,
    grid: tuple[float, ...],
    n_archetypes: int,
    budget: int,
    horizon: int,
    device: torch.device | None = None,
) -> AdapterController:
    """Deployment wrapper for use with ``HorizonEnv(override=...)``."""
    return AdapterController(
        model,
        grid=grid,
        n_archetypes=n_archetypes,
        budget=budget,
        horizon=horizon,
        device=device,
        greedy=True,
    )
