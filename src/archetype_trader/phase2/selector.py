"""Phase-2 selector networks behind a single interface.

The selector answers one question per horizon: *which archetype should run for
the next ``h`` bars?* It never emits a position — that is the frozen Phase-1
decoder's job — so the action space is ``K`` (+1 for the explicit cash archetype)
regardless of how many positions the market environment admits.

Two implementations share :class:`SelectorTrunk` and the :class:`SelectorPolicy`
protocol so the training code can swap algorithms without touching the
environment:

* :class:`RecurrentActorCritic` — categorical policy + value head, for PPO.
* :class:`RecurrentQNetwork` — Q head over archetypes, for recurrent DQN.

Recurrence is *across horizons*, not across bars: the selector must remember
that it has been repeatedly stopped out or that it just flipped direction, and a
horizon-level GRU state is the cheapest place to keep that.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@runtime_checkable
class SelectorPolicy(Protocol):
    """Minimal contract the Phase-2 trainer and the evaluator rely on."""

    n_archetypes: int
    hidden_dim: int

    def init_state(self, batch: int, device: torch.device) -> Tensor: ...

    def act(
        self, context: Tensor, portfolio: Tensor, state: Tensor, *, greedy: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One macro step. Returns ``(action, log_prob_or_zero, new_state)``."""
        ...


class SelectorTrunk(nn.Module):
    """Encodes ``(market context window, portfolio state)`` plus horizon memory.

    The market context is summarised by its own GRU over bars, then merged with
    the portfolio vector and fed to a ``GRUCell`` that carries state *between*
    horizons.
    """

    def __init__(
        self,
        n_features: int,
        n_portfolio: int,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.market_rnn = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.portfolio_proj = nn.Sequential(nn.Linear(n_portfolio, hidden_dim), nn.Tanh())
        self.merge = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.core = nn.GRUCell(hidden_dim, hidden_dim)

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def encode_obs(self, context: Tensor, portfolio: Tensor) -> Tensor:
        """``context`` ``(B, C, F)`` / ``(B, T, C, F)``, ``portfolio`` matching -> merged."""
        if context.dim() == 4:
            b, t, c, f = context.shape
            flat = context.reshape(b * t, c, f)
            pf = portfolio.reshape(b * t, -1)
        else:
            flat, pf = context, portfolio
        out, _ = self.market_rnn(flat)
        merged = self.merge(torch.cat([out[:, -1], self.portfolio_proj(pf)], dim=-1))
        if context.dim() == 4:
            return merged.reshape(context.shape[0], context.shape[1], self.hidden_dim)
        return merged

    def step(self, context: Tensor, portfolio: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """One macro step -> ``(features (B, H), new_state (B, H))``."""
        new_state = self.core(self.encode_obs(context, portfolio), state)
        return new_state, new_state

    def sequence(self, context: Tensor, portfolio: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Replay a whole macro-step sequence. ``context`` ``(B, T, C, F)``.

        PPO needs this: the recurrent state at step ``t`` of an update must be
        recomputed from the *current* parameters, not read back from the rollout.
        """
        merged = self.encode_obs(context, portfolio)
        outs = []
        h = state
        for t in range(merged.shape[1]):
            h = self.core(merged[:, t], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


class RecurrentActorCritic(nn.Module):
    """Categorical policy over archetypes plus a state-value head (PPO)."""

    def __init__(
        self,
        n_features: int,
        n_portfolio: int,
        n_archetypes: int,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_archetypes = n_archetypes
        self.hidden_dim = hidden_dim
        self.trunk = SelectorTrunk(n_features, n_portfolio, hidden_dim=hidden_dim)
        self.policy = nn.Linear(hidden_dim, n_archetypes)
        self.value = nn.Linear(hidden_dim, 1)
        # Small policy init keeps the first update from committing to one
        # archetype before any return signal has arrived.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return self.trunk.init_state(batch, device)

    def forward(
        self, context: Tensor, portfolio: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sequence forward -> ``(logits (B, T, K), values (B, T), final_state)``."""
        feats, new_state = self.trunk.sequence(context, portfolio, state)
        return self.policy(feats), self.value(feats).squeeze(-1), new_state

    @torch.no_grad()
    def act(
        self,
        context: Tensor,
        portfolio: Tensor,
        state: Tensor,
        *,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        feats, new_state = self.trunk.step(context, portfolio, state)
        logits = self.policy(feats)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        log_prob = torch.log_softmax(logits, dim=-1).gather(1, action[:, None]).squeeze(1)
        return action, log_prob, new_state

    @torch.no_grad()
    def action_probs(self, context: Tensor, portfolio: Tensor, state: Tensor) -> Tensor:
        feats, _ = self.trunk.step(context, portfolio, state)
        return torch.softmax(self.policy(feats), dim=-1)

    def evaluate_actions(
        self, context: Tensor, portfolio: Tensor, state: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(log_probs, entropy, values, logits)`` for a stored sequence."""
        logits, values, _ = self(context, portfolio, state)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return chosen, entropy, values, logits


class RecurrentQNetwork(nn.Module):
    """Q values over archetypes (recurrent DQN alternative).

    Present so the "which discrete RL algorithm?" question is answerable by
    experiment rather than by assertion; it shares the trunk so the comparison is
    about the learning rule, not the encoder.
    """

    def __init__(
        self,
        n_features: int,
        n_portfolio: int,
        n_archetypes: int,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_archetypes = n_archetypes
        self.hidden_dim = hidden_dim
        self.trunk = SelectorTrunk(n_features, n_portfolio, hidden_dim=hidden_dim)
        # Dueling head: the advantage stream matters because most archetypes are
        # near-equivalent in quiet regimes and the value baseline absorbs that.
        self.adv = nn.Linear(hidden_dim, n_archetypes)
        self.val = nn.Linear(hidden_dim, 1)

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return self.trunk.init_state(batch, device)

    def forward(self, context: Tensor, portfolio: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        feats, new_state = self.trunk.sequence(context, portfolio, state)
        adv = self.adv(feats)
        q = self.val(feats) + adv - adv.mean(dim=-1, keepdim=True)
        return q, new_state

    @torch.no_grad()
    def act(
        self,
        context: Tensor,
        portfolio: Tensor,
        state: Tensor,
        *,
        greedy: bool = False,
        epsilon: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        feats, new_state = self.trunk.step(context, portfolio, state)
        adv = self.adv(feats)
        q = self.val(feats) + adv - adv.mean(dim=-1, keepdim=True)
        action = q.argmax(dim=-1)
        if not greedy and epsilon > 0.0:
            explore = torch.rand(action.shape, device=q.device, generator=generator) < epsilon
            random_a = torch.randint(
                0, self.n_archetypes, action.shape, device=q.device, generator=generator
            )
            action = torch.where(explore, random_a, action)
        # DQN has no log-probability; zeros keep the tuple shape of the protocol.
        return action, torch.zeros_like(action, dtype=q.dtype), new_state

    @torch.no_grad()
    def action_probs(self, context: Tensor, portfolio: Tensor, state: Tensor) -> Tensor:
        """Softmax over Q as a stand-in for archetype probabilities in logs."""
        feats, _ = self.trunk.step(context, portfolio, state)
        adv = self.adv(feats)
        q = self.val(feats) + adv - adv.mean(dim=-1, keepdim=True)
        return F.softmax(q, dim=-1)


def build_selector(
    algo: str,
    n_features: int,
    n_portfolio: int,
    n_archetypes: int,
    *,
    hidden_dim: int = 128,
) -> RecurrentActorCritic | RecurrentQNetwork:
    if algo == "recurrent_ppo":
        return RecurrentActorCritic(n_features, n_portfolio, n_archetypes, hidden_dim=hidden_dim)
    if algo == "recurrent_dqn":
        return RecurrentQNetwork(n_features, n_portfolio, n_archetypes, hidden_dim=hidden_dim)
    raise ValueError(f"unknown phase-2 algorithm {algo!r}")
