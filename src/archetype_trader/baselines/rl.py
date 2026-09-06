"""Bar-level RL baselines: recurrent PPO, recurrent DQN, and a continuous-position SAC.

These are the strategies ArchetypeTrader has to beat to justify its extra
machinery, so they are given every advantage that does not amount to leakage:
the same features, the same environment, the same costs, the same risk shield,
and a training budget set by the same config knobs.

Three deliberate design points:

* **They act every bar.** That is the whole comparison — a flat policy choosing a
  target position at 10-minute resolution versus a hierarchy that commits to an
  archetype for ``h`` bars. If the flat policy wins, the hierarchy is not earning
  its complexity, and that is a result worth reporting.
* **They are recurrent.** The observation is a lookback window plus portfolio
  state; a feed-forward policy on a windowed observation would be a weaker
  baseline for no principled reason.
* **SAC is the continuous-position ablation.** It emits a position in
  ``[-1, +1]`` (or ``[0, 1]`` in spot) which is snapped to the environment's
  discrete grid, so ``env.position_levels`` controls how much of the continuity
  survives. At ``position_levels=1`` it degenerates to the discrete grid, which
  is exactly the point of running the ablation at several grid resolutions.

Everything here trains on a training segment and is evaluated on another segment
through the *same* environment class as every other strategy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from archetype_trader.config import BaselineConfig, EnvConfig, ShieldConfig
from archetype_trader.data.types import MarketData
from archetype_trader.env.trading_env import TradingEnv
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.splits import Segment

BarAlgo = Literal["recurrent_ppo", "dqn", "sac_continuous"]


# --------------------------------------------------------------------------- #
# Shared trunk
# --------------------------------------------------------------------------- #
class BarTrunk(nn.Module):
    """GRU over the lookback window, merged with the portfolio vector.

    Shared by all three baselines so that a performance difference between them is
    attributable to the learning rule rather than to one of them having a bigger
    encoder.
    """

    def __init__(self, n_features: int, n_portfolio: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.market_rnn = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.portfolio_proj = nn.Sequential(nn.Linear(n_portfolio, hidden_dim), nn.Tanh())
        self.merge = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh()
        )

    def forward(self, market: Tensor, portfolio: Tensor) -> Tensor:
        """``market`` ``(B, L, F)`` or ``(B, T, L, F)`` -> ``(B, H)`` / ``(B, T, H)``."""
        four = market.dim() == 4
        if four:
            b, t, lb, f = market.shape
            flat = market.reshape(b * t, lb, f)
            pf = portfolio.reshape(b * t, -1)
        else:
            flat, pf = market, portfolio
        out, _ = self.market_rnn(flat)
        merged = self.merge(torch.cat([out[:, -1], self.portfolio_proj(pf)], dim=-1))
        if four:
            return merged.reshape(market.shape[0], market.shape[1], self.hidden_dim)
        return merged


class BarActorCritic(nn.Module):
    """Categorical policy over target positions plus a value head (PPO)."""

    def __init__(
        self, n_features: int, n_portfolio: int, n_actions: int, *, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.trunk = BarTrunk(n_features, n_portfolio, hidden_dim=hidden_dim)
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def sequence(
        self, market: Tensor, portfolio: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replay a rollout under current parameters -> ``(logits, values, final_state)``."""
        merged = self.trunk(market, portfolio)
        outs = []
        h = state
        for t in range(merged.shape[1]):
            h = self.core(merged[:, t], h)
            outs.append(h)
        feats = torch.stack(outs, dim=1)
        return self.policy(feats), self.value(feats).squeeze(-1), h

    @torch.no_grad()
    def act(
        self,
        market: Tensor,
        portfolio: Tensor,
        state: Tensor,
        *,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.core(self.trunk(market, portfolio), state)
        logits = self.policy(h)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        logp = torch.log_softmax(logits, dim=-1).gather(1, action[:, None]).squeeze(1)
        return action, logp, self.value(h).squeeze(-1), h


class BarQNetwork(nn.Module):
    """Recurrent Q-network over the discrete target-position grid (DQN)."""

    def __init__(
        self, n_features: int, n_portfolio: int, n_actions: int, *, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.trunk = BarTrunk(n_features, n_portfolio, hidden_dim=hidden_dim)
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, n_actions)

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def step(self, market: Tensor, portfolio: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        h = self.core(self.trunk(market, portfolio), state)
        return self.q(h), h

    def sequence(self, market: Tensor, portfolio: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        merged = self.trunk(market, portfolio)
        outs = []
        h = state
        for t in range(merged.shape[1]):
            h = self.core(merged[:, t], h)
            outs.append(h)
        feats = torch.stack(outs, dim=1)
        return self.q(feats), h


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian over a scalar target position (SAC).

    ``tanh`` bounds the position to ``[-1, 1]`` structurally, which matters: an
    unbounded actor plus a clamp gives zero gradient exactly where the policy
    spends most of its time, and the entropy term would then be measuring the
    spread of a quantity the environment never sees.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, n_features: int, n_portfolio: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = BarTrunk(n_features, n_portfolio, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(
        self, market: Tensor, portfolio: Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Returns ``(action in [-1, 1], log_prob)`` with the tanh Jacobian correction."""
        mu, log_std = self.head(self.trunk(market, portfolio)).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        noise = torch.randn(mu.shape, generator=generator, device="cpu").to(mu.device)
        pre = mu + std * noise
        action = torch.tanh(pre)
        logp = (-0.5 * ((pre - mu) / std) ** 2 - log_std - 0.9189385332046727).sum(-1)
        logp = logp - torch.log1p(-action.pow(2) + 1e-6).sum(-1)
        return action.squeeze(-1), logp

    @torch.no_grad()
    def mean_action(self, market: Tensor, portfolio: Tensor) -> Tensor:
        mu, _ = self.head(self.trunk(market, portfolio)).chunk(2, dim=-1)
        return torch.tanh(mu).squeeze(-1)


class ContinuousCritic(nn.Module):
    """Twin Q heads on ``(observation, scalar position)``. The 'TQC-lite' part.

    Full TQC's quantile critics are omitted on purpose: with a one-dimensional
    action and a reward that is already a log return, the twin-minimum is the part
    of TQC that does the work, and the extra quantile machinery would add
    hyperparameters this baseline has no budget to tune.
    """

    def __init__(self, n_features: int, n_portfolio: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = BarTrunk(n_features, n_portfolio, hidden_dim=hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, market: Tensor, portfolio: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        z = torch.cat([self.trunk(market, portfolio), action.unsqueeze(-1)], dim=-1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# Rollout plumbing
# --------------------------------------------------------------------------- #
@dataclass
class BarRollout:
    """One contiguous on-policy segment of bar-level experience."""

    market: list[np.ndarray] = field(default_factory=list)
    portfolio: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    last_value: float = 0.0

    def __len__(self) -> int:
        return len(self.actions)

    def tensors(self, device: torch.device) -> dict[str, Tensor]:
        return {
            "market": torch.from_numpy(np.stack(self.market)).float().unsqueeze(0).to(device),
            "portfolio": torch.from_numpy(np.stack(self.portfolio)).float().unsqueeze(0).to(device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device).unsqueeze(0),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device).unsqueeze(
                0
            ),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device).unsqueeze(0),
        }


@dataclass
class Transition:
    """One off-policy transition. Stored as arrays; the buffer is small by design."""

    market: np.ndarray
    portfolio: np.ndarray
    action: float
    reward: float
    next_market: np.ndarray
    next_portfolio: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity uniform replay. No prioritisation — one fewer knob to tune."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, tr: Transition) -> None:
        self._buf.append(tr)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, len(self._buf), size=min(n, len(self._buf)))
        rows = [self._buf[int(i)] for i in idx]
        return {
            "market": np.stack([r.market for r in rows]),
            "portfolio": np.stack([r.portfolio for r in rows]),
            "action": np.asarray([r.action for r in rows], dtype=np.float32),
            "reward": np.asarray([r.reward for r in rows], dtype=np.float32),
            "next_market": np.stack([r.next_market for r in rows]),
            "next_portfolio": np.stack([r.next_portfolio for r in rows]),
            "done": np.asarray([r.done for r in rows], dtype=np.float32),
        }


@dataclass
class BarRLResult:
    """Trained bar-level baseline plus its training trace."""

    algo: BarAlgo
    model: nn.Module
    history: list[dict[str, float]]
    n_env_steps: int

    @property
    def final_train_return(self) -> float:
        return float(self.history[-1]["episode_net_log_return"]) if self.history else 0.0


def _make_env(
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    env_cfg: EnvConfig,
    shield_cfg: ShieldConfig,
    segment: Segment,
    *,
    bars_per_year: int,
    max_bars: int | None = None,
) -> TradingEnv:
    start = max(segment.start, features.valid_from, features.lookback - 1)
    cfg = env_cfg if max_bars is None else env_cfg.model_copy(update={"max_episode_bars": max_bars})
    return TradingEnv(
        data,
        features,
        scaler,
        cfg,
        shield_cfg,
        start_index=start,
        end_index=segment.stop,
        bars_per_year=bars_per_year,
    )


def _episode_start(env: TradingEnv, rng: np.random.Generator, n_steps: int) -> int:
    """Uniform random start leaving room for ``n_steps`` bars.

    Random starts matter more here than in Phase 2: a bar-level policy trained
    always from the same bar memorises the segment's opening regime.
    """
    lo = env.start_index
    hi = max(lo + 1, env.end_index - n_steps - 2)
    return int(rng.integers(lo, hi)) if hi > lo else lo


def _gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in reversed(range(n)):
        nonterminal = 0.0 if dones[t] else 1.0
        next_v = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
def train_bar_ppo(
    env: TradingEnv,
    cfg: BaselineConfig,
    *,
    device: torch.device,
    seed: int = 0,
    reward_scale: float = 100.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    entropy_coef: float = 0.01,
    epochs_per_update: int = 4,
) -> BarRLResult:
    """Recurrent PPO over discrete target positions, one decision per bar."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model = BarActorCritic(
        env.features.n_features,
        env.observation_space["portfolio"].shape[0],  # type: ignore[index]
        len(env.positions),
        hidden_dim=cfg.rl_hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    history: list[dict[str, float]] = []
    total_steps = 0

    for update in range(cfg.rl_updates):
        roll = BarRollout()
        obs, _ = env.reset(
            options={"start_index": _episode_start(env, rng, cfg.rl_steps_per_update)}
        )
        state = model.init_state(1, device)
        net_sum = 0.0
        for _ in range(cfg.rl_steps_per_update):
            m = torch.from_numpy(obs["market"]).float().unsqueeze(0).to(device)
            p = torch.from_numpy(obs["portfolio"]).float().unsqueeze(0).to(device)
            action, logp, value, state = model.act(m, p, state, generator=gen)
            a = int(action.item())
            roll.market.append(np.asarray(obs["market"], dtype=np.float32))
            roll.portfolio.append(np.asarray(obs["portfolio"], dtype=np.float32))
            roll.actions.append(a)
            roll.log_probs.append(float(logp.item()))
            roll.values.append(float(value.item()))
            obs, reward, term, trunc, info = env.step(a)
            roll.rewards.append(float(reward) * reward_scale)
            roll.dones.append(bool(term))
            net_sum += float(info["net_log_return"])
            total_steps += 1
            if term or trunc:
                break
        # A one-step rollout has no advantage spread to normalise by; dividing by
        # a NaN std would poison every subsequent forward pass with NaN logits.
        if len(roll) < 2:
            continue
        with torch.no_grad():
            m = torch.from_numpy(obs["market"]).float().unsqueeze(0).to(device)
            p = torch.from_numpy(obs["portfolio"]).float().unsqueeze(0).to(device)
            # Bootstrap only if the episode did not terminate; a terminated
            # episode has no future value by definition.
            last_v = 0.0 if roll.dones[-1] else float(model.act(m, p, state, greedy=True)[2].item())
        adv, ret = _gae(roll.rewards, roll.values, roll.dones, last_v, gamma, gae_lambda)
        t = roll.tensors(device)
        adv_t = torch.from_numpy(adv).float().unsqueeze(0).to(device)
        ret_t = torch.from_numpy(ret).float().unsqueeze(0).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats: dict[str, float] = {}
        for _ in range(epochs_per_update):
            logits, values, _ = model.sequence(
                t["market"], t["portfolio"], model.init_state(1, device)
            )
            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(2, t["actions"].unsqueeze(-1)).squeeze(-1)
            ratio = (logp - t["log_probs"]).exp()
            pg = -torch.min(
                ratio * adv_t, ratio.clamp(1 - clip_range, 1 + clip_range) * adv_t
            ).mean()
            vl = F.mse_loss(values, ret_t)
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            loss = pg + 0.5 * vl - entropy_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            stats = {
                "policy_loss": float(pg.detach()),
                "value_loss": float(vl.detach()),
                "entropy": float(ent.detach()),
            }
        history.append(
            {
                "update": float(update),
                "steps": float(len(roll)),
                "episode_net_log_return": net_sum,
                "mean_reward": float(np.mean(roll.rewards)),
                **stats,
            }
        )
    return BarRLResult(algo="recurrent_ppo", model=model, history=history, n_env_steps=total_steps)


# --------------------------------------------------------------------------- #
# DQN
# --------------------------------------------------------------------------- #
def train_bar_dqn(
    env: TradingEnv,
    cfg: BaselineConfig,
    *,
    device: torch.device,
    seed: int = 0,
    reward_scale: float = 100.0,
    gamma: float = 0.99,
    target_sync_every: int = 5,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
) -> BarRLResult:
    """Recurrent DQN over discrete target positions.

    Trained on whole on-policy segments rather than from a replay buffer of single
    transitions: the Q-network is recurrent, and replaying isolated transitions
    would destroy the hidden state that makes it recurrent in the first place.
    Sequence-level updates with a target network are the standard R2D2-style
    compromise, minus the distributed machinery.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_actions = len(env.positions)
    n_portfolio = env.observation_space["portfolio"].shape[0]  # type: ignore[index]
    online = BarQNetwork(
        env.features.n_features, n_portfolio, n_actions, hidden_dim=cfg.rl_hidden
    ).to(device)
    target = BarQNetwork(
        env.features.n_features, n_portfolio, n_actions, hidden_dim=cfg.rl_hidden
    ).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=3e-4)
    history: list[dict[str, float]] = []
    total_steps = 0

    for update in range(cfg.rl_updates):
        frac = update / max(cfg.rl_updates - 1, 1)
        eps = eps_start + frac * (eps_end - eps_start)
        markets: list[np.ndarray] = []
        portfolios: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        obs, _ = env.reset(
            options={"start_index": _episode_start(env, rng, cfg.rl_steps_per_update)}
        )
        state = online.init_state(1, device)
        net_sum = 0.0
        for _ in range(cfg.rl_steps_per_update):
            m = torch.from_numpy(obs["market"]).float().unsqueeze(0).to(device)
            p = torch.from_numpy(obs["portfolio"]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q, state = online.step(m, p, state)
            a = int(rng.integers(n_actions)) if rng.random() < eps else int(q.argmax(dim=-1).item())
            markets.append(np.asarray(obs["market"], dtype=np.float32))
            portfolios.append(np.asarray(obs["portfolio"], dtype=np.float32))
            actions.append(a)
            obs, reward, term, trunc, info = env.step(a)
            rewards.append(float(reward) * reward_scale)
            dones.append(bool(term))
            net_sum += float(info["net_log_return"])
            total_steps += 1
            if term or trunc:
                break
        n = len(actions)
        if n < 2:
            continue
        # The last stored observation has no recorded successor inside this
        # segment, so the final transition is dropped rather than bootstrapped
        # from an observation that does not belong to it.
        mk = torch.from_numpy(np.stack(markets)).float().unsqueeze(0).to(device)
        pf = torch.from_numpy(np.stack(portfolios)).float().unsqueeze(0).to(device)
        act = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
        rew = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(0)
        nonterm = torch.tensor(
            [0.0 if d else 1.0 for d in dones], dtype=torch.float32, device=device
        ).unsqueeze(0)

        q_all, _ = online.sequence(mk, pf, online.init_state(1, device))
        q_taken = q_all.gather(2, act.unsqueeze(-1)).squeeze(-1)[:, :-1]
        with torch.no_grad():
            q_next_online, _ = online.sequence(mk, pf, online.init_state(1, device))
            q_next_target, _ = target.sequence(mk, pf, target.init_state(1, device))
            # Double DQN: online picks the argmax, target evaluates it.
            best = q_next_online[:, 1:].argmax(dim=-1, keepdim=True)
            boot = q_next_target[:, 1:].gather(2, best).squeeze(-1)
            tgt = rew[:, :-1] + gamma * nonterm[:, :-1] * boot
        loss = F.smooth_l1_loss(q_taken, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        if (update + 1) % target_sync_every == 0:
            target.load_state_dict(online.state_dict())
        history.append(
            {
                "update": float(update),
                "steps": float(n),
                "epsilon": eps,
                "loss": float(loss.detach()),
                "episode_net_log_return": net_sum,
                "mean_q": float(q_all.detach().mean()),
            }
        )
    return BarRLResult(algo="dqn", model=online, history=history, n_env_steps=total_steps)


# --------------------------------------------------------------------------- #
# SAC (continuous position)
# --------------------------------------------------------------------------- #
class SACModules(nn.Module):
    """Container so the actor and critic checkpoint and move device together."""

    def __init__(self, actor: SquashedGaussianActor, critic: ContinuousCritic) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic


def train_bar_sac(
    env: TradingEnv,
    cfg: BaselineConfig,
    *,
    device: torch.device,
    seed: int = 0,
    reward_scale: float = 100.0,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 128,
    warmup_steps: int = 512,
    updates_per_step: int = 1,
) -> BarRLResult:
    """SAC on a continuous target position, snapped to the environment's grid.

    The snapping is done by the environment's own action grid, so the reward the
    critic learns from is the reward of the position that was actually held — not
    of the unrounded number the actor emitted. Automatic entropy tuning targets
    ``-1`` nat (one action dimension), the usual heuristic.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_portfolio = env.observation_space["portfolio"].shape[0]  # type: ignore[index]
    grid = np.asarray(env.positions, dtype=np.float64)
    actor = SquashedGaussianActor(
        env.features.n_features, n_portfolio, hidden_dim=cfg.rl_hidden
    ).to(device)
    critic = ContinuousCritic(env.features.n_features, n_portfolio, hidden_dim=cfg.rl_hidden).to(
        device
    )
    critic_target = ContinuousCritic(
        env.features.n_features, n_portfolio, hidden_dim=cfg.rl_hidden
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())
    log_alpha = torch.zeros(1, device=device, requires_grad=True)
    target_entropy = -1.0
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
    buf = ReplayBuffer(20_000)
    history: list[dict[str, float]] = []
    total_steps = 0

    def snap(x: float) -> int:
        return int(np.abs(grid - x).argmin())

    for update in range(cfg.rl_updates):
        obs, _ = env.reset(
            options={"start_index": _episode_start(env, rng, cfg.rl_steps_per_update)}
        )
        net_sum = 0.0
        losses: list[tuple[float, float, float]] = []
        for _ in range(cfg.rl_steps_per_update):
            m = torch.from_numpy(obs["market"]).float().unsqueeze(0).to(device)
            p = torch.from_numpy(obs["portfolio"]).float().unsqueeze(0).to(device)
            if total_steps < warmup_steps:
                raw = float(rng.uniform(-1.0, 1.0))
            else:
                with torch.no_grad():
                    raw = float(actor(m, p, generator=gen)[0].item())
            idx = snap(raw)
            prev_obs = obs
            obs, reward, term, trunc, info = env.step(idx)
            # The stored action is the *executed* grid position, so the critic is
            # never trained on a position the environment did not hold.
            buf.push(
                Transition(
                    market=np.asarray(prev_obs["market"], dtype=np.float32),
                    portfolio=np.asarray(prev_obs["portfolio"], dtype=np.float32),
                    action=float(grid[idx]),
                    reward=float(reward) * reward_scale,
                    next_market=np.asarray(obs["market"], dtype=np.float32),
                    next_portfolio=np.asarray(obs["portfolio"], dtype=np.float32),
                    done=bool(term),
                )
            )
            net_sum += float(info["net_log_return"])
            total_steps += 1
            if len(buf) >= batch_size and total_steps >= warmup_steps:
                for _ in range(updates_per_step):
                    losses.append(
                        _sac_update(
                            buf,
                            rng,
                            actor,
                            critic,
                            critic_target,
                            log_alpha,
                            opt_a,
                            opt_c,
                            opt_alpha,
                            device=device,
                            gamma=gamma,
                            tau=tau,
                            batch_size=batch_size,
                            target_entropy=target_entropy,
                            generator=gen,
                        )
                    )
            if term or trunc:
                break
        arr = np.asarray(losses, dtype=np.float64) if losses else np.zeros((1, 3))
        history.append(
            {
                "update": float(update),
                "episode_net_log_return": net_sum,
                "critic_loss": float(arr[:, 0].mean()),
                "actor_loss": float(arr[:, 1].mean()),
                "alpha": float(arr[:, 2].mean()),
                "buffer": float(len(buf)),
            }
        )
    return BarRLResult(
        algo="sac_continuous",
        model=SACModules(actor, critic),
        history=history,
        n_env_steps=total_steps,
    )


def _sac_update(
    buf: ReplayBuffer,
    rng: np.random.Generator,
    actor: SquashedGaussianActor,
    critic: ContinuousCritic,
    critic_target: ContinuousCritic,
    log_alpha: Tensor,
    opt_a: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    opt_alpha: torch.optim.Optimizer,
    *,
    device: torch.device,
    gamma: float,
    tau: float,
    batch_size: int,
    target_entropy: float,
    generator: torch.Generator | None,
) -> tuple[float, float, float]:
    b = buf.sample(batch_size, rng)
    m = torch.from_numpy(b["market"]).float().to(device)
    p = torch.from_numpy(b["portfolio"]).float().to(device)
    a = torch.from_numpy(b["action"]).float().to(device)
    r = torch.from_numpy(b["reward"]).float().to(device)
    nm = torch.from_numpy(b["next_market"]).float().to(device)
    npf = torch.from_numpy(b["next_portfolio"]).float().to(device)
    nonterm = 1.0 - torch.from_numpy(b["done"]).float().to(device)
    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        na, nlogp = actor(nm, npf, generator=generator)
        tq1, tq2 = critic_target(nm, npf, na)
        target = r + gamma * nonterm * (torch.min(tq1, tq2) - alpha * nlogp)
    q1, q2 = critic(m, p, a)
    closs = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    opt_c.zero_grad(set_to_none=True)
    closs.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    opt_c.step()

    pa, plogp = actor(m, p, generator=generator)
    pq1, pq2 = critic(m, p, pa)
    aloss = (alpha * plogp - torch.min(pq1, pq2)).mean()
    opt_a.zero_grad(set_to_none=True)
    aloss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    opt_a.step()

    alpha_loss = -(log_alpha.exp() * (plogp.detach() + target_entropy)).mean()
    opt_alpha.zero_grad(set_to_none=True)
    alpha_loss.backward()
    opt_alpha.step()

    with torch.no_grad():
        for tp, sp in zip(critic_target.parameters(), critic.parameters(), strict=True):
            tp.mul_(1 - tau).add_(tau * sp)
    return float(closs.detach()), float(aloss.detach()), float(log_alpha.exp().detach().item())


# --------------------------------------------------------------------------- #
# Deterministic evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_bar_policy(
    result: BarRLResult, env: TradingEnv, *, device: torch.device, start_index: int | None = None
) -> dict[str, Any]:
    """Greedy/mean-action roll-out of a trained baseline over a whole segment.

    Greedy at evaluation for all three algorithms: a stochastic policy evaluated
    once would confound policy quality with sampling noise, and the seed
    distribution over training runs is where the variance belongs.
    """
    obs, _ = env.reset(
        options={"start_index": env.start_index if start_index is None else start_index}
    )
    grid = np.asarray(env.positions, dtype=np.float64)
    model = result.model
    model.eval()
    if not isinstance(model, BarActorCritic | BarQNetwork | SACModules):
        raise TypeError(f"cannot roll out {type(model).__name__}")
    state = (
        model.init_state(1, device)
        if isinstance(model, BarActorCritic | BarQNetwork)
        else torch.zeros(1)
    )
    while True:
        m = torch.from_numpy(obs["market"]).float().unsqueeze(0).to(device)
        p = torch.from_numpy(obs["portfolio"]).float().unsqueeze(0).to(device)
        if isinstance(model, BarActorCritic):
            action, _, _, state = model.act(m, p, state, greedy=True)
            idx = int(action.item())
        elif isinstance(model, BarQNetwork):
            q, state = model.step(m, p, state)
            idx = int(q.argmax(dim=-1).item())
        else:
            raw = float(model.actor.mean_action(m, p).item())
            idx = int(np.abs(grid - raw).argmin())
        obs, _, term, trunc, _ = env.step(idx)
        if term or trunc:
            break
    summary = env.episode_summary()
    summary["name"] = result.algo
    summary["mean_position"] = float(np.mean(summary["positions"])) if summary["n_steps"] else 0.0
    return summary


def train_bar_baseline(
    algo: BarAlgo,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    env_cfg: EnvConfig,
    shield_cfg: ShieldConfig,
    cfg: BaselineConfig,
    train_segment: Segment,
    *,
    device: torch.device,
    seed: int = 0,
    bars_per_year: int = 52_560,
) -> BarRLResult:
    """Train one bar-level baseline on ``train_segment`` and nothing else."""
    env = _make_env(
        data,
        features,
        scaler,
        env_cfg,
        shield_cfg,
        train_segment,
        bars_per_year=bars_per_year,
        max_bars=cfg.rl_steps_per_update + 2,
    )
    if algo == "recurrent_ppo":
        return train_bar_ppo(env, cfg, device=device, seed=seed)
    if algo == "dqn":
        return train_bar_dqn(env, cfg, device=device, seed=seed)
    if algo == "sac_continuous":
        return train_bar_sac(env, cfg, device=device, seed=seed)
    raise KeyError(f"unknown bar-level baseline {algo!r}")
