"""Phase-2 training: learn *which* archetype to run, one choice per horizon.

Two algorithms behind one entry point (:func:`train_phase2`):

* ``recurrent_ppo`` — categorical recurrent PPO with GAE. Sequences are replayed
  from a zero state on every update so the recurrent state is consistent with the
  current parameters rather than stale rollout state.
* ``recurrent_dqn`` — dueling recurrent DQN over sequence replay, provided so the
  choice of algorithm is an experiment rather than an assumption.

The reward is the horizon's realised, fully-costed reward from
:class:`TradingEnv` (net log equity change minus the configured penalties),
summed over the bars of the horizon. The unshaped net log return is carried
alongside so the report can show economics that no penalty ever touched.

The optional auxiliary term is a cross-entropy toward the Phase-1
demonstration's archetype label at the same bar. It is *training-only
supervision from a training-only planner* — the labels exist for training start
indices and nowhere else, and the selector never receives them as an input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from archetype_trader.config import ExperimentConfig
from archetype_trader.phase2.horizon_env import HorizonEnv
from archetype_trader.phase2.selector import (
    RecurrentActorCritic,
    RecurrentQNetwork,
    build_selector,
)
from archetype_trader.utils.logging import get_logger, log_event

logger = get_logger(__name__)

_NO_LABEL = -1


@dataclass
class Rollout:
    """One macro episode of selector experience."""

    context: np.ndarray  # (T, C, F)
    portfolio: np.ndarray  # (T, P)
    actions: np.ndarray  # (T,)
    log_probs: np.ndarray  # (T,)
    values: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    net_log_returns: np.ndarray  # (T,) unshaped
    labels: np.ndarray  # (T,) -1 where no demonstration label exists
    terminated: bool
    last_value: float

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclass
class UpdateStats:
    update: int
    n_episodes: int
    n_horizons: int
    mean_reward: float
    mean_net_log_return: float
    policy_loss: float
    value_loss: float
    entropy: float
    aux_loss: float
    approx_kl: float
    grad_norm: float
    switch_rate: float
    cash_fraction: float
    epsilon: float
    archetype_usage: list[float] = field(default_factory=list)


@dataclass
class Phase2Result:
    policy: RecurrentActorCritic | RecurrentQNetwork
    history: list[UpdateStats]
    best_update: int
    best_objective: float
    meta: dict[str, Any] = field(default_factory=dict)

    def history_table(self) -> list[dict[str, Any]]:
        return [vars(h) for h in self.history]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def label_array(n_bars: int, starts: np.ndarray, code_indices: np.ndarray) -> np.ndarray:
    """Per-bar archetype label from Phase-1 demonstrations; ``-1`` elsewhere.

    Only training start indices are populated, so a label can never appear in a
    validation or test fold — the array is all ``-1`` there and the auxiliary
    term silently contributes nothing.
    """
    labels = np.full(n_bars, _NO_LABEL, dtype=np.int64)
    labels[starts] = code_indices
    return labels


# --------------------------------------------------------------------------- #
# Rollout collection
# --------------------------------------------------------------------------- #
def _collect_episode(
    henv: HorizonEnv,
    policy: RecurrentActorCritic | RecurrentQNetwork,
    *,
    start_index: int,
    labels: np.ndarray,
    device: torch.device,
    max_horizons: int,
    greedy: bool = False,
    epsilon: float = 0.0,
    generator: torch.Generator | None = None,
) -> Rollout:
    obs = henv.reset(start_index=start_index)
    state = policy.init_state(1, device)
    ctxs, pfs, acts, logps, vals, rews, nets, labs = [], [], [], [], [], [], [], []
    terminated = False
    last_value = 0.0

    for _ in range(max_horizons):
        ctx = torch.from_numpy(np.ascontiguousarray(obs["context"])).unsqueeze(0).to(device)
        pf = torch.from_numpy(np.ascontiguousarray(obs["portfolio"])).unsqueeze(0).to(device)
        bar = henv.env._t
        ctxs.append(obs["context"].copy())
        pfs.append(obs["portfolio"].copy())
        labs.append(int(labels[bar]) if bar < labels.shape[0] else _NO_LABEL)

        if isinstance(policy, RecurrentActorCritic):
            action, log_prob, new_state = policy.act(
                ctx, pf, state, greedy=greedy, generator=generator
            )
            with torch.no_grad():
                value = policy.value(new_state).squeeze(-1)
            vals.append(float(value.item()))
            logps.append(float(log_prob.item()))
        else:
            action, _, new_state = policy.act(
                ctx, pf, state, greedy=greedy, epsilon=epsilon, generator=generator
            )
            vals.append(0.0)
            logps.append(0.0)

        a = int(action.item())
        obs, reward, term, trunc, outcome = henv.step(a)
        acts.append(a)
        rews.append(reward)
        nets.append(outcome.net_log_return)
        state = new_state
        if term or trunc:
            terminated = term
            if trunc and not term and isinstance(policy, RecurrentActorCritic):
                # Truncation is not the end of the world: bootstrap from the value
                # of the state we stopped in, or the episode return is biased low.
                ctx = torch.from_numpy(np.ascontiguousarray(obs["context"])).unsqueeze(0).to(device)
                pf = (
                    torch.from_numpy(np.ascontiguousarray(obs["portfolio"])).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    feats, _ = policy.trunk.step(ctx, pf, state)
                    last_value = float(policy.value(feats).squeeze(-1).item())
            break

    return Rollout(
        context=np.stack(ctxs).astype(np.float32),
        portfolio=np.stack(pfs).astype(np.float32),
        actions=np.asarray(acts, dtype=np.int64),
        log_probs=np.asarray(logps, dtype=np.float32),
        values=np.asarray(vals, dtype=np.float32),
        rewards=np.asarray(rews, dtype=np.float32),
        net_log_returns=np.asarray(nets, dtype=np.float64),
        labels=np.asarray(labs, dtype=np.int64),
        terminated=terminated,
        last_value=last_value,
    )


def _gae(rollout: Rollout, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation for one episode."""
    t = len(rollout)
    adv = np.zeros(t, dtype=np.float32)
    next_value = 0.0 if rollout.terminated else rollout.last_value
    gae = 0.0
    for i in range(t - 1, -1, -1):
        delta = rollout.rewards[i] + gamma * next_value - rollout.values[i]
        gae = delta + gamma * lam * gae
        adv[i] = gae
        next_value = rollout.values[i]
    return adv, adv + rollout.values


def _pad_batch(
    rollouts: list[Rollout],
    advantages: list[np.ndarray],
    returns: list[np.ndarray],
    device: torch.device,
) -> dict[str, Tensor]:
    """Right-pad variable-length episodes and return a mask."""
    t_max = max(len(r) for r in rollouts)
    b = len(rollouts)
    c, f = rollouts[0].context.shape[1:]
    p = rollouts[0].portfolio.shape[1]
    ctx = np.zeros((b, t_max, c, f), dtype=np.float32)
    pf = np.zeros((b, t_max, p), dtype=np.float32)
    act = np.zeros((b, t_max), dtype=np.int64)
    logp = np.zeros((b, t_max), dtype=np.float32)
    adv = np.zeros((b, t_max), dtype=np.float32)
    ret = np.zeros((b, t_max), dtype=np.float32)
    lab = np.full((b, t_max), _NO_LABEL, dtype=np.int64)
    mask = np.zeros((b, t_max), dtype=np.float32)
    for i, r in enumerate(rollouts):
        n = len(r)
        ctx[i, :n] = r.context
        pf[i, :n] = r.portfolio
        act[i, :n] = r.actions
        logp[i, :n] = r.log_probs
        adv[i, :n] = advantages[i]
        ret[i, :n] = returns[i]
        lab[i, :n] = r.labels
        mask[i, :n] = 1.0
    to = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return {
        "context": to(ctx),
        "portfolio": to(pf),
        "actions": to(act),
        "log_probs": to(logp),
        "advantages": to(adv),
        "returns": to(ret),
        "labels": to(lab),
        "mask": to(mask),
    }


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def train_phase2(
    cfg: ExperimentConfig,
    henv: HorizonEnv,
    *,
    labels: np.ndarray,
    device: torch.device,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> Phase2Result:
    """Train the archetype selector on ``henv`` (a *training*-segment episode)."""
    p2 = cfg.phase2
    if p2.algo == "recurrent_ppo":
        return _train_ppo(
            cfg, henv, labels=labels, device=device, seed=seed, checkpoint_path=checkpoint_path
        )
    if p2.algo == "recurrent_dqn":
        return _train_dqn(
            cfg, henv, labels=labels, device=device, seed=seed, checkpoint_path=checkpoint_path
        )
    raise ValueError(f"unknown phase-2 algorithm {p2.algo!r}")


def _make_policy(cfg: ExperimentConfig, henv: HorizonEnv, device: torch.device):
    return build_selector(
        cfg.phase2.algo,
        henv.n_features,
        henv.n_portfolio,
        henv.n_archetypes,
        hidden_dim=cfg.phase2.hidden_dim,
    ).to(device)


def _episode_starts(
    henv: HorizonEnv, n: int, rng: np.random.Generator, horizons: int
) -> np.ndarray:
    """Random episode starts leaving room for ``horizons`` full horizons."""
    hi = henv.latest_start() - (horizons - 1) * henv.horizon
    hi = max(hi, henv.min_start)
    return rng.integers(henv.min_start, hi + 1, size=n, dtype=np.int64)


def _rollout_diagnostics(
    rollouts: list[Rollout], n_archetypes: int, cash: int
) -> tuple[list[float], float, float]:
    chosen = np.concatenate([r.actions for r in rollouts])
    counts = np.bincount(chosen, minlength=n_archetypes).astype(np.float64)
    usage = (counts / max(chosen.size, 1)).tolist()
    switches = float(
        np.mean([float((np.diff(r.actions) != 0).mean()) for r in rollouts if len(r) > 1] or [0.0])
    )
    cash_frac = float(counts[cash] / max(chosen.size, 1)) if cash >= 0 else 0.0
    return usage, switches, cash_frac


def _train_ppo(
    cfg: ExperimentConfig,
    henv: HorizonEnv,
    *,
    labels: np.ndarray,
    device: torch.device,
    seed: int,
    checkpoint_path: str | Path | None,
) -> Phase2Result:
    p2 = cfg.phase2
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    policy = _make_policy(cfg, henv, device)
    if not isinstance(policy, RecurrentActorCritic):
        raise TypeError("PPO training requires a recurrent actor-critic policy")
    # The decoder is frozen unless explicitly fine-tuned: Phase 2 is a selection
    # problem, and letting gradients into the decoder would quietly re-define
    # what the archetypes mean mid-training.
    for param in henv.model.parameters():
        param.requires_grad_(p2.finetune_decoder)
    opt = torch.optim.AdamW(policy.parameters(), lr=p2.lr)

    # Episode length in macro steps: enough horizons to matter, bounded by the
    # segment so the sampler is not forced into a corner.
    max_horizons = max(1, min(16, (henv.latest_start() - henv.min_start) // max(henv.horizon, 1)))
    eps_per_update = max(1, p2.horizons_per_update // max_horizons)

    history: list[UpdateStats] = []
    best_obj = -float("inf")
    best_state: dict[str, Tensor] | None = None
    best_update = -1

    for update in range(p2.updates):
        rollouts: list[Rollout] = []
        starts = _episode_starts(henv, eps_per_update, rng, max_horizons)
        for start in starts:
            rollouts.append(
                _collect_episode(
                    henv,
                    policy,
                    start_index=int(start),
                    labels=labels,
                    device=device,
                    max_horizons=max_horizons,
                    generator=gen,
                )
            )
        rollouts = [r for r in rollouts if len(r) > 0]
        if not rollouts:
            raise RuntimeError("no macro steps were collected; the segment is too short")

        advs, rets = zip(*(_gae(r, p2.gamma, p2.gae_lambda) for r in rollouts), strict=True)
        advs, rets = list(advs), list(rets)
        flat = np.concatenate(advs)
        mean, std = float(flat.mean()), float(flat.std() + 1e-8)
        advs = [(a - mean) / std for a in advs]

        pg_l = v_l = ent_l = aux_l = kl = gnorm = 0.0
        n_batches = 0
        for _ in range(p2.epochs_per_update):
            order = rng.permutation(len(rollouts))
            for i in range(0, order.shape[0], p2.minibatch_size):
                sel = order[i : i + p2.minibatch_size]
                batch = _pad_batch(
                    [rollouts[j] for j in sel],
                    [advs[j] for j in sel],
                    [rets[j] for j in sel],
                    device,
                )
                state = policy.init_state(sel.shape[0], device)
                logp, entropy, values, logits = policy.evaluate_actions(
                    batch["context"], batch["portfolio"], state, batch["actions"]
                )
                mask = batch["mask"]
                ratio = torch.exp(logp - batch["log_probs"])
                unclipped = ratio * batch["advantages"]
                clipped = (
                    torch.clamp(ratio, 1 - p2.clip_range, 1 + p2.clip_range) * batch["advantages"]
                )
                policy_loss = -_masked_mean(torch.min(unclipped, clipped), mask)
                value_loss = _masked_mean((values - batch["returns"]).pow(2), mask)
                ent = _masked_mean(entropy, mask)

                aux = torch.zeros((), device=device)
                has_label = (batch["labels"] >= 0) & (mask > 0)
                if p2.aux_label_coef > 0 and bool(has_label.any()):
                    # Labels index the codebook; the cash archetype has no label,
                    # so only real codes contribute.
                    aux = F.cross_entropy(
                        logits[has_label],
                        batch["labels"][has_label].clamp(max=logits.shape[-1] - 1),
                    )

                loss = (
                    policy_loss
                    + p2.value_coef * value_loss
                    - p2.entropy_coef * ent
                    + p2.aux_label_coef * aux
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = float(nn.utils.clip_grad_norm_(policy.parameters(), p2.grad_clip))
                opt.step()

                pg_l += float(policy_loss.detach())
                v_l += float(value_loss.detach())
                ent_l += float(ent.detach())
                aux_l += float(aux.detach())
                kl += float(_masked_mean(batch["log_probs"] - logp, mask).detach())
                n_batches += 1

        usage, switch_rate, cash_frac = _rollout_diagnostics(
            rollouts, henv.n_archetypes, henv.cash_archetype
        )
        mean_reward = float(np.mean([r.rewards.sum() for r in rollouts]))
        mean_net = float(np.mean([r.net_log_returns.sum() for r in rollouts]))
        stats = UpdateStats(
            update=update,
            n_episodes=len(rollouts),
            n_horizons=int(sum(len(r) for r in rollouts)),
            mean_reward=mean_reward,
            mean_net_log_return=mean_net,
            policy_loss=pg_l / max(n_batches, 1),
            value_loss=v_l / max(n_batches, 1),
            entropy=ent_l / max(n_batches, 1),
            aux_loss=aux_l / max(n_batches, 1),
            approx_kl=kl / max(n_batches, 1),
            grad_norm=gnorm,
            switch_rate=switch_rate,
            cash_fraction=cash_frac,
            epsilon=0.0,
            archetype_usage=usage,
        )
        history.append(stats)
        log_event(logger, "phase2.update", **vars(stats))

        # Selection objective is the *unshaped* net log return: penalties are a
        # training device, not the thing we want to be best at.
        if mean_net > best_obj:
            best_obj = mean_net
            best_update = update
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)

    result = Phase2Result(
        policy=policy,
        history=history,
        best_update=best_update,
        best_objective=best_obj,
        meta={
            "algo": "recurrent_ppo",
            "n_archetypes": henv.n_archetypes,
            "cash_archetype": henv.cash_archetype,
            "horizon": henv.horizon,
            "context_bars": henv.context_bars,
            "max_horizons_per_episode": max_horizons,
            "episodes_per_update": eps_per_update,
            "seed": seed,
            "device": str(device),
            "finetune_decoder": p2.finetune_decoder,
        },
    )
    if checkpoint_path is not None:
        save_selector(result, henv, checkpoint_path, cfg=cfg)
    return result


def _train_dqn(
    cfg: ExperimentConfig,
    henv: HorizonEnv,
    *,
    labels: np.ndarray,
    device: torch.device,
    seed: int,
    checkpoint_path: str | Path | None,
) -> Phase2Result:
    """Dueling recurrent DQN over sequence replay.

    Sequences, not transitions: the selector's state is recurrent, so a replayed
    transition without its prefix would be evaluated under a hidden state that
    never existed.
    """
    p2 = cfg.phase2
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    policy = _make_policy(cfg, henv, device)
    if not isinstance(policy, RecurrentQNetwork):
        raise TypeError("DQN training requires a recurrent Q-network policy")
    target = _make_policy(cfg, henv, device)
    if not isinstance(target, RecurrentQNetwork):
        raise TypeError("DQN target must be a recurrent Q-network policy")
    target.load_state_dict(policy.state_dict())
    for param in henv.model.parameters():
        param.requires_grad_(p2.finetune_decoder)
    opt = torch.optim.AdamW(policy.parameters(), lr=p2.lr)

    max_horizons = max(1, min(16, (henv.latest_start() - henv.min_start) // max(henv.horizon, 1)))
    eps_per_update = max(1, p2.horizons_per_update // max_horizons)
    buffer: list[Rollout] = []
    buffer_cap = max(4 * eps_per_update, 32)

    history: list[UpdateStats] = []
    best_obj = -float("inf")
    best_state: dict[str, Tensor] | None = None
    best_update = -1

    for update in range(p2.updates):
        frac = update / max(p2.updates - 1, 1)
        epsilon = p2.eps_start + frac * (p2.eps_end - p2.eps_start)
        fresh: list[Rollout] = []
        for start in _episode_starts(henv, eps_per_update, rng, max_horizons):
            fresh.append(
                _collect_episode(
                    henv,
                    policy,
                    start_index=int(start),
                    labels=labels,
                    device=device,
                    max_horizons=max_horizons,
                    epsilon=epsilon,
                    generator=gen,
                )
            )
        buffer.extend(r for r in fresh if len(r) > 0)
        buffer = buffer[-buffer_cap:]
        if not fresh:
            raise RuntimeError("no macro steps were collected; the segment is too short")

        td_l = aux_l = gnorm = 0.0
        n_batches = 0
        for _ in range(p2.epochs_per_update):
            sel = rng.choice(len(buffer), size=min(p2.minibatch_size, len(buffer)), replace=False)
            chunk = [buffer[j] for j in sel]
            zeros = [np.zeros(len(r), dtype=np.float32) for r in chunk]
            batch = _pad_batch(chunk, zeros, zeros, device)
            mask = batch["mask"]
            state = policy.init_state(len(chunk), device)
            q, _ = policy(batch["context"], batch["portfolio"], state)
            q_taken = q.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)

            rewards = torch.zeros_like(mask)
            done = torch.zeros_like(mask)
            for i, r in enumerate(chunk):
                rewards[i, : len(r)] = torch.from_numpy(r.rewards).to(device)
                if r.terminated:
                    done[i, len(r) - 1] = 1.0
            with torch.no_grad():
                tstate = target.init_state(len(chunk), device)
                q_next_t, _ = target(batch["context"], batch["portfolio"], tstate)
                q_next_p, _ = policy(batch["context"], batch["portfolio"], state)
                # Double DQN: the online net picks, the target net evaluates.
                best_a = q_next_p.argmax(dim=-1, keepdim=True)
                bootstrap = q_next_t.gather(-1, best_a).squeeze(-1)
                shifted = torch.zeros_like(bootstrap)
                shifted[:, :-1] = bootstrap[:, 1:]
                td_target = rewards + p2.gamma * shifted * (1.0 - done)
            td = _masked_mean(F.smooth_l1_loss(q_taken, td_target, reduction="none"), mask)

            aux = torch.zeros((), device=device)
            has_label = (batch["labels"] >= 0) & (mask > 0)
            if p2.aux_label_coef > 0 and bool(has_label.any()):
                aux = F.cross_entropy(
                    q[has_label], batch["labels"][has_label].clamp(max=q.shape[-1] - 1)
                )
            loss = td + p2.aux_label_coef * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(nn.utils.clip_grad_norm_(policy.parameters(), p2.grad_clip))
            opt.step()
            td_l += float(td.detach())
            aux_l += float(aux.detach())
            n_batches += 1

        if (update + 1) % p2.target_sync_every == 0:
            target.load_state_dict(policy.state_dict())

        usage, switch_rate, cash_frac = _rollout_diagnostics(
            fresh, henv.n_archetypes, henv.cash_archetype
        )
        mean_net = float(np.mean([r.net_log_returns.sum() for r in fresh]))
        stats = UpdateStats(
            update=update,
            n_episodes=len(fresh),
            n_horizons=int(sum(len(r) for r in fresh)),
            mean_reward=float(np.mean([r.rewards.sum() for r in fresh])),
            mean_net_log_return=mean_net,
            policy_loss=td_l / max(n_batches, 1),
            value_loss=td_l / max(n_batches, 1),
            entropy=0.0,
            aux_loss=aux_l / max(n_batches, 1),
            approx_kl=0.0,
            grad_norm=gnorm,
            switch_rate=switch_rate,
            cash_fraction=cash_frac,
            epsilon=epsilon,
            archetype_usage=usage,
        )
        history.append(stats)
        log_event(logger, "phase2.update", **vars(stats))
        if mean_net > best_obj:
            best_obj, best_update = mean_net, update
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
    result = Phase2Result(
        policy=policy,
        history=history,
        best_update=best_update,
        best_objective=best_obj,
        meta={
            "algo": "recurrent_dqn",
            "n_archetypes": henv.n_archetypes,
            "cash_archetype": henv.cash_archetype,
            "horizon": henv.horizon,
            "context_bars": henv.context_bars,
            "seed": seed,
            "device": str(device),
        },
    )
    if checkpoint_path is not None:
        save_selector(result, henv, checkpoint_path, cfg=cfg)
    return result


# --------------------------------------------------------------------------- #
# Evaluation / persistence
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_selector(
    henv: HorizonEnv,
    policy: RecurrentActorCritic | RecurrentQNetwork,
    *,
    device: torch.device,
    start_index: int | None = None,
    greedy: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Roll the selector across a whole segment and return the episode statistics.

    This is the deployed path: sampling is off by default, no labels are read and
    no planner is reachable from here.
    """
    obs = henv.reset(start_index=start_index)
    state = policy.init_state(1, device)
    probs_log: list[list[float]] = []
    while True:
        ctx = torch.from_numpy(np.ascontiguousarray(obs["context"])).unsqueeze(0).to(device)
        pf = torch.from_numpy(np.ascontiguousarray(obs["portfolio"])).unsqueeze(0).to(device)
        probs_log.append(policy.action_probs(ctx, pf, state).squeeze(0).cpu().tolist())
        if isinstance(policy, RecurrentActorCritic):
            action, _, state = policy.act(ctx, pf, state, greedy=greedy, generator=generator)
        else:
            action, _, state = policy.act(ctx, pf, state, greedy=greedy, generator=generator)
        obs, _, term, trunc, _ = henv.step(int(action.item()))
        if term or trunc:
            break
    stats = henv.episode_stats()
    stats["archetype_probs"] = probs_log
    stats["mean_archetype_probs"] = np.asarray(probs_log, dtype=np.float64).mean(axis=0).tolist()
    return stats


def save_selector(
    result: Phase2Result,
    henv: HorizonEnv,
    path: str | Path,
    *,
    cfg: ExperimentConfig | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.policy.state_dict(),
            "algo": cfg.phase2.algo if cfg is not None else result.meta.get("algo"),
            "n_features": henv.n_features,
            "n_portfolio": henv.n_portfolio,
            "n_archetypes": henv.n_archetypes,
            "hidden_dim": result.policy.hidden_dim,
            "phase2": cfg.phase2.model_dump() if cfg is not None else None,
            "history": result.history_table(),
            "meta": result.meta,
        },
        p,
    )
    return p


def load_selector(
    path: str | Path, *, device: torch.device
) -> tuple[RecurrentActorCritic | RecurrentQNetwork, dict[str, Any]]:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    policy = build_selector(
        blob["algo"],
        blob["n_features"],
        blob["n_portfolio"],
        blob["n_archetypes"],
        hidden_dim=blob["hidden_dim"],
    ).to(device)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy, blob
