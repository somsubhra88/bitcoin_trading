"""Phase-3 training: learn *when* to deviate from the archetype.

Training runs offline on sampled horizons rather than inside
:class:`~archetype_trader.env.trading_env.TradingEnv`, for one reason: the
quantity Phase 3 is optimising is *regret against the best bounded adaptation*,
and that reference (:func:`~archetype_trader.phase3.hindsight.best_override`) is
defined in the planner's log-space accounting. Scoring the adapter in the same
accounting makes improvement and regret exactly comparable. The trained adapter
is then evaluated through the real environment — with fees, spread, slippage,
latency and the risk shield — via :class:`AdapterController`.

Objective per horizon (all terms configurable, several default to zero)::

    reward = (realized - base)                      # actual improvement
           - regret_coef   * (best_bounded - realized)
           - cost_coef     * (cost_paid - base_cost_paid)
           - turnover_penalty * turnover
           - drawdown_penalty * max_drawdown
           - downside_penalty * downside_deviation

optimised by REINFORCE with a learned per-step baseline, plus an optional
imitation cross-entropy toward the hindsight target and an entropy bonus.

The improvement term is decomposed per step (it is a sum of per-bar
contributions), so the baseline sees return-to-go rather than one scalar per
horizon. The regret and penalty terms are terminal.

Archetypes are sampled **uniformly** during training, not drawn from the Phase-2
selector. The adapter should work under whatever the selector picks, and
training on the selector's current distribution would make the two co-adapt to a
policy that is still moving.
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
from archetype_trader.data.types import MarketData
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.phase1.models import ArchetypeVQ
from archetype_trader.phase2.horizon_env import build_scaled_features
from archetype_trader.phase3.adapter import OverrideAdapter, budget_mask, build_step_features
from archetype_trader.phase3.hindsight import adaptation_target, best_override
from archetype_trader.planner.dp import default_cost_frac, plan_value, planning_returns
from archetype_trader.splits import Segment
from archetype_trader.utils.logging import get_logger, log_event

logger = get_logger(__name__)

_EPS = 1e-12


@dataclass
class HorizonDataset:
    """Sampled horizons with the archetype's plan and its hindsight ceiling.

    Built from a **training** segment only: ``best_actions`` is hindsight and
    ``returns`` spans the whole horizon, so nothing here may touch validation or
    test ranges. The caller is the single place that decides the segment.
    """

    starts: np.ndarray  # (N,)
    archetypes: np.ndarray  # (N,)
    base_actions: np.ndarray  # (N, h)
    best_actions: np.ndarray  # (N, h)
    target_classes: np.ndarray  # (N, h) imitation labels incl. the retain class
    returns: np.ndarray  # (N, h)
    base_value: np.ndarray  # (N,)
    best_value: np.ndarray  # (N,)
    features: np.ndarray  # (T, F) scaled market matrix
    grid: tuple[float, ...]
    horizon: int
    cost_frac: float
    budget: int
    n_archetypes: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    @property
    def headroom(self) -> np.ndarray:
        """Return the best bounded adaptation could have added, per horizon."""
        return self.best_value - self.base_value

    def obs_batch(self, rows: np.ndarray) -> np.ndarray:
        idx = self.starts[rows][:, None] + np.arange(self.horizon)[None, :]
        return self.features[idx]

    def split(self, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        perm = rng.permutation(len(self))
        cut = max(1, int(len(self) * (1.0 - val_fraction)))
        return perm[:cut], perm[cut:]


@dataclass
class UpdateStats:
    epoch: int
    loss: float
    policy_loss: float
    value_loss: float
    imitation_loss: float
    entropy: float
    mean_reward: float
    mean_improvement: float
    mean_regret: float
    override_rate: float
    mean_overrides: float
    grad_norm: float
    val_improvement: float
    val_regret: float
    val_override_rate: float


@dataclass
class Phase3Result:
    model: OverrideAdapter
    dataset_meta: dict[str, Any]
    history: list[UpdateStats]
    best_epoch: int
    best_val_improvement: float
    meta: dict[str, Any] = field(default_factory=dict)

    def history_table(self) -> list[dict[str, float]]:
        return [vars(h) for h in self.history]


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _decoder_plans(
    model: ArchetypeVQ,
    obs: np.ndarray,
    archetypes: np.ndarray,
    *,
    device: torch.device,
    flat_action: int,
    cash_archetype: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Roll the frozen decoder to get each horizon's base action sequence."""
    model.eval()
    n, h, _ = obs.shape
    out = np.empty((n, h), dtype=np.int64)
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        arche = archetypes[sl]
        block = torch.from_numpy(np.ascontiguousarray(obs[sl])).to(device)
        codes = torch.from_numpy(np.clip(arche, 0, model.codebook_size - 1)).to(device)
        pred = model.rollout(block, codes, greedy=True).cpu().numpy()
        # The cash archetype is not a code: it is a hard flat plan.
        pred[arche == cash_archetype] = flat_action
        out[sl] = pred
    return out


def build_horizon_dataset(
    cfg: ExperimentConfig,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    segment: Segment,
    model: ArchetypeVQ,
    *,
    rng: np.random.Generator,
    device: torch.device,
    n_episodes: int | None = None,
    include_cash_archetype: bool = True,
) -> HorizonDataset:
    """Sample horizons inside ``segment`` and solve each one's override ceiling."""
    horizon = cfg.phase1.horizon
    budget = cfg.phase3.override_budget
    grid = cfg.env.positions
    n = n_episodes if n_episodes is not None else cfg.phase3.n_episodes

    lookback = features.lookback
    lo = max(segment.start, features.valid_from, lookback - 1)
    hi = segment.stop - horizon - 1  # the planner's last return is open[s+h+1]
    if hi <= lo:
        raise ValueError(
            f"segment [{segment.start}, {segment.stop}) is too short for horizon {horizon}"
        )
    admissible = np.arange(lo, hi, dtype=np.int64)
    replace = n > admissible.shape[0]
    starts = np.sort(rng.choice(admissible, size=n, replace=replace))

    n_arche = model.codebook_size + (1 if include_cash_archetype else 0)
    cash_archetype = model.codebook_size if include_cash_archetype else -1
    archetypes = rng.integers(0, n_arche, size=n).astype(np.int64)

    scaled = build_scaled_features(features, scaler)
    obs = scaled[starts[:, None] + np.arange(horizon)[None, :]]
    flat_action = int(np.argmin(np.abs(np.asarray(grid) - 0.0)))
    base_actions = _decoder_plans(
        model,
        obs,
        archetypes,
        device=device,
        flat_action=flat_action,
        cash_archetype=cash_archetype,
    )

    returns = planning_returns(data.open, starts, horizon)
    cost_frac = (
        cfg.planner.cost_bps * 1e-4
        if cfg.planner.cost_bps is not None
        else default_cost_frac(
            cfg.env.costs.taker_fee_bps,
            float(np.median(data.half_spread_frac(cfg.env.costs.fallback_half_spread_bps)) * 1e4),
        )
    )
    override = best_override(
        base_actions, returns, grid=grid, cost_frac=cost_frac, budget=budget, init_position=0.0
    )
    base_value = plan_value(base_actions, returns, grid=grid, cost_frac=cost_frac)
    targets = adaptation_target(override.actions, base_actions, retain_index=len(grid))

    meta = {
        "n_episodes": int(n),
        "unique_starts": int(np.unique(starts).shape[0]),
        "horizon": horizon,
        "budget": budget,
        "cost_frac": cost_frac,
        "n_archetypes": n_arche,
        "mean_base_value": float(base_value.mean()),
        "mean_best_value": float(override.value.mean()),
        "mean_headroom": float(override.headroom.mean()),
        "median_headroom": float(np.median(override.headroom)),
        "frac_horizons_with_headroom": float((override.headroom > 1e-9).mean()),
        "mean_overrides_used": float(override.n_overrides.mean()),
    }
    log_event(logger, "phase3.dataset", **meta)
    return HorizonDataset(
        starts=starts,
        archetypes=archetypes,
        base_actions=base_actions,
        best_actions=override.actions,
        target_classes=targets,
        returns=returns,
        base_value=base_value,
        best_value=override.value,
        features=scaled,
        grid=grid,
        horizon=horizon,
        cost_frac=cost_frac,
        budget=budget,
        n_archetypes=n_arche,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# batched rollout
# --------------------------------------------------------------------------- #
@dataclass
class RolloutBatch:
    """One batched pass of the adapter over a set of horizons."""

    log_prob: Tensor  # (B, h)
    entropy: Tensor  # (B, h)
    value: Tensor  # (B, h)
    raw_logits: Tensor  # (B, h, A+1) pre-mask, for the imitation term
    actions: Tensor  # (B, h)
    improvement: Tensor  # (B, h) per-step realized minus base
    realized: Tensor  # (B,)
    base: Tensor  # (B,)
    extra_cost: Tensor  # (B,) cost paid above the base plan
    turnover: Tensor  # (B,)
    max_drawdown: Tensor  # (B,)
    downside: Tensor  # (B,)
    n_overrides: Tensor  # (B,)


def rollout_batch(
    model: OverrideAdapter,
    obs: Tensor,
    base_actions: Tensor,
    returns: Tensor,
    archetypes: Tensor,
    *,
    positions: Tensor,
    cost_frac: float,
    budget: int,
    n_archetypes: int,
    init_action: int,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> RolloutBatch:
    """Vectorised training-time twin of :class:`AdapterController`.

    Every masking and budget rule here matches the controller exactly; that
    equality is asserted by the Phase-3 parity test. The scalars fed to the
    network at step ``t`` use only prefix sums over ``returns[:, :t]``, so the
    *inputs* stay causal even though the array itself is a training-only object.
    """
    b, h, _ = obs.shape
    n_actions = positions.shape[0]
    retain = model.retain_index
    dev = obs.device

    arche_oh = F.one_hot(archetypes.clamp(max=n_archetypes - 1), n_archetypes).to(obs.dtype)
    state = model.init_state(b, dev)

    prev_action = torch.full((b,), init_action, dtype=torch.long, device=dev)
    base_prev = torch.full((b,), init_action, dtype=torch.long, device=dev)
    budget_left = torch.full((b,), budget, dtype=torch.long, device=dev)
    in_override = torch.zeros(b, dtype=torch.bool, device=dev)
    real_cum = torch.zeros(b, device=dev, dtype=obs.dtype)
    base_cum = torch.zeros(b, device=dev, dtype=obs.dtype)
    peak = torch.zeros(b, device=dev, dtype=obs.dtype)
    max_dd = torch.zeros(b, device=dev, dtype=obs.dtype)
    turnover = torch.zeros(b, device=dev, dtype=obs.dtype)
    cost_paid = torch.zeros(b, device=dev, dtype=obs.dtype)
    base_cost_paid = torch.zeros(b, device=dev, dtype=obs.dtype)
    n_ovr = torch.zeros(b, device=dev, dtype=obs.dtype)

    lps, ents, vals, raws, acts, imps, steps = [], [], [], [], [], [], []
    for t in range(h):
        base_t = base_actions[:, t]
        scalars = torch.stack(
            [
                real_cum,
                base_cum,
                real_cum - base_cum,
                torch.full((b,), (h - t) / h, device=dev, dtype=obs.dtype),
                budget_left.to(obs.dtype) / max(budget, 1),
            ],
            dim=-1,
        )
        feats = build_step_features(
            obs[:, t],
            arche_oh,
            F.one_hot(base_t, n_actions).to(obs.dtype),
            F.one_hot(prev_action, n_actions).to(obs.dtype),
            scalars,
        )
        raw, value, state = model(feats, state)
        logits = budget_mask(raw, base_t, budget_left, in_override, retain_index=retain)
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        if greedy:
            cls = logits.argmax(dim=-1)
        else:
            cls = torch.multinomial(probs.detach(), 1, generator=generator).squeeze(-1)
        # Masked classes have log-prob -inf; zeroing them keeps the entropy finite
        # instead of producing 0 * -inf = nan.
        ent = -(probs * torch.nan_to_num(logp, neginf=0.0)).sum(dim=-1)
        lp = logp.gather(1, cls.unsqueeze(1)).squeeze(1)

        action = torch.where(cls == retain, base_t, cls.clamp(max=n_actions - 1))
        pos_now = positions[action]
        pos_prev = positions[prev_action]
        d = (pos_now - pos_prev).abs()
        cost = torch.log1p(-cost_frac * d)
        step_v = pos_now * returns[:, t] + cost

        base_pos = positions[base_t]
        base_d = (base_pos - positions[base_prev]).abs()
        base_cost = torch.log1p(-cost_frac * base_d)
        base_step = base_pos * returns[:, t] + base_cost

        ovr = action != base_t
        started = ovr & ~in_override
        n_ovr = n_ovr + started.to(obs.dtype)
        budget_left = budget_left - started.long()
        in_override = ovr
        prev_action = action
        base_prev = base_t

        real_cum = real_cum + step_v
        base_cum = base_cum + base_step
        peak = torch.maximum(peak, real_cum)
        max_dd = torch.maximum(max_dd, peak - real_cum)
        turnover = turnover + d
        cost_paid = cost_paid - cost
        base_cost_paid = base_cost_paid - base_cost

        lps.append(lp)
        ents.append(ent)
        vals.append(value)
        raws.append(raw)
        acts.append(action)
        imps.append(step_v - base_step)
        steps.append(step_v)

    step_stack = torch.stack(steps, dim=1)
    downside = torch.sqrt(F.relu(-step_stack).pow(2).mean(dim=1) + _EPS)
    return RolloutBatch(
        log_prob=torch.stack(lps, dim=1),
        entropy=torch.stack(ents, dim=1),
        value=torch.stack(vals, dim=1),
        raw_logits=torch.stack(raws, dim=1),
        actions=torch.stack(acts, dim=1),
        improvement=torch.stack(imps, dim=1),
        realized=real_cum,
        base=base_cum,
        extra_cost=cost_paid - base_cost_paid,
        turnover=turnover,
        max_drawdown=max_dd,
        downside=downside,
        n_overrides=n_ovr,
    )


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _terminal_penalty(
    cfg: ExperimentConfig, roll: RolloutBatch, best_value: Tensor
) -> tuple[Tensor, Tensor]:
    """``(penalty, regret)``, both ``(B,)`` and both non-negative."""
    p3 = cfg.phase3
    regret = F.relu(best_value - roll.realized)
    penalty = (
        p3.regret_coef * regret
        + p3.cost_coef * F.relu(roll.extra_cost)
        + p3.turnover_penalty * roll.turnover
        + p3.drawdown_penalty * roll.max_drawdown
        + p3.downside_penalty * roll.downside
    )
    return penalty, regret


def _tensors(
    ds: HorizonDataset, rows: np.ndarray, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    obs = torch.from_numpy(np.ascontiguousarray(ds.obs_batch(rows))).to(device).float()
    if not torch.isfinite(obs).all():
        raise AssertionError("non-finite observation reached the Phase-3 batch")
    base = torch.from_numpy(np.ascontiguousarray(ds.base_actions[rows])).to(device)
    rets = torch.from_numpy(np.ascontiguousarray(ds.returns[rows])).to(device).float()
    arche = torch.from_numpy(np.ascontiguousarray(ds.archetypes[rows])).to(device)
    best = torch.from_numpy(np.ascontiguousarray(ds.best_value[rows])).to(device).float()
    tgt = torch.from_numpy(np.ascontiguousarray(ds.target_classes[rows])).to(device)
    return obs, base, rets, arche, best, tgt


def train_phase3(
    cfg: ExperimentConfig,
    ds: HorizonDataset,
    *,
    device: torch.device,
    seed: int,
    val_fraction: float = 0.15,
    checkpoint_path: str | Path | None = None,
) -> Phase3Result:
    """Fit the override adapter on ``ds``. Deterministic given ``seed``."""
    p3 = cfg.phase3
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)  # weight init and dropout use the global RNG

    train_rows, val_rows = ds.split(val_fraction, rng)
    if val_rows.size == 0:
        val_rows = train_rows[: max(1, train_rows.size // 10)]

    n_actions = len(ds.grid)
    model = OverrideAdapter(
        int(ds.features.shape[1]), n_actions, ds.n_archetypes, hidden_dim=p3.hidden_dim
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p3.lr)
    positions = torch.tensor(ds.grid, dtype=torch.float32, device=device)
    init_action = int(np.argmin(np.abs(np.asarray(ds.grid) - 0.0)))

    history: list[UpdateStats] = []
    best_val = -float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1

    for epoch in range(p3.epochs):
        model.train()
        perm = rng.permutation(train_rows)
        agg: dict[str, list[float]] = {
            k: []
            for k in (
                "loss",
                "policy",
                "value",
                "imitation",
                "entropy",
                "reward",
                "improvement",
                "regret",
                "override_rate",
                "overrides",
                "grad",
            )
        }
        for i in range(0, perm.shape[0], p3.batch_size):
            rows = perm[i : i + p3.batch_size]
            obs, base, rets, arche, best_v, tgt = _tensors(ds, rows, device)
            roll = rollout_batch(
                model,
                obs,
                base,
                rets,
                arche,
                positions=positions,
                cost_frac=ds.cost_frac,
                budget=ds.budget,
                n_archetypes=ds.n_archetypes,
                init_action=init_action,
                greedy=False,
                generator=gen,
            )
            penalty, regret = _terminal_penalty(cfg, roll, best_v)

            # Per-step reward: improvement is additive over bars, so it can carry a
            # baseline; the terminal terms land on the last step.
            reward = roll.improvement.clone()
            reward[:, -1] = reward[:, -1] - penalty
            # Reverse cumulative sum = return-to-go (undiscounted: a horizon is
            # short and every bar's PnL counts equally).
            rtg = reward.detach().flip(1).cumsum(1).flip(1)
            adv = rtg - roll.value.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

            policy_loss = -(roll.log_prob * adv).mean()
            value_loss = F.mse_loss(roll.value, rtg)
            imitation = F.cross_entropy(roll.raw_logits.reshape(-1, n_actions + 1), tgt.reshape(-1))
            entropy = roll.entropy.mean()
            loss = (
                policy_loss
                + 0.5 * value_loss
                + p3.imitation_coef * imitation
                - p3.entropy_coef * entropy
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), p3.grad_clip))
            opt.step()

            agg["loss"].append(float(loss.detach()))
            agg["policy"].append(float(policy_loss.detach()))
            agg["value"].append(float(value_loss.detach()))
            agg["imitation"].append(float(imitation.detach()))
            agg["entropy"].append(float(entropy.detach()))
            agg["reward"].append(float(reward.sum(dim=1).mean().detach()))
            agg["improvement"].append(float((roll.realized - roll.base).mean().detach()))
            agg["regret"].append(float(regret.mean().detach()))
            agg["override_rate"].append(float((roll.n_overrides > 0).float().mean()))
            agg["overrides"].append(float(roll.n_overrides.mean()))
            agg["grad"].append(grad)

        val = evaluate_adapter(cfg, model, ds, val_rows, device=device)
        stats = UpdateStats(
            epoch=epoch,
            loss=float(np.mean(agg["loss"])),
            policy_loss=float(np.mean(agg["policy"])),
            value_loss=float(np.mean(agg["value"])),
            imitation_loss=float(np.mean(agg["imitation"])),
            entropy=float(np.mean(agg["entropy"])),
            mean_reward=float(np.mean(agg["reward"])),
            mean_improvement=float(np.mean(agg["improvement"])),
            mean_regret=float(np.mean(agg["regret"])),
            override_rate=float(np.mean(agg["override_rate"])),
            mean_overrides=float(np.mean(agg["overrides"])),
            grad_norm=float(np.mean(agg["grad"])),
            val_improvement=val["improvement"],
            val_regret=val["regret"],
            val_override_rate=val["override_rate"],
        )
        history.append(stats)
        log_event(logger, "phase3.epoch", **vars(stats))

        # Selection objective is *greedy* improvement over the unmodified
        # archetype: the deployed adapter is greedy, and improvement is the thing
        # Phase 3 exists to produce.
        if val["improvement"] > best_val:
            best_val = val["improvement"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final = evaluate_adapter(cfg, model, ds, val_rows, device=device)
    meta = {
        "n_train_rows": int(train_rows.size),
        "n_val_rows": int(val_rows.size),
        "best_epoch": best_epoch,
        "best_val_improvement": best_val,
        "val_improvement": final["improvement"],
        "val_regret": final["regret"],
        "val_headroom": final["headroom"],
        "val_regret_ratio": final["regret_ratio"],
        "val_override_rate": final["override_rate"],
        "val_max_overrides": final["max_overrides"],
        "budget": ds.budget,
        "seed": seed,
        "device": str(device),
    }
    if final["max_overrides"] > ds.budget:
        raise AssertionError(
            f"adapter used {final['max_overrides']} overrides with budget {ds.budget}"
        )
    log_event(logger, "phase3.done", **meta)
    result = Phase3Result(
        model=model,
        dataset_meta=ds.meta,
        history=history,
        best_epoch=best_epoch,
        best_val_improvement=best_val,
        meta=meta,
    )
    if checkpoint_path is not None:
        save_adapter(result, ds, checkpoint_path, cfg=cfg)
    return result


@torch.no_grad()
def evaluate_adapter(
    cfg: ExperimentConfig,
    model: OverrideAdapter,
    ds: HorizonDataset,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    """Greedy (deployed-mode) improvement, regret and override usage."""
    model.eval()
    positions = torch.tensor(ds.grid, dtype=torch.float32, device=device)
    init_action = int(np.argmin(np.abs(np.asarray(ds.grid) - 0.0)))
    retain = len(ds.grid)
    imp, reg, ovr, head, n_max = [], [], [], [], 0
    hits = steps = tp = fp = fn = 0
    for i in range(0, rows.shape[0], batch_size):
        batch = rows[i : i + batch_size]
        obs, base, rets, arche, best_v, tgt = _tensors(ds, batch, device)
        roll = rollout_batch(
            model,
            obs,
            base,
            rets,
            arche,
            positions=positions,
            cost_frac=ds.cost_frac,
            budget=ds.budget,
            n_archetypes=ds.n_archetypes,
            init_action=init_action,
            greedy=True,
        )
        imp.append((roll.realized - roll.base).cpu().numpy())
        reg.append(F.relu(best_v - roll.realized).cpu().numpy())
        ovr.append(roll.n_overrides.cpu().numpy())
        head.append((best_v - roll.base).cpu().numpy())
        n_max = max(n_max, int(roll.n_overrides.max()))
        # Behavioural class: overriding to an action that happens to equal the base
        # action is retention, whatever logit produced it.
        pred = torch.where(roll.actions == base, retain, roll.actions)
        hits += int((pred == tgt).sum())
        steps += int(tgt.numel())
        want, got = tgt != retain, pred != retain
        tp += int((want & got & (pred == tgt)).sum())
        fp += int((got & ~want).sum()) + int((got & want & (pred != tgt)).sum())
        fn += int((want & ~got).sum())
    improvement = float(np.concatenate(imp).mean())
    regret = float(np.concatenate(reg).mean())
    headroom = float(np.concatenate(head).mean())
    return {
        "improvement": improvement,
        "regret": regret,
        "headroom": headroom,
        # Agreement with the hindsight target, reported because it separates "the
        # adapter cannot learn" from "the target is unpredictable".
        "target_accuracy": hits / max(steps, 1),
        "override_precision": tp / max(tp + fp, 1),
        "override_recall": tp / max(tp + fn, 1),
        # Fraction of the available headroom left on the table. 1.0 means the
        # adapter captured nothing; 0.0 means it matched hindsight.
        "regret_ratio": regret / max(headroom, _EPS),
        "override_rate": float((np.concatenate(ovr) > 0).mean()),
        "mean_overrides": float(np.concatenate(ovr).mean()),
        "max_overrides": float(n_max),
    }


def save_adapter(
    result: Phase3Result,
    ds: HorizonDataset,
    path: str | Path,
    *,
    cfg: ExperimentConfig | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "n_features": int(ds.features.shape[1]),
            "n_actions": len(ds.grid),
            "n_archetypes": ds.n_archetypes,
            "grid": list(ds.grid),
            "horizon": ds.horizon,
            "budget": ds.budget,
            "cost_frac": ds.cost_frac,
            "phase3": cfg.phase3.model_dump() if cfg is not None else None,
            "history": result.history_table(),
            "dataset_meta": result.dataset_meta,
            "meta": result.meta,
        },
        p,
    )
    return p
