"""Phase-1 training loop: fit the archetype codebook on hindsight demonstrations.

The objective is *action reconstruction*, not return maximisation::

    loss = CE(decoder logits, planner actions)
         + codebook_loss_weight * ||sg[z_e] - z_q||^2
         + commitment_beta      * ||z_e - sg[z_q]||^2

Phase 1 is therefore behaviour cloning of a training-only oracle, compressed
through a ``K``-way bottleneck. What makes it useful downstream is the bottleneck,
not the accuracy: Phase 2 selects a *code*, so the codes must be few, distinct and
alive. That is why the loop tracks codebook health every epoch and restarts dead
codes, and why early stopping watches validation loss rather than accuracy (a
collapsed codebook can raise accuracy while destroying the representation).

Model selection here uses a *random row split inside the training segment*. It is
an optimisation guard only; the real model-selection folds are separate, later
time ranges handled by the walk-forward plan.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from archetype_trader.config import ExperimentConfig
from archetype_trader.phase1.demonstrations import DemonstrationSet
from archetype_trader.phase1.diagnostics import (
    BehaviouralDistinctness,
    CodebookHealth,
    behavioural_distinctness,
    codebook_health,
)
from archetype_trader.phase1.models import ArchetypeVQ
from archetype_trader.utils.logging import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_recon: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    perplexity: float
    utilization: float
    dead_codes: int
    restarted: int
    grad_norm: float
    lr: float


@dataclass
class Phase1Result:
    """Everything Phase 2 and the report need from a Phase-1 run."""

    model: ArchetypeVQ
    history: list[EpochStats]
    best_epoch: int
    best_val_loss: float
    #: Code assignment for every demonstration row, in row order.
    code_indices: np.ndarray
    health: CodebookHealth
    distinctness: BehaviouralDistinctness
    train_rows: np.ndarray
    val_rows: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def history_table(self) -> list[dict[str, float]]:
        return [vars(h) for h in self.history]


def _batches(rows: np.ndarray, batch_size: int, rng: np.random.Generator, *, shuffle: bool):
    idx = rng.permutation(rows) if shuffle else rows
    for i in range(0, idx.shape[0], batch_size):
        yield idx[i : i + batch_size]


def _to_tensors(
    demos: DemonstrationSet, rows: np.ndarray, device: torch.device
) -> tuple[Tensor, Tensor]:
    obs = torch.from_numpy(np.ascontiguousarray(demos.obs_batch(rows))).to(device)
    actions = torch.from_numpy(np.ascontiguousarray(demos.actions[rows])).to(device)
    if not torch.isfinite(obs).all():
        # Warm-up rows are zeroed in build_demonstrations, so a NaN here means a
        # containment bug, not a legitimately unavailable feature.
        raise AssertionError("non-finite observation reached the Phase-1 batch")
    return obs, actions


def _scheduled_forward(
    model: ArchetypeVQ,
    obs: Tensor,
    actions: Tensor,
    *,
    teacher_forcing: float,
    gen: torch.Generator,
) -> tuple[Tensor, Any]:
    """Step-wise forward that mixes ground-truth and sampled previous actions.

    Used only when ``teacher_forcing < 1``. It closes the train/deploy gap: at
    inference the decoder conditions on its *own* previous action, so training
    purely on ground-truth prefixes can leave it brittle. Costs an h-step Python
    loop, hence opt-in.
    """
    z_e = model.encoder(obs, actions)
    vq = model.quantizer(z_e)
    code = model.decoder.code_proj(vq.z_q)
    b, h, _ = obs.shape
    state = model.decoder.init_state(b, obs.device, obs.dtype)
    prev = model.decoder.start_token.unsqueeze(0).expand(b, model.n_actions)
    logits_out = []
    for t in range(h):
        x = torch.cat([obs[:, t], prev, code], dim=-1).unsqueeze(1)
        out, state = model.decoder.rnn(x, state)
        logits = model.decoder.head(out[:, 0])
        logits_out.append(logits)
        truth = F.one_hot(actions[:, t], model.n_actions).to(obs.dtype)
        sampled = F.one_hot(logits.argmax(dim=-1), model.n_actions).to(obs.dtype)
        use_truth = (torch.rand(b, 1, device=obs.device, generator=gen) < teacher_forcing).to(
            obs.dtype
        )
        prev = (use_truth * truth + (1.0 - use_truth) * sampled).detach()
    return torch.stack(logits_out, dim=1), vq


def train_phase1(
    cfg: ExperimentConfig,
    demos: DemonstrationSet,
    *,
    device: torch.device,
    seed: int,
    val_fraction: float = 0.1,
    checkpoint_path: str | Path | None = None,
) -> Phase1Result:
    """Fit :class:`ArchetypeVQ` on ``demos``.

    Deterministic given ``seed``: row shuffling, dropout, dead-code restarts and
    scheduled sampling all draw from seeded generators.
    """
    p1 = cfg.phase1
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device).manual_seed(seed)
    # Weight init and dropout draw from the *global* torch RNG, so seeding it here
    # is what makes two calls with the same seed bit-identical. Without this the
    # determinism test passes on the data pipeline and fails on the model.
    torch.manual_seed(seed)

    train_rows, val_rows = demos.split(val_fraction, rng)
    if val_rows.size == 0:  # tiny smoke datasets
        val_rows = train_rows[: max(1, train_rows.size // 10)]

    model = ArchetypeVQ(
        demos.n_features,
        demos.n_actions,
        codebook_size=p1.codebook_size,
        code_dim=p1.code_dim,
        hidden_dim=p1.hidden_dim,
        layers=p1.layers,
        dropout=p1.dropout,
        rnn=p1.encoder,
        commitment_beta=p1.commitment_beta,
        codebook_loss_weight=p1.codebook_loss_weight,
        ema=p1.ema_codebook,
        ema_decay=p1.ema_decay,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p1.lr, weight_decay=p1.weight_decay)

    # AMP only on CUDA: autocast on CPU is bf16 and gives no speed-up here, and
    # MPS is excluded upstream for determinism.
    use_amp = p1.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # pyright: ignore[reportPrivateImportUsage]

    history: list[EpochStats] = []
    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    since_best = 0
    zero_use_epochs = np.zeros(p1.codebook_size, dtype=np.int64)

    for epoch in range(p1.epochs):
        model.train()
        model.quantizer.reset_usage()
        sum_loss = sum_recon = 0.0
        n_seen = correct = total = 0
        last_grad = 0.0
        last_z_e: Tensor | None = None
        perplexities: list[float] = []

        for batch in _batches(train_rows, p1.batch_size, rng, shuffle=True):
            obs, actions = _to_tensors(demos, batch, device)
            ctx = torch.autocast("cuda", enabled=True) if use_amp else nullcontext()
            with ctx:
                if p1.teacher_forcing >= 1.0:
                    logits, vq = model(obs, actions)
                else:
                    logits, vq = _scheduled_forward(
                        model, obs, actions, teacher_forcing=p1.teacher_forcing, gen=torch_gen
                    )
                recon = F.cross_entropy(
                    logits.reshape(-1, model.n_actions).float(), actions.reshape(-1)
                )
                loss = (
                    recon
                    + p1.codebook_loss_weight * vq.codebook_loss
                    + p1.commitment_beta * vq.commitment_loss
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            last_grad = float(nn.utils.clip_grad_norm_(model.parameters(), p1.grad_clip))
            scaler.step(opt)
            scaler.update()

            bs = obs.shape[0]
            sum_loss += float(loss.detach()) * bs
            sum_recon += float(recon.detach()) * bs
            n_seen += bs
            correct += int((logits.argmax(dim=-1) == actions).sum())
            total += int(actions.numel())
            perplexities.append(float(vq.perplexity.detach()))
            last_z_e = vq.z_e.detach()

        usage = model.quantizer.reset_usage().cpu().numpy()
        zero_use_epochs = np.where(usage == 0, zero_use_epochs + 1, 0)
        restarted = 0
        dead_mask = zero_use_epochs >= p1.dead_code_restart_after
        if dead_mask.any() and last_z_e is not None:
            restarted = model.quantizer.restart_dead_codes(
                last_z_e, torch.from_numpy(dead_mask).to(device), generator=torch_gen
            )
            zero_use_epochs[dead_mask] = 0

        val_loss, val_acc = _evaluate(model, demos, val_rows, device, p1.batch_size)
        stats = EpochStats(
            epoch=epoch,
            train_loss=sum_loss / max(n_seen, 1),
            train_recon=sum_recon / max(n_seen, 1),
            train_accuracy=correct / max(total, 1),
            val_loss=val_loss,
            val_accuracy=val_acc,
            perplexity=float(np.mean(perplexities)) if perplexities else 0.0,
            utilization=float((usage > 0).mean()),
            dead_codes=int((usage == 0).sum()),
            restarted=restarted,
            grad_norm=last_grad,
            lr=float(opt.param_groups[0]["lr"]),
        )
        history.append(stats)
        log_event(logger, "phase1.epoch", **vars(stats))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= p1.early_stopping_patience:
                log_event(logger, "phase1.early_stop", epoch=epoch, best_epoch=best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    code_indices = assign_codes(model, demos, device=device, batch_size=p1.batch_size)
    health = codebook_health(
        code_indices,
        demos.actions,
        codebook_size=p1.codebook_size,
        n_actions=demos.n_actions,
        min_perplexity=p1.min_perplexity,
    )
    if health.collapsed:
        # Loud but not fatal: a collapsed codebook is a *finding*, and the
        # evaluation report must be able to state it rather than crash on it.
        logger.warning("phase-1 codebook collapsed: %s", "; ".join(health.reasons))

    probe = val_rows[: min(512, val_rows.size)]
    distinct = code_distinctness(model, demos, probe, device=device, code_indices=code_indices)
    if distinct.degenerate:
        logger.warning(
            "phase-1 archetypes are behaviourally degenerate: %s", "; ".join(distinct.reasons)
        )

    meta: dict[str, Any] = {
        "n_train_rows": int(train_rows.size),
        "n_val_rows": int(val_rows.size),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_val_accuracy": history[-1].val_accuracy if history else float("nan"),
        "codebook_collapsed": health.collapsed,
        "perplexity": health.perplexity,
        "behaviourally_degenerate": distinct.degenerate,
        "distinct_behaviours": distinct.n_distinct_behaviours,
        "code_gain": distinct.code_gain,
        "mean_pairwise_disagreement": distinct.mean_pairwise_disagreement,
        "amp": use_amp,
        "device": str(device),
        "seed": seed,
    }
    result = Phase1Result(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        code_indices=code_indices,
        health=health,
        distinctness=distinct,
        train_rows=train_rows,
        val_rows=val_rows,
        meta=meta,
    )
    if checkpoint_path is not None:
        save_checkpoint(result, demos, checkpoint_path, cfg=cfg)
    return result


@torch.no_grad()
def _evaluate(
    model: ArchetypeVQ,
    demos: DemonstrationSet,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Teacher-forced validation loss and per-step action accuracy."""
    model.eval()
    total_loss = 0.0
    n_seen = correct = total = 0
    for i in range(0, rows.shape[0], batch_size):
        batch = rows[i : i + batch_size]
        obs, actions = _to_tensors(demos, batch, device)
        logits, vq = model(obs, actions)
        loss = (
            F.cross_entropy(logits.reshape(-1, model.n_actions), actions.reshape(-1))
            + model.quantizer.codebook_loss_weight * vq.codebook_loss
            + model.quantizer.commitment_beta * vq.commitment_loss
        )
        total_loss += float(loss) * batch.shape[0]
        n_seen += batch.shape[0]
        correct += int((logits.argmax(dim=-1) == actions).sum())
        total += int(actions.numel())
    return total_loss / max(n_seen, 1), correct / max(total, 1)


@torch.no_grad()
def assign_codes(
    model: ArchetypeVQ,
    demos: DemonstrationSet,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Archetype label for every demonstration, in row order.

    Uses the encoder, which sees the whole demonstration — legitimate because
    these labels only ever supervise training (Phase-2 auxiliary CE, archetype
    profiling). Nothing on the inference path calls this.
    """
    model.eval()
    n = len(demos)
    out = np.empty(n, dtype=np.int64)
    for i in range(0, n, batch_size):
        rows = np.arange(i, min(i + batch_size, n))
        obs, actions = _to_tensors(demos, rows, device)
        vq = model.quantizer(model.encoder(obs, actions))
        out[rows] = vq.indices.cpu().numpy()
    return out


@torch.no_grad()
def rollout_accuracy(
    model: ArchetypeVQ,
    demos: DemonstrationSet,
    rows: np.ndarray,
    *,
    device: torch.device,
    code_indices: np.ndarray | None = None,
) -> float:
    """Accuracy of the *autoregressive* rollout, i.e. the deployed decoder path.

    Reported alongside teacher-forced accuracy because the gap between them is
    the exposure bias Phase 2 will actually inherit.
    """
    obs, actions = _to_tensors(demos, rows, device)
    if code_indices is None:
        codes = model.quantizer(model.encoder(obs, actions)).indices
    else:
        codes = torch.from_numpy(code_indices[rows]).to(device)
    pred = model.rollout(obs, codes, greedy=True)
    return float((pred == actions).float().mean())


@torch.no_grad()
def code_distinctness(
    model: ArchetypeVQ,
    demos: DemonstrationSet,
    rows: np.ndarray,
    *,
    device: torch.device,
    code_indices: np.ndarray,
) -> BehaviouralDistinctness:
    """Roll every code over the same windows and measure behavioural spread.

    The one Phase-1 check that can fail while every loss curve looks healthy: if
    the decoder ignores the code, Phase-2 selection is choosing between identical
    strategies and the whole three-phase structure is decoration.
    """
    obs, actions = _to_tensors(demos, rows, device)
    per_code = np.stack(
        [
            model.rollout(obs, torch.full((rows.shape[0],), k, dtype=torch.long, device=device))
            .cpu()
            .numpy()
            for k in range(model.codebook_size)
        ]
    )
    own = model.rollout(obs, torch.from_numpy(code_indices[rows]).to(device)).cpu().numpy()
    return behavioural_distinctness(
        per_code,
        actions.cpu().numpy(),
        own,
        positions=np.asarray(demos.grid, dtype=np.float64),
    )


def save_checkpoint(
    result: Phase1Result,
    demos: DemonstrationSet,
    path: str | Path,
    *,
    cfg: ExperimentConfig | None = None,
) -> Path:
    """Persist weights plus the shape/grid contract needed to rebuild the model."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "n_features": demos.n_features,
            "n_actions": demos.n_actions,
            "grid": list(demos.grid),
            "horizon": demos.horizon,
            "phase1": cfg.phase1.model_dump() if cfg is not None else None,
            "history": result.history_table(),
            "health": result.health.to_dict(),
            "distinctness": result.distinctness.to_dict(),
            "meta": result.meta,
        },
        p,
    )
    return p


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[ArchetypeVQ, dict[str, Any]]:
    """Rebuild a Phase-1 model from a checkpoint. Inference-safe (no planner)."""
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    p1 = blob.get("phase1") or {}
    model = ArchetypeVQ(
        blob["n_features"],
        blob["n_actions"],
        codebook_size=p1.get("codebook_size", 10),
        code_dim=p1.get("code_dim", 16),
        hidden_dim=p1.get("hidden_dim", 128),
        layers=p1.get("layers", 1),
        dropout=0.0,
        rnn=p1.get("encoder", "gru"),
        commitment_beta=p1.get("commitment_beta", 0.25),
        codebook_loss_weight=p1.get("codebook_loss_weight", 1.0),
        ema=p1.get("ema_codebook", True),
        ema_decay=p1.get("ema_decay", 0.99),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
