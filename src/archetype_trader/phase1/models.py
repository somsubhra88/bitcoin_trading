"""Phase-1 networks: trajectory encoder, VQ codebook, causal action decoder.

Architecture (defaults ``K=10``, ``code_dim=16``, ``hidden=128``)::

    demo trajectory (obs_t, a_t)_{t<h}  ->  GRU/LSTM  ->  z_e in R^16
    z_e  ->  nearest of K codebook vectors  ->  z_q  (straight-through)
    (obs_t, a_{t-1}, z_q)  ->  GRU  ->  logits over positions   [causal decoder]

The encoder is the only component that sees the whole demonstration, and it is
used **only** to assign a training label. At inference the code comes from the
Phase-2 selector and the decoder consumes one observation at a time, so no future
observation is ever an input.

The decoder is deliberately autoregressive in its own past actions rather than in
future observations: that is what lets a single code express a *strategy* (e.g.
"enter early and hold") instead of memorising a return path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class VQOutput:
    """Result of quantising a batch of encoder outputs."""

    z_q: Tensor  # (B, D) straight-through quantised vector
    z_e: Tensor  # (B, D) raw encoder output
    indices: Tensor  # (B,) long
    codebook_loss: Tensor
    commitment_loss: Tensor
    perplexity: Tensor


class TrajectoryEncoder(nn.Module):
    """Encodes a full demonstration ``(obs, action)`` sequence into ``R^code_dim``."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        hidden_dim: int = 128,
        code_dim: int = 16,
        layers: int = 1,
        dropout: float = 0.0,
        rnn: Literal["gru", "lstm"] = "gru",
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        rnn_cls = nn.GRU if rnn == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=n_features + n_actions,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, code_dim),
        )

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        """``obs`` ``(B, h, F)``, ``actions`` ``(B, h)`` long -> ``(B, code_dim)``."""
        onehot = F.one_hot(actions, num_classes=self.n_actions).to(obs.dtype)
        out, _ = self.rnn(torch.cat([obs, onehot], dim=-1))
        return self.head(out[:, -1])


class VectorQuantizer(nn.Module):
    """VQ layer with optional EMA codebook updates and dead-code restarts.

    EMA is the default: with a small codebook (``K=10``) and a strong
    reconstruction signal, gradient-updated codebooks collapse readily, whereas
    EMA tracks the encoder distribution and keeps rare codes alive long enough
    for the restart heuristic to act.
    """

    # Declared so buffers type as tensors rather than ``Tensor | Module``.
    # ``embedding`` is a buffer under EMA and a Parameter otherwise; Parameter is
    # a Tensor subclass, so one annotation covers both.
    embedding: Tensor
    ema_cluster_size: Tensor
    ema_weight: Tensor
    usage_count: Tensor

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        *,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        ema: bool = True,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_beta = commitment_beta
        self.codebook_loss_weight = codebook_loss_weight
        self.ema = ema
        self.ema_decay = ema_decay
        self.eps = eps

        embedding = torch.randn(codebook_size, code_dim) * 0.5
        if ema:
            # Buffers, not parameters: EMA statistics must not receive gradients
            # but must travel with the checkpoint and the device.
            self.register_buffer("embedding", embedding)
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_weight", embedding.clone())
        else:
            self.embedding = nn.Parameter(embedding)
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))

    def forward(self, z_e: Tensor) -> VQOutput:
        emb = self.embedding
        # (B, K) squared distances without materialising (B, K, D).
        dist = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ emb.t()
            + emb.pow(2).sum(dim=1)[None, :]
        )
        indices = dist.argmin(dim=1)
        z_q = emb[indices]

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        one_hot = F.one_hot(indices, self.codebook_size).to(z_e.dtype)
        if self.training:
            self.usage_count += one_hot.sum(dim=0).long()
            if self.ema:
                self._ema_update(z_e.detach(), one_hot)
                z_q = self.embedding[indices]
                codebook_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)

        # Straight-through: decoder gradients flow to the encoder unchanged.
        z_q_st = z_e + (z_q - z_e).detach()

        probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return VQOutput(
            z_q=z_q_st,
            z_e=z_e,
            indices=indices,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    @torch.no_grad()
    def _ema_update(self, z_e: Tensor, one_hot: Tensor) -> None:
        cluster_size = one_hot.sum(dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        dw = one_hot.t() @ z_e
        self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
        n = self.ema_cluster_size.sum()
        # Laplace smoothing keeps an empty cluster's denominator away from zero.
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        self.embedding.copy_(self.ema_weight / smoothed.unsqueeze(1))

    @torch.no_grad()
    def restart_dead_codes(
        self, z_e: Tensor, dead: Tensor, *, generator: torch.Generator | None = None
    ) -> int:
        """Re-seed unused codes from encoder outputs in the current batch.

        Without this a collapsed codebook is a permanent loss of capacity: a code
        with zero assignments receives no gradient and no EMA mass, so it can
        never recover on its own.
        """
        n_dead = int(dead.sum())
        if n_dead == 0 or z_e.shape[0] == 0:
            return 0
        pick = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device, generator=generator)
        fresh = z_e[pick] + 0.01 * torch.randn(
            (n_dead, self.code_dim), device=z_e.device, dtype=z_e.dtype, generator=generator
        )
        idx = dead.nonzero(as_tuple=True)[0]
        self.embedding[idx] = fresh
        if self.ema:
            self.ema_weight[idx] = fresh
            self.ema_cluster_size[idx] = 1.0
        return n_dead

    @torch.no_grad()
    def reset_usage(self) -> Tensor:
        counts = self.usage_count.clone()
        self.usage_count.zero_()
        return counts


class CausalActionDecoder(nn.Module):
    """Predicts the action sequence from a code and *available* observations.

    Inputs at step ``t``: the observation at bar ``t``, a one-hot of the action
    taken at ``t-1`` (a learned start token at ``t=0``), and the code embedding.
    Nothing from ``t+1`` onward enters, which is what makes rollout at inference
    identical to training-time teacher forcing.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        code_dim: int,
        *,
        hidden_dim: int = 128,
        layers: int = 1,
        dropout: float = 0.0,
        rnn: Literal["gru", "lstm"] = "gru",
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn_kind = rnn
        rnn_cls = nn.GRU if rnn == "gru" else nn.LSTM
        self.code_proj = nn.Sequential(nn.Linear(code_dim, hidden_dim), nn.Tanh())
        self.rnn = rnn_cls(
            input_size=n_features + n_actions + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_actions))
        # Learned "no previous action" token so step 0 is not forced to claim a
        # position it never held.
        self.start_token = nn.Parameter(torch.zeros(n_actions))

    def _prev_onehot(self, actions: Tensor) -> Tensor:
        """``(B, h, A)`` one-hot of the previous action, with a start token at t=0."""
        b, h = actions.shape
        onehot = F.one_hot(actions, self.n_actions).to(self.start_token.dtype)
        prev = torch.empty(b, h, self.n_actions, device=actions.device, dtype=onehot.dtype)
        prev[:, 0] = self.start_token
        if h > 1:
            prev[:, 1:] = onehot[:, :-1]
        return prev

    def forward(self, obs: Tensor, code: Tensor, prev_actions: Tensor) -> Tensor:
        """Teacher-forced pass. Returns logits ``(B, h, A)``.

        ``prev_actions`` is the *ground-truth* action sequence; the module shifts
        it internally, so a caller cannot accidentally feed ``a_t`` at step ``t``.
        """
        b, h, _ = obs.shape
        code_h = self.code_proj(code).unsqueeze(1).expand(b, h, self.hidden_dim)
        x = torch.cat([obs, self._prev_onehot(prev_actions), code_h], dim=-1)
        out, _ = self.rnn(x)
        return self.head(out)

    def init_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | tuple[Tensor, Tensor]:
        h = torch.zeros(self.layers, batch, self.hidden_dim, device=device, dtype=dtype)
        if self.rnn_kind == "lstm":
            return (h, torch.zeros_like(h))
        return h

    def step(
        self,
        obs_t: Tensor,
        code: Tensor,
        prev_action: Tensor | None,
        state: Tensor | tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        """One causal step. ``obs_t`` ``(B, F)``; returns ``(logits (B, A), state)``.

        This is the deployed path: it takes a single observation and cannot be
        handed a future one.
        """
        b = obs_t.shape[0]
        if prev_action is None:
            prev = self.start_token.unsqueeze(0).expand(b, self.n_actions)
        else:
            prev = F.one_hot(prev_action, self.n_actions).to(obs_t.dtype)
        code_h = self.code_proj(code)
        x = torch.cat([obs_t, prev, code_h], dim=-1).unsqueeze(1)
        out, new_state = self.rnn(x, state)
        return self.head(out[:, 0]), new_state


class ArchetypeVQ(nn.Module):
    """Encoder + quantiser + decoder, checkpointed as one unit."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        codebook_size: int = 10,
        code_dim: int = 16,
        hidden_dim: int = 128,
        layers: int = 1,
        dropout: float = 0.0,
        rnn: Literal["gru", "lstm"] = "gru",
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        ema: bool = True,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.encoder = TrajectoryEncoder(
            n_features,
            n_actions,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            layers=layers,
            dropout=dropout,
            rnn=rnn,
        )
        self.quantizer = VectorQuantizer(
            codebook_size,
            code_dim,
            commitment_beta=commitment_beta,
            codebook_loss_weight=codebook_loss_weight,
            ema=ema,
            ema_decay=ema_decay,
        )
        self.decoder = CausalActionDecoder(
            n_features,
            n_actions,
            code_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            rnn=rnn,
        )

    def forward(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, VQOutput]:
        """Teacher-forced training pass. Returns ``(logits, vq)``."""
        z_e = self.encoder(obs, actions)
        vq = self.quantizer(z_e)
        logits = self.decoder(obs, vq.z_q, actions)
        return logits, vq

    def code_embedding(self, indices: Tensor) -> Tensor:
        """Look up codebook rows for given archetype indices (inference path)."""
        return self.quantizer.embedding[indices]

    @torch.no_grad()
    def rollout(
        self,
        obs: Tensor,
        code_indices: Tensor,
        *,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressive action rollout for a batch. ``obs`` ``(B, h, F)``.

        Only observation ``t`` is visible at step ``t``. Used for evaluation and
        by Phase 2/3; ``obs`` is passed as a block purely for convenience and is
        consumed one step at a time.
        """
        self.eval()
        b, h, _ = obs.shape
        code = self.code_embedding(code_indices)
        state = self.decoder.init_state(b, obs.device, obs.dtype)
        prev: Tensor | None = None
        out = torch.empty(b, h, dtype=torch.long, device=obs.device)
        for t in range(h):
            logits, state = self.decoder.step(obs[:, t], code, prev, state)
            if greedy:
                prev = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                prev = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            out[:, t] = prev
        return out
