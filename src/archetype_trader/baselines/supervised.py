"""Supervised forecast + threshold policy: the honest "just predict the return" baseline.

The most common straw man in the literature is a return forecaster whose labels
were computed across the split boundary. Two rules prevent that here:

1. A training row at bar ``t`` needs the realised return over ``(t, t + h]``, so
   every row with ``t + h >= train.stop`` is **dropped**. The label window is
   never allowed to peek into the embargo, let alone into validation.
2. The forecaster consumes the same scaled feature row the learned policies see,
   fitted by the same train-only :class:`FeatureScaler`, and the decision
   threshold is a quantile of *training* predictions only.

The policy itself is deliberately trivial — sign of the forecast, gated by a
magnitude threshold, sized at full exposure — because the point of the baseline
is to measure how much of the edge is just "a return forecast plus a filter".
Anything cleverer would be a competing strategy, not a reference point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from archetype_trader.config import BaselineConfig, FeatureConfig
from archetype_trader.data.types import MarketData
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.splits import Segment


class ForecastMLP(nn.Module):
    """Two-layer MLP on one scaled feature row. Predicts a forward log return."""

    def __init__(self, n_features: int, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ForecastModel:
    """Fitted forecaster plus the train-only decision threshold."""

    model: ForecastMLP
    threshold: float
    horizon: int
    train_rows: int
    train_loss: float
    val_loss: float
    #: Spearman-free, scale-free sanity check: fraction of training rows whose
    #: predicted sign matched the realised sign. ~0.5 means the forecaster
    #: learned nothing and the baseline should trade almost never.
    train_sign_accuracy: float

    @torch.no_grad()
    def predict(self, x: np.ndarray, *, device: torch.device | None = None) -> np.ndarray:
        self.model.eval()
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if device is not None:
            t = t.to(device)
        return self.model(t).detach().cpu().numpy().astype(np.float64)


def forward_log_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """``log(close[t+h] / close[t])``, NaN where ``t + h`` runs off the end.

    NaN rather than 0.0: a silently zero-filled tail would train the forecaster
    to predict "no move" at exactly the bars where it has no information.
    """
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if horizon < n:
        out[: n - horizon] = np.log(close[horizon:] / close[: n - horizon])
    return out


def training_rows(segment: Segment, features: FeatureMatrix, horizon: int) -> np.ndarray:
    """Bars inside ``segment`` whose whole label window also lies inside it.

    This is where label leakage would enter, so it is one small function with one
    job and a test pointed straight at it.
    """
    lo = max(segment.start, features.valid_from)
    hi = segment.stop - horizon
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    return np.arange(lo, hi, dtype=np.int64)


def fit_forecaster(
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    segment: Segment,
    cfg: BaselineConfig,
    feature_cfg: FeatureConfig,
    *,
    horizon: int,
    device: torch.device | None = None,
    seed: int = 0,
    val_fraction: float = 0.15,
) -> ForecastModel:
    """Fit the forecaster on ``segment`` only and pick the threshold on it too.

    The internal validation split is the **last** ``val_fraction`` of the training
    segment, chronologically — a random split would let the model see bars after
    the ones it is early-stopped on.
    """
    del feature_cfg  # kept for signature symmetry with the rule baselines
    dev = device or torch.device("cpu")
    rows = training_rows(segment, features, horizon)
    if rows.size < 64:
        raise ValueError(
            f"supervised baseline needs >=64 labelled training rows, got {rows.size}; "
            f"segment {segment.start}:{segment.stop} with horizon {horizon}"
        )
    scaled = scaler.transform(features.values).astype(np.float32)
    y_all = forward_log_return(data.close, horizon)
    x = scaled[rows]
    y = y_all[rows]
    if not np.isfinite(y).all():
        raise AssertionError("labelled training rows contain a non-finite forward return")

    cut = max(1, int(rows.size * (1.0 - val_fraction)))
    xt = torch.from_numpy(x[:cut]).to(dev)
    yt = torch.from_numpy(y[:cut].astype(np.float32)).to(dev)
    xv = torch.from_numpy(x[cut:]).to(dev)
    yv = torch.from_numpy(y[cut:].astype(np.float32)).to(dev)

    torch.manual_seed(seed)
    model = ForecastMLP(scaled.shape[1], hidden_dim=cfg.rl_hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = 256
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_loss = float("inf")
    for _ in range(cfg.supervised_epochs):
        model.train()
        perm = torch.randperm(xt.shape[0], generator=gen)
        total = 0.0
        for i in range(0, perm.numel(), batch):
            idx = perm[i : i + batch].to(dev)
            opt.zero_grad(set_to_none=True)
            # Huber: forward crypto returns are fat-tailed, and MSE would let a
            # handful of jump bars set the entire forecast scale.
            loss = nn.functional.huber_loss(model(xt[idx]), yt[idx], delta=0.01)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * idx.numel()
        train_loss = total / max(perm.numel(), 1)
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.huber_loss(model(xv), yv, delta=0.01)) if xv.numel() else 0.0
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    fitted = ForecastModel(
        model=model,
        threshold=0.0,
        horizon=horizon,
        train_rows=int(rows.size),
        train_loss=train_loss,
        val_loss=best_val,
        train_sign_accuracy=0.0,
    )
    pred_train = fitted.predict(x, device=dev)
    fitted.threshold = float(np.quantile(np.abs(pred_train), cfg.threshold_quantile))
    fitted.train_sign_accuracy = float(np.mean(np.sign(pred_train) == np.sign(y)))
    return fitted


def threshold_targets(
    fitted: ForecastModel,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    *,
    allow_short: bool,
    device: torch.device | None = None,
) -> np.ndarray:
    """Target position per bar: ``sign(pred)`` when ``|pred| >= threshold``, else flat.

    The forecast at bar ``t`` uses only features observable at ``t`` and the
    position it implies is executed at the next bar's open by the environment, so
    the one-bar shift the rule baselines apply explicitly is applied here by the
    execution model instead.
    """
    scaled = scaler.transform(features.values).astype(np.float32)
    pred = fitted.predict(scaled, device=device)
    pred[: features.valid_from] = 0.0
    side = np.sign(pred)
    if not allow_short:
        side = np.clip(side, 0.0, 1.0)
    targets = np.where(np.abs(pred) >= fitted.threshold, side, 0.0)
    targets[: features.valid_from] = 0.0
    return targets.astype(np.float64)
