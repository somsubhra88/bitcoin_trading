"""Feature assembly, warm-up masking and train-only fitted scaling.

Two rules are enforced here rather than left to callers:

1. **Warm-up is dropped, not filled.** ``valid_from`` is the first index at
   which every column is finite; earlier rows are unusable and any consumer
   that ignores ``valid_from`` will see ``NaN`` and fail loudly.
2. **Scalers are fitted on a train slice only.** :func:`fit_scaler` takes an
   explicit index range and refuses to look outside it, so a scaler can never
   absorb validation or test statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from archetype_trader.config import FeatureConfig
from archetype_trader.data.types import MarketData
from archetype_trader.features.ohlcv import ohlcv_features
from archetype_trader.features.orderbook import orderbook_features


@dataclass(frozen=True)
class FeatureMatrix:
    """Precomputed causal market features aligned to bar closes.

    ``values[t]`` is observable at the close of bar ``t`` and at no earlier
    time. Rows before ``valid_from`` contain warm-up ``NaN``s.
    """

    values: np.ndarray
    names: tuple[str, ...]
    valid_from: int
    lookback: int
    max_lookback: int
    definition_hash: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {self.values.shape}")
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"values has {self.values.shape[1]} columns but {len(self.names)} names"
            )

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def window(self, end: int) -> np.ndarray:
        """The ``lookback``-bar window ending at (and including) bar ``end``."""
        start = end - self.lookback + 1
        if start < 0:
            raise IndexError(f"window ending at {end} needs {self.lookback} bars of history")
        if end >= len(self):
            raise IndexError(f"bar {end} is beyond the feature matrix ({len(self)} rows)")
        return self.values[start : end + 1]

    def is_finite_from(self, index: int) -> bool:
        return bool(np.isfinite(self.values[index:]).all())


def build_features(data: MarketData, cfg: FeatureConfig) -> FeatureMatrix:
    """Assemble the market feature matrix for ``data``."""
    names, market = ohlcv_features(data, cfg)
    if cfg.include_orderbook:
        if not data.has_book:
            raise ValueError("features.include_orderbook is set but the data has no order book")
        ob_names, ob = orderbook_features(data, cfg)
        names = names + ob_names
        market = np.column_stack([market, ob]) if ob.shape[1] else market

    # First row where every column is finite. Warm-up lengths differ per column,
    # so take the max of each column's first finite index.
    finite = np.isfinite(market)
    if not finite.any():
        raise ValueError("no finite feature rows were produced")
    first_finite_per_col = np.argmax(finite, axis=0)
    never_finite = ~finite.any(axis=0)
    if never_finite.any():
        bad = [names[i] for i in np.flatnonzero(never_finite)]
        raise ValueError(f"features never become finite: {bad}")
    valid_from = int(first_finite_per_col.max())

    # Any NaN *after* valid_from means a mid-series hole, not warm-up, and would
    # otherwise propagate silently into the models.
    if not np.isfinite(market[valid_from:]).all():
        n_bad = int((~np.isfinite(market[valid_from:])).sum())
        raise ValueError(f"{n_bad} non-finite feature values after warm-up index {valid_from}")

    definition = {
        "names": names,
        "config": cfg.model_dump(mode="json"),
        "n_bars": len(data),
        "levels": data.levels,
    }
    from archetype_trader.utils.provenance import sha256_json

    return FeatureMatrix(
        values=np.ascontiguousarray(market, dtype=np.float64),
        names=tuple(names),
        valid_from=max(valid_from, cfg.lookback - 1),
        lookback=cfg.lookback,
        max_lookback=cfg.max_lookback,
        definition_hash=sha256_json(definition),
    )


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
@dataclass
class FeatureScaler:
    """Robust per-column standardisation fitted on a train range only.

    Median/IQR rather than mean/std: 10-minute crypto features have heavy tails
    and a single flash-crash bar can otherwise set the scale for a whole fold.
    """

    center: np.ndarray
    scale: np.ndarray
    names: tuple[str, ...]
    clip_sigma: float
    fit_start: int
    fit_stop: int
    fit_rows: int
    portfolio_names: tuple[str, ...] = ()

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardise then winsorise. Accepts ``(F,)``, ``(T, F)`` or ``(B, T, F)``."""
        if x.shape[-1] != self.center.shape[0]:
            raise ValueError(f"expected {self.center.shape[0]} feature columns, got {x.shape[-1]}")
        out = (x - self.center) / self.scale
        return np.clip(out, -self.clip_sigma, self.clip_sigma)

    def state_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center"] = self.center.tolist()
        d["scale"] = self.scale.tolist()
        d["names"] = list(self.names)
        d["portfolio_names"] = list(self.portfolio_names)
        return d

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), indent=2))
        return p

    @staticmethod
    def load(path: str | Path) -> FeatureScaler:
        d = json.loads(Path(path).read_text())
        return FeatureScaler(
            center=np.asarray(d["center"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            names=tuple(d["names"]),
            clip_sigma=float(d["clip_sigma"]),
            fit_start=int(d["fit_start"]),
            fit_stop=int(d["fit_stop"]),
            fit_rows=int(d["fit_rows"]),
            portfolio_names=tuple(d.get("portfolio_names", ())),
        )


def fit_scaler(
    features: FeatureMatrix,
    train_start: int,
    train_stop: int,
    *,
    clip_sigma: float | None = None,
    portfolio_names: tuple[str, ...] = (),
) -> FeatureScaler:
    """Fit a :class:`FeatureScaler` on ``[train_start, train_stop)`` only.

    ``train_start`` is raised to ``features.valid_from`` so warm-up rows never
    enter the fit.
    """
    if not 0 <= train_start < train_stop <= len(features):
        raise IndexError(
            f"invalid train range [{train_start}:{train_stop}) for {len(features)} rows"
        )
    start = max(train_start, features.valid_from)
    if start >= train_stop:
        raise ValueError(
            f"train range [{train_start}:{train_stop}) is entirely inside the warm-up region "
            f"(valid_from={features.valid_from})"
        )
    block = features.values[start:train_stop]
    center = np.median(block, axis=0)
    q75, q25 = np.percentile(block, [75, 25], axis=0)
    iqr = q75 - q25
    # Constant columns (e.g. is_weekend on a short synthetic sample) get unit
    # scale rather than exploding to infinity.
    scale = np.where(iqr > 1e-12, iqr / 1.349, 1.0)
    std = block.std(axis=0)
    scale = np.where(iqr > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return FeatureScaler(
        center=center,
        scale=scale,
        names=features.names,
        clip_sigma=clip_sigma if clip_sigma is not None else 8.0,
        fit_start=start,
        fit_stop=train_stop,
        fit_rows=int(block.shape[0]),
        portfolio_names=portfolio_names,
    )


@dataclass
class FeatureBundle:
    """Everything a stage needs: data, features, and the train-fitted scaler."""

    data: MarketData
    features: FeatureMatrix
    scaler: FeatureScaler
    extra: dict[str, Any] = field(default_factory=dict)

    def scaled_window(self, end: int) -> np.ndarray:
        return self.scaler.transform(self.features.window(end))
