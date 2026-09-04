"""CPU-only DeepLOB-style order-book signal model.

The research DeepLOB architecture is a CNN/LSTM that requires a trained model
artifact. This module keeps the strategy deployable without an opaque artifact:
it builds the same kind of level-by-time order-book representation, applies
temporal convolution features, and fits a small ridge prediction head using
only labels that were already observable before the current snapshot.

It is therefore an online, leakage-safe DeepLOB-style baseline, not a claim to
be a pretrained reproduction of the paper. The model returns HOLD until the
book history is both deep enough and sufficiently valid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import numpy as np

from btc_trader.db.models import OrderBookSnapshot
from btc_trader.enums import TradingSignal
from btc_trader.strategy import SignalDecision

if TYPE_CHECKING:
    from btc_trader.config import Settings

MIN_TRAINING_ROWS = 16
CONVOLUTION_WIDTHS = (3, 5, 7)


@dataclass(frozen=True, slots=True)
class DeepLOBParameters:
    """Runtime controls for the rolling order-book model."""

    window: int = 32
    horizon: int = 3
    training_window: int = 256
    levels: int = 10
    threshold_percent: Decimal = Decimal("0.01")
    min_quality_percent: Decimal = Decimal("80")

    def __post_init__(self) -> None:
        if self.window < 8:
            raise ValueError("DeepLOB window must be at least 8 snapshots")
        if self.horizon < 1:
            raise ValueError("DeepLOB horizon must be positive")
        if self.training_window < MIN_TRAINING_ROWS:
            raise ValueError("DeepLOB training window is too small")
        if self.levels < 1:
            raise ValueError("DeepLOB levels must be positive")
        if self.threshold_percent <= 0:
            raise ValueError("DeepLOB threshold must be positive")
        if not 0 < self.min_quality_percent <= 100:
            raise ValueError("DeepLOB minimum quality must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class _BookPoint:
    timestamp: datetime
    mid: float
    bid: float
    ask: float
    channels: tuple[float, ...]


def parameters_from_settings(settings: Settings) -> DeepLOBParameters:
    return DeepLOBParameters(
        window=settings.strategy_deeplob_window,
        horizon=settings.strategy_deeplob_horizon,
        training_window=settings.strategy_deeplob_training_window,
        levels=settings.strategy_deeplob_levels,
        threshold_percent=settings.strategy_deeplob_threshold_percent,
        min_quality_percent=settings.strategy_deeplob_min_quality_percent,
    )


def evaluate_order_book(
    snapshots: Sequence[OrderBookSnapshot],
    *,
    parameters: DeepLOBParameters | None = None,
) -> SignalDecision | None:
    """Predict the next mid-price return from historical order-book snapshots.

    Training rows end at ``n - horizon``. Consequently, the current snapshot
    is never labelled with a future value that is not yet known.
    """
    parameters = parameters or DeepLOBParameters()
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.received_at)
    if not ordered:
        return None

    points = _valid_points(ordered, parameters.levels)
    quality = Decimal(len(points)) / Decimal(len(ordered)) * Decimal("100")
    if not points:
        return None
    latest = points[-1]
    base_metadata: dict[str, str | float] = {
        "model": "deeplob_style_ridge",
        "order_book_quality_percent": float(quality),
        "order_book_imbalance": latest.channels[0],
        "microprice_pressure": latest.channels[1],
        "spread_bps": latest.channels[2],
        "source_timestamp": latest.timestamp.isoformat(),
        "best_bid": latest.bid,
        "best_ask": latest.ask,
    }
    if quality < parameters.min_quality_percent:
        return _hold(
            latest,
            f"DeepLOB waiting for reliable order-book history ({quality:.1f}% valid)",
            base_metadata,
        )

    minimum_rows = parameters.window + parameters.horizon + MIN_TRAINING_ROWS
    if len(points) < minimum_rows:
        return _hold(
            latest,
            f"DeepLOB waiting for history ({len(points)}/{minimum_rows} snapshots)",
            base_metadata,
        )

    examples: list[np.ndarray] = []
    targets: list[float] = []
    last_labelled_end = len(points) - parameters.horizon - 1
    first_end = parameters.window - 1
    for end in range(first_end, last_labelled_end + 1):
        start = end - parameters.window + 1
        examples.append(_temporal_features(points[start : end + 1]))
        current_mid = points[end].mid
        future_mid = points[end + parameters.horizon].mid
        targets.append((future_mid - current_mid) / current_mid)
    if len(examples) > parameters.training_window:
        examples = examples[-parameters.training_window :]
        targets = targets[-parameters.training_window :]
    if len(examples) < MIN_TRAINING_ROWS:
        return _hold(
            latest,
            f"DeepLOB waiting for labelled history ({len(examples)}/{MIN_TRAINING_ROWS} rows)",
            base_metadata,
        )

    matrix = np.asarray(examples, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardised = (matrix - mean) / scale
    design = np.column_stack((standardised, np.ones(len(standardised))))
    regularisation = np.eye(design.shape[1], dtype=np.float64) * 1e-3
    regularisation[-1, -1] = 0.0
    try:
        weights = np.linalg.solve(
            design.T @ design + regularisation,
            design.T @ target_array,
        )
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(design, target_array, rcond=None)[0]

    current_features = (_temporal_features(points[-parameters.window :]) - mean) / scale
    predicted_return = float(np.clip(np.dot(np.append(current_features, 1.0), weights), -1.0, 1.0))
    threshold = float(parameters.threshold_percent / Decimal("100"))
    signal = (
        TradingSignal.BUY
        if predicted_return > threshold
        else TradingSignal.SELL
        if predicted_return < -threshold
        else TradingSignal.HOLD
    )
    confidence = min(abs(predicted_return) / threshold, 1.0)
    metadata: dict[str, str | float] = {
        **base_metadata,
        "predicted_return": predicted_return,
        "training_rows": float(len(examples)),
        "signal_threshold": threshold,
        "signal_strength": abs(predicted_return),
    }
    return SignalDecision(
        signal=signal,
        confidence=Decimal(str(round(confidence, 6))),
        price=Decimal(str(latest.mid)),
        reason=(
            f"DeepLOB-style {predicted_return:+.2%} next-return forecast "
            f"(threshold ±{threshold:.2%}, {len(examples)} training rows)"
        ),
        metadata=metadata,
    )


def snapshot_quotes(snapshot: OrderBookSnapshot) -> tuple[Decimal, Decimal, Decimal] | None:
    """Return ``(mid, bid, ask)`` for a snapshot that has an executable top of book."""
    bid_level = _decimal_level(snapshot.bids)
    ask_level = _decimal_level(snapshot.asks)
    if bid_level is None or ask_level is None:
        return None
    bid, _ = bid_level
    ask, _ = ask_level
    if ask <= bid:
        return None
    mid = (
        snapshot.mid_price
        if snapshot.mid_price is not None and snapshot.mid_price > 0
        else (bid + ask) / 2
    )
    if not mid.is_finite() or mid <= 0:
        return None
    return mid, bid, ask


def _valid_points(snapshots: Sequence[OrderBookSnapshot], levels: int) -> list[_BookPoint]:
    points: list[_BookPoint] = []
    previous_mid: float | None = None
    for snapshot in snapshots:
        extracted = _extract_point(snapshot, levels, previous_mid)
        if extracted is None:
            continue
        points.append(extracted)
        previous_mid = extracted.mid
    return points


def _extract_point(
    snapshot: OrderBookSnapshot, levels: int, previous_mid: float | None
) -> _BookPoint | None:
    bids = _levels(snapshot.bids, levels)
    asks = _levels(snapshot.asks, levels)
    if not bids or not asks:
        return None
    if any(bids[index][0] <= bids[index + 1][0] for index in range(len(bids) - 1)):
        return None
    if any(asks[index][0] >= asks[index + 1][0] for index in range(len(asks) - 1)):
        return None
    quote = snapshot_quotes(snapshot)
    if quote is None:
        return None
    mid_decimal, bid_decimal, ask_decimal = quote
    bid, bid_size = float(bid_decimal), bids[0][1]
    ask, ask_size = float(ask_decimal), asks[0][1]
    mid = float(mid_decimal)
    bid_depth = sum(size / (index + 1) for index, (_, size) in enumerate(bids))
    ask_depth = sum(size / (index + 1) for index, (_, size) in enumerate(asks))
    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        return None
    imbalance = (bid_depth - ask_depth) / total_depth
    microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    microprice_pressure = (microprice - mid) / mid
    spread_bps = (ask - bid) / mid * 10_000
    depth_imbalance = (bid_depth - ask_depth) / total_depth
    mid_return = (mid - previous_mid) / previous_mid if previous_mid else 0.0
    return _BookPoint(
        timestamp=snapshot.received_at,
        mid=mid,
        bid=bid,
        ask=ask,
        channels=(imbalance, microprice_pressure, spread_bps / 10_000, depth_imbalance, mid_return),
    )


def _levels(value: Any, levels: int) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    result: list[tuple[float, float]] = []
    for level in value[:levels]:
        if not isinstance(level, dict):
            return []
        try:
            price = Decimal(str(level["price"]))
            size = Decimal(str(level["size"]))
        except (KeyError, InvalidOperation, TypeError, ValueError):
            return []
        if not price.is_finite() or not size.is_finite() or price <= 0 or size <= 0:
            return []
        result.append((float(price), float(size)))
    return result


def _decimal_level(value: Any) -> tuple[Decimal, Decimal] | None:
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return None
    try:
        price = Decimal(str(value[0]["price"]))
        size = Decimal(str(value[0]["size"]))
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None
    if not price.is_finite() or not size.is_finite() or price <= 0 or size <= 0:
        return None
    return price, size


def _temporal_features(points: Sequence[_BookPoint]) -> np.ndarray:
    channels = np.asarray([point.channels for point in points], dtype=np.float64)
    features: list[float] = []
    for channel in channels.T:
        features.extend((float(channel[-1]), float(channel.mean()), float(channel.std())))
        features.append(float(channel[-1] - channel[0]))
        for width in CONVOLUTION_WIDTHS:
            kernel = np.arange(width, dtype=np.float64) - (width - 1) / 2
            kernel /= np.linalg.norm(kernel)
            features.append(float(np.dot(channel[-width:], kernel)))
    return np.asarray(features, dtype=np.float64)


def _hold(point: _BookPoint, reason: str, metadata: dict[str, Any]) -> SignalDecision:
    return SignalDecision(
        signal=TradingSignal.HOLD,
        confidence=Decimal("0"),
        price=Decimal(str(point.mid)),
        reason=reason,
        metadata=metadata,
    )


__all__ = [
    "DeepLOBParameters",
    "evaluate_order_book",
    "parameters_from_settings",
    "snapshot_quotes",
]
