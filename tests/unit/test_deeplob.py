from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from btc_trader.db.models import OrderBookSnapshot
from btc_trader.deeplob import DeepLOBParameters, evaluate_order_book, snapshot_quotes
from btc_trader.enums import TradingSignal


def _snapshots(count: int, *, invalid_at: int | None = None) -> list[OrderBookSnapshot]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    rows: list[OrderBookSnapshot] = []
    for index in range(count):
        mid = Decimal("100") + Decimal(index) / Decimal("20")
        rows.append(
            OrderBookSnapshot(
                exchange="test",
                product_code="BTC_JPY",
                received_at=start + timedelta(seconds=index),
                mid_price=mid,
                depth_levels=3,
                bids=[
                    {"price": str(mid - Decimal("0.5") - level), "size": "2"} for level in range(3)
                ],
                asks=[
                    {"price": str(mid + Decimal("0.5") + level), "size": "1"} for level in range(3)
                ],
            )
        )
    if invalid_at is not None:
        rows[invalid_at].asks = [{"price": "0", "size": "0"}]
    return rows


def _parameters() -> DeepLOBParameters:
    return DeepLOBParameters(
        window=8,
        horizon=2,
        training_window=64,
        levels=3,
        threshold_percent=Decimal("0.01"),
        min_quality_percent=Decimal("80"),
    )


def test_deeplob_produces_a_directional_signal_from_reliable_history() -> None:
    decision = evaluate_order_book(_snapshots(80), parameters=_parameters())

    assert decision is not None
    assert decision.signal is TradingSignal.BUY
    assert decision.confidence > 0
    assert decision.metadata["model"] == "deeplob_style_ridge"
    assert decision.metadata["training_rows"] == 64.0


def test_deeplob_rejects_unreliable_order_book_history() -> None:
    decision = evaluate_order_book(_snapshots(80, invalid_at=0), parameters=_parameters())

    assert decision is not None
    assert decision.signal is TradingSignal.BUY
    assert "reliable" not in decision.reason

    invalid = _snapshots(80)
    for index in range(20):
        invalid[index].asks = [{"price": "0", "size": "0"}]
    rejected = evaluate_order_book(invalid, parameters=_parameters())

    assert rejected is not None
    assert rejected.signal is TradingSignal.HOLD
    assert "reliable" in rejected.reason


def test_snapshot_quotes_preserve_decimal_prices() -> None:
    snapshot = _snapshots(1)[0]

    assert snapshot_quotes(snapshot) == (Decimal("100"), Decimal("99.5"), Decimal("100.5"))
