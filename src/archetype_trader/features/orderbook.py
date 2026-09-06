"""Causal level-2 order-book features.

All quantities are computed from the snapshot at bar ``t`` (and, for flow
measures, ``t-1``), so nothing looks ahead. Prices enter only as ratios to the
mid, keeping the block scale-free like the OHLCV block.
"""

from __future__ import annotations

import numpy as np

from archetype_trader.config import FeatureConfig
from archetype_trader.data.types import MarketData
from archetype_trader.features import rolling as rr

_EPS = 1e-12


def orderbook_features(data: MarketData, cfg: FeatureConfig) -> tuple[list[str], np.ndarray]:
    """Build the L2 feature block. Returns empty arrays when no book exists."""
    if not data.has_book:
        return [], np.zeros((len(data), 0), dtype=np.float64)

    bid_px = data.bid_px
    bid_sz = data.bid_sz
    ask_px = data.ask_px
    ask_sz = data.ask_sz
    assert bid_px is not None and bid_sz is not None  # noqa: S101
    assert ask_px is not None and ask_sz is not None  # noqa: S101

    best_bid, best_ask = bid_px[:, 0], ask_px[:, 0]
    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid

    names: list[str] = []
    cols: list[np.ndarray] = []

    def add(name: str, col: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(col, dtype=np.float64))

    add("spread_bps", spread / np.maximum(mid, _EPS) * 1e4)
    add("rel_spread_log", np.log(np.maximum(spread, _EPS) / np.maximum(mid, _EPS)))

    # Microprice: size-weighted touch price, the fair value a taker faces.
    touch_sz = bid_sz[:, 0] + ask_sz[:, 0]
    microprice = (best_bid * ask_sz[:, 0] + best_ask * bid_sz[:, 0]) / np.maximum(touch_sz, _EPS)
    add("microprice_dev_bps", (microprice - mid) / np.maximum(mid, _EPS) * 1e4)

    depths = tuple(d for d in cfg.orderbook_depths if d <= data.levels)
    for d in depths:
        bsum = bid_sz[:, :d].sum(axis=1)
        asum = ask_sz[:, :d].sum(axis=1)
        total = bsum + asum
        add(f"imbalance_{d}", (bsum - asum) / np.maximum(total, _EPS))
        add(f"log_depth_{d}", np.log1p(total))
        # Weighted midprice using cumulative depth at this level count.
        wmid = (best_bid * asum + best_ask * bsum) / np.maximum(total, _EPS)
        add(f"weighted_mid_dev_bps_{d}", (wmid - mid) / np.maximum(mid, _EPS) * 1e4)

    # Depth slope: how fast cumulative size grows per bp away from the mid.
    for side, px, sz in (("bid", bid_px, bid_sz), ("ask", ask_px, ask_sz)):
        d = max(depths) if depths else 1
        dist_bps = np.abs(px[:, :d] - mid[:, None]) / np.maximum(mid[:, None], _EPS) * 1e4
        cum_sz = np.cumsum(sz[:, :d], axis=1)
        add(f"{side}_depth_slope", _ls_slope(dist_bps, cum_sz))

    # Order-flow imbalance at the touch (Cont/Kukanov/Stoikov style): change in
    # resting size adjusted for price moves, using only t and t-1.
    add("ofi_touch", _ofi(bid_px[:, 0], bid_sz[:, 0], ask_px[:, 0], ask_sz[:, 0]))
    d_full = max(depths) if depths else 1
    add(
        "ofi_depth",
        _ofi(
            bid_px[:, 0],
            bid_sz[:, :d_full].sum(axis=1),
            ask_px[:, 0],
            ask_sz[:, :d_full].sum(axis=1),
        ),
    )

    # Liquidity / volatility interaction: realised vol of the mid per unit of
    # touch depth — a proxy for how expensive it is to trade in this regime.
    mid_ret = np.zeros_like(mid)
    mid_ret[1:] = np.diff(np.log(np.maximum(mid, _EPS)))
    for w in cfg.vol_windows:
        add(f"mid_realized_vol_{w}", rr.rolling_std(mid_ret, w))
        add(f"spread_mean_bps_{w}", rr.rolling_mean(spread / np.maximum(mid, _EPS) * 1e4, w))
    add("vol_per_depth", rr.rolling_std(mid_ret, min(cfg.vol_windows)) / np.log1p(touch_sz + _EPS))

    matrix = np.column_stack(cols) if cols else np.zeros((len(data), 0))
    return names, matrix


def _ls_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row least-squares slope of ``y`` on ``x`` (no intercept centring bug).

    Rows with zero x-variance (a degenerate single-level book) return 0.
    """
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=np.float64)
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    denom = (xc * xc).sum(axis=1)
    slope = np.where(denom > _EPS, (xc * yc).sum(axis=1) / np.maximum(denom, _EPS), 0.0)
    return slope


def _ofi(
    bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray
) -> np.ndarray:
    """Order-flow imbalance from consecutive snapshots, normalised by size.

    On the bid: a higher price means new demand (``+size``), a lower price means
    demand was pulled (``-prev_size``), an unchanged price contributes the size
    delta. The ask is mirrored. Element 0 is 0 (no previous snapshot).
    """
    out = np.zeros(bid_px.shape[0], dtype=np.float64)
    if bid_px.shape[0] < 2:
        return out
    pb, cb = bid_px[:-1], bid_px[1:]
    ps_b, cs_b = bid_sz[:-1], bid_sz[1:]
    pa, ca = ask_px[:-1], ask_px[1:]
    ps_a, cs_a = ask_sz[:-1], ask_sz[1:]

    bid_flow = np.where(cb > pb, cs_b, np.where(cb < pb, -ps_b, cs_b - ps_b))
    ask_flow = np.where(ca < pa, cs_a, np.where(ca > pa, -ps_a, cs_a - ps_a))
    scale = np.maximum(cs_b + cs_a + ps_b + ps_a, _EPS)
    out[1:] = (bid_flow - ask_flow) / scale
    return out
