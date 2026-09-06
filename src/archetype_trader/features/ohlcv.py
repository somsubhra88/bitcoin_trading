"""Causal OHLCV features.

Each block returns ``(names, columns)`` so the feature matrix carries readable
column labels into diagnostics, plots and the saved feature definition.

Scale discipline: everything is expressed as a return, a ratio, a log-ratio or
a bounded oscillator. Nothing is a raw price level, so a model trained on one
price regime is not extrapolating on another.
"""

from __future__ import annotations

import numpy as np

from archetype_trader.config import FeatureConfig
from archetype_trader.data.types import MarketData
from archetype_trader.features import rolling as rr

_EPS = 1e-12


def ohlcv_features(data: MarketData, cfg: FeatureConfig) -> tuple[list[str], np.ndarray]:
    """Build the OHLCV feature block. Column ``t`` uses bars ``<= t`` only."""
    close = data.close
    high = data.high
    low = data.low
    open_ = data.open
    volume = data.volume
    log_close = np.log(close)
    log_ret = data.log_returns()

    names: list[str] = []
    cols: list[np.ndarray] = []

    def add(name: str, col: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(col, dtype=np.float64))

    # --- returns ---------------------------------------------------------- #
    for w in cfg.return_windows:
        add(f"logret_{w}", log_close - rr.shift(log_close, w))

    # --- ranges, gaps, body ---------------------------------------------- #
    add("hl_range", np.log(np.maximum(high, _EPS) / np.maximum(low, _EPS)))
    add("body_frac", (close - open_) / np.maximum(high - low, _EPS))
    add("upper_wick", (high - np.maximum(open_, close)) / np.maximum(high - low, _EPS))
    add("lower_wick", (np.minimum(open_, close) - low) / np.maximum(high - low, _EPS))
    add("gap_open", np.log(np.maximum(open_, _EPS)) - rr.shift(log_close, 1))
    add("close_in_range", (close - low) / np.maximum(high - low, _EPS))

    # --- volume ----------------------------------------------------------- #
    log_vol = np.log1p(volume)
    add("log_volume_z", _zscore_rolling(log_vol, max(cfg.vol_windows)))
    add("volume_change", log_vol - rr.shift(log_vol, 1))
    for w in cfg.vol_windows:
        mean_v = rr.rolling_mean(volume, w)
        add(f"volume_ratio_{w}", volume / np.maximum(mean_v, _EPS))

    # --- realised / downside volatility ---------------------------------- #
    for w in cfg.vol_windows:
        add(f"realized_vol_{w}", rr.rolling_std(log_ret, w))
    neg = np.minimum(log_ret, 0.0)
    dv = np.sqrt(np.maximum(rr.rolling_mean(neg * neg, cfg.downside_vol_window), 0.0))
    add(f"downside_vol_{cfg.downside_vol_window}", dv)
    ref_vol = rr.rolling_std(log_ret, max(cfg.vol_windows))
    add("vol_of_vol", rr.rolling_std(np.nan_to_num(ref_vol, nan=0.0), max(cfg.vol_windows)))
    # Short/long realised-vol ratio: a compact volatility-regime indicator.
    short_vol = rr.rolling_std(log_ret, min(cfg.vol_windows))
    add("vol_ratio_short_long", short_vol / np.maximum(ref_vol, _EPS))

    # --- ATR -------------------------------------------------------------- #
    atr_abs = rr.atr(high, low, close, cfg.atr_window)
    add(f"atr_{cfg.atr_window}_frac", atr_abs / np.maximum(close, _EPS))

    # --- oscillators ------------------------------------------------------ #
    add(f"rsi_{cfg.rsi_window}", (rr.rsi(close, cfg.rsi_window) - 50.0) / 50.0)
    fast, slow, sig = cfg.macd
    macd_line, macd_sig, macd_hist = rr.macd(close, fast, slow, sig)
    scale = np.maximum(close, _EPS)
    add("macd_line", macd_line / scale)
    add("macd_signal", macd_sig / scale)
    add("macd_hist", macd_hist / scale)

    # --- moving-average distance ------------------------------------------ #
    for w in cfg.ma_windows:
        ma = rr.rolling_mean(close, w)
        add(f"ma_dist_{w}", log_close - np.log(np.maximum(ma, _EPS)))
        ema = rr.ewma(close, w)
        add(f"ema_dist_{w}", log_close - np.log(np.maximum(ema, _EPS)))

    # --- momentum, normalised by realised vol ----------------------------- #
    for w in cfg.momentum_windows:
        raw = log_close - rr.shift(log_close, w)
        vol_w = rr.rolling_std(log_ret, w) * np.sqrt(w)
        add(f"momentum_{w}", raw)
        add(f"momentum_z_{w}", raw / np.maximum(vol_w, _EPS))

    # --- position within trailing range ---------------------------------- #
    for w in cfg.ma_windows:
        hi = rr.rolling_max(high, w)
        lo = rr.rolling_min(low, w)
        add(f"range_pos_{w}", (close - lo) / np.maximum(hi - lo, _EPS))

    # --- calendar --------------------------------------------------------- #
    if cfg.include_time_encodings:
        for name, col in _time_encodings(data.ts).items():
            add(name, col)

    matrix = np.column_stack(cols) if cols else np.zeros((len(data), 0))
    return names, matrix


def _zscore_rolling(x: np.ndarray, window: int) -> np.ndarray:
    mean = rr.rolling_mean(x, window)
    std = rr.rolling_std(x, window)
    return (x - mean) / np.maximum(std, _EPS)


def _time_encodings(ts: np.ndarray) -> dict[str, np.ndarray]:
    """Cyclical time-of-day / day-of-week encodings.

    Sin/cos pairs rather than integers so midnight is adjacent to 23:50 and
    Sunday is adjacent to Monday.
    """
    epoch_s = ts.astype("datetime64[s]").astype("int64")
    seconds_of_day = epoch_s % 86_400
    tod = seconds_of_day / 86_400.0
    # 1970-01-01 was a Thursday, hence the +3 to make Monday == 0.
    dow = ((epoch_s // 86_400) + 3) % 7
    return {
        "tod_sin": np.sin(2 * np.pi * tod),
        "tod_cos": np.cos(2 * np.pi * tod),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype(np.float64),
    }
