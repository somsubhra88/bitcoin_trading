"""Deterministic rule baselines: cash, buy-and-hold, EMA cross, MACD, vol-target trend.

Every signal is computed once over the whole series and then *shifted by one bar*
before use, so the position held during bar ``t`` was decided from data up to
``t-1`` and executed at the open of ``t`` — the same timing the learned policies
get. :func:`assert_causal_signal` is the mechanical check, and the leakage test
calls it with a deliberately forward-looking signal to confirm it fires.

Baselines exist to be beaten honestly. They are charged identical fees, spread,
slippage, funding and latency, and they run through the same risk shield, so a
comparison cannot flatter the learned policy by giving the baseline worse
execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from archetype_trader.config import BaselineConfig, EnvConfig, FeatureConfig, ShieldConfig
from archetype_trader.data.types import MarketData
from archetype_trader.env.trading_env import TradingEnv
from archetype_trader.features.pipeline import FeatureMatrix, FeatureScaler
from archetype_trader.features.rolling import ewma, rolling_std, shift
from archetype_trader.features.rolling import macd as macd_lines

#: Signature of a baseline signal: market data plus config -> target position per bar.
SignalFn = Callable[[MarketData, BaselineConfig, FeatureConfig], np.ndarray]


@dataclass
class BaselineRun:
    """One baseline evaluated over one segment."""

    name: str
    summary: dict[str, Any]
    targets: np.ndarray

    @property
    def final_equity(self) -> float:
        return float(self.summary["final_equity"])


def assert_causal_signal(
    fn: SignalFn,
    data: MarketData,
    cfg: BaselineConfig,
    fcfg: FeatureConfig,
    *,
    name: str = "signal",
    probe: int | None = None,
) -> None:
    """Fail if ``fn`` reacts to a bar it should not have seen.

    Multiplies every price from ``probe`` onward by 1.5 and recomputes. Because
    the position held during bar ``t`` comes from ``signal[t]``, which may only
    use closes up to ``t-1``, nothing in ``signal[: probe + 1]`` is allowed to
    move. A signal that peeks changes an earlier entry and trips the assertion.
    The leakage test calls this with a deliberately forward-looking signal to
    confirm it fires rather than silently passing.
    """
    base = np.asarray(fn(data, cfg, fcfg), dtype=np.float64)
    if base.shape[0] != len(data):
        raise ValueError(f"{name}: signal {base.shape} does not align with {len(data)} bars")
    p = len(data) // 2 if probe is None else int(probe)
    bumped = replace(
        data,
        open=data.open.copy(),
        high=data.high.copy(),
        low=data.low.copy(),
        close=data.close.copy(),
    )
    for arr in (bumped.open, bumped.high, bumped.low, bumped.close):
        arr[p:] *= 1.5
    other = np.asarray(fn(bumped, cfg, fcfg), dtype=np.float64)
    a, b = np.nan_to_num(base[: p + 1]), np.nan_to_num(other[: p + 1])
    if not np.allclose(a, b, atol=1e-12, rtol=0.0):
        bad = int(np.argmax(np.abs(a - b)))
        raise AssertionError(
            f"{name}: signal[{bad}] changed from {a[bad]} to {b[bad]} when bars >= {p} "
            f"were perturbed — the signal is not causal"
        )


def cash_signal(data: MarketData, cfg: BaselineConfig, fcfg: FeatureConfig) -> np.ndarray:
    """Hold nothing. The floor every strategy must clear after costs."""
    return np.zeros(len(data), dtype=np.float64)


def buy_and_hold_signal(data: MarketData, cfg: BaselineConfig, fcfg: FeatureConfig) -> np.ndarray:
    """Fully long from the first tradable bar. The bar every BTC strategy must clear."""
    return np.ones(len(data), dtype=np.float64)


def ema_cross_signal(data: MarketData, cfg: BaselineConfig, fcfg: FeatureConfig) -> np.ndarray:
    fast = ewma(data.close, cfg.ema_fast)
    slow = ewma(data.close, cfg.ema_slow)
    raw = np.sign(np.nan_to_num(fast - slow, nan=0.0))
    return shift(raw, 1, fill=0.0)


def macd_signal(data: MarketData, cfg: BaselineConfig, fcfg: FeatureConfig) -> np.ndarray:
    fast, slow, sig_span = fcfg.macd
    line, sig, _ = macd_lines(data.close, fast, slow, sig_span)
    raw = np.sign(np.nan_to_num(line - sig, nan=0.0))
    return shift(raw, 1, fill=0.0)


def vol_target_trend_signal(
    data: MarketData, cfg: BaselineConfig, fcfg: FeatureConfig
) -> np.ndarray:
    """Trend direction scaled to a volatility target, the standard managed-futures rule.

    This is the baseline most likely to be competitive on BTC, which is why it is
    included: a trend signal sized by inverse realised volatility captures much of
    what a discretionary crypto trend follower does, and beating it is a much
    stronger claim than beating a raw EMA cross.
    """
    fast = ewma(data.close, cfg.ema_fast)
    slow = ewma(data.close, cfg.ema_slow)
    direction = np.sign(np.nan_to_num(fast - slow, nan=0.0))
    logret = np.zeros(len(data), dtype=np.float64)
    logret[1:] = np.diff(np.log(data.close))
    vol = rolling_std(logret, max(fcfg.vol_windows))
    bars_per_year = 365.0 * 24.0 * 60.0 / max(data.bar_minutes, 1)
    ann = np.nan_to_num(vol, nan=0.0) * np.sqrt(bars_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(ann > 1e-9, cfg.vol_target_annual / ann, 0.0)
    scale = np.clip(np.nan_to_num(scale, nan=0.0), 0.0, 1.0)
    return shift(direction * scale, 1, fill=0.0)


SIGNALS: dict[str, SignalFn] = {
    "cash": cash_signal,
    "buy_and_hold": buy_and_hold_signal,
    "ema_cross": ema_cross_signal,
    "macd": macd_signal,
    "vol_target_trend": vol_target_trend_signal,
}


def snap_to_grid(targets: np.ndarray, grid: tuple[float, ...]) -> np.ndarray:
    """Nearest admissible target position for each bar, as action indices."""
    g = np.asarray(grid, dtype=np.float64)
    return np.abs(targets[:, None] - g[None, :]).argmin(axis=1).astype(np.int64)


def run_targets(env: TradingEnv, targets: np.ndarray, *, start_index: int) -> dict[str, Any]:
    """Step ``env`` through a precomputed target-position series."""
    actions = snap_to_grid(targets, env.positions)
    env.reset(options={"start_index": start_index})
    while True:
        # env._t is the bar being decided on; the target was fixed from data < t.
        idx = int(actions[env._t])
        _, _, term, trunc, _ = env.step(idx)
        if term or trunc:
            break
    return env.episode_summary()


def run_rule_baseline(
    name: str,
    data: MarketData,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    env_cfg: EnvConfig,
    shield_cfg: ShieldConfig,
    baseline_cfg: BaselineConfig,
    feature_cfg: FeatureConfig,
    *,
    start_index: int,
    end_index: int,
) -> BaselineRun:
    """Evaluate one named rule baseline over ``[start_index, end_index)``."""
    if name not in SIGNALS:
        raise KeyError(f"unknown rule baseline {name!r}; have {sorted(SIGNALS)}")
    targets = SIGNALS[name](data, baseline_cfg, feature_cfg)
    assert_causal_signal(SIGNALS[name], data, baseline_cfg, feature_cfg, name=name)
    # buy_and_hold and cash have no warm-up; the indicator rules do, and holding a
    # position during their warm-up would be a look-ahead-free but dishonest edge.
    if name not in ("cash", "buy_and_hold"):
        targets = targets.copy()
        targets[: features.valid_from] = 0.0
    env = TradingEnv(
        data,
        features,
        scaler,
        env_cfg,
        shield_cfg,
        start_index=start_index,
        end_index=end_index,
    )
    summary = run_targets(env, targets, start_index=start_index)
    summary["name"] = name
    summary["mean_position"] = float(np.mean(summary["positions"])) if summary["n_steps"] else 0.0
    return BaselineRun(name=name, summary=summary, targets=targets)
