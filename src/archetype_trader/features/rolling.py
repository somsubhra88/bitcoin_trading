"""Vectorised causal rolling primitives.

Every function here returns an array where element ``t`` depends only on input
elements ``<= t``. Warm-up positions are ``NaN``, never back-filled: silently
filling them would make an under-determined statistic look like a real
observation, and the pipeline drops the warm-up region explicitly instead.

Cumulative-sum tricks are used over ``np.lib.stride_tricks`` windows where the
result is numerically safe, because the arrays are long (10-minute bars over
years) and materialising windows costs ``O(T * W)`` memory.
"""

from __future__ import annotations

import numpy as np


def _validate(x: np.ndarray, window: int) -> None:
    if x.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {x.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")


def rolling_windows(x: np.ndarray, window: int) -> np.ndarray:
    """Read-only causal window view of shape ``(T - window + 1, window)``."""
    _validate(x, window)
    return np.lib.stride_tricks.sliding_window_view(x, window)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` bars ending at ``t`` (inclusive)."""
    _validate(x, window)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def rolling_std(x: np.ndarray, window: int, *, ddof: int = 0) -> np.ndarray:
    """Trailing standard deviation; ``NaN`` until ``window`` samples exist."""
    _validate(x, window)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] < window or window - ddof <= 0:
        return out
    win = rolling_windows(x, window)
    # Two-pass within the window: cheap enough here and avoids the catastrophic
    # cancellation that E[x^2] - E[x]^2 shows on price-level inputs.
    out[window - 1 :] = win.std(axis=1, ddof=ddof)
    return out


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    _validate(x, window)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out[window - 1 :] = csum[window:] - csum[:-window]
    return out


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    _validate(x, window)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] >= window:
        out[window - 1 :] = rolling_windows(x, window).max(axis=1)
    return out


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    _validate(x, window)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] >= window:
        out[window - 1 :] = rolling_windows(x, window).min(axis=1)
    return out


def ewma(x: np.ndarray, span: int, *, warmup_nan: bool = True) -> np.ndarray:
    """Recursive EWMA with ``alpha = 2 / (span + 1)`` (pandas ``adjust=False``).

    Seeded with ``x[0]``; the first ``span - 1`` values are marked ``NaN`` when
    ``warmup_nan`` so the caller cannot mistake the transient for a converged
    average.
    """
    _validate(x, span)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    acc = x[0]
    for i in range(x.shape[0]):
        acc = alpha * x[i] + (1.0 - alpha) * acc if i else x[0]
        out[i] = acc
    if warmup_nan:
        out[: min(span - 1, out.shape[0])] = np.nan
    return out


def wilder_ema(x: np.ndarray, window: int) -> np.ndarray:
    """Wilder smoothing (``alpha = 1 / window``), seeded with the first mean.

    Used by RSI and ATR so they match the standard definitions rather than a
    simple moving average.
    """
    _validate(x, window)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    n = x.shape[0]
    if n < window:
        return out
    acc = float(np.mean(x[:window]))
    out[window - 1] = acc
    alpha = 1.0 / window
    for i in range(window, n):
        acc = acc + alpha * (x[i] - acc)
        out[i] = acc
    return out


def rsi(close: np.ndarray, window: int) -> np.ndarray:
    """Wilder RSI in [0, 100]; ``NaN`` during warm-up."""
    _validate(close, window)
    delta = np.zeros_like(close, dtype=np.float64)
    delta[1:] = np.diff(close)
    gain = wilder_ema(np.maximum(delta, 0.0)[1:], window)
    loss = wilder_ema(np.maximum(-delta, 0.0)[1:], window)
    out = np.full(close.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
        val = 100.0 - 100.0 / (1.0 + rs)
    # loss == 0 with gain > 0 is an unbroken up-run: RSI is 100 by definition.
    val = np.where((loss == 0) & (gain > 0), 100.0, val)
    val = np.where((loss == 0) & (gain == 0), 50.0, val)
    out[1:] = val
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range; element 0 falls back to the bar range (no previous close)."""
    prev_close = np.concatenate(([close[0]], close[:-1]))
    return np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    return wilder_ema(true_range(high, low, close), window)


def macd(
    close: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(macd_line, signal_line, histogram)``, all warm-up masked."""
    if fast >= slow:
        raise ValueError(f"macd fast must be < slow, got {fast} >= {slow}")
    line = ewma(close, fast) - ewma(close, slow)
    # ``ewma`` needs finite input; smooth the finite tail and re-mask.
    finite = np.isfinite(line)
    sig = np.full_like(line, np.nan)
    if finite.any():
        first = int(np.argmax(finite))
        tail = ewma(line[first:], signal)
        sig[first:] = tail
    return line, sig, line - sig


def drawdown(equity: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak, in [0, 1]."""
    peak = np.maximum.accumulate(equity)
    return 1.0 - equity / np.maximum(peak, 1e-12)


def shift(x: np.ndarray, periods: int = 1, fill: float = np.nan) -> np.ndarray:
    """Shift forward in time: ``out[t] = x[t - periods]``.

    Positive ``periods`` only. This is the single place a lag is introduced, so
    the leakage tests have one function to reason about.
    """
    if periods < 0:
        raise ValueError("shift only supports non-negative (causal) lags")
    out = np.full(x.shape, fill, dtype=np.float64)
    if periods == 0:
        out[:] = x
    elif periods < x.shape[0]:
        out[periods:] = x[:-periods]
    return out
