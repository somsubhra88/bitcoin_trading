"""Portfolio-state features appended to the market observation at run time.

These cannot be precomputed: they depend on the agent's own trajectory. The
environment calls :func:`portfolio_features` each step with the state *as of the
current bar close*, so the vector is causal by construction.

The names/order in :data:`PORTFOLIO_FEATURES` are the contract between the
environment, the models and the saved feature definition — appending is safe,
reordering is not.
"""

from __future__ import annotations

import numpy as np

PORTFOLIO_FEATURES: tuple[str, ...] = (
    "target_position",
    "filled_position",
    "position_gap",
    "cash_frac",
    "equity_log_ratio",
    "unrealized_pnl_frac",
    "drawdown",
    "turnover_recent",
    "holding_time_frac",
    "risk_budget_left",
    "steps_left_frac",
    "leverage_used",
)

N_PORTFOLIO_FEATURES = len(PORTFOLIO_FEATURES)


def portfolio_features(
    *,
    target_position: float,
    filled_position: float,
    cash: float,
    equity: float,
    initial_equity: float,
    unrealized_pnl: float,
    drawdown: float,
    turnover_recent: float,
    holding_bars: int,
    horizon: int,
    steps_left: int,
    risk_budget_left: float,
    max_leverage: float,
) -> np.ndarray:
    """Return the portfolio-state vector, ordered as :data:`PORTFOLIO_FEATURES`.

    Every entry is dimensionless and roughly O(1) so it can share a scaler with
    the market block without dominating it.
    """
    eq = max(equity, 1e-9)
    init = max(initial_equity, 1e-9)
    hz = max(horizon, 1)
    vec = np.array(
        [
            target_position,
            filled_position,
            target_position - filled_position,
            cash / eq,
            np.log(eq / init),
            unrealized_pnl / eq,
            drawdown,
            turnover_recent,
            min(holding_bars / hz, 4.0),
            risk_budget_left,
            steps_left / hz,
            abs(filled_position) / max(max_leverage, 1e-9),
        ],
        dtype=np.float64,
    )
    if vec.shape[0] != N_PORTFOLIO_FEATURES:  # pragma: no cover - guards the contract
        raise AssertionError(
            f"portfolio vector length {vec.shape[0]} != {N_PORTFOLIO_FEATURES}; "
            "PORTFOLIO_FEATURES and this function must stay in sync"
        )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def zero_portfolio_features() -> np.ndarray:
    """Neutral portfolio state (flat, full equity, no history)."""
    return portfolio_features(
        target_position=0.0,
        filled_position=0.0,
        cash=1.0,
        equity=1.0,
        initial_equity=1.0,
        unrealized_pnl=0.0,
        drawdown=0.0,
        turnover_recent=0.0,
        holding_bars=0,
        horizon=1,
        steps_left=0,
        risk_budget_left=1.0,
        max_leverage=1.0,
    )
