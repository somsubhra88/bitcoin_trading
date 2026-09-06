"""Friction stress tests: re-run a strategy at 0.5x / 1x / 2x / 4x costs.

A strategy whose edge is real degrades smoothly as friction rises. A strategy
that is fitting the cost model collapses. That is the whole diagnostic, and it is
the single most informative robustness check available without more data, because
it needs no assumptions about the future — only about the exchange.

Multiplied together: fees, half-spread, slippage coefficient and floor, funding,
and latency. Latency is multiplied and *ceiled* rather than scaled continuously
because it is an integer number of bars; at 0.5x it stays at the floor of 1 bar,
so next-bar execution is never relaxed into same-bar execution by a stress
sweep — that would make the strategy look better under "lower friction", which is
not the question being asked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from archetype_trader.config import CostConfig, EnvConfig


@dataclass(frozen=True)
class StressPoint:
    """One friction multiplier and the resulting summary."""

    multiplier: float
    summary: dict[str, Any]

    @property
    def final_equity(self) -> float:
        return float(self.summary["final_equity"])


def scale_costs(costs: CostConfig, multiplier: float) -> CostConfig:
    """Every friction knob scaled by ``multiplier``; latency floored at 1 bar."""
    if multiplier <= 0:
        raise ValueError(f"stress multiplier must be positive, got {multiplier}")
    return costs.model_copy(
        update={
            "taker_fee_bps": costs.taker_fee_bps * multiplier,
            "maker_fee_bps": costs.maker_fee_bps * multiplier,
            "fallback_half_spread_bps": costs.fallback_half_spread_bps * multiplier,
            "slippage_coef": costs.slippage_coef * multiplier,
            "slippage_floor_bps": costs.slippage_floor_bps * multiplier,
            "funding_bps_per_8h": costs.funding_bps_per_8h * multiplier,
            "latency_bars": max(1, math.ceil(costs.latency_bars * multiplier)),
            # Tighter participation is part of "worse conditions": at 4x the
            # strategy may consume a quarter as much of each bar's volume, which
            # is what actually happens when liquidity dries up.
            "max_participation": min(1.0, costs.max_participation / multiplier),
        }
    )


def stress_env_config(env_cfg: EnvConfig, multiplier: float) -> EnvConfig:
    return env_cfg.model_copy(update={"costs": scale_costs(env_cfg.costs, multiplier)})


def collapse_verdict(
    points: list[StressPoint], *, initial_equity: float, tolerated_multiplier: float = 2.0
) -> dict[str, Any]:
    """Decide whether the strategy survives modest friction increases.

    "Survives" is defined narrowly and in advance: positive net PnL at every
    multiplier up to and including ``tolerated_multiplier``, and no bankruptcy or
    liquidation anywhere in the sweep. Anything else is reported as fragile. The
    threshold is a config choice, not a fitted one, and it is stated in the report
    so a reader can disagree with it explicitly.
    """
    by_mult = {p.multiplier: p for p in sorted(points, key=lambda p: p.multiplier)}
    survives = True
    reasons: list[str] = []
    for mult, point in by_mult.items():
        if mult > tolerated_multiplier:
            continue
        if point.final_equity <= initial_equity:
            survives = False
            reasons.append(f"non-positive net PnL at {mult}x friction")
        if point.summary.get("bankrupt"):
            survives = False
            reasons.append(f"bankrupt at {mult}x friction")
        if int(point.summary.get("n_liquidations", 0)) > 0:
            survives = False
            reasons.append(f"liquidated at {mult}x friction")
    base = by_mult.get(1.0)
    ratios = {
        f"equity_ratio_{mult}x": (
            (p.final_equity - initial_equity) / (base.final_equity - initial_equity)
            if base is not None and abs(base.final_equity - initial_equity) > 1e-9
            else 0.0
        )
        for mult, p in by_mult.items()
    }
    return {
        "survives_friction": survives,
        "tolerated_multiplier": tolerated_multiplier,
        "reasons": reasons,
        "final_equity_by_multiplier": {m: p.final_equity for m, p in by_mult.items()},
        **ratios,
    }
