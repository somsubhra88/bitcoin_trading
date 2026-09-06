"""Deterministic risk shield.

The shield is a pure function of *observable* state and never learns. It sits
between any policy (archetype, adapter, baseline, or a human) and the execution
simulator, so its interventions are measurable and identical in backtest and
paper trading.

Both the proposed and the executed target are returned so the environment can
log an intervention rate. A shield that is never triggered is as suspicious as
one that is always triggered, and only logging makes that visible.

Checks are ordered hard-stop first: a kill switch or breached loss limit
short-circuits before the softer sizing rules, and the reason string names the
binding constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from archetype_trader.config import ShieldConfig


class ShieldReason(StrEnum):
    OK = "ok"
    KILL_SWITCH = "kill_switch"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    COOLDOWN = "cooldown"
    STALE_DATA = "stale_data"
    WIDE_SPREAD = "wide_spread"
    THIN_LIQUIDITY = "thin_liquidity"
    SLIPPAGE_TOO_HIGH = "slippage_too_high"
    LIQUIDATION_DISTANCE = "liquidation_distance"
    VOL_TARGET_SCALED = "vol_target_scaled"
    EXPOSURE_CLAMPED = "exposure_clamped"
    LEVERAGE_CLAMPED = "leverage_clamped"


@dataclass
class ShieldState:
    """Mutable shield memory across a single episode."""

    cooldown_left: int = 0
    loss_streak: int = 0
    day_start_equity: float = 0.0
    day_index: int = -1
    flattened: bool = False
    n_proposals: int = 0
    n_interventions: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def intervention_rate(self) -> float:
        return self.n_interventions / self.n_proposals if self.n_proposals else 0.0

    def record(self, intervened: bool, reasons: tuple[str, ...]) -> None:
        self.n_proposals += 1
        if intervened:
            self.n_interventions += 1
        for r in reasons:
            self.reason_counts[r] = self.reason_counts.get(r, 0) + 1


@dataclass(frozen=True)
class ShieldDecision:
    """What the shield allowed, and why it differs from the proposal."""

    proposed_position: float
    executed_position: float
    reasons: tuple[str, ...]
    forced_flat: bool
    scale_applied: float

    @property
    def intervened(self) -> bool:
        return abs(self.executed_position - self.proposed_position) > 1e-12

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else ShieldReason.OK.value


class RiskShield:
    """Applies :class:`ShieldConfig` limits to a proposed target position."""

    def __init__(
        self,
        cfg: ShieldConfig,
        *,
        mode: str,
        bars_per_year: int,
        maintenance_margin: float = 0.005,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        self.bars_per_year = bars_per_year
        # Owned by EnvConfig (it is an exchange property, not a risk policy) and
        # injected here for the liquidation-distance check.
        self.maintenance_margin = maintenance_margin

    def reset(self, initial_equity: float) -> ShieldState:
        return ShieldState(day_start_equity=initial_equity, day_index=-1)

    def apply(
        self,
        state: ShieldState,
        *,
        proposed_position: float,
        current_position: float,
        equity: float,
        drawdown: float,
        realized_vol_per_bar: float,
        spread_bps: float,
        bar_volume: float,
        est_slippage_bps: float,
        is_stale: bool,
        day_index: int,
        last_step_return: float,
        max_positions: tuple[float, ...],
    ) -> ShieldDecision:
        """Return the executable target position for this bar.

        All inputs are observable at the decision time. ``last_step_return`` is
        the previous step's realised net return, used for the loss-streak
        cooldown.
        """
        reasons: list[str] = []
        if not self.cfg.enabled:
            state.record(False, ())
            return ShieldDecision(proposed_position, proposed_position, (), False, 1.0)

        # --- roll the daily loss window ----------------------------------- #
        if day_index != state.day_index:
            state.day_index = day_index
            state.day_start_equity = equity

        # --- loss streak / cooldown bookkeeping --------------------------- #
        if last_step_return < 0:
            state.loss_streak += 1
        elif last_step_return > 0:
            state.loss_streak = 0
        if state.loss_streak >= self.cfg.loss_streak_trigger:
            state.cooldown_left = max(state.cooldown_left, self.cfg.cooldown_bars)
            state.loss_streak = 0

        # --- hard stops: force flat --------------------------------------- #
        hard: str | None = None
        if self.cfg.kill_switch:
            hard = ShieldReason.KILL_SWITCH.value
        elif drawdown >= self.cfg.max_drawdown:
            hard = ShieldReason.MAX_DRAWDOWN.value
        elif state.day_start_equity > 0 and (
            equity / state.day_start_equity - 1.0 <= -self.cfg.max_daily_loss
        ):
            hard = ShieldReason.MAX_DAILY_LOSS.value
        if hard is not None:
            state.flattened = True
            state.cooldown_left = max(state.cooldown_left, self.cfg.cooldown_bars)
            reasons.append(hard)
            decision = ShieldDecision(proposed_position, 0.0, tuple(reasons), True, 0.0)
            state.record(decision.intervened, decision.reasons)
            return decision

        target = float(proposed_position)

        # --- market-condition gates: block *increases* only --------------- #
        gates: list[str] = []
        if is_stale:
            gates.append(ShieldReason.STALE_DATA.value)
        if spread_bps > self.cfg.max_spread_bps:
            gates.append(ShieldReason.WIDE_SPREAD.value)
        if bar_volume < self.cfg.min_bar_volume:
            gates.append(ShieldReason.THIN_LIQUIDITY.value)
        if est_slippage_bps > self.cfg.max_est_slippage_bps:
            gates.append(ShieldReason.SLIPPAGE_TOO_HIGH.value)
        if state.cooldown_left > 0:
            gates.append(ShieldReason.COOLDOWN.value)
        if self.mode == "perp":
            distance = self._liquidation_distance(current_position)
            if distance < self.cfg.min_liquidation_distance:
                gates.append(ShieldReason.LIQUIDATION_DISTANCE.value)

        if gates and self._increases_risk(target, current_position):
            reasons.extend(gates)
            target = current_position

        # --- volatility targeting ----------------------------------------- #
        scale = 1.0
        if self.cfg.vol_target_annual is not None and target != 0.0:
            per_bar_target = self.cfg.vol_target_annual / math.sqrt(max(self.bars_per_year, 1))
            scale = per_bar_target / max(realized_vol_per_bar, 1e-8)
            scale = min(max(scale, 0.0), 1.0)  # de-risk only, never lever up
            if scale < 1.0 - 1e-9:
                reasons.append(ShieldReason.VOL_TARGET_SCALED.value)
                target = target * scale

        # --- exposure and leverage caps ----------------------------------- #
        clamped = float(np.clip(target, -self.cfg.max_net_exposure, self.cfg.max_net_exposure))
        if abs(clamped - target) > 1e-12:
            reasons.append(ShieldReason.EXPOSURE_CLAMPED.value)
            target = clamped
        if abs(target) > self.cfg.max_gross_exposure:
            reasons.append(ShieldReason.EXPOSURE_CLAMPED.value)
            target = math.copysign(self.cfg.max_gross_exposure, target)
        if abs(target) > self.cfg.max_leverage:
            reasons.append(ShieldReason.LEVERAGE_CLAMPED.value)
            target = math.copysign(self.cfg.max_leverage, target)
        if self.mode == "spot" and target < 0.0:
            reasons.append(ShieldReason.EXPOSURE_CLAMPED.value)
            target = 0.0

        # Snap back onto the discrete action grid so the executed target is
        # always a position the environment can actually represent.
        target = _snap(target, max_positions)

        if state.cooldown_left > 0:
            state.cooldown_left -= 1

        decision = ShieldDecision(
            proposed_position=float(proposed_position),
            executed_position=float(target),
            reasons=tuple(dict.fromkeys(reasons)),
            forced_flat=False,
            scale_applied=scale,
        )
        state.record(decision.intervened, decision.reasons)
        return decision

    def _liquidation_distance(self, position: float) -> float:
        """Fractional adverse price move that would wipe out margin.

        For exposure ``p`` (notional/equity), equity hits the maintenance
        requirement after an adverse move of roughly ``(1 - mm*|p|) / |p|``.
        Flat or tiny positions are effectively un-liquidatable.
        """
        p = abs(position)
        if p <= 1e-9:
            return float("inf")
        return max(0.0, (1.0 - self.maintenance_margin * p) / p)

    @staticmethod
    def _increases_risk(target: float, current: float) -> bool:
        """True when moving to ``target`` raises gross exposure or flips sign."""
        if abs(target) > abs(current) + 1e-12:
            return True
        return target * current < -1e-12


def _snap(value: float, grid: tuple[float, ...]) -> float:
    """Nearest representable target position, biased toward less risk on ties."""
    if not grid:
        return value
    arr = np.asarray(grid, dtype=np.float64)
    dist = np.abs(arr - value)
    best = float(dist.min())
    candidates = arr[dist <= best + 1e-12]
    # Tie-break toward the smaller absolute exposure.
    return float(candidates[np.argmin(np.abs(candidates))])
