"""Turn a target BTC exposure into at most one order.

Shared by the paper ledger and the backtest so that both apply identical
sizing, banding, lot rounding, risk limits and costs. Nothing here touches the
database or an exchange.

Rules, in order:

1. Equity is cash plus BTC marked at the bid (what it could be sold for).
2. The desired BTC value is ``target_exposure * equity``, capped by
   ``MAX_POSITION_JPY``.
3. Trend entries/exits (``force=True``), a first entry, and a full exit always
   trade. A volatility-only resize trades only when it moves the holding by at
   least ``STRATEGY_REBALANCE_BAND_PERCENT`` of the current holding — small
   resizes would mostly pay fees.
4. Buys are capped by ``MAX_ORDER_JPY`` and available cash, and are blocked once
   the daily loss limit is reached. Sells reduce risk and are never blocked by
   order-size limits.
5. Sizes are rounded down to the exchange lot step; anything under the minimum
   lot is not sent. A sell that would leave less than one minimum lot sells the
   whole holding instead of leaving unsellable dust.
6. ``STRATEGY_EXIT`` sells must clear ``MIN_PROFIT_PERCENT`` net of costs
   (PROJECT.md §21). Stop and volatility exits are risk controls and are exempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from btc_trader.config import Settings
from btc_trader.enums import ExitReason, Side

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_TEN_THOUSAND = Decimal("10000")


@dataclass(frozen=True, slots=True)
class RebalanceRequest:
    cash: Decimal
    btc: Decimal
    bid: Decimal
    ask: Decimal
    target_exposure: Decimal
    #: Bypass the band: the trend signal itself changed and has not been acted on.
    force: bool = False
    #: Recorded on a sell. Only ``STRATEGY_EXIT`` is subject to the profit gate.
    reduction_reason: ExitReason = ExitReason.RISK_EXIT
    average_entry_price: Decimal | None = None
    daily_loss_reached: bool = False
    cost_multiplier: Decimal = _ONE
    apply_jpy_limits: bool = True


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    side: Side | None
    size: Decimal
    price: Decimal
    fee: Decimal
    equity: Decimal
    current_exposure: Decimal
    target_exposure: Decimal
    exit_reason: ExitReason | None
    reason: str

    @property
    def trades(self) -> bool:
        return self.side is not None and self.size > 0


def normalise_order_size(
    size: Decimal, settings: Settings, price: Decimal | None = None
) -> Decimal | None:
    """Apply the lot step, minimum size and minimum notional to an order."""
    if size <= 0:
        return None
    step = settings.order_size_step_btc
    if step > 0:
        size = (size / step).to_integral_value(rounding=ROUND_DOWN) * step
    if settings.enforce_exchange_minimums:
        if size < settings.min_order_size_btc:
            return None
        if (
            price is not None
            and settings.min_order_notional_jpy > 0
            and size * price < settings.min_order_notional_jpy
        ):
            return None
    return size if size > 0 else None


def plan_rebalance(request: RebalanceRequest, settings: Settings) -> RebalancePlan:
    fee_rate = settings.taker_fee_percent / _HUNDRED * request.cost_multiplier
    slippage = settings.paper_slippage_bps / _TEN_THOUSAND * request.cost_multiplier
    bid, ask = request.bid, request.ask
    current_value = request.btc * bid
    equity = request.cash + current_value
    current_exposure = current_value / equity if equity > 0 else _ZERO
    target = max(_ZERO, request.target_exposure)

    def hold(reason: str, side: Side | None = None) -> RebalancePlan:
        return RebalancePlan(
            side=None,
            size=_ZERO,
            price=bid if side is Side.SELL else ask,
            fee=_ZERO,
            equity=equity,
            current_exposure=current_exposure,
            target_exposure=target,
            exit_reason=None,
            reason=reason,
        )

    if equity <= 0 or bid <= 0 or ask <= 0:
        return hold("no equity or no executable price")

    desired_value = target * equity
    if request.apply_jpy_limits and settings.max_position_jpy > 0:
        desired_value = min(desired_value, settings.max_position_jpy)
    delta = desired_value - current_value

    full_exit = target == 0 and request.btc > 0
    entry = request.btc == 0 and delta > 0
    band = settings.strategy_rebalance_band_percent / _HUNDRED
    if not (request.force or full_exit or entry) and abs(delta) < band * current_value:
        return hold(f"within rebalance band: holding {current_exposure:.1%}, target {target:.1%}")
    if delta == 0:
        return hold("already at target")

    if delta > 0:
        if request.daily_loss_reached:
            return hold("daily loss limit reached; buys paused")
        price = ask * (_ONE + slippage)
        notional = delta
        if request.apply_jpy_limits and settings.max_order_jpy > 0:
            notional = min(notional, settings.max_order_jpy)
        notional = min(notional, request.cash / (_ONE + fee_rate))
        size = normalise_order_size(notional / price, settings, price)
        if size is None:
            return hold(
                f"buy of ¥{notional:,.0f} is below the exchange minimum "
                f"({settings.min_order_size_btc} BTC)",
                Side.BUY,
            )
        return RebalancePlan(
            side=Side.BUY,
            size=size,
            price=price,
            fee=price * size * fee_rate,
            equity=equity,
            current_exposure=current_exposure,
            target_exposure=target,
            exit_reason=None,
            reason=f"raise exposure {current_exposure:.1%} -> {target:.1%}",
        )

    price = bid * (_ONE - slippage)
    size = request.btc if full_exit else min(request.btc, -delta / bid)
    if request.btc - size < settings.min_order_size_btc:
        size = request.btc
    rounded = normalise_order_size(size, settings, price)
    if rounded is None:
        return hold("holding is below the exchange minimum lot; nothing sellable", Side.SELL)
    reason = request.reduction_reason
    if reason is ExitReason.STRATEGY_EXIT and request.average_entry_price:
        proceeds = price * (_ONE - fee_rate)
        net_return = proceeds / request.average_entry_price - _ONE
        if net_return < settings.min_profit_percent / _HUNDRED:
            return hold(
                f"strategy sell blocked: net {net_return:.2%} is below the "
                f"{settings.min_profit_percent}% minimum profit",
                Side.SELL,
            )
    return RebalancePlan(
        side=Side.SELL,
        size=rounded,
        price=price,
        fee=price * rounded * fee_rate,
        equity=equity,
        current_exposure=current_exposure,
        target_exposure=target,
        exit_reason=reason,
        reason=f"reduce exposure {current_exposure:.1%} -> {target:.1%} ({reason.value})",
    )


__all__ = ["RebalancePlan", "RebalanceRequest", "normalise_order_size", "plan_rebalance"]
