from __future__ import annotations

from decimal import Decimal

import pytest

from btc_trader.config import Settings
from btc_trader.enums import ExitReason, Side
from btc_trader.rebalance import RebalanceRequest, normalise_order_size, plan_rebalance

PRICE = Decimal("10000000")


def _settings(settings: Settings, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "max_position_jpy": Decimal("0"),
        "max_order_jpy": Decimal("0"),
        "paper_slippage_bps": Decimal("0"),
    }
    values.update(overrides)
    return settings.model_copy(update=values)


def _request(**overrides: object) -> RebalanceRequest:
    values: dict[str, object] = {
        "cash": Decimal("1000000"),
        "btc": Decimal("0"),
        "bid": PRICE,
        "ask": PRICE,
        "target_exposure": Decimal("0.5"),
    }
    values.update(overrides)
    return RebalanceRequest(**values)  # type: ignore[arg-type]


def test_an_entry_buys_the_target_share_of_equity_net_of_fees(settings: Settings) -> None:
    plan = plan_rebalance(_request(), _settings(settings))

    assert plan.side is Side.BUY
    assert plan.size == Decimal("0.05")
    assert plan.fee == PRICE * plan.size * Decimal("0.0015")
    assert plan.price * plan.size + plan.fee <= Decimal("1000000")


def test_a_buy_never_spends_more_cash_than_exists(settings: Settings) -> None:
    plan = plan_rebalance(_request(target_exposure=Decimal("1")), _settings(settings))

    assert plan.side is Side.BUY
    assert plan.price * plan.size + plan.fee <= Decimal("1000000")


def test_small_volatility_resizes_stay_inside_the_band(settings: Settings) -> None:
    held = _request(btc=Decimal("0.05"), cash=Decimal("500000"), target_exposure=Decimal("0.55"))

    within = plan_rebalance(held, _settings(settings))
    forced = plan_rebalance(
        _request(
            btc=Decimal("0.05"), cash=Decimal("500000"), target_exposure=Decimal("0.55"), force=True
        ),
        _settings(settings),
    )

    assert not within.trades
    assert "rebalance band" in within.reason
    assert forced.side is Side.BUY


def test_a_large_resize_trades_without_force(settings: Settings) -> None:
    plan = plan_rebalance(
        _request(btc=Decimal("0.05"), cash=Decimal("500000"), target_exposure=Decimal("0.3")),
        _settings(settings),
    )

    assert plan.side is Side.SELL
    assert plan.size == Decimal("0.02")
    assert plan.exit_reason is ExitReason.RISK_EXIT


def test_a_full_exit_sells_everything_with_its_reason(settings: Settings) -> None:
    plan = plan_rebalance(
        _request(
            btc=Decimal("0.0123"),
            target_exposure=Decimal("0"),
            reduction_reason=ExitReason.TRAILING_STOP,
        ),
        _settings(settings, strategy_rebalance_band_percent=Decimal("100")),
    )

    assert plan.side is Side.SELL
    assert plan.size == Decimal("0.0123")
    assert plan.exit_reason is ExitReason.TRAILING_STOP


def test_position_cap_limits_the_target(settings: Settings) -> None:
    plan = plan_rebalance(
        _request(target_exposure=Decimal("1")),
        _settings(settings, max_position_jpy=Decimal("200000")),
    )

    assert plan.side is Side.BUY
    assert plan.size == Decimal("0.02")


def test_order_cap_limits_buys_but_never_blocks_an_exit(settings: Settings) -> None:
    capped = _settings(settings, max_order_jpy=Decimal("100000"))

    buy = plan_rebalance(_request(), capped)
    sell = plan_rebalance(
        _request(btc=Decimal("0.08"), cash=Decimal("200000"), target_exposure=Decimal("0")),
        capped,
    )

    assert buy.size == Decimal("0.01")
    assert sell.size == Decimal("0.08")


def test_daily_loss_limit_blocks_buys_only(settings: Settings) -> None:
    buy = plan_rebalance(_request(daily_loss_reached=True), _settings(settings))
    sell = plan_rebalance(
        _request(btc=Decimal("0.05"), target_exposure=Decimal("0"), daily_loss_reached=True),
        _settings(settings),
    )

    assert not buy.trades
    assert "daily loss limit" in buy.reason
    assert sell.side is Side.SELL


def test_orders_below_the_exchange_minimum_are_not_sent(settings: Settings) -> None:
    plan = plan_rebalance(
        _request(cash=Decimal("50000"), target_exposure=Decimal("0.1")), _settings(settings)
    )

    assert not plan.trades
    assert "below the exchange minimum" in plan.reason


def test_a_sell_never_leaves_unsellable_dust(settings: Settings) -> None:
    plan = plan_rebalance(
        _request(btc=Decimal("0.0015"), cash=Decimal("0"), target_exposure=Decimal("0.5")),
        _settings(settings, strategy_rebalance_band_percent=Decimal("0")),
    )

    assert plan.side is Side.SELL
    assert plan.size == Decimal("0.0015")


def test_strategy_exits_must_clear_the_minimum_profit_but_risk_exits_need_not(
    settings: Settings,
) -> None:
    losing = {
        "btc": Decimal("0.05"),
        "target_exposure": Decimal("0"),
        "average_entry_price": PRICE * Decimal("1.05"),
    }

    strategy_exit = plan_rebalance(
        _request(**losing, reduction_reason=ExitReason.STRATEGY_EXIT), _settings(settings)
    )
    risk_exit = plan_rebalance(
        _request(**losing, reduction_reason=ExitReason.TRAILING_STOP), _settings(settings)
    )
    profitable = plan_rebalance(
        _request(
            btc=Decimal("0.05"),
            target_exposure=Decimal("0"),
            average_entry_price=PRICE * Decimal("0.95"),
            reduction_reason=ExitReason.STRATEGY_EXIT,
        ),
        _settings(settings),
    )

    assert not strategy_exit.trades
    assert "minimum profit" in strategy_exit.reason
    assert risk_exit.side is Side.SELL
    assert profitable.side is Side.SELL


def test_costs_scale_with_the_stress_multiplier(settings: Settings) -> None:
    base = plan_rebalance(_request(), _settings(settings, paper_slippage_bps=Decimal("10")))
    doubled = plan_rebalance(
        _request(cost_multiplier=Decimal("2")),
        _settings(settings, paper_slippage_bps=Decimal("10")),
    )

    assert doubled.price - PRICE == pytest.approx(2 * (base.price - PRICE))
    assert doubled.fee / (doubled.price * doubled.size) == Decimal("0.003")


def test_lot_rounding_and_minimums(settings: Settings) -> None:
    assert normalise_order_size(Decimal("0.0012345678912"), settings) == Decimal("0.00123456")
    assert normalise_order_size(Decimal("0.0009"), settings) is None
    lenient = settings.model_copy(update={"enforce_exchange_minimums": False})
    assert normalise_order_size(Decimal("0.0009"), lenient) == Decimal("0.0009")
    notional = settings.model_copy(update={"min_order_notional_jpy": Decimal("20000")})
    assert normalise_order_size(Decimal("0.001"), notional, PRICE) is None
