"""Live-trading readiness check.

    python -m btc_trader.scripts.live_check
    python -m btc_trader.scripts.live_check --test-order buy --confirm-real-money

Read-only by default: verifies the API key's permissions, the account's
balances and fee tier, the exchange state, and every live safety gate, then
prints what the bot would trade with. Nothing is written or ordered.

``--test-order`` sends ONE minimum-size (0.001 BTC) limit IOC order through the
same submission path the bot uses, so a first deliberate trade can confirm the
whole lifecycle (PROJECT.md §50 Phase 11). It spends real money and requires
``--confirm-real-money``; it ignores the trading-mode switches but never the
kill switch.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from btc_trader.config import Settings, get_settings
from btc_trader.db.session import dispose_engine, session_scope
from btc_trader.enums import Side
from btc_trader.errors import BtcTraderError
from btc_trader.exchange.bitflyer import BitflyerClient
from btc_trader.exchange.models import OrderRequest
from btc_trader.live import in_maintenance_window, load_account, submit_order
from btc_trader.monitoring.logging import configure_logging
from btc_trader.runtime_settings import get_runtime_settings

REQUIRED_PERMISSIONS = (
    "/v1/me/getbalance",
    "/v1/me/gettradingcommission",
    "/v1/me/sendchildorder",
    "/v1/me/getchildorders",
    "/v1/me/getexecutions",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check live-trading readiness")
    parser.add_argument("--test-order", choices=["buy", "sell"], default=None)
    parser.add_argument("--confirm-real-money", action="store_true")
    return parser.parse_args(argv)


def _gates(settings: Settings) -> list[tuple[str, bool]]:
    return [
        ("TRADING_MODE=live", settings.trading_mode.value == "live"),
        ("ENABLE_AUTOMATIC_TRADING=true", settings.enable_automatic_trading),
        ("LIVE_TRADING_ACKNOWLEDGED=true", settings.live_trading_acknowledged),
        ("TRADING_KILL_SWITCH=false", not settings.trading_kill_switch),
        ("API credentials configured", settings.has_private_credentials),
        ("LIVE_CAPITAL_JPY > 0", settings.live_capital_jpy > 0),
        ("STRATEGY_NAME=trend_ensemble", settings.strategy_name == "trend_ensemble"),
    ]


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = get_settings()
    configure_logging(base.log_level, base.log_format)
    if not base.has_private_credentials:
        print("BITFLYER_API_KEY / BITFLYER_API_SECRET are not set.", file=sys.stderr)
        return 1
    client = BitflyerClient(base)
    try:
        async with session_scope(base) as session:
            settings = await get_runtime_settings(session, base)
        permissions = set(await client.get_permissions())
        missing = [path for path in REQUIRED_PERMISSIONS if path not in permissions]
        account = await load_account(client, settings)
        health = await client.get_health(settings.product_code)

        print("API key permissions:", "OK" if not missing else f"MISSING {missing}")
        if any("withdraw" in path for path in permissions):
            print("  WARNING: this key can withdraw funds; create a trade-only key.")
        print(f"Fee tier: {account.fee_rate * 100:.3f}% per trade")
        print(
            f"Balances: ¥{account.jpy:,.0f} (available ¥{account.jpy_available:,.0f}), "
            f"{account.btc} BTC (available {account.btc_available})"
        )
        print(
            f"Market: bid ¥{account.bid:,.0f} / ask ¥{account.ask:,.0f}; exchange "
            f"{health.state.value}, board {health.board_state}"
            f"{'; MAINTENANCE WINDOW' if in_maintenance_window() else ''}"
        )
        bot_cash = account.bot_cash(settings)
        print(
            f"Bot capital: LIVE_CAPITAL_JPY ¥{settings.live_capital_jpy:,.0f} -> spendable "
            f"¥{bot_cash:,.0f} + BTC worth ¥{account.btc * account.bid:,.0f}"
        )
        print("Live gates:")
        for label, ok in _gates(settings):
            print(f"  [{'x' if ok else ' '}] {label}")

        if args.test_order is None:
            return 0 if not missing else 1
        if not args.confirm_real_money:
            print("--test-order spends real money; add --confirm-real-money.", file=sys.stderr)
            return 2
        if settings.trading_kill_switch or missing or not health.accepts_orders:
            print("Refusing the test order: kill switch, permissions or exchange state.")
            return 1
        side = Side.BUY if args.test_order == "buy" else Side.SELL
        slip = settings.live_max_slippage_bps / Decimal("10000")
        price = (
            (account.ask * (1 + slip)).quantize(Decimal("1"), ROUND_CEILING)
            if side is Side.BUY
            else (account.bid * (1 - slip)).quantize(Decimal("1"), ROUND_FLOOR)
        )
        request = OrderRequest(
            product_code=settings.product_code,
            side=side,
            size=settings.min_order_size_btc,
            price=price,
        )
        async with session_scope(settings) as session:
            order = await submit_order(
                session, settings, client, request, context={"reason": "live_check test order"}
            )
        print(
            f"Test order {order.client_order_id}: {order.status.value}, filled "
            f"{order.filled_size} BTC @ {order.average_fill_price}, exchange id "
            f"{order.exchange_order_id}, {order.rejection_reason or ''}"
        )
        return 0 if order.filled_size > 0 else 1
    except BtcTraderError as exc:
        print(f"live check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()
        await dispose_engine()


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(asyncio.run(main()))
