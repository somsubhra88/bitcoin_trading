"""Recorded bitFlyer public API responses.

Public payloads were captured from ``api.bitflyer.com`` (values lightly
trimmed); private ones follow the documented shapes. They are kept verbatim so parsing tests fail if bitFlyer's actual field names
or timestamp format are ever misremembered (PROJECT.md §9, §46).

Numbers are written as JSON text exactly as bitFlyer sends them — unquoted — so
the tests exercise the ``parse_float=Decimal`` decoding path rather than
sidestepping it with pre-made ``Decimal`` objects.
"""

from __future__ import annotations

TICKER_JSON = """
{
  "product_code": "BTC_JPY",
  "state": "RUNNING",
  "timestamp": "2026-09-01T17:47:23.553",
  "tick_id": 7350202,
  "best_bid": 12412066.0,
  "best_ask": 12420247.0,
  "best_bid_size": 0.0161117,
  "best_ask_size": 0.0224848,
  "total_bid_depth": 154.883013,
  "total_ask_depth": 107.6899998,
  "market_bid_size": 0.0,
  "market_ask_size": 0.0,
  "ltp": 12411935.0,
  "volume": 1155.80728167,
  "volume_by_product": 269.26826031
}
"""

BOARD_STATE_JSON = '{"health": "NORMAL", "state": "RUNNING"}'
BOARD_STATE_BUSY_JSON = '{"health": "VERY BUSY", "state": "RUNNING"}'
BOARD_STATE_NO_ORDER_JSON = '{"health": "NO ORDER", "state": "CLOSED"}'

# Private API shapes as documented at lightning.bitflyer.com/docs (and confirmed
# against a read-only call to a real account).
BALANCE_JSON = """
[
  {"currency_code": "JPY", "amount": 1024078, "available": 508000},
  {"currency_code": "BTC", "amount": 0.01234567, "available": 0.01234567},
  {"currency_code": "ETH", "amount": 20.48, "available": 16.38}
]
"""

COMMISSION_JSON = '{"commission_rate": 0.0015}'

SEND_ORDER_JSON = '{"child_order_acceptance_id": "JRF20260901-050237-639234"}'

CHILD_ORDERS_JSON = """
[
  {
    "id": 138398,
    "child_order_id": "JOR20260901-084555-022523",
    "product_code": "BTC_JPY",
    "side": "BUY",
    "child_order_type": "LIMIT",
    "price": 12450000,
    "average_price": 12420247,
    "size": 0.01,
    "child_order_state": "COMPLETED",
    "expire_date": "2026-09-01T08:46:53",
    "child_order_date": "2026-09-01T08:45:53",
    "child_order_acceptance_id": "JRF20260901-050237-639234",
    "outstanding_size": 0,
    "cancel_size": 0,
    "executed_size": 0.01,
    "total_commission": 0.000015,
    "time_in_force": "IOC"
  }
]
"""

MY_EXECUTIONS_JSON = """
[
  {
    "id": 37233,
    "child_order_id": "JOR20260901-084555-022523",
    "side": "BUY",
    "price": 12420247,
    "size": 0.01,
    "commission": 0.000015,
    "exec_date": "2026-09-01T08:45:53.397",
    "child_order_acceptance_id": "JRF20260901-050237-639234"
  }
]
"""

#: Prices/sizes as decimal *strings*, for assertions that must not go via float.
EXPECTED_TICKER = {
    "ltp": "12411935.0",
    "best_bid": "12412066.0",
    "best_ask": "12420247.0",
    "volume": "1155.80728167",
    "tick_id": 7350202,
    "state": "RUNNING",
}
