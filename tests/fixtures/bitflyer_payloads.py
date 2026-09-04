"""Recorded bitFlyer public API responses.

These are real payloads captured from ``api.bitflyer.com`` (values only lightly
trimmed), kept verbatim so parsing tests fail if bitFlyer's actual field names
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

BOARD_JSON = """
{
  "mid_price": 12416157.0,
  "bids": [
    {"price": 12412059.0, "size": 0.25032835},
    {"price": 12412067.0, "size": 0.07977624},
    {"price": 12412066.0, "size": 0.0161117}
  ],
  "asks": [
    {"price": 12420702.0, "size": 0.0224664},
    {"price": 12420247.0, "size": 0.0224848},
    {"price": 12420249.0, "size": 0.367}
  ]
}
"""

EXECUTIONS_JSON = """
[
  {
    "id": 2650513084,
    "side": "BUY",
    "price": 12411935.0,
    "size": 0.023552,
    "exec_date": "2026-09-01T17:47:04.587",
    "buy_child_order_acceptance_id": "JRF20260901-174704-011635",
    "sell_child_order_acceptance_id": "JRF20260901-174648-011506"
  },
  {
    "id": 2650513078,
    "side": "SELL",
    "price": 12411056.0,
    "size": 0.00171919,
    "exec_date": "2026-09-01T17:46:46.293",
    "buy_child_order_acceptance_id": "JRF20260901-164224-013454",
    "sell_child_order_acceptance_id": "JRF20260901-174645-012182"
  },
  {
    "id": 2650513076,
    "side": "",
    "price": 12411057.0,
    "size": 0.02939722,
    "exec_date": "2026-09-01T17:46:45.98",
    "buy_child_order_acceptance_id": "JRF20260901-174645-012164",
    "sell_child_order_acceptance_id": "JRF20260901-174645-012168"
  }
]
"""

HEALTH_JSON = '{"status": "NORMAL"}'
HEALTH_STOP_JSON = '{"status": "STOP"}'

#: Prices/sizes as decimal *strings*, for assertions that must not go via float.
EXPECTED_TICKER = {
    "ltp": "12411935.0",
    "best_bid": "12412066.0",
    "best_ask": "12420247.0",
    "volume": "1155.80728167",
    "tick_id": 7350202,
    "state": "RUNNING",
}

EXPECTED_EXECUTION_IDS = ["2650513084", "2650513078", "2650513076"]
