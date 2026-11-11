"""JSON-safe payload conversion (PROJECT.md §13)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

from btc_trader.utils.serialization import jsonable


def test_decimals_become_plain_strings() -> None:
    assert jsonable(Decimal("12411935.0")) == "12411935.0"
    assert jsonable(Decimal("0.00171919")) == "0.00171919"


def test_exponent_notation_is_expanded() -> None:
    """``format(..., "f")`` avoids storing ``1E+3`` where a reader expects 1000."""
    assert jsonable(Decimal("1E+3")) == "1000"
    assert jsonable(Decimal("1e-8")) == "0.00000001"


def test_datetimes_become_iso_strings() -> None:
    assert jsonable(datetime(2026, 9, 1, 12, 0, tzinfo=UTC)) == "2026-09-01T12:00:00+00:00"
    assert jsonable(date(2026, 9, 1)) == "2026-09-01"


def test_nested_structures_are_converted_recursively() -> None:
    payload = {
        "price": Decimal("1.5"),
        "levels": [{"price": Decimal("2")}, {"price": Decimal("3")}],
        "at": datetime(2026, 9, 1, tzinfo=UTC),
        "nested": ({"size": Decimal("0.5")},),
    }
    converted = jsonable(payload)
    assert converted == {
        "price": "1.5",
        "levels": [{"price": "2"}, {"price": "3"}],
        "at": "2026-09-01T00:00:00+00:00",
        "nested": [{"size": "0.5"}],
    }
    # The whole point: the result must survive json.dumps.
    assert json.loads(json.dumps(converted)) == converted


def test_primitives_pass_through_untouched() -> None:
    assert jsonable({"a": 1, "b": True, "c": None, "d": "x"}) == {
        "a": 1,
        "b": True,
        "c": None,
        "d": "x",
    }


def test_non_string_keys_are_stringified() -> None:
    assert jsonable({1: Decimal("2")}) == {"1": "2"}


def test_round_trip_preserves_the_exact_decimal() -> None:
    original = Decimal("12411935.12345678")
    restored = Decimal(jsonable(original))
    assert restored == original
