"""JSON-safe conversion for payloads that contain ``Decimal`` values.

Raw exchange payloads are parsed with ``parse_float=Decimal`` so no precision is
lost, but ``Decimal`` is not JSON-serialisable.  Persisting the *string* form
keeps the stored payload byte-faithful to what the exchange sent.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively convert a payload into JSON-serialisable primitives."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [jsonable(item) for item in value]
    return value
