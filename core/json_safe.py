from __future__ import annotations

import json
import math
from numbers import Real
from typing import Any


def to_json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers with null for strict JSON output."""

    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        try:
            as_float = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return as_float if math.isfinite(as_float) else None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, str):
        try:
            scalar = item()
        except (AttributeError, TypeError, ValueError):
            return value
        if isinstance(scalar, Real):
            return to_json_safe(scalar)
    return value


def json_dumps(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("allow_nan", False)
    return json.dumps(to_json_safe(value), **kwargs)
