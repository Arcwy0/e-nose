"""JSON sanitation for API responses.

Model metadata legitimately contains non-finite floats (e.g. the H2S sensor's
upper range bound is ``inf``). Starlette's ``JSONResponse`` serializes with
``allow_nan=False`` and browsers' ``JSON.parse`` can't read ``Infinity``/``NaN``
either, so any such value must become ``null`` before it goes on the wire.
"""

from __future__ import annotations

import math
import numbers
from typing import Any


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN / ±inf) with None.

    Handles nested dicts/lists/tuples and numpy floating types (which register
    as ``numbers.Real``). Leaves everything else untouched.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, numbers.Real) and not isinstance(obj, numbers.Integral):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj
