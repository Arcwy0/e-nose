"""Live sensor streaming endpoints.

The client's LivePublisher pushes one sample at a time to ``POST /sensor/live/push``;
the browser UI polls ``GET /sensor/live/recent?since=<id>`` roughly twice per
second to draw a scrolling chart. ``DELETE /sensor/live/clear`` is offered so
the UI can reset between runs.

Why polling instead of WebSockets: the existing tuna.am tunnel is a plain
TCP forward and works perfectly for HTTP; adding WS would require a second
tunnel and extra client code for reconnection handling. At ~2 Hz polling the
overhead is negligible and the plot feels smooth.

If the request supplies ``classify=true`` we also run the sample through the
fitted classifier and attach the prediction / top-k probabilities to the
buffer entry. That way the Live tab can overlay a confidence time-series
without a second round-trip per sample.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from enose.config import ALL_SENSORS

from .. import state
from ..live_buffer import buffer

router = APIRouter(prefix="/sensor/live")


class LivePushPayload(BaseModel):
    """One sample published by the client. All fields optional except ``sample``.

    ``client_t`` is the client's own timestamp — useful when the UI wants to
    plot using client wall-clock time rather than server receive time (the
    two can drift a few seconds over a long session).
    """

    sample: Dict[str, float]
    client_t: Optional[float] = None
    label: Optional[str] = None        # pre-assigned label, e.g. during training
    classify: bool = False             # ask server to also run /smell/classify
    session_id: Optional[str] = None   # groups samples from the same run


def _run_classifier(sample: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Best-effort classification. Returns None if the classifier is missing
    or not fitted — never raises, so a publisher with ``classify=true`` doesn't
    crash the whole push when the server happens to be mid-retrain."""
    clf = state.smell_classifier
    if clf is None or not getattr(clf, "is_fitted", False):
        return None
    try:
        pred = clf.predict(sample)[0]
        probs = clf.predict_proba(sample)[0]
        prob_map = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
        # Keep payload small — top-3 only; the full probs vector is available
        # via /smell/classify for anyone who needs it.
        top3 = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return {
            "predicted": str(pred),
            "confidence": float(max(probs)),
            "top3": top3,
        }
    except Exception as e:  # pragma: no cover — defensive
        print(f"[live] classify failed: {e}")
        return None


@router.post("/push")
async def push_sample(payload: LivePushPayload) -> Dict[str, Any]:
    """Accept one sample from the client publisher; optionally classify it."""
    if not isinstance(payload.sample, dict) or not payload.sample:
        raise HTTPException(status_code=400, detail="empty sample")
    # Coerce to float + only keep known sensor names so stray keys don't
    # bloat the buffer. Missing sensors stay missing — the UI chart can cope.
    clean: Dict[str, float] = {}
    for k in ALL_SENSORS:
        if k in payload.sample:
            try:
                clean[k] = float(payload.sample[k])
            except (TypeError, ValueError):
                continue

    entry: Dict[str, Any] = {"sample": clean}
    if payload.client_t is not None:
        entry["client_t"] = float(payload.client_t)
    if payload.label:
        entry["label"] = str(payload.label)
    if payload.session_id:
        entry["session_id"] = str(payload.session_id)

    if payload.classify:
        info = _run_classifier(clean)
        if info:
            entry.update(info)

    eid = buffer.push(entry)
    return {"ok": True, "id": eid, "buffered": len(buffer)}


@router.get("/recent")
async def recent(
    since: Optional[int] = Query(None, description="Only return entries with id > since"),
    limit: Optional[int] = Query(None, ge=1, le=2000),
) -> Dict[str, Any]:
    """Fetch recent samples. UI uses ``?since=<last_seen_id>`` for cheap polling."""
    items = buffer.recent(since_id=since, limit=limit)
    last_id = items[-1]["id"] if items else (since if since is not None else -1)
    return {
        "items": items,
        "last_id": last_id,
        "buffered": len(buffer),
        "capacity": buffer.capacity,
    }


@router.delete("/clear")
async def clear() -> Dict[str, Any]:
    """Reset the buffer — handy between runs so old traces don't linger on the chart."""
    n = buffer.clear()
    return {"cleared": n}


@router.get("/stats")
async def stats() -> Dict[str, Any]:
    """Aggregate stats over the current buffer — per-sensor mean/std, sample count.

    Used by the drift endpoint and for a quick sanity glance in the UI.
    """
    items = buffer.snapshot()
    n = len(items)
    if n == 0:
        return {"n": 0, "per_sensor": {}}
    per: Dict[str, Dict[str, float]] = {}
    for name in ALL_SENSORS:
        vals = [e["sample"].get(name) for e in items if name in e.get("sample", {})]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        per[name] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    sessions = sorted({e.get("session_id") for e in items if e.get("session_id")})
    return {"n": n, "per_sensor": per, "sessions": sessions}
