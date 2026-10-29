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

import json
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator

from enose.config import ALL_SENSORS
from enose.utils.acquisition import (
    acquisition_fingerprint, live_baseline_error, prediction_error_code, prediction_unit_error,
    resistance_sample_error,
)

from .. import state
from ..live_buffer import buffer

router = APIRouter(prefix="/sensor/live")


class LivePose(BaseModel):
    """Optional robot pose attached to a live sample.

    Sent by the robot's mission node so every sample is spatially anchored —
    required for Idea 2 (smell-aware navigation / spatial maps), useful for
    Idea 1 (auditing where a training sample was taken). ``frame_id`` defaults
    to ``"map"`` (the SLAM map frame); callers may override (e.g. ``"odom"``
    when SLAM isn't running). The buffer stores it opaquely and the UI / robot
    consumers interpret it.
    """

    x: float
    y: float
    theta: float
    frame_id: str = "map"


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
    pose: Optional[LivePose] = None    # robot pose at sample time (Phase 0+)
    acquisition: Optional[Dict[str, Any]] = None  # diagnostics, never classifier inputs

    @field_validator("acquisition")
    @classmethod
    def validate_acquisition(cls, value):
        if value is None:
            return None
        # Bound the optional extension and discard unknown keys. JSON values
        # only: no nested arbitrary objects or non-finite sensor diagnostics.
        if len(json.dumps(value, allow_nan=False)) > 8192:
            raise ValueError("acquisition metadata exceeds 8192 bytes")
        allowed = {
            "backend", "resistance_unit", "environment_source", "frame_index",
            "received_at", "sample_age_seconds", "device_id", "firmware_version",
            "temp0_c", "temp1_c", "duty_permille", "flags", "dropped_frames", "source_timestamp",
            "valid", "config", "segment_id", "stale_after_seconds",
            "heater",
            "temp_actual_c",
        }
        clean = {k: v for k, v in value.items() if k in allowed}
        for key, item in clean.items():
            if key in {"config", "heater"}:
                if not isinstance(item, dict) or len(item) > 32:
                    raise ValueError("acquisition config must contain at most 32 fields")
                # Short arrays (e.g. 17 per-channel gains) are useful, but
                # deeply nested structures do not belong in each live frame.
                for config_value in item.values():
                    parts = config_value if isinstance(config_value, list) else [config_value]
                    if len(parts) > 32 or any(not isinstance(p, (str, int, float, bool, type(None))) for p in parts):
                        raise ValueError("acquisition config values must be scalar or short scalar lists")
            elif not isinstance(item, (str, int, float, bool, type(None))):
                raise ValueError(f"acquisition {key} must be scalar")
            if isinstance(item, str) and len(item) > 160:
                raise ValueError(f"acquisition {key} exceeds 160 characters")
        return clean


def _run_classifier(sample: Dict[str, float], acquisition=None, session_id=None) -> Optional[Dict[str, Any]]:
    """Best-effort classification. Returns None if the classifier is missing
    or not fitted — never raises, so a publisher with ``classify=true`` doesn't
    crash the whole push when the server happens to be mid-retrain."""
    clf = state.smell_classifier
    if clf is None or not getattr(clf, "is_fitted", False):
        return None
    if live_baseline_error(clf, acquisition, session_id):
        return None
    try:
        probs = clf.predict_proba(sample)[0]
        pred = str(clf.classes_[int(np.argmax(probs))])
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
                if not math.isfinite(clean[k]):
                    raise HTTPException(status_code=400, detail=f"non-finite sensor {k}")
            except (TypeError, ValueError):
                continue

    entry: Dict[str, Any] = {"sample": clean}
    if payload.client_t is not None:
        entry["client_t"] = float(payload.client_t)
    if payload.label:
        entry["label"] = str(payload.label)
    if payload.session_id:
        entry["session_id"] = str(payload.session_id)
    if payload.pose is not None:
        entry["pose"] = payload.pose.model_dump()
    if payload.acquisition is not None:
        entry["acquisition"] = payload.acquisition

    sample_error = resistance_sample_error(clean, payload.acquisition)
    if sample_error:
        raise HTTPException(status_code=400, detail=sample_error)
    clf = state.smell_classifier
    # Observe identity changes even when classify=false: returning to an old
    # board later must not resurrect its previously invalidated baseline.
    baseline_error = live_baseline_error(clf, payload.acquisition, payload.session_id)
    if payload.classify:
        error = prediction_unit_error(clf, payload.acquisition) or baseline_error
        if clf is None or not getattr(clf, "is_fitted", False):
            error = "No fitted classifier is available; train or load a model first."
        if error:
            entry["prediction_error"] = error
            entry["prediction_error_code"] = prediction_error_code(error)
        else:
            info = _run_classifier(clean, payload.acquisition, payload.session_id)
            if info:
                entry.update(info)
            else:
                entry["prediction_error"] = "Classifier failed to produce a prediction; inspect server diagnostics."
                entry["prediction_error_code"] = "inference_failed"

    eid = buffer.push(entry)
    result = {"ok": True, "id": eid, "buffered": len(buffer)}
    result.update({k: entry[k] for k in ("predicted", "confidence", "top3", "prediction_error", "prediction_error_code") if k in entry})
    return result


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
        "server_time": time.time(),
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
    clf = state.smell_classifier
    latest_acquisition = items[-1].get("acquisition")
    return {
        "n": n, "per_sensor": per, "sessions": sessions,
        "acquisition": latest_acquisition,
        "acquisition_fingerprint": acquisition_fingerprint(latest_acquisition),
        "baseline_captured": bool(getattr(clf, "live_baseline_captured_", False)),
        "baseline_context": getattr(clf, "live_baseline_context_", None),
        "training_acquisition_provenance": getattr(clf, "acquisition_provenance_", None),
    }
