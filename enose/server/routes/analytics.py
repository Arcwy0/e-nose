"""Analytics endpoints: visualization generation, data quality, environmental analysis, drift."""

from __future__ import annotations

from typing import Any, Dict, List

import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from enose.config import ALL_SENSORS, DATA_DIR, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS

from .. import state
from ..live_buffer import buffer as live_buffer

router = APIRouter(prefix="/smell")


@router.get("/visualize_data")
async def visualize_data() -> Dict[str, Any]:
    clf = state.require_fitted_classifier()
    try:
        plots = clf.generate_visualizations(DATA_DIR)
        if not plots:
            return {"message": "No visualizations generated", "plots": []}
        names = list(plots.keys())
        return {
            "message": f"Generated {len(names)} visualizations",
            "plots": names,
            "base_url": f"/{DATA_DIR}/",
            "full_urls": [f"/{DATA_DIR}/{p}" for p in names],
            "feature_info": {
                "total_features": 22,
                "resistance_sensors": len(RESISTANCE_SENSORS),
                "environmental_sensors": len(ENVIRONMENTAL_SENSORS),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[visualize_data] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze_data")
async def analyze_data() -> Dict[str, Any]:
    clf = state.require_classifier()
    if clf.last_training_data is None:
        raise HTTPException(status_code=404, detail="No training data available for analysis")
    try:
        analysis = clf.analyze_data_quality(output_dir=DATA_DIR)
        return {
            "message": "Data quality analysis completed",
            "analysis": analysis,
            "feature_info": {
                "total_features": 22,
                "resistance_sensors": len(RESISTANCE_SENSORS),
                "environmental_sensors": len(ENVIRONMENTAL_SENSORS),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[analyze_data] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift")
async def drift_report() -> Dict[str, Any]:
    """Compare the live buffer's per-sensor distribution to the training set.

    For each sensor we report:
        * ``train_mean`` / ``train_std`` — reference computed from the
          classifier's stored training frame (``last_training_data``).
        * ``live_mean`` / ``live_std`` — same stats over the current live
          buffer.
        * ``z_shift`` — ``(live_mean - train_mean) / train_std``. A |z| > 3
          typically means the sensor has drifted out of its training range
          (or the rig is reading a genuinely different environment).
        * ``std_ratio`` — ``live_std / train_std``. Values far from 1 mean
          the live variance is very different from training (noisy sensor
          or a narrow single-odour run).
        * ``status`` — coarse ok/warn/out so the UI can traffic-light each
          row without re-implementing the thresholds.

    Returns a summary ``overall_status`` so the drift badge can be computed
    with a single lookup.
    """
    clf = state.require_classifier()
    if clf.last_training_data is None or not len(clf.last_training_data):
        raise HTTPException(
            status_code=404,
            detail="No training data on the classifier yet — retrain to enable drift monitoring.",
        )

    # Training reference — use the same raw frame that feeds the model so
    # drift is measured in the same space as the features the classifier sees.
    train_df = clf.last_training_data
    live_items = live_buffer.snapshot()
    n_live = len(live_items)

    per_sensor: Dict[str, Dict[str, Any]] = {}
    worst_status = "ok"

    for name in ALL_SENSORS:
        if name not in train_df.columns:
            continue
        train_vals = train_df[name].astype(float).values
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size < 2:
            continue
        t_mean = float(train_vals.mean())
        t_std = float(train_vals.std(ddof=0)) or 1e-9

        live_vals = np.array(
            [e["sample"].get(name) for e in live_items if name in e.get("sample", {})],
            dtype=float,
        )
        live_vals = live_vals[np.isfinite(live_vals)]
        if live_vals.size < 2:
            per_sensor[name] = {
                "train_mean": round(t_mean, 4),
                "train_std": round(t_std, 4),
                "live_mean": None,
                "live_std": None,
                "live_n": int(live_vals.size),
                "z_shift": None,
                "std_ratio": None,
                "status": "na",
            }
            continue

        l_mean = float(live_vals.mean())
        l_std = float(live_vals.std(ddof=0))
        z_shift = (l_mean - t_mean) / t_std
        std_ratio = l_std / t_std if t_std > 0 else float("inf")

        # Thresholds are conservative — researchers would rather see a
        # yellow flag than miss a drifted sensor.
        if abs(z_shift) > 3.0 or std_ratio > 4.0 or std_ratio < 0.25:
            st = "out"
        elif abs(z_shift) > 1.5 or std_ratio > 2.0 or std_ratio < 0.5:
            st = "warn"
        else:
            st = "ok"
        if st == "out" or (st == "warn" and worst_status == "ok"):
            worst_status = st

        per_sensor[name] = {
            "train_mean": round(t_mean, 4),
            "train_std": round(t_std, 4),
            "live_mean": round(l_mean, 4),
            "live_std": round(l_std, 4),
            "live_n": int(live_vals.size),
            "z_shift": round(float(z_shift), 3),
            "std_ratio": round(float(std_ratio), 3),
            "status": st,
        }

    # Rank the worst offenders for the UI summary line.
    ranked: List[Dict[str, Any]] = [
        {"sensor": n, **v} for n, v in per_sensor.items()
        if v.get("z_shift") is not None
    ]
    ranked.sort(key=lambda r: abs(r["z_shift"] or 0.0), reverse=True)

    return {
        "live_buffer_n": n_live,
        "train_n": int(len(train_df)),
        "per_sensor": per_sensor,
        "overall_status": worst_status,
        "top_shifts": ranked[:5],
        "thresholds": {
            "z_warn": 1.5, "z_out": 3.0,
            "std_ratio_warn": [0.5, 2.0], "std_ratio_out": [0.25, 4.0],
        },
    }


@router.get("/settling")
async def settling(
    window: float = Query(10.0, gt=0.0, le=300.0, description="Look-back window in seconds"),
    threshold: float = Query(
        0.02,
        gt=0.0,
        description=(
            "Relative drift threshold: |slope / mean| (per second). A sensor is "
            "'settled' when its drift falls below this. 0.02 ≈ 2% per second."
        ),
    ),
    min_samples: int = Query(4, ge=2, description="Minimum samples required in the window"),
    session_id: Optional[str] = Query(None, description="Restrict to a session_id if given"),
) -> Dict[str, Any]:
    """Per-resistance-sensor stability over the last ``window`` seconds.

    Used by the robot mission policy's SETTLE state: poll this and only
    proceed to RECORD when ``settled`` stays True for K consecutive seconds.
    A small, dedicated endpoint here is cheaper than streaming raw samples
    and recomputing on the client — the buffer already lives server-side.

    Per-sensor fields:

    * ``mean``, ``std`` — window stats.
    * ``slope`` — least-squares ``dR/dt`` over the window.
    * ``rel_slope`` — ``slope / max(|mean|, eps)``. Dimensionless drift / s.
    * ``settled`` — ``|rel_slope| < threshold``.

    Top-level ``settled`` is True iff ALL ``RESISTANCE_SENSORS`` are settled,
    the window contains ≥ ``min_samples`` samples, and the time span covers
    ≥ ``0.5 * window`` seconds (so we don't declare "settled" on too short
    a slice).
    """
    items = live_buffer.snapshot()
    if session_id:
        items = [e for e in items if e.get("session_id") == session_id]
    if not items:
        return {
            "settled": False,
            "reason": "live buffer empty" + (f" for session_id={session_id}" if session_id else ""),
            "n": 0,
            "window": window,
            "threshold": threshold,
        }

    now = time.time()
    cutoff = now - window
    win = [e for e in items if e.get("t", 0.0) >= cutoff]
    n = len(win)
    if n < min_samples:
        return {
            "settled": False,
            "reason": f"not enough samples in window ({n} < {min_samples})",
            "n": n,
            "window": window,
            "threshold": threshold,
        }

    ts = np.asarray([e["t"] for e in win], dtype=float)
    span = float(ts.max() - ts.min())
    if span < 0.5 * window:
        return {
            "settled": False,
            "reason": f"time span too short ({span:.2f}s < {0.5 * window:.2f}s)",
            "n": n,
            "span_s": span,
            "window": window,
            "threshold": threshold,
        }

    eps = 1e-9
    per_sensor: Dict[str, Dict[str, Any]] = {}
    all_settled = True
    worst_rel: float = 0.0
    worst_name: Optional[str] = None

    for name in RESISTANCE_SENSORS:
        vals = np.asarray(
            [e["sample"].get(name) for e in win if name in e.get("sample", {})],
            dtype=float,
        )
        # Re-index ts to only the samples that actually have this sensor.
        ts_for = np.asarray(
            [e["t"] for e in win if name in e.get("sample", {})],
            dtype=float,
        )
        mask = np.isfinite(vals) & np.isfinite(ts_for)
        vals = vals[mask]
        ts_for = ts_for[mask]
        if vals.size < min_samples or (ts_for.max() - ts_for.min()) < 0.5 * window:
            per_sensor[name] = {
                "n": int(vals.size),
                "mean": None, "std": None,
                "slope": None, "rel_slope": None,
                "settled": None,
                "reason": "insufficient samples for this sensor",
            }
            all_settled = False
            continue

        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        # Linear least-squares slope: dR/dt.
        # polyfit gives [slope, intercept].
        slope = float(np.polyfit(ts_for, vals, 1)[0])
        rel_slope = slope / max(abs(mean), eps)
        settled = bool(abs(rel_slope) < threshold)
        per_sensor[name] = {
            "n": int(vals.size),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "slope": round(slope, 6),
            "rel_slope": round(rel_slope, 6),
            "settled": settled,
        }
        if not settled:
            all_settled = False
            if abs(rel_slope) > worst_rel:
                worst_rel = abs(rel_slope)
                worst_name = name

    return {
        "settled": bool(all_settled),
        "n": n,
        "span_s": span,
        "window": window,
        "threshold": threshold,
        "worst_sensor": worst_name,
        "worst_rel_slope": round(worst_rel, 6) if worst_name else 0.0,
        "per_sensor": per_sensor,
    }


@router.get("/environmental_analysis")
async def environmental_analysis() -> Dict[str, Any]:
    clf = state.require_classifier()
    if clf.last_training_data is None:
        raise HTTPException(status_code=404, detail="No training data available for environmental analysis")
    try:
        env_stats = clf.analyze_environmental_sensors(output_dir=DATA_DIR)
        return {
            "message": "Environmental sensor analysis completed",
            "environmental_stats": env_stats,
            "environmental_sensors": ENVIRONMENTAL_SENSORS,
            "resistance_sensors_count": len(RESISTANCE_SENSORS),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[environmental_analysis] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
