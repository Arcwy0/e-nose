"""Analytics endpoints: visualization generation, data quality, environmental analysis, drift."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException

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
