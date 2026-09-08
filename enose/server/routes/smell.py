"""Inference endpoints: single-sample classify, console test, diagnostics."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from fastapi.responses import JSONResponse

from enose.config import ALL_SENSORS, ENV_DEFAULTS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS
from enose.utils.plateau import relative_slope_score

from .. import state
from ..jsonsafe import json_safe
from ..live_buffer import buffer as live_buffer
from ..schemas import BaselineData, ConsoleSensorData, SensorData

router = APIRouter(prefix="/smell")

_NO_STORE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}


def _recent_live_frame(window: float, session_id: Optional[str]) -> pd.DataFrame:
    items = live_buffer.snapshot()
    if session_id:
        items = [item for item in items if item.get("session_id") == session_id]
    if not items:
        raise HTTPException(status_code=409, detail="live sensor buffer is empty")
    newest = max(float(item.get("t", 0.0)) for item in items)
    items = [item for item in items if float(item.get("t", 0.0)) >= newest - window]
    records: List[Dict[str, Any]] = []
    for item in items:
        sample = dict(item.get("sample") or {})
        sample["_timestamp"] = pd.to_datetime(float(item.get("t", time.time())), unit="s")
        records.append(sample)
    return pd.DataFrame(records)


def _stable_live_frame(
    window: float,
    session_id: Optional[str],
    max_relative_slope: float,
    min_samples: int,
) -> tuple[pd.DataFrame, float]:
    frame = _recent_live_frame(window, session_id)
    _require_resistance_columns(frame)
    if len(frame) < min_samples:
        raise HTTPException(
            status_code=409,
            detail=f"need at least {min_samples} recent samples; got {len(frame)}",
        )
    span = float((frame["_timestamp"].max() - frame["_timestamp"].min()).total_seconds())
    if span < window * 0.5:
        raise HTTPException(
            status_code=409,
            detail=f"recent samples span only {span:.1f}s; need at least {window * 0.5:.1f}s",
        )
    score = relative_slope_score(frame)
    if score > max_relative_slope:
        raise HTTPException(
            status_code=409,
            detail=(
                f"sensor response is still transitioning: stability_score={score:.6f} "
                f"> {max_relative_slope:.6f}"
            ),
        )
    return frame, score


def _require_resistance_columns(frame: pd.DataFrame) -> None:
    missing = [sensor for sensor in RESISTANCE_SENSORS if sensor not in frame.columns]
    if missing:
        raise HTTPException(status_code=409, detail=f"live samples are missing resistance sensors: {missing}")


def _validate_live_span(frame: pd.DataFrame, window: float, min_samples: int) -> float:
    if len(frame) < min_samples:
        raise HTTPException(
            status_code=409,
            detail=f"need at least {min_samples} recent samples; got {len(frame)}",
        )
    span = float((frame["_timestamp"].max() - frame["_timestamp"].min()).total_seconds())
    if span < window * 0.5:
        raise HTTPException(
            status_code=409,
            detail=f"recent samples span only {span:.1f}s; need at least {window * 0.5:.1f}s",
        )
    return span


def _median_live_sample(frame: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, Any]]:
    sample: Dict[str, float] = {}
    coverage: Dict[str, float] = {}
    for sensor in ALL_SENSORS:
        if sensor not in frame.columns:
            sample[sensor] = ENV_DEFAULTS.get(sensor, 0.0)
            if sensor in RESISTANCE_SENSORS:
                coverage[sensor] = 0.0
            continue
        values = pd.to_numeric(frame[sensor], errors="coerce")
        if sensor in RESISTANCE_SENSORS:
            positive = values.where(values > 0.0).dropna()
            coverage[sensor] = float(len(positive) / max(len(frame), 1))
            value = positive.median() if len(positive) else float("nan")
        else:
            finite = values.dropna()
            value = finite.median() if len(finite) else float("nan")
        sample[sensor] = float(value) if pd.notna(value) else ENV_DEFAULTS.get(sensor, 0.0)
    active = [sensor for sensor, fraction in coverage.items() if fraction > 0.0]
    quality = {
        "active_resistance_sensors": active,
        "n_active_resistance_sensors": len(active),
        "sparse_resistance_sensors": [
            sensor for sensor, fraction in coverage.items() if 0.0 < fraction < 0.5
        ],
        "missing_resistance_sensors": [sensor for sensor in RESISTANCE_SENSORS if sensor not in active],
        "positive_coverage": coverage,
    }
    return sample, quality


def _require_live_baseline(clf) -> None:
    mode = getattr(getattr(clf, "config", None), "baseline_mode", "none") or "none"
    if mode != "none" and not bool(getattr(clf, "live_baseline_captured_", False)):
        raise HTTPException(
            status_code=409,
            detail="capture a clean-air baseline for this live session before classifying",
        )


@router.get("/class_examples")
async def class_examples() -> JSONResponse:
    """Representative sensor vector per known class — powers the UI's dynamic
    "Try: <class>" buttons.

    ``examples`` maps each class to a 22-value list in ``order`` (ALL_SENSORS)
    order, computed as the per-class mean of the classifier's retained training
    data. Returns empty maps when the model is unfitted or holds no training
    data, so the UI can fall back to its built-in examples.
    """
    clf = state.smell_classifier
    examples: Dict[str, Any] = {}
    if clf is not None and getattr(clf, "is_fitted", False) and hasattr(clf, "class_example_vectors"):
        try:
            examples = clf.class_example_vectors()
        except Exception as e:  # pragma: no cover — best-effort
            print(f"[class_examples] error: {e}")
            examples = {}
    return JSONResponse(
        json_safe({"order": list(ALL_SENSORS), "classes": list(examples.keys()), "examples": examples}),
        headers=_NO_STORE,
    )


@router.post("/classify")
async def classify_smell(sensor_data: SensorData) -> Dict[str, Any]:
    """Single 22-feature classification → label + per-class probabilities.

    Response also carries an ``ood`` block with the mean |z-score| across
    features plus the minimum distance to any class centroid, and a coarse
    ``status`` ("ok"/"warn"/"out") so the UI can draw a traffic light
    without re-deriving the thresholds.

    Thresholds are lifted from BalancedRFClassifier.predict_with_ood so both
    code paths label OOD the same way.
    """
    clf = state.require_fitted_classifier()
    try:
        data_dict = sensor_data.dict()
        # predict()/predict_proba() run process_sensor_data internally
        # (see BalancedRFClassifier._model_input). Pass the raw dict so we
        # don't scale + log1p twice and collapse everything near zero.
        prediction = clf.predict(data_dict)[0]
        probabilities = clf.predict_proba(data_dict)[0]

        ood_payload: Dict[str, Any] = {"available": False}
        if hasattr(clf, "diagnose_sample"):
            try:
                diag = clf.diagnose_sample(data_dict)
                ood_score = float(diag.get("ood_score") or 0.0)
                nearest = diag.get("nearest_centroid_L2") or {}
                min_centroid = (
                    float(min(nearest.values())) if nearest else None
                )
                # Matches predict_with_ood's OOD gate (score>3 or centroid>5).
                if ood_score > 3.0 or (min_centroid is not None and min_centroid > 5.0):
                    status = "out"
                elif ood_score > 2.0 or (min_centroid is not None and min_centroid > 3.5):
                    status = "warn"
                else:
                    status = "ok"
                ood_payload = {
                    "available": True,
                    "score": round(ood_score, 3),
                    "min_centroid_L2": (round(min_centroid, 3) if min_centroid is not None else None),
                    "nearest_centroid_L2": {k: round(float(v), 3) for k, v in nearest.items()},
                    "status": status,
                    "thresholds": {"warn": 2.0, "out": 3.0, "centroid_warn": 3.5, "centroid_out": 5.0},
                }
            except Exception as e:  # pragma: no cover — diagnostics are best-effort
                print(f"[classify] OOD diagnostics failed: {e}")

        return {
            "predicted_smell": prediction,
            "probabilities": {cls: float(p) for cls, p in zip(clf.classes_, probabilities)},
            "confidence": float(max(probabilities)),
            "sensor_input": data_dict,
            "features_used": len(clf.selected_features or []) or 22,
            "ood": ood_payload,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[classify] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test_console")
async def test_console_input(data: ConsoleSensorData) -> Dict[str, Any]:
    """Comma-separated 22 values → prediction. Ergonomic for curl/console testing."""
    clf = state.require_fitted_classifier()

    try:
        values = [float(x.strip()) for x in data.values.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid sensor values format: {e}")
    # Accept either 17 (R-only, for models trained with use_env_sensors=False)
    # or 22 (R + T/H/CO2/H2S/CH2O). The classifier's process_sensor_data will
    # fill any missing env columns with defaults; when the model was trained
    # without env, those columns are dropped at the model boundary anyway.
    if len(values) not in (17, 22):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 17 or 22 values, got {len(values)}",
        )

    sensor_dict: Dict[str, float] = {}
    for i, name in enumerate(RESISTANCE_SENSORS):
        sensor_dict[name] = values[i]
    if len(values) == 22:
        for i, name in enumerate(ENVIRONMENTAL_SENSORS):
            sensor_dict[name] = values[17 + i]
    else:
        # Keep the response payload and classifier input complete for R-only
        # requests.  Previously preprocessing filled these values only in its
        # private DataFrame, then response construction indexed the original
        # dict and raised KeyError("T") after prediction had succeeded.
        sensor_dict.update(ENV_DEFAULTS)

    # Raw dict → predict; predict() re-runs process_sensor_data internally.
    prediction = clf.predict(sensor_dict)[0]
    probabilities = clf.predict_proba(sensor_dict)[0]
    prob_dict = {cls: float(p) for cls, p in zip(clf.classes_, probabilities)}
    sorted_probs = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)

    # Same OOD payload shape as /smell/classify so the UI's traffic-light
    # renderer doesn't need two branches. Wrapped in try/except so a
    # diagnostics failure never blocks the prediction.
    ood_payload: Dict[str, Any] = {"available": False}
    if hasattr(clf, "diagnose_sample"):
        try:
            diag = clf.diagnose_sample(sensor_dict)
            ood_score = float(diag.get("ood_score") or 0.0)
            nearest = diag.get("nearest_centroid_L2") or {}
            min_centroid = float(min(nearest.values())) if nearest else None
            if ood_score > 3.0 or (min_centroid is not None and min_centroid > 5.0):
                status = "out"
            elif ood_score > 2.0 or (min_centroid is not None and min_centroid > 3.5):
                status = "warn"
            else:
                status = "ok"
            ood_payload = {
                "available": True,
                "score": round(ood_score, 3),
                "min_centroid_L2": (round(min_centroid, 3) if min_centroid is not None else None),
                "nearest_centroid_L2": {k: round(float(v), 3) for k, v in nearest.items()},
                "status": status,
                "thresholds": {"warn": 2.0, "out": 3.0, "centroid_warn": 3.5, "centroid_out": 5.0},
            }
        except Exception as e:  # pragma: no cover
            print(f"[test_console] OOD diagnostics failed: {e}")

    print(f"[test_console] prediction={prediction} conf={max(probabilities):.3f}")
    return {
        "predicted_smell": prediction,
        "confidence": float(max(probabilities)),
        "all_probabilities": prob_dict,
        "sorted_probabilities": sorted_probs,
        "sensor_input": {
            "resistance_sensors": {name: sensor_dict[name] for name in RESISTANCE_SENSORS},
            "environmental_sensors": {name: sensor_dict[name] for name in ENVIRONMENTAL_SENSORS},
        },
        "available_classes": clf.classes_.tolist(),
        "feature_info": {"total_features": 22, "resistance_sensors": 17, "environmental_sensors": 5},
        "ood": ood_payload,
    }


@router.post("/debug_input")
async def debug_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnostics for one sample: z-scores, OOD score, nearest centroids, top-3 probs.

    Accepts either {"values": [22 floats]} or a named-sensor dict.
    """
    clf = state.require_fitted_classifier()

    try:
        if "values" in payload:
            vals = payload["values"]
            if not isinstance(vals, list) or len(vals) != 22:
                raise HTTPException(status_code=400, detail="Expected 'values' with 22 numbers")
            data_dict = {name: float(vals[i]) for i, name in enumerate(RESISTANCE_SENSORS)}
            for i, name in enumerate(ENVIRONMENTAL_SENSORS):
                data_dict[name] = float(vals[17 + i])
        else:
            data_dict = {k: float(v) for k, v in payload.items()}

        # diagnose_sample re-runs process_sensor_data internally; feed raw dict.
        diag = clf.diagnose_sample(data_dict)

        prob_map = {c: float(p) for c, p in zip(diag["classes"], diag["probs"])}
        top3 = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return {
            "predicted": max(prob_map, key=prob_map.get),
            "top3": top3,
            "ood_score": diag["ood_score"],
            "nearest_centroid_L2": diag["nearest_centroid_L2"],
            "z_scores": {k: round(v, 2) for k, v in diag["z_scores"].items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[debug_input] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/baseline")
async def set_baseline(data: BaselineData) -> Dict[str, Any]:
    """Set the current session's clean-air drift baseline from air readings.

    Call this once at the start of a session (~30–60 s of clean air) so a
    drift-robust model references today's samples to today's air. Harmless /
    no-op for a model trained in absolute mode (`baseline_mode='none'`): the
    baseline is stored but never used, and classification behaves as before.
    """
    clf = state.require_fitted_classifier()
    if not data.sensor_data:
        raise HTTPException(status_code=400, detail="sensor_data (clean-air readings) required")
    mode = getattr(getattr(clf, "config", None), "baseline_mode", "none") or "none"
    if not hasattr(clf, "update_baseline"):
        return {"applied": False, "baseline_mode": mode,
                "message": "classifier backend does not support baseline capture"}
    baseline_sample, quality = _median_live_sample(pd.DataFrame(data.sensor_data))
    if quality["n_active_resistance_sensors"] < 12:
        raise HTTPException(status_code=400, detail="fewer than 12 resistance sensors are active")
    baseline = clf.update_baseline([baseline_sample], ema=bool(data.ema))
    clf.live_baseline_captured_ = bool(baseline)
    return {
        "applied": mode != "none",
        "baseline_mode": mode,
        "n_sensors": len(baseline),
        "n_air_samples": len(data.sensor_data),
        "sensor_quality": quality,
        "message": (
            "baseline updated" if mode != "none"
            else "stored but unused (model trained in absolute mode)"
        ),
    }


@router.post("/baseline/live")
async def set_baseline_from_live(
    window: float = Query(60.0, gt=5.0, le=300.0),
    max_relative_slope: float = Query(0.002, gt=0.0),
    min_samples: int = Query(10, ge=5),
    session_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Capture a clean-air baseline from a stable recent live window."""
    clf = state.require_fitted_classifier()
    frame, score = _stable_live_frame(window, session_id, max_relative_slope, min_samples)
    baseline_sample, quality = _median_live_sample(frame)
    if quality["n_active_resistance_sensors"] < 12:
        raise HTTPException(status_code=409, detail="fewer than 12 resistance sensors are active")
    baseline = clf.update_baseline([baseline_sample], ema=False)
    clf.live_baseline_captured_ = bool(baseline)
    mode = getattr(clf.config, "baseline_mode", "none") or "none"
    return {
        "applied": mode != "none",
        "baseline_mode": mode,
        "n_air_samples": len(frame),
        "n_sensors": len(baseline),
        "stability_score": score,
        "sensor_quality": quality,
        "session_id": session_id,
    }


@router.post("/classify_stable")
async def classify_stable_live_window(
    window: float = Query(60.0, gt=5.0, le=300.0),
    max_relative_slope: float = Query(0.002, gt=0.0),
    min_samples: int = Query(10, ge=5),
    session_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Classify the median of a stable recent window, never a transient frame."""
    clf = state.require_fitted_classifier()
    _require_live_baseline(clf)
    frame, score = _stable_live_frame(window, session_id, max_relative_slope, min_samples)
    sample, quality = _median_live_sample(frame)
    if quality["n_active_resistance_sensors"] < 12:
        raise HTTPException(status_code=409, detail="fewer than 12 resistance sensors are active")
    probabilities = clf.predict_proba(sample)[0]
    prediction = str(clf.classes_[int(np.argmax(probabilities))])
    return {
        "stable": True,
        "stability_score": score,
        "n_samples": len(frame),
        "predicted_smell": prediction,
        "confidence": float(max(probabilities)),
        "all_probabilities": {
            str(label): float(probability)
            for label, probability in zip(clf.classes_, probabilities)
        },
        "sensor_input": sample,
        "sensor_quality": quality,
    }


@router.post("/classify_window")
async def classify_recent_window(
    window: float = Query(15.0, gt=3.0, le=120.0),
    bin_seconds: float = Query(5.0, gt=1.0, le=30.0),
    min_samples: int = Query(5, ge=3),
    max_relative_slope: float = Query(0.002, gt=0.0),
    min_confidence: float = Query(0.45, ge=0.0, le=1.0),
    min_margin: float = Query(0.10, ge=0.0, le=1.0),
    session_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Fast, noise-reduced prediction from a short live window.

    Probabilities are averaged across short median bins. A changing response is
    allowed and reported as provisional; low-confidence/tied results abstain as
    ``unknown`` rather than forcing the wrong odor.
    """
    compute_started = time.perf_counter()
    clf = state.require_fitted_classifier()
    _require_live_baseline(clf)
    frame = _recent_live_frame(window, session_id).sort_values("_timestamp").reset_index(drop=True)
    _require_resistance_columns(frame)
    span = _validate_live_span(frame, window, min_samples)
    score = relative_slope_score(frame)
    start = frame["_timestamp"].min()
    bin_index = ((frame["_timestamp"] - start).dt.total_seconds() / bin_seconds).astype(int)
    bin_samples: List[Dict[str, float]] = []
    for _, part in frame.groupby(bin_index, sort=True):
        if len(part) < 2:
            continue
        sample, quality = _median_live_sample(part)
        if quality["n_active_resistance_sensors"] < 12:
            continue
        bin_samples.append(sample)
    if not bin_samples:
        raise HTTPException(
            status_code=409,
            detail="no short bins contain readings from at least 12 resistance sensors",
        )
    # One batched forest call is substantially faster than one call per bin.
    probability_rows = np.asarray(clf.predict_proba(pd.DataFrame(bin_samples)), dtype=float)
    probabilities = np.mean(probability_rows, axis=0)
    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    confidence = float(probabilities[top])
    margin = float(confidence - probabilities[order[1]]) if len(order) > 1 else confidence
    candidate = str(clf.classes_[top])
    accepted = confidence >= min_confidence and margin >= min_margin
    overall_sample, overall_quality = _median_live_sample(frame)
    return {
        "predicted_smell": candidate if accepted else "unknown",
        "candidate_smell": candidate,
        "accepted": accepted,
        "abstained": not accepted,
        "confidence": confidence,
        "confidence_margin": margin,
        "all_probabilities": {
            str(label): float(probability)
            for label, probability in zip(clf.classes_, probabilities)
        },
        "measurement_latency_seconds": span,
        "inference_compute_ms": round((time.perf_counter() - compute_started) * 1000.0, 3),
        "requested_window_seconds": window,
        "bins_used": len(bin_samples),
        "stable": score <= max_relative_slope,
        "provisional": score > max_relative_slope,
        "stability_score": score,
        "sensor_quality": overall_quality,
        "sensor_input": overall_sample,
        "session_id": session_id,
    }


@router.get("/recovery")
async def recovery_status(
    window: float = Query(15.0, gt=3.0, le=120.0),
    response_threshold: float = Query(0.12, gt=0.0, le=1.0),
    max_relative_slope: float = Query(0.002, gt=0.0),
    min_samples: int = Query(5, ge=3),
    session_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Report whether the array has returned to its captured clean-air baseline."""
    clf = state.require_fitted_classifier()
    _require_live_baseline(clf)
    frame = _recent_live_frame(window, session_id)
    _require_resistance_columns(frame)
    span = _validate_live_span(frame, window, min_samples)
    sample, quality = _median_live_sample(frame)
    baseline = dict(getattr(clf, "sensor_baseline_", {}) or {})
    deviations = []
    for sensor in RESISTANCE_SENSORS:
        current = float(sample.get(sensor, 0.0))
        reference = float(baseline.get(sensor, 0.0))
        if current > 0.0 and reference > 0.0:
            deviations.append(abs(float(np.log(current / reference))))
    if len(deviations) < 12:
        raise HTTPException(status_code=409, detail="fewer than 12 sensors can be compared to baseline")
    response_score = float(np.quantile(deviations, 0.8))
    stability_score = relative_slope_score(frame)
    stable = stability_score <= max_relative_slope
    recovered = response_score <= response_threshold and stable
    return {
        "recovered": recovered,
        "stable": stable,
        "response_score": response_score,
        "response_threshold": response_threshold,
        "stability_score": stability_score,
        "max_relative_slope": max_relative_slope,
        "measurement_latency_seconds": span,
        "n_sensors_compared": len(deviations),
        "sensor_quality": quality,
        "session_id": session_id,
    }
