"""Inference endpoints: single-sample classify, console test, diagnostics."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from enose.config import ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS

from .. import state
from ..schemas import ConsoleSensorData, SensorData

router = APIRouter(prefix="/smell")


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
