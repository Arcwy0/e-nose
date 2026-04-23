"""Liveness, root metadata, and model introspection endpoints."""

from __future__ import annotations

import datetime

from fastapi import APIRouter, HTTPException

from enose.config import (
    ENVIRONMENTAL_SENSORS,
    RESISTANCE_SENSORS,
    SMELL_MODEL_PATH,
    TRAINING_DATA_PATH,
    VLM_MODEL_PATH,
)

from .. import state

router = APIRouter()


@router.get("/")
async def root() -> dict:
    """Server status — mirrors the legacy root payload so existing clients keep working."""
    return {
        "message": "E-Nose Multimodal Server",
        "version": "3.1.0",
        "status": "running",
        "sensor_configuration": {
            "resistance_sensors": RESISTANCE_SENSORS,
            "environmental_sensors": ENVIRONMENTAL_SENSORS,
            "total_features": 22,
            "preprocessing": {
                "resistance": "StandardScaler applied",
                "environmental": "Raw values (no scaling)",
            },
        },
        "model_paths": {
            "vlm_model": VLM_MODEL_PATH,
            "smell_model": SMELL_MODEL_PATH,
            "training_data": TRAINING_DATA_PATH,
        },
        "models": {
            "vlm_loaded": state.vlm_model is not None,
            "smell_classifier_loaded": state.smell_classifier is not None,
            "smell_classifier_fitted": (
                state.smell_classifier.is_fitted if state.smell_classifier else False
            ),
            # Exposed so the browser UI and remote clients can show which
            # backend is actually running (BalancedRF vs XGBTabular etc.) —
            # the server picks one at startup via ENOSE_CLASSIFIER.
            "smell_classifier_backend": (
                type(state.smell_classifier).__name__
                if state.smell_classifier is not None
                else None
            ),
        },
    }


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "3.1.0",
        "features": {
            "total_features": 22,
            "resistance_sensors": len(RESISTANCE_SENSORS),
            "environmental_sensors": len(ENVIRONMENTAL_SENSORS),
        },
        "models": {
            "vlm": state.vlm_model is not None,
            "smell_classifier": state.smell_classifier is not None and state.smell_classifier.is_fitted,
        },
    }


@router.get("/smell/model_info")
async def get_model_info() -> dict:
    if state.smell_classifier is None:
        return {"model_loaded": False, "message": "Smell classifier not available"}
    try:
        return state.smell_classifier.get_model_info()
    except Exception as e:
        print(f"[model_info] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
