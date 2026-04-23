"""Training endpoints: online learning, CSV batch learning, vision+smell pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from io import StringIO
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from enose.classifier.training import retrain_with_all_data
from enose.config import (
    ALL_SENSORS,
    DATA_DIR,
    ENV_DEFAULTS,
    ENVIRONMENTAL_SENSORS,
    RESISTANCE_SENSORS,
    TRAINED_MODELS_DIR,
)
from enose.vision.florence import process_image_with_vlm

from .. import state
from ..model_loader import reload_smell_classifier, save_training_data
from ..schemas import CSVLearningData, OnlineLearningData

router = APIRouter()


def _detect_new_classes_or_mismatch(clf, incoming_classes) -> tuple[set, bool]:
    """Return (new_classes, classes_mismatch). Both drive the "retrain from scratch" path."""
    current = set(clf.classes_)
    new_classes = set(incoming_classes) - current
    mismatch = False
    if getattr(clf, "class_weights_", None):
        mismatch = set(clf.class_weights_.keys()) != current
    return new_classes, mismatch


def _fill_missing_sensors(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure the 22 canonical columns exist; fill missing R with 0 and env with defaults."""
    X = X.copy()
    for sensor in ALL_SENSORS:
        if sensor not in X.columns:
            X[sensor] = ENV_DEFAULTS.get(sensor, 0.0) if sensor in ENVIRONMENTAL_SENSORS else 0.0
    return X[ALL_SENSORS]


@router.post("/smell/online_learning")
async def online_learning(data: OnlineLearningData) -> Dict[str, Any]:
    """Incremental training. Retrains from scratch when new classes or weight mismatch appear."""
    clf = state.require_classifier()

    if not data.sensor_data or not data.labels:
        raise HTTPException(status_code=400, detail="Both sensor_data and labels required")
    if len(data.sensor_data) != len(data.labels):
        raise HTTPException(status_code=400, detail="Sensor data and labels length mismatch")

    print(f"[online_learning] {len(data.sensor_data)} samples; labels={sorted(set(data.labels))}")

    try:
        save_training_data(data.sensor_data, data.labels)
        df = clf.process_sensor_data(data.sensor_data)
        new_labels = pd.Series(data.labels)
        new_classes: set = set()
        mismatch = False

        if clf.is_fitted:
            new_classes, mismatch = _detect_new_classes_or_mismatch(clf, new_labels.unique())
            if new_classes or mismatch:
                print(f"[online_learning] retraining — new_classes={new_classes}, mismatch={mismatch}")
                ok, accuracy, fresh = retrain_with_all_data(
                    clf, df, new_labels, use_augmentation=True, n_augmentations=3,
                )
                if not ok or fresh is None:
                    raise HTTPException(status_code=500, detail="Retraining failed")
                state.set_classifier(fresh)
                clf = fresh
                update_type = "retrain_for_consistency"
            else:
                clf.online_update(df, new_labels, use_augmentation=True, n_augmentations=2)
                update_type = "online_update"
                accuracy = clf.training_history["accuracy"][-1] if clf.training_history["accuracy"] else 0.0
        else:
            print("[online_learning] initial training")
            ok, accuracy, fresh = retrain_with_all_data(
                clf, df, new_labels,
                use_augmentation=True,
                n_augmentations=5,
            )
            if not ok or fresh is None:
                raise HTTPException(status_code=500, detail="Initial training failed")
            state.set_classifier(fresh)
            clf = fresh
            update_type = "initial_training"

        model_path = clf.save_model(TRAINED_MODELS_DIR)
        reloaded = reload_smell_classifier()
        clf = state.smell_classifier  # pick up the reloaded instance

        return {
            "success": True,
            "update_type": update_type,
            "samples_processed": len(data.labels),
            "current_accuracy": accuracy,
            "model_saved_at": model_path,
            "classes": clf.classes_.tolist(),
            "n_features": len(clf.selected_features or []),
            "feature_breakdown": {
                "resistance_sensors": len(RESISTANCE_SENSORS),
                "environmental_sensors": len(ENVIRONMENTAL_SENSORS),
                "total_features": 22,
            },
            "model_reloaded": reloaded,
            "new_classes_detected": len(new_classes) > 0,
            "inconsistency_fixed": mismatch,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[online_learning] error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smell/learn_from_csv")
async def learn_from_csv(data: CSVLearningData) -> Dict[str, Any]:
    """Batch training from a CSV string. Same retrain-on-new-class logic as online learning."""
    clf = state.require_classifier()

    if not data.csv_data:
        raise HTTPException(status_code=400, detail="CSV data required")

    try:
        df = pd.read_csv(StringIO(data.csv_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    if data.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{data.target_column}' not found. Available: {df.columns.tolist()}",
        )

    try:
        X = df.drop(columns=[data.target_column, "Gas label", "timestamp"], errors="ignore")
        y = df[data.target_column].astype(str)
        if data.lowercase_labels:
            y = y.str.lower()
        X = _fill_missing_sensors(X)

        new_classes: set = set()
        mismatch = False

        if clf.is_fitted:
            new_classes, mismatch = _detect_new_classes_or_mismatch(clf, y.unique())
            if new_classes or mismatch:
                print(f"[learn_from_csv] retraining — new_classes={new_classes}, mismatch={mismatch}")
                ok, accuracy, fresh = retrain_with_all_data(
                    clf, X, y,
                    use_augmentation=data.use_augmentation,
                    n_augmentations=data.n_augmentations,
                )
                if not ok or fresh is None:
                    raise HTTPException(status_code=500, detail="Retraining failed")
                state.set_classifier(fresh)
                clf = fresh
                update_type = "retrain_for_consistency"
            else:
                clf.online_update(
                    X, y,
                    use_augmentation=data.use_augmentation,
                    n_augmentations=max(1, data.n_augmentations // 2),
                )
                update_type = "online_update"
                accuracy = clf.training_history["accuracy"][-1] if clf.training_history["accuracy"] else 0.0
        else:
            # Route initial training through retrain_with_all_data so the
            # majority-class cap + group-aware split apply here too. Without
            # this, the first CSV load would train on the raw (air-dominated)
            # distribution and every subsequent call would retrain on a mix
            # that was never balanced.
            ok, accuracy, fresh = retrain_with_all_data(
                clf, X, y,
                use_augmentation=data.use_augmentation,
                n_augmentations=data.n_augmentations,
            )
            if not ok or fresh is None:
                raise HTTPException(status_code=500, detail="Initial training failed")
            state.set_classifier(fresh)
            clf = fresh
            update_type = "initial_training"

        model_path = clf.save_model(TRAINED_MODELS_DIR)
        reloaded = reload_smell_classifier()
        clf = state.smell_classifier

        try:
            plots = clf.generate_visualizations(DATA_DIR)
            plot_urls = [f"/{DATA_DIR}/{p}" for p in plots.keys()]
        except Exception as e:
            print(f"[learn_from_csv] visualization error: {e}")
            plot_urls = []

        return {
            "success": True,
            "update_type": update_type,
            "samples_processed": len(df),
            "current_accuracy": accuracy,
            "model_saved_at": model_path,
            "classes": clf.classes_.tolist(),
            "n_features": len(clf.selected_features or []),
            "feature_breakdown": {
                "resistance_sensors": len(RESISTANCE_SENSORS),
                "environmental_sensors": len(ENVIRONMENTAL_SENSORS),
                "total_features": 22,
                "sensor_columns_found": X.columns.tolist(),
            },
            "visualizations": plot_urls,
            "model_reloaded": reloaded,
            "new_classes_detected": len(new_classes) > 0,
            "inconsistency_fixed": mismatch,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[learn_from_csv] error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training_pipeline")
async def full_training_pipeline(
    image: UploadFile = File(...),
    object_name: str = Form(...),
    sensor_data: str = Form(...),  # JSON-encoded list of sensor dicts
) -> Dict[str, Any]:
    """Vision + smell: detect `object_name` in the image, then online-train on provided samples."""
    temp_image: str | None = None
    try:
        try:
            sensor_data_list = json.loads(sensor_data)
            if not isinstance(sensor_data_list, list):
                raise ValueError("Sensor data must be a list")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid sensor data: {e}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            temp_image = f.name
            f.write(await image.read())

        if state.vlm_model is None or state.vlm_processor is None:
            raise HTTPException(status_code=503, detail="VLM not loaded")

        detection_result = process_image_with_vlm(
            state.vlm_model, state.vlm_processor,
            temp_image, "<OPEN_VOCABULARY_DETECTION>", f"Find {object_name}",
        )

        learning_data = OnlineLearningData(
            sensor_data=sensor_data_list,
            labels=[object_name] * len(sensor_data_list),
        )
        learning_result = await online_learning(learning_data)

        return {
            "detection_result": detection_result,
            "learning_result": learning_result,
            "pipeline_status": "success",
            "message": f"Processed {object_name} detection and smell training",
            "feature_info": {
                "total_features": 22,
                "resistance_sensors": len(RESISTANCE_SENSORS),
                "environmental_sensors": len(ENVIRONMENTAL_SENSORS),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[training_pipeline] error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_image and os.path.exists(temp_image):
            os.unlink(temp_image)
