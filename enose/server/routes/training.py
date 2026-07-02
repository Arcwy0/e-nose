"""Training endpoints: online learning, CSV batch learning, vision+smell pipeline."""

from __future__ import annotations

import base64
import datetime
import json
import os
import tempfile
import traceback
import uuid
from io import StringIO
from typing import Any, Dict, Optional

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
from ..schemas import CommitProvenance, CSVLearningData, OnlineLearningData

router = APIRouter()


PROVENANCE_DIR = os.path.join(DATA_DIR, "provenance")


def _persist_provenance(
    provenance: CommitProvenance,
    labels: list[str],
    n_samples: int,
) -> Optional[Dict[str, Any]]:
    """Persist a commit's provenance to ``data/provenance/<id>.{json,png}``.

    Returns the saved metadata (with ``provenance_id`` + ``image_path``) on
    success, ``None`` on failure (never raises — provenance is best-effort).
    """
    try:
        os.makedirs(PROVENANCE_DIR, exist_ok=True)
        pid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        image_path: Optional[str] = None
        if provenance.image_b64:
            try:
                img_bytes = base64.b64decode(provenance.image_b64)
                ext = os.path.splitext(provenance.image_filename or "img.png")[1] or ".png"
                image_path = os.path.join(PROVENANCE_DIR, f"{pid}{ext}")
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
            except Exception as e:  # pragma: no cover — defensive
                print(f"[provenance] image decode failed: {e}")
                image_path = None
        meta = {
            "provenance_id": pid,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "label": labels[0] if labels else None,
            "n_samples": n_samples,
            "grounding_score": float(provenance.grounding_score),
            "bbox": provenance.bbox,
            "pose": provenance.pose,
            "session_id": provenance.session_id,
            "detector_task": provenance.detector_task,
            "image_path": image_path,
            "extras": provenance.extras or {},
        }
        with open(os.path.join(PROVENANCE_DIR, f"{pid}.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return meta
    except Exception as e:  # pragma: no cover — defensive
        print(f"[provenance] persist failed: {e}")
        return None


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

    provenance_meta: Optional[Dict[str, Any]] = None
    if data.provenance is not None:
        provenance_meta = _persist_provenance(data.provenance, data.labels, len(data.sensor_data))
        if provenance_meta:
            print(
                f"[online_learning] provenance saved id={provenance_meta['provenance_id']} "
                f"score={provenance_meta['grounding_score']:.3f}"
            )

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
                # Refit from the full accumulated history + this batch. We route
                # even same-class commits through retrain_with_all_data rather
                # than the in-memory online_update: after a model reload the
                # classifier's last_training_data is empty, so online_update
                # would train on this single-class batch alone and crash with
                # "target y needs more than 1 class". retrain_with_all_data pulls
                # history from the persistent CSV, so training stays multi-class.
                ok, accuracy, fresh = retrain_with_all_data(
                    clf, df, new_labels, use_augmentation=True, n_augmentations=2,
                )
                if not ok or fresh is None:
                    raise HTTPException(status_code=500, detail="Retraining failed")
                state.set_classifier(fresh)
                clf = fresh
                update_type = "online_update"
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
            "provenance": provenance_meta,
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


@router.get("/smell/provenance")
async def list_provenance(
    limit: int = 100,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """List recent autonomous-training commits. Used by ``scripts/audit_auto_labels.py``.

    Returns the metadata JSONs in ``data/provenance/`` sorted by ``saved_at``
    descending (newest first). Each entry is the same dict that
    ``/smell/online_learning`` returns under the ``provenance`` key, so the
    audit tool sees one consistent schema.
    """
    if not os.path.isdir(PROVENANCE_DIR):
        return {"count": 0, "entries": []}
    entries: list[Dict[str, Any]] = []
    for fname in sorted(os.listdir(PROVENANCE_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(PROVENANCE_DIR, fname), "r") as f:
                meta = json.load(f)
        except Exception as e:  # pragma: no cover — corrupt file
            print(f"[provenance] read {fname} failed: {e}")
            continue
        if label and meta.get("label") != label:
            continue
        entries.append(meta)
        if len(entries) >= max(1, limit):
            break
    return {"count": len(entries), "entries": entries}


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
