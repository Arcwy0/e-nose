"""Training endpoints: online learning, CSV batch learning, vision+smell pipeline."""

from __future__ import annotations

import base64
import datetime
import json
import os
import tempfile
import traceback
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from enose.classifier.training import retrain_with_all_data
from enose.utils.csv_io import parse_uploaded_enose_csv
from enose.utils.plateau import build_plateau_training_frame
from enose.utils.quality import analyze_recording_quality, summarize_online_capture
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
from ..model_loader import reload_smell_classifier
from ..schemas import CommitProvenance, CSVLearningData, OnlineLearningData

router = APIRouter()


PROVENANCE_DIR = os.path.join(DATA_DIR, "provenance")


def _persist_provenance(
    provenance: CommitProvenance,
    labels: list[str],
    n_samples: int,
    provenance_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Persist a commit's provenance to ``data/provenance/<id>.{json,png}``.

    Returns the saved metadata (with ``provenance_id`` + ``image_path``) on
    success, ``None`` on failure (never raises — provenance is best-effort).
    """
    try:
        os.makedirs(PROVENANCE_DIR, exist_ok=True)
        pid = provenance_id or (
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        )
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
    """Fill canonical sensors while preserving grouping/provenance columns."""
    X = X.copy()
    for sensor in ALL_SENSORS:
        if sensor not in X.columns:
            X[sensor] = ENV_DEFAULTS.get(sensor, 0.0) if sensor in ENVIRONMENTAL_SENSORS else 0.0
    extras = [c for c in X.columns if c not in ALL_SENSORS]
    return X[ALL_SENSORS + extras]


@router.post("/smell/online_learning")
async def online_learning(data: OnlineLearningData) -> Dict[str, Any]:
    """Incremental training. Retrains from scratch when new classes or weight mismatch appear."""
    clf = state.require_classifier()

    if not data.sensor_data or not data.labels:
        raise HTTPException(status_code=400, detail="Both sensor_data and labels required")
    if len(data.sensor_data) != len(data.labels):
        raise HTTPException(status_code=400, detail="Sensor data and labels length mismatch")

    print(f"[online_learning] {len(data.sensor_data)} samples; labels={sorted(set(data.labels))}")

    # A baseline-relative Idea-1 update must pair today's confirmed odor with
    # today's clean air. Training it against the persisted fallback would bake
    # session drift into the new class instead of cancelling it.
    mode = getattr(getattr(clf, "config", None), "baseline_mode", "none") or "none"
    session_baseline: Optional[Dict[str, float]] = None
    if clf.is_fitted and mode != "none":
        if not bool(getattr(clf, "live_baseline_captured_", False)):
            raise HTTPException(
                status_code=409,
                detail="capture today's clean-air baseline before online learning",
            )
        session_baseline = dict(getattr(clf, "sensor_baseline_", {}) or {})

    try:
        # Reduce one physical sniff to a few ordered, zero-aware stable medians.
        # This prevents a 100-frame capture from getting 100 votes and filters
        # transient/noisy chunks before they enter persistent history.
        raw = _fill_missing_sensors(pd.DataFrame(data.sensor_data))
        label_series = pd.Series(data.labels).astype(str).reset_index(drop=True)
        run_ids = label_series.ne(label_series.shift()).cumsum()
        root_exposure = (
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        )
        session_id = (
            data.provenance.session_id
            if data.provenance is not None and data.provenance.session_id
            else datetime.datetime.now().strftime("%Y-%m-%d")
        )
        frames = []
        labels = []
        capture_reports = []
        for run_number in sorted(run_ids.unique()):
            mask = run_ids.eq(run_number)
            run_label = str(label_series.loc[mask].iloc[0]).strip()
            exposure_id = f"{root_exposure}:{int(run_number)}"
            try:
                aggregated, capture_report = summarize_online_capture(
                    raw.loc[mask].reset_index(drop=True)
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            aggregated["session_id"] = session_id
            aggregated["exposure_id"] = exposure_id
            frames.append(aggregated)
            labels.extend([run_label] * len(aggregated))
            capture_report.update({"label": run_label, "exposure_id": exposure_id})
            capture_reports.append(capture_report)

            # Pair each new odor exposure with the current clean-air baseline,
            # so training and inference both construct log(R/R0) in this same
            # physical session. Do not duplicate it for an explicit air batch.
            if session_baseline and run_label.lower() != "air":
                baseline_row = {
                    sensor: float(session_baseline.get(sensor, 0.0))
                    for sensor in RESISTANCE_SENSORS
                }
                baseline_row.update(ENV_DEFAULTS)
                baseline_row.update({"session_id": session_id, "exposure_id": exposure_id})
                frames.append(pd.DataFrame([baseline_row]))
                labels.append("air")

        df = _fill_missing_sensors(pd.concat(frames, ignore_index=True))
        new_labels = pd.Series(labels)
        odor_windows = int(sum(report["output_windows"] for report in capture_reports))
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

        provenance_meta: Optional[Dict[str, Any]] = None
        if data.provenance is not None:
            provenance_meta = _persist_provenance(
                data.provenance,
                data.labels,
                len(data.sensor_data),
                provenance_id=root_exposure,
            )
            if provenance_meta:
                print(
                    f"[online_learning] provenance saved id={provenance_meta['provenance_id']} "
                    f"score={provenance_meta['grounding_score']:.3f}"
                )

        model_path = clf.save_model(TRAINED_MODELS_DIR)
        reloaded = reload_smell_classifier()
        clf = state.smell_classifier  # pick up the reloaded instance
        if session_baseline and clf is not None and hasattr(clf, "update_baseline"):
            # Reload correctly resets live-session state. We are still in the
            # same running session, so reapply the already validated baseline
            # in memory without persisting the captured flag to disk.
            clf.update_baseline([session_baseline], ema=False)

        use_env = bool(getattr(getattr(clf, "config", None), "use_env_sensors", False))

        return {
            "success": True,
            "update_type": update_type,
            "samples_received": len(data.labels),
            "samples_processed": odor_windows,
            "training_rows_added": len(new_labels),
            "capture_report": capture_reports,
            "current_accuracy": accuracy,
            "model_saved_at": model_path,
            "classes": clf.classes_.tolist(),
            "n_features": len(clf.selected_features or []),
            "feature_breakdown": {
                "resistance_sensors": len(RESISTANCE_SENSORS),
                "environmental_sensors": len(ENVIRONMENTAL_SENSORS) if use_env else 0,
                "total_features": len(RESISTANCE_SENSORS) + (
                    len(ENVIRONMENTAL_SENSORS) if use_env else 0
                ),
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
    """Train from CSV, merging history by default or replacing it explicitly."""
    clf = state.require_classifier()

    if not data.csv_data:
        raise HTTPException(status_code=400, detail="CSV data required")

    try:
        df, csv_parse_report = parse_uploaded_enose_csv(
            data.csv_data, label_column=data.target_column
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    if data.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{data.target_column}' not found. Available: {df.columns.tolist()}",
        )

    profile = data.training_profile.strip().lower()
    if profile not in {"raw", "plateau"}:
        raise HTTPException(status_code=400, detail="training_profile must be 'raw' or 'plateau'")
    backend = data.classifier_backend.strip().lower()
    if backend not in {"balanced_rf", "xgboost", "two_stage"}:
        raise HTTPException(status_code=400, detail="unsupported classifier_backend")
    baseline_mode = data.baseline_mode.strip().lower()
    if baseline_mode not in {"none", "delta", "ratio", "logratio"}:
        raise HTTPException(status_code=400, detail="unsupported baseline_mode")

    try:
        samples_received = len(df)
        input_quality = analyze_recording_quality(df, data.target_column)
        input_quality["environment_reliable"] = bool(
            csv_parse_report.get("environment_reliable", False)
        )
        if not input_quality["environment_reliable"]:
            input_quality["warnings"].append(
                "environmental fields are absent or ambiguous; training uses R1-R17 only"
            )
        profile_report: Dict[str, Any] = {
            "profile": "raw", "input_rows": samples_received,
            "csv_parser": csv_parse_report,
            "input_quality": input_quality,
        }
        if profile == "plateau":
            df, profile_report = build_plateau_training_frame(df, label_col=data.target_column)
            profile_report["csv_parser"] = csv_parse_report
            profile_report["input_quality"] = input_quality
            # The plateau builder emits the canonical label name.
            target_column = "Gas name"
        else:
            target_column = data.target_column

        X = df.drop(columns=[target_column, "Gas label", "Gas class", "timestamp", "Timestamp"], errors="ignore")
        y = df[target_column].astype(str)
        if data.lowercase_labels:
            y = y.str.lower()
        X = _fill_missing_sensors(X)
        existing_classes = set(clf.classes_) if clf.is_fitted else set()
        new_classes = set(y.unique()) - existing_classes
        mismatch = False

        ok, accuracy, fresh = retrain_with_all_data(
            clf, X, y,
            use_augmentation=data.use_augmentation,
            n_augmentations=data.n_augmentations,
            merge_history=data.merge_history,
            classifier_backend=backend,
            baseline_mode=baseline_mode,
            snv=data.snv,
        )
        if not ok or fresh is None:
            raise HTTPException(status_code=500, detail="Training failed")
        state.set_classifier(fresh)
        clf = fresh
        update_type = "historical_retrain" if data.merge_history else "replacement_training"

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
            "samples_received": samples_received,
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
            "merge_history": data.merge_history,
            "training_profile": profile,
            "profile_report": profile_report,
            "classifier_backend": getattr(clf, "backend_name", backend),
            "baseline_mode": getattr(clf.config, "baseline_mode", baseline_mode),
            "snv": bool(getattr(clf.config, "snv", data.snv)),
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
