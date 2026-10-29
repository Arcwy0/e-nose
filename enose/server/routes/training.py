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
from starlette.concurrency import run_in_threadpool

from enose.classifier.training import retrain_with_all_data
from enose.utils.training_preview import prepare_csv_training, csv_training_preview
from enose.utils.quality import summarize_online_capture
from enose.utils.acquisition import (frame_resistance_unit, normalize_resistance_unit, training_resistance_unit,
                                     live_baseline_error, bind_live_baseline, acquisition_provenance)
from enose.utils.acquisition import resistance_sample_error
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


def _retrain_with_units(*args, **kwargs):
    """Map provenance conflicts to an actionable response, before disk writes."""
    try:
        return retrain_with_all_data(*args, **kwargs)
    except ValueError as exc:
        if "unit" in str(exc).lower():
            raise HTTPException(status_code=409, detail=str(exc))
        raise


@router.post("/smell/online_learning")
async def online_learning(data: OnlineLearningData) -> Dict[str, Any]:
    from .models import deployment_lock
    async with deployment_lock:
        return await _online_learning(data)


async def _online_learning(data: OnlineLearningData) -> Dict[str, Any]:
    """Incremental training. Retrains from scratch when new classes or weight mismatch appear."""
    clf = state.require_classifier()
    previous_provenance = dict(getattr(clf, "acquisition_provenance_", {}) or {})
    previous_fingerprints = list(getattr(clf, "acquisition_fingerprints_", []) or [])
    previous_fitted = bool(clf.is_fitted)

    if not data.sensor_data or not data.labels:
        raise HTTPException(status_code=400, detail="Both sensor_data and labels required")
    if len(data.sensor_data) != len(data.labels):
        raise HTTPException(status_code=400, detail="Sensor data and labels length mismatch")

    acquisition = data.acquisition or {}
    if acquisition.get("valid") is False:
        raise HTTPException(status_code=400, detail="Invalid acquisition cannot enter training history")
    for sample in data.sensor_data:
        error = resistance_sample_error(sample, acquisition)
        if error:
            raise HTTPException(status_code=400, detail=error)
    incoming_unit = normalize_resistance_unit(acquisition.get("resistance_unit"))
    if acquisition.get("backend") in {"pcb", "replay"} and incoming_unit is None:
        raise HTTPException(status_code=400, detail="PCB/replay online learning requires explicit ohm/kohm/mohm resistance units")
    try:
        training_resistance_unit(clf, incoming_unit)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    print(f"[online_learning] {len(data.sensor_data)} samples; labels={sorted(set(data.labels))}")

    # A baseline-relative Idea-1 update must pair today's confirmed odor with
    # today's clean air. Training it against the persisted fallback would bake
    # session drift into the new class instead of cancelling it.
    mode = getattr(getattr(clf, "config", None), "baseline_mode", "none") or "none"
    session_baseline: Optional[Dict[str, float]] = None
    baseline_context = dict(getattr(clf, "live_baseline_context_", {}) or {})
    if clf.is_fitted and mode != "none":
        context_error = live_baseline_error(clf, acquisition, data.session_id or (
                                           data.provenance.session_id if data.provenance else None))
        if context_error:
            raise HTTPException(status_code=409, detail=context_error)
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
                ok, accuracy, fresh = await run_in_threadpool(_retrain_with_units,
                    clf, df, new_labels, use_augmentation=True, n_augmentations=3,
                    acquisition_resistance_unit=incoming_unit,
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
                ok, accuracy, fresh = await run_in_threadpool(_retrain_with_units,
                    clf, df, new_labels, use_augmentation=True, n_augmentations=2,
                    acquisition_resistance_unit=incoming_unit,
                )
                if not ok or fresh is None:
                    raise HTTPException(status_code=500, detail="Retraining failed")
                state.set_classifier(fresh)
                clf = fresh
                update_type = "online_update"
        else:
            print("[online_learning] initial training")
            ok, accuracy, fresh = await run_in_threadpool(_retrain_with_units,
                clf, df, new_labels,
                use_augmentation=True,
                n_augmentations=5,
                acquisition_resistance_unit=incoming_unit,
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

        provenance = acquisition_provenance([acquisition] if acquisition else [])
        if previous_fitted:
            provenance["verified"] = bool(provenance["verified"] and previous_provenance.get("verified"))
            provenance["fingerprints"] = sorted(set(provenance["fingerprints"]) | set(previous_fingerprints))
            if not provenance["verified"]:
                provenance["warning"] = "Incoming or historical physical acquisition provenance is incomplete."
        clf.acquisition_provenance_ = provenance
        clf.acquisition_fingerprints_ = provenance["fingerprints"]
        model_path = clf.save_model(TRAINED_MODELS_DIR)
        reloaded = reload_smell_classifier()
        clf = state.smell_classifier  # pick up the reloaded instance
        if session_baseline and clf is not None and hasattr(clf, "update_baseline"):
            # Reload correctly resets live-session state. We are still in the
            # same running session, so reapply the already validated baseline
            # in memory without persisting the captured flag to disk.
            clf.update_baseline([session_baseline], ema=False)
            bind_live_baseline(clf, acquisition, baseline_context.get("session_id"))

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


@router.post("/smell/preview_csv")
async def preview_csv(data: CSVLearningData) -> Dict[str, Any]:
    """Inspect exactly the selected CSV windows, without fitting or disk writes."""
    try:
        source, selected, target, report = await run_in_threadpool(prepare_csv_training, data, allow_unlabelled=True)
        preview = await run_in_threadpool(csv_training_preview, source, selected, target, report)
        clf = state.smell_classifier
        if data.merge_history and clf is not None and clf.is_fitted and not selected.empty:
            available = set(preview["class_counts"]) | set(map(str, clf.classes_))
            if len(available) >= 2:
                preview["training_ready"] = True
                preview["message"] = "Selected windows can update the existing classes. History and unit compatibility will be checked during training."
        return preview
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/smell/learn_from_csv")
async def learn_from_csv(data: CSVLearningData) -> Dict[str, Any]:
    from .models import deployment_lock

    async with deployment_lock:
        return await _learn_from_csv(data)


async def _learn_from_csv(data: CSVLearningData) -> Dict[str, Any]:
    """Train from CSV, merging history by default or replacing it explicitly."""
    clf = state.require_classifier()

    if not data.csv_data:
        raise HTTPException(status_code=400, detail="CSV data required")

    try:
        source, df, target_column, profile_report = await run_in_threadpool(prepare_csv_training, data)
        csv_parse_report = profile_report["csv_parser"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    combined_classes = set(df[target_column])
    if data.merge_history and clf.is_fitted:
        combined_classes.update(clf.classes_)
    if df.empty or len(combined_classes) < 2:
        raise HTTPException(
            status_code=400,
            detail="No valid labelled training windows from at least two classes; use Preview CSV to inspect labels and extraction settings.",
        )

    try:
        incoming_unit = (
            normalize_resistance_unit(csv_parse_report.get("resistance_unit"))
            if csv_parse_report.get("parser") == "pcb_resistance_csv"
            else frame_resistance_unit(df)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        training_resistance_unit(clf, incoming_unit, merge_history=data.merge_history)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

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
        samples_received = len(source)
        X = df.drop(columns=[target_column, "Gas label", "Gas class", "timestamp", "Timestamp", "_source_row", "_label_block"], errors="ignore")
        y = df[target_column].astype(str)
        if data.lowercase_labels:
            y = y.str.lower()
        X = _fill_missing_sensors(X)
        existing_classes = set(clf.classes_) if clf.is_fitted else set()
        new_classes = set(y.unique()) - existing_classes
        mismatch = False

        ok, accuracy, fresh = await run_in_threadpool(_retrain_with_units,
            clf, X, y,
            use_augmentation=data.use_augmentation,
            n_augmentations=data.n_augmentations,
            merge_history=data.merge_history,
            classifier_backend=backend,
            baseline_mode=baseline_mode,
            snv=data.snv,
            acquisition_resistance_unit=incoming_unit,
            persist=not data.stage_candidate,
        )
        if not ok or fresh is None:
            raise HTTPException(status_code=500, detail="Training failed")
        provenance = dict(profile_report["acquisition_provenance"])
        if data.merge_history and clf.is_fitted:
            previous = getattr(clf, "acquisition_provenance_", {}) or {}
            provenance["verified"] = bool(provenance["verified"] and previous.get("verified"))
            provenance["fingerprints"] = sorted(set(provenance.get("fingerprints", [])) | set(getattr(clf, "acquisition_fingerprints_", []) or []))
            if not provenance["verified"]:
                provenance["warning"] = "Historical or incoming acquisition provenance is incomplete; physical board compatibility is unverified."
        fresh.acquisition_provenance_ = provenance
        fresh.acquisition_fingerprints_ = list(provenance.get("fingerprints", []))
        if data.stage_candidate:
            from enose.classifier.registry import create_candidate

            candidate_id = create_candidate(fresh, os.path.join(TRAINED_MODELS_DIR, "candidates"))
            return {
                "success": True, "staged": True, "candidate_id": candidate_id,
                "update_type": "candidate_staged", "samples_received": samples_received,
                "samples_processed": len(df), "current_accuracy": accuracy,
                "classes": fresh.classes_.tolist(), "merge_history": data.merge_history,
                "training_profile": profile, "profile_report": profile_report,
                "classifier_backend": getattr(fresh, "backend_name", backend),
                "baseline_mode": getattr(fresh.config, "baseline_mode", baseline_mode),
                "snv": bool(getattr(fresh.config, "snv", data.snv)),
                "validation": fresh.get_model_info().get("validation", {}),
                "calibration": fresh.get_model_info().get("calibration", {}),
                "message": "Candidate saved separately. Active model and canonical history are unchanged. Promotion requires an explicit decision; this holdout is not a paired comparison.",
            }
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

        from .vision import run_vision_inference
        detection_result = await run_vision_inference(
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
