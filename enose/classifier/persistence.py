"""Joblib save/load for BalancedRFClassifier with legacy-artifact fallback.

Why the loader is messy: we've shipped multiple formats — old sklearn pipelines,
dicts without scalers, current dict format. This module hides that pain so the
classifier body stays focused on training/inference.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np


def build_save_payload(classifier) -> Dict[str, Any]:
    """Extract persistent state from a BalancedRFClassifier.

    Saves enough of the pipeline configuration (log1p flag, scaler kind) that
    a model trained with one preprocessing regime can be loaded and used
    consistently; predictions MUST go through the same transforms they were
    trained on.
    """
    return {
        "model": classifier.model,
        "scaler_r": classifier.scaler_r,
        "classes_": classifier.classes_,
        "class_weights_": classifier.class_weights_,
        "selected_features": classifier.selected_features,
        "original_features": classifier.original_features,
        "training_history": classifier.training_history,
        "env_ranges": classifier.config.env_ranges,
        "env_medians": classifier.config.env_medians,
        "r_clip_": classifier.r_clip_,
        "r_median_": classifier.r_median_,
        "feature_means_": classifier.feature_means_,
        "feature_stds_": classifier.feature_stds_,
        "class_centroids_": classifier.class_centroids_,
        # Pipeline regime flags — must be restored on load for inference to
        # match training. Older payloads lacking these keys default to
        # "log1p off, standard scaler" on the read side.
        "use_log1p": bool(getattr(classifier.config, "use_log1p", False)),
        "log1p_applied": bool(getattr(classifier, "_log1p_applied", False)),
        "scaler_kind": str(getattr(classifier.config, "scaler_kind", "standard")),
        "use_env_sensors": bool(getattr(classifier.config, "use_env_sensors", True)),
        "dead_env_features": tuple(
            getattr(classifier.config, "dead_env_features", ()) or ()
        ),
        # Per-class metrics captured at the last training run. Persisted so
        # they're available via /smell/model_info after a server restart
        # without having to retrain.
        "per_class_metrics_": dict(getattr(classifier, "per_class_metrics_", {}) or {}),
        "confusion_matrix_": getattr(classifier, "confusion_matrix_", None),
        "confusion_labels_": list(getattr(classifier, "confusion_labels_", []) or []),
        "last_test_size_": int(getattr(classifier, "last_test_size_", 0) or 0),
    }


def save(classifier, out_dir: str = "trained_models") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"smell_classifier_22f_{ts}.joblib")
    payload = build_save_payload(classifier)
    joblib.dump(payload, path)

    # The server's reload step reads ``smell_classifier_sgd_latest.joblib``
    # (see enose.config.SMELL_MODEL_PATH). Keep that file in sync with the
    # fresh save so post-train reload doesn't pick up a stale artifact and
    # clobber the classifier we just fit.
    latest_path = os.path.join(out_dir, "smell_classifier_sgd_latest.joblib")
    joblib.dump(payload, latest_path)

    # Human-readable manifest
    manifest = {
        "path": path,
        "latest_path": latest_path,
        "n_features": len(classifier.ALL_SENSORS),
        "features": classifier.ALL_SENSORS,
        "classes": list(map(str, classifier.classes_)),
        "balanced_accuracy_last": (
            classifier.training_history["balanced_accuracy"][-1]
            if classifier.training_history["balanced_accuracy"]
            else None
        ),
        "saved_at": ts,
    }
    with open(os.path.join(out_dir, f"smell_classifier_22f_{ts}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def _load_legacy_estimator(payload, cls) -> "cls":
    """Payload is a raw sklearn estimator/pipeline — wrap it but skip internal scaling."""
    obj = cls()
    obj.is_fitted = True
    obj.model = payload
    obj.classes_ = getattr(payload, "classes_", np.array([]))
    obj._legacy_pipeline = True
    obj._use_internal_scaler = False
    return obj


def _extract_model(payload: dict) -> Tuple[Any, bool]:
    """Return (model, is_legacy_pipeline). Raise if no estimator found."""
    if "model" in payload and hasattr(payload["model"], "predict"):
        return payload["model"], False
    if "pipeline" in payload and hasattr(payload["pipeline"], "predict"):
        return payload["pipeline"], True
    for v in payload.values():
        if hasattr(v, "predict"):
            return v, True
    raise ValueError("Unrecognized model payload: no estimator found")


def load(path: str, cls) -> "cls":
    """Restore a classifier. Handles new dict format and legacy artifacts."""
    payload = joblib.load(path)

    # Case A: bare sklearn estimator/pipeline
    if hasattr(payload, "predict"):
        return _load_legacy_estimator(payload, cls)

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported model payload type: {type(payload)}")

    # Case B: dict container
    obj = cls()
    obj.is_fitted = True
    obj._legacy_pipeline = False
    obj._use_internal_scaler = True

    model, is_legacy = _extract_model(payload)
    obj.model = model
    if is_legacy:
        obj._legacy_pipeline = True
        obj._use_internal_scaler = False

    obj.classes_ = np.array(
        payload.get("classes_", getattr(obj.model, "classes_", np.array([])))
    )

    # Scaler: new format uses scaler_r; legacy may have plain "scaler" inside a Pipeline
    if "scaler_r" in payload:
        obj.scaler_r = payload["scaler_r"]
        obj._use_internal_scaler = True
    elif "scaler" in payload:
        obj.scaler_r = payload["scaler"]
        obj._use_internal_scaler = False
    else:
        obj._use_internal_scaler = False

    obj.class_weights_ = payload.get("class_weights_", {})
    obj.selected_features = payload.get("selected_features", list(obj.ALL_SENSORS))
    obj.original_features = payload.get("original_features", list(obj.ALL_SENSORS))
    obj.training_history = payload.get(
        "training_history", {"accuracy": [], "balanced_accuracy": []}
    )
    obj.r_clip_ = payload.get("r_clip_", {})
    obj.r_median_ = payload.get("r_median_", {})
    obj.feature_means_ = payload.get("feature_means_", {})
    obj.feature_stds_ = payload.get("feature_stds_", {})
    obj.class_centroids_ = payload.get("class_centroids_", None)

    if "env_ranges" in payload:
        obj.config.env_ranges = payload["env_ranges"]
    if "env_medians" in payload:
        obj.config.env_medians = payload["env_medians"]

    # Pipeline regime — for legacy payloads without these keys, default to
    # the old behavior (no log1p, standard scaler) so existing trained models
    # keep predicting the way they were trained.
    obj.config.use_log1p = bool(payload.get("use_log1p", False))
    obj.config.scaler_kind = str(payload.get("scaler_kind", "standard"))
    obj._log1p_applied = bool(payload.get("log1p_applied", obj.config.use_log1p))
    # Default True on legacy payloads — every pre-flag model was trained on
    # all 22 features, so restoring them at inference matches training.
    obj.config.use_env_sensors = bool(payload.get("use_env_sensors", True))
    # Legacy artifacts predate the dead-feature filter; an empty tuple means
    # "no features to exclude" and matches the old behavior.
    dead = payload.get("dead_env_features", ())
    obj.config.dead_env_features = tuple(dead) if dead else ()

    # Per-class metrics — empty on legacy payloads; the UI shows a "retrain to
    # populate" hint when these come back blank.
    obj.per_class_metrics_ = dict(payload.get("per_class_metrics_", {}) or {})
    obj.confusion_matrix_ = payload.get("confusion_matrix_", None)
    obj.confusion_labels_ = list(payload.get("confusion_labels_", []) or [])
    obj.last_test_size_ = int(payload.get("last_test_size_", 0) or 0)
    return obj
