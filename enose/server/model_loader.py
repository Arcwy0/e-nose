"""Persistence glue: reload the classifier from disk, append training data to the CSV."""

from __future__ import annotations

import datetime
import os
import traceback
from typing import Dict, List

import pandas as pd

from enose.classifier import BalancedRFClassifier, get_classifier_backend
from enose.config import ALL_SENSORS, ENV_DEFAULTS, SMELL_MODEL_PATH, TRAINED_MODELS_DIR, TRAINING_DATA_PATH

from . import state


def _selected_backend_name() -> str:
    """Read the classifier backend from ``ENOSE_CLASSIFIER`` (env).

    Accepts: ``balanced_rf`` / ``rf`` / ``brf`` (default), or ``xgb`` /
    ``xgboost``. Unknown values fall back to balanced_rf with a log line.
    """
    return os.environ.get("ENOSE_CLASSIFIER", "balanced_rf")


def reload_smell_classifier() -> bool:
    """Swap `state.smell_classifier` for a fresh load of the latest saved artifact.

    Looks for `trained_models/smell_classifier_sgd_latest.joblib`. Returns True on success.
    """
    latest_path = os.path.join(TRAINED_MODELS_DIR, "smell_classifier_sgd_latest.joblib")
    if not os.path.exists(latest_path):
        print(f"[reload] no artifact at {latest_path}")
        return False
    try:
        cls = get_classifier_backend(_selected_backend_name())
        clf = cls.load_model(latest_path)
        state.set_classifier(clf)
        print(f"[reload] ✓ reloaded {cls.__name__} from {latest_path}; classes={list(clf.classes_)}")
        return True
    except Exception as e:
        print(f"[reload] failed: {e}")
        traceback.print_exc()
        return False


def save_training_data(
    sensor_data: List[Dict[str, float]],
    labels: List[str],
    path: str = TRAINING_DATA_PATH,
) -> str:
    """Append a batch of labelled samples to the persistent CSV. Missing sensors get defaults."""
    df = pd.DataFrame(sensor_data)
    df["smell_label"] = labels
    df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for sensor in ALL_SENSORS:
        if sensor not in df.columns:
            df[sensor] = ENV_DEFAULTS.get(sensor, 0.0)

    file_exists = os.path.isfile(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)
    print(f"[save_training_data] appended {len(labels)} samples → {path}")
    return path


def load_or_create_classifier() -> BalancedRFClassifier:
    """Startup: load latest model if present, else return a fresh unfitted classifier.

    Backend selected via the ``ENOSE_CLASSIFIER`` env var (default: balanced_rf).
    The return type annotation stays ``BalancedRFClassifier`` because the XGB
    drop-in subclasses it — callers see the same interface either way.
    """
    cls = get_classifier_backend(_selected_backend_name())
    print(f"[startup] classifier backend: {cls.__name__}")
    if os.path.exists(SMELL_MODEL_PATH):
        try:
            clf = cls.load_model(SMELL_MODEL_PATH)
            print(f"[startup] ✓ loaded classifier; classes={list(clf.classes_)}")
            return clf
        except Exception as e:
            print(f"[startup] load failed ({e}); starting fresh")

    print("[startup] no existing model — creating empty classifier")
    return cls(online_learning=True)
