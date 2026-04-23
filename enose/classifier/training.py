"""Retrain-from-scratch with merged history.

`retrain_with_all_data` is the workhorse behind `/smell/online_learning` and
`/smell/learn_from_csv`. The SGD / calibrated-RF backends can't learn new
classes incrementally, so we concatenate the new batch with all prior training
data, rebuild a fresh classifier, fit, save. Returns (ok, balanced_accuracy).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from enose.config import ALL_SENSORS, TRAINING_DATA_PATH
from enose.utils.csv_io import (
    ensure_canonical_columns,
    load_training_csv,
    save_training_samples,
)

from .balanced_rf import BalancedRFClassifier
from .xgb_tabular import get_classifier_backend


DEFAULT_CSV_CANDIDATES: Tuple[str, ...] = (
    TRAINING_DATA_PATH,
    "data/smell_training_data.csv",
    "smell_training_data.csv",
    "data/smell_training_data_22features.csv",
)


def _frame_from_classifier_memory(classifier) -> Optional[pd.DataFrame]:
    """Fallback: rebuild a training frame from the classifier's in-memory history."""
    last = getattr(classifier, "last_training_data", None)
    if last is None or not len(last):
        return None
    df = last.copy()
    if "Gas name" in df.columns and "smell_label" not in df.columns:
        df = df.rename(columns={"Gas name": "smell_label"})
    return ensure_canonical_columns(df)


def _load_history(classifier) -> Optional[pd.DataFrame]:
    """Find prior training data: CSV on disk first, then classifier's in-memory record."""
    df = load_training_csv(DEFAULT_CSV_CANDIDATES)
    if df is not None:
        return ensure_canonical_columns(df)
    return _frame_from_classifier_memory(classifier)


_GROUP_CANDIDATES: Tuple[str, ...] = (
    # segment_id is the tightest meaningful group for continuous recordings:
    # two windows from the same bottle-swap segment are highly correlated,
    # so keeping them on one side of the split is what actually prevents
    # leakage. session_id is coarser but still useful when segment_id is
    # absent. The other names are kept for historical CSVs.
    "segment_id",
    "session_id",
    "session",
    "recording_id",
    "source",
    "source_file",
)


def _extract_groups(combined: pd.DataFrame) -> Optional[pd.Series]:
    """Pick a group column if the training frame carries one.

    Used by ``retrain_with_all_data`` to enable session-aware splitting when
    the incoming CSV has a column identifying which recording each row came
    from. Without this, consecutive frames from one run leak between train
    and test and inflate the reported accuracy.
    """
    for col in _GROUP_CANDIDATES:
        if col in combined.columns:
            s = combined[col].astype(str)
            if s.nunique() >= 2:
                print(f"[training] using '{col}' as group for GroupShuffleSplit ({s.nunique()} groups)")
                return s
    return None


def _cap_per_class(
    df: pd.DataFrame,
    label_col: str = "smell_label",
    multiplier: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cap every class at ``multiplier * min_class_count`` rows.

    Stricter than capping only the runner-up: on our dataset the top two
    classes ('air' and the next-largest recording) are both huge, and the
    old (runner_up * 1.15) rule left them tied and still drowning out the
    minority classes. Using multiplier * min_count scales the whole
    distribution against the smallest class instead.

    Sampling is seed-stable so retrains are reproducible. Rows are
    shuffled after the cut so dedup/stratify downstream see an interleaved
    stream rather than class-contiguous blocks.
    """
    if df.empty or multiplier <= 0:
        return df
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        return df
    min_n = int(counts.min())
    target = max(min_n, int(np.ceil(min_n * multiplier)))
    touched = counts[counts > target]
    if touched.empty:
        print(f"[training] per-class cap: no class exceeds {target}; no-op")
        return df

    print(
        f"[training] per-class cap: min_class={min_n}, "
        f"target=min*{multiplier:g}={target}; capping "
        f"{sorted(touched.index.tolist())}"
    )
    parts = []
    for label, n in counts.items():
        sub = df[df[label_col] == label]
        if n > target:
            sub = sub.sample(n=target, random_state=random_state)
            print(f"[training]   '{label}': {n} -> {target}")
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _drop_sensor_off_air_rows(
    df: pd.DataFrame,
    label_col: str = "smell_label",
    air_label: str = "air",
    required_nonzero_cols: Tuple[str, ...] = ("R1", "R4"),
) -> pd.DataFrame:
    """Remove 'air' rows whose key resistance columns are exactly 0.

    Analysis showed ~22% of air rows have R1==0 and R4==0 — these are
    sensor-off / startup frames that teach the model "all-zeros means
    air". Any live sensor reading lands far from that cluster, which is
    a useless decision boundary for actual clean-background detection.

    Only 'air' is filtered; other classes with zero values (~10% of
    coffee) are left alone since those classes are small and we can't
    afford to trim further. The log lines let us check the impact per
    retrain.
    """
    if df.empty or label_col not in df.columns:
        return df
    air_mask = df[label_col].astype(str).str.lower() == air_label
    if not air_mask.any():
        return df
    missing = [c for c in required_nonzero_cols if c not in df.columns]
    if missing:
        print(f"[training] zero-air filter: columns {missing} not present; skipping")
        return df

    zero_mask = np.logical_and.reduce(
        [df[c].astype(float).values == 0.0 for c in required_nonzero_cols]
    )
    drop_mask = air_mask.values & zero_mask
    n_drop = int(drop_mask.sum())
    if n_drop == 0:
        return df
    pct = 100.0 * n_drop / int(air_mask.sum())
    print(
        f"[training] dropped {n_drop} sensor-off 'air' rows "
        f"({pct:.1f}% of air class) where {list(required_nonzero_cols)} all == 0"
    )
    return df[~drop_mask].reset_index(drop=True)


def _auto_run_ids(
    df: pd.DataFrame,
    label_col: str = "smell_label",
) -> pd.Series:
    """Assign a run-id per contiguous stretch of identical labels.

    The historical CSV is a concatenation of 8 recordings — when the label
    column changes row-to-row, a new recording has begun. Using the
    resulting run-id as the group for ``GroupShuffleSplit`` keeps an entire
    recording on one side of the split, which is the only honest way to
    estimate generalization on a time-series dataset where adjacent rows
    are near-duplicates by construction.
    """
    labels = df[label_col].astype(str)
    run_id = (labels != labels.shift()).cumsum()
    return run_id.astype(str).reset_index(drop=True)


# Back-compat shim: old callers (if any remain) still import _cap_majority_class.
# The new call-site inside retrain_with_all_data uses _cap_per_class.
_cap_majority_class = _cap_per_class


def retrain_with_all_data(
    classifier,
    new_X: pd.DataFrame,
    new_y: pd.Series,
    use_augmentation: bool = True,
    n_augmentations: int = 1,
    combined_save_path: str = "data/smell_training_data_22features.csv",
    model_out_dir: str = "trained_models",
    groups: Optional[pd.Series] = None,
    balance_majority: bool = True,
    per_class_cap_multiplier: Optional[float] = None,
    drop_sensor_off_air: Optional[bool] = None,
) -> Tuple[bool, float, Optional[BalancedRFClassifier]]:
    """Merge new samples with historical data, train a fresh classifier, save everything.

    Returns (ok, balanced_accuracy, new_classifier). The new classifier is returned
    so the caller (server) can swap it into its global reference atomically.

    Cleanup order (all run before splitting so train and test see the same
    distribution):
      1. Drop unusable labels ("" / NaN / "unknown").
      2. Drop sensor-off 'air' rows where R1==0 AND R4==0
         (config.drop_sensor_off_air_rows; ~22% of air historically).
      3. Per-class cap at ``multiplier * min_class_count`` rows
         (config.per_class_cap_multiplier, default 3x).
      4. Auto-assign a run-id per contiguous label stretch if no explicit
         group column exists, so GroupShuffleSplit keeps whole recordings
         on one side of the train/test split.
    """
    # Pull cleanup thresholds from the classifier's config unless the caller
    # overrides. Keeps behavior tunable from a single place.
    cfg = getattr(classifier, "config", None)
    if per_class_cap_multiplier is None:
        per_class_cap_multiplier = float(
            getattr(cfg, "per_class_cap_multiplier", 3.0)
        ) if cfg else 3.0
    if drop_sensor_off_air is None:
        drop_sensor_off_air = bool(
            getattr(cfg, "drop_sensor_off_air_rows", True)
        ) if cfg else True

    history = _load_history(classifier)

    new_frame = new_X.copy()
    new_frame["smell_label"] = pd.Series(new_y).astype(str).values
    new_frame = ensure_canonical_columns(new_frame)

    if history is not None and len(history):
        combined = pd.concat([history, new_frame], ignore_index=True)
    else:
        combined = new_frame
        print("[training] no history found — training on new batch only")

    # drop rows with unusable labels
    mask = combined["smell_label"].notna() & (combined["smell_label"] != "") & (combined["smell_label"] != "unknown")
    combined = combined[mask].reset_index(drop=True)
    combined["smell_label"] = combined["smell_label"].astype(str).str.lower().str.strip()

    if drop_sensor_off_air:
        combined = _drop_sensor_off_air_rows(combined, label_col="smell_label")

    # Compute auto run-ids BEFORE any shuffling / capping so contiguous-label
    # stretches still map to sessions. The cap reshuffles rows, which would
    # turn label-adjacent runs into per-row groups (useless).
    combined = combined.reset_index(drop=True)
    combined["_run_id"] = _auto_run_ids(combined, label_col="smell_label").values

    if balance_majority:
        combined = _cap_per_class(
            combined,
            label_col="smell_label",
            multiplier=per_class_cap_multiplier,
        )

    classes = sorted(combined["smell_label"].unique().tolist())
    if len(classes) < 2:
        print(f"[training] need ≥2 classes, got {classes}")
        return False, 0.0, None
    print(f"[training] {len(combined)} samples across {len(classes)} classes: {classes}")
    print(f"[training] class counts: {combined['smell_label'].value_counts().to_dict()}")

    X_all = combined[ALL_SENSORS]
    y_all = combined["smell_label"]
    if groups is None:
        groups = _extract_groups(combined)
    if groups is None:
        # Fall back to the run-id we stamped before capping. It survived the
        # shuffle as a column, so row-alignment is preserved.
        groups = combined["_run_id"].astype(str).reset_index(drop=True)
        n_groups = groups.nunique()
        print(
            f"[training] no group column found; using contiguous-label run-id "
            f"as group ({n_groups} runs across {len(combined)} rows)"
        )

    # Honor ENOSE_CLASSIFIER so a retrain triggered via /smell/learn_from_csv
    # uses the same backend the server was started with; returning a
    # different class would break the atomic swap in state.set_classifier.
    cls = get_classifier_backend(os.environ.get("ENOSE_CLASSIFIER", "balanced_rf"))
    fresh = cls(online_learning=True)
    fresh.original_features = list(ALL_SENSORS)
    bal_acc = fresh.train(
        X_all, y_all,
        use_augmentation=use_augmentation,
        n_augmentations=n_augmentations,
        test_size=0.1,
        groups=groups,
    )

    model_path = fresh.save_model(model_out_dir)
    # Drop the internal run-id column before persisting to CSV — it's
    # regenerated on the next retrain and shouldn't be part of the stored
    # schema.
    save_training_samples(combined.drop(columns=["_run_id"], errors="ignore"), combined_save_path)
    print(f"[training] saved model → {model_path}")
    print(f"[training] saved combined data → {combined_save_path}")
    return True, float(bal_acc), fresh


def list_known_classes(classifier) -> List[str]:
    """Shorthand: classifier.classes_ as plain Python strings."""
    return [str(c) for c in getattr(classifier, "classes_", [])]
