"""Auditable quality checks for uploaded e-nose recordings."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from enose.config import ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS


def summarize_online_capture(
    frame: pd.DataFrame,
    max_windows: int = 5,
    min_frames_per_window: int = 5,
    min_valid_sensors: int = 12,
    max_relative_mad: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reduce a sequential online capture to zero-aware median windows.

    Sending every 5–10 Hz frame to the classifier gives one physical exposure
    hundreds of votes and preserves multiplexing zeros.  A few ordered median
    windows retain genuine response variation while limiting correlation and
    rejecting captures where too little of the array was active.
    """
    if frame.empty:
        raise ValueError("online capture is empty")
    n_windows = min(
        max(1, int(max_windows)),
        max(1, len(frame) // max(1, int(min_frames_per_window))),
    )
    chunks = np.array_split(np.arange(len(frame)), n_windows)
    rows = []
    stability_scores = []
    rejected = 0
    for indices in chunks:
        part = frame.iloc[indices]
        row: Dict[str, float] = {}
        valid_resistance = 0
        relative_mads = []
        for sensor in RESISTANCE_SENSORS:
            values = (
                pd.to_numeric(part[sensor], errors="coerce")
                if sensor in part.columns else pd.Series(dtype=float)
            )
            positive = values.where(values > 0.0).dropna()
            if len(positive):
                median = float(positive.median())
                row[sensor] = median
                valid_resistance += 1
                relative_mads.append(
                    float((positive - median).abs().median()) / max(abs(median), 1e-9)
                )
            else:
                row[sensor] = 0.0
        for sensor in ENVIRONMENTAL_SENSORS:
            values = (
                pd.to_numeric(part[sensor], errors="coerce").dropna()
                if sensor in part.columns else pd.Series(dtype=float)
            )
            row[sensor] = float(values.median()) if len(values) else float("nan")
        stability_score = (
            float(np.quantile(relative_mads, 0.8)) if relative_mads else float("inf")
        )
        if valid_resistance < min_valid_sensors or stability_score > max_relative_mad:
            rejected += 1
            continue
        row["capture_window"] = len(rows)
        row["source_frames"] = int(len(part))
        row["valid_resistance_sensors"] = valid_resistance
        rows.append(row)
        stability_scores.append(stability_score)
    if not rows:
        raise ValueError(
            "no online windows passed sensor coverage and stability checks "
            f"(need {min_valid_sensors} active sensors and relative MAD <= {max_relative_mad})"
        )
    return pd.DataFrame(rows), {
        "input_frames": int(len(frame)),
        "output_windows": int(len(rows)),
        "rejected_windows": int(rejected),
        "aggregation": "ordered_zero_aware_median",
        "max_relative_mad": float(max_relative_mad),
        "stability_scores": [round(value, 6) for value in stability_scores],
    }


def analyze_recording_quality(
    frame: pd.DataFrame,
    label_column: str = "Gas name",
) -> Dict[str, Any]:
    """Summarize sensor coverage without modifying the recording.

    Old recorder versions use zero as a multiplexing ``not updated`` marker.
    Zero is therefore counted as missing for R1-R17. The CSV-training API
    returns this report so an operator can distinguish a large file from a
    genuinely information-rich experiment.
    """
    n_rows = int(len(frame))
    resistance: Dict[str, Dict[str, Any]] = {}
    active_per_row = np.zeros(n_rows, dtype=int)
    dead = []
    sparse = []
    for sensor in RESISTANCE_SENSORS:
        if sensor in frame.columns:
            values = pd.to_numeric(frame[sensor], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & (values > 0.0)
            unique_positive = int(pd.Series(values[valid]).nunique()) if valid.any() else 0
        else:
            valid = np.zeros(n_rows, dtype=bool)
            unique_positive = 0
        active_per_row += valid.astype(int)
        positive_fraction = float(valid.mean()) if n_rows else 0.0
        if positive_fraction < 0.10 or unique_positive < 2:
            status = "dead"
            dead.append(sensor)
        elif positive_fraction < 0.50:
            status = "sparse"
            sparse.append(sensor)
        else:
            status = "ok"
        resistance[sensor] = {
            "positive_fraction": round(positive_fraction, 4),
            "zero_or_invalid_fraction": round(1.0 - positive_fraction, 4),
            "unique_positive_values": unique_positive,
            "status": status,
        }

    labels: Dict[str, int] = {}
    if label_column in frame.columns:
        normalized = frame[label_column].astype(str).str.strip().str.lower()
        labels = {str(k): int(v) for k, v in normalized.value_counts().items()}

    environment = {}
    for sensor in ENVIRONMENTAL_SENSORS:
        if sensor not in frame.columns:
            environment[sensor] = {"available": False, "finite_fraction": 0.0}
            continue
        values = pd.to_numeric(frame[sensor], errors="coerce").to_numpy(float)
        finite_fraction = float(np.isfinite(values).mean()) if n_rows else 0.0
        environment[sensor] = {
            "available": bool(finite_fraction > 0.0),
            "finite_fraction": round(finite_fraction, 4),
        }

    rows_with_12 = float((active_per_row >= 12).mean()) if n_rows else 0.0
    warnings = []
    if dead:
        warnings.append(f"dead or constant resistance channels: {', '.join(dead)}")
    if sparse:
        warnings.append(f"sparse resistance channels (<50% positive): {', '.join(sparse)}")
    if rows_with_12 < 0.80:
        warnings.append(
            f"only {rows_with_12:.1%} of rows contain at least 12 positive resistance readings"
        )
    if len(labels) < 2:
        warnings.append("fewer than two smell classes are present")

    return {
        "rows": n_rows,
        "label_counts": labels,
        "resistance_sensors": resistance,
        "dead_resistance_sensors": dead,
        "sparse_resistance_sensors": sparse,
        "row_coverage": {
            "mean_active_resistance_sensors": round(float(active_per_row.mean()), 3)
            if n_rows else 0.0,
            "fraction_with_at_least_12": round(rows_with_12, 4),
        },
        "environmental_sensors": environment,
        "warnings": warnings,
    }
