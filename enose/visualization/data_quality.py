"""Data-quality summary: counts, class balance, per-sensor descriptive stats."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

from enose.config import ALL_SENSORS


def analyze_data_quality(
    df: pd.DataFrame,
    label_col: str = "Gas name",
    sensors: Sequence[str] = ALL_SENSORS,
) -> Dict[str, Any]:
    """Row counts, class distribution, sensor describe-stats. Pure — no plotting."""
    if df is None or len(df) == 0:
        return {"summary": "No training data available."}

    present_sensors = [s for s in sensors if s in df.columns]
    return {
        "n_samples": int(len(df)),
        "classes": sorted(list(df[label_col].unique())),
        "class_counts": df[label_col].value_counts().to_dict(),
        "sensor_stats": df[present_sensors].describe().to_dict(),
        "n_sensors": len(present_sensors),
    }
