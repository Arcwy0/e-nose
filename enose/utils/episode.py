"""Low-latency episode extraction for continuous e-nose recordings.

Unlike the plateau profile, this profile deliberately learns the early part of
the response.  Each exposure is paired with either its preceding clean-air
tail or, for legacy recordings that did not save recovery air, a short onset
proxy.  The emitted values stay in raw resistance space so the existing
baseline-relative classifier owns the log-ratio transformation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from enose.config import ALL_SENSORS, ENV_DEFAULTS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS
from enose.utils.plateau import _canonical_labels


@dataclass(frozen=True)
class EarlyResponseConfig:
    latencies_seconds: Tuple[float, ...] = (20.0, 30.0, 60.0, 120.0)
    onset_baseline_seconds: float = 10.0
    response_window_seconds: float = 10.0
    preceding_air_seconds: float = 60.0
    max_preceding_air_gap_seconds: float = 900.0
    max_gap_seconds: float = 300.0
    min_window_samples: int = 3
    min_valid_sensors: int = 12
    air_label: str = "air"


def _timestamp(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:  # pandas < 2.0
        return pd.to_datetime(values, errors="coerce")


def _median_row(frame: pd.DataFrame) -> Tuple[Dict[str, float], int]:
    row: Dict[str, float] = {}
    valid_resistance = 0
    for sensor in ALL_SENSORS:
        if sensor not in frame.columns:
            row[sensor] = ENV_DEFAULTS.get(sensor, 0.0)
            continue
        values = pd.to_numeric(frame[sensor], errors="coerce")
        if sensor in RESISTANCE_SENSORS:
            values = values.where(values > 0.0)
        finite = values.dropna()
        value = finite.median() if len(finite) else float("nan")
        if pd.notna(value):
            row[sensor] = float(value)
            if sensor in RESISTANCE_SENSORS:
                valid_resistance += 1
        else:
            row[sensor] = ENV_DEFAULTS.get(sensor, 0.0)
    return row, valid_resistance


def build_early_response_training_frame(
    df: pd.DataFrame,
    label_col: str = "Gas name",
    timestamp_col: str = "Timestamp",
    config: Optional[EarlyResponseConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return early response windows and their paired baseline rows."""
    cfg = config or EarlyResponseConfig()
    if label_col not in df.columns or timestamp_col not in df.columns:
        raise ValueError("early-response training requires Timestamp and the selected label column")
    missing = [sensor for sensor in RESISTANCE_SENSORS if sensor not in df.columns]
    if missing:
        raise ValueError(f"early-response training requires R1-R17; missing {missing}")

    work = df.copy()
    work["_timestamp"] = _timestamp(work[timestamp_col])
    invalid_time = int(work["_timestamp"].isna().sum())
    work = work.dropna(subset=["_timestamp"]).sort_values("_timestamp").reset_index(drop=True)
    work["_label"], aliases = _canonical_labels(work, label_col)
    gaps = work["_timestamp"].diff().dt.total_seconds()
    starts = work["_label"].ne(work["_label"].shift()) | gaps.gt(cfg.max_gap_seconds)
    work["_segment"] = starts.cumsum().astype(int)
    segments = [group.reset_index(drop=True) for _, group in work.groupby("_segment", sort=False)]

    air_label = cfg.air_label.lower().strip()
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for index, exposure in enumerate(segments):
        label = str(exposure["_label"].iloc[0])
        if label == air_label:
            continue
        start = exposure["_timestamp"].iloc[0]
        end = exposure["_timestamp"].iloc[-1]
        preceding = segments[index - 1] if index else None
        use_air = False
        if preceding is not None and str(preceding["_label"].iloc[0]) == air_label:
            gap = float((start - preceding["_timestamp"].iloc[-1]).total_seconds())
            use_air = 0.0 <= gap <= cfg.max_preceding_air_gap_seconds
        if use_air:
            baseline_frame = preceding[
                preceding["_timestamp"]
                >= preceding["_timestamp"].iloc[-1]
                - pd.Timedelta(seconds=cfg.preceding_air_seconds)
            ]
            baseline_kind = "preceding_air"
        else:
            baseline_frame = exposure[
                exposure["_timestamp"]
                < start + pd.Timedelta(seconds=cfg.onset_baseline_seconds)
            ]
            baseline_kind = "onset_proxy"
        baseline, baseline_valid = _median_row(baseline_frame)
        if len(baseline_frame) < cfg.min_window_samples or baseline_valid < cfg.min_valid_sensors:
            details.append({
                "label": label, "start": str(start), "kept": False,
                "reason": "insufficient baseline sensor coverage",
                "valid_baseline_sensors": baseline_valid,
            })
            continue

        session_id = str(start.date())
        segment_id = int(exposure["_segment"].iloc[0])
        exposure_id = f"{session_id}:{segment_id}:{label}"
        baseline.update({
            "Gas name": air_label, "phase": "baseline_air",
            "baseline_kind": baseline_kind, "latency_seconds": 0.0,
            "session_id": session_id, "exposure_id": exposure_id,
            "segment_id": segment_id, "n_frames": int(len(baseline_frame)),
            "valid_sensors": baseline_valid, "t_start": str(baseline_frame["_timestamp"].iloc[0]),
            "t_end": str(baseline_frame["_timestamp"].iloc[-1]),
        })
        exposure_rows: List[Dict[str, Any]] = []
        for latency in sorted(set(float(value) for value in cfg.latencies_seconds)):
            response_end = start + pd.Timedelta(seconds=latency)
            if response_end > end:
                continue
            response_start = response_end - pd.Timedelta(seconds=cfg.response_window_seconds)
            # When the segment itself supplies the baseline, do not overlap it
            # with the response window.
            if baseline_kind == "onset_proxy":
                response_start = max(
                    response_start,
                    start + pd.Timedelta(seconds=cfg.onset_baseline_seconds),
                )
            window = exposure[
                (exposure["_timestamp"] >= response_start)
                & (exposure["_timestamp"] <= response_end)
            ]
            response, valid = _median_row(window)
            if len(window) < cfg.min_window_samples or valid < cfg.min_valid_sensors:
                continue
            response.update({
                "Gas name": label, "phase": "early_response",
                "baseline_kind": baseline_kind, "latency_seconds": latency,
                "session_id": session_id, "exposure_id": exposure_id,
                "segment_id": segment_id, "n_frames": int(len(window)),
                "valid_sensors": valid, "t_start": str(window["_timestamp"].iloc[0]),
                "t_end": str(window["_timestamp"].iloc[-1]),
            })
            exposure_rows.append(response)
        if exposure_rows:
            rows.append(baseline)
            rows.extend(exposure_rows)
        details.append({
            "exposure_id": exposure_id, "label": label, "start": str(start),
            "baseline_kind": baseline_kind, "valid_baseline_sensors": baseline_valid,
            "latencies_kept": [row["latency_seconds"] for row in exposure_rows],
            "kept": bool(exposure_rows),
        })

    if not rows:
        raise ValueError("early-response extraction produced no usable exposures")
    result = pd.DataFrame(rows)
    report: Dict[str, Any] = {
        "profile": "early_response", "config": asdict(cfg),
        "input_rows": int(len(df)), "invalid_timestamp_rows": invalid_time,
        "segments": int(len(segments)),
        "exposures": int(result["exposure_id"].nunique()),
        "sessions": int(result["session_id"].nunique()),
        "output_windows": int(len(result)),
        "class_windows": {
            str(key): int(value) for key, value in result["Gas name"].value_counts().items()
        },
        "baseline_kinds": {
            str(key): int(value)
            for key, value in result[result["phase"] == "baseline_air"]["baseline_kind"].value_counts().items()
        },
        "label_aliases": aliases, "exposure_details": details,
    }
    return result, report
