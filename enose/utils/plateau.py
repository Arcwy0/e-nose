"""Turn continuous labelled e-nose recordings into stable exposure windows.

The operator changes the label at the same instant as the bottle/valve.  Raw
segments therefore contain a response transient, while the following ``air``
segment contains the reverse recovery transient.  A point classifier cannot
infer the direction of travel from one row.  This module keeps the raw CSV
untouched and emits a compact, auditable training frame containing only:

* clean-air windows immediately *before* an exposure; and
* late, low-slope windows from the corresponding analyte exposure.

Every emitted row carries both ``session_id`` (calendar day, used for honest
train/test splitting) and ``exposure_id`` (used to pair an analyte with its own
clean-air baseline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from enose.config import ALL_SENSORS, ENV_DEFAULTS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS


@dataclass(frozen=True)
class PlateauConfig:
    window_seconds: float = 60.0
    stride_seconds: float = 60.0
    baseline_tail_seconds: float = 600.0
    plateau_fraction: float = 0.25
    tail_margin_seconds: float = 30.0
    stability_quantile: float = 0.80
    max_relative_slope: float = 0.002
    min_window_samples: int = 5
    # Prevent a many-hour exposure from overwhelming shorter repetitions.
    # Windows are sampled evenly across the stable tail when this cap applies.
    max_windows_per_phase: int = 30
    # End an exposure across recorder restarts or overnight pauses even when
    # the operator left the same label selected.
    max_gap_seconds: float = 300.0
    air_label: str = "air"

    def __post_init__(self):
        for name in ("window_seconds", "stride_seconds", "baseline_tail_seconds", "max_gap_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.plateau_fraction <= 1 or not 0 <= self.stability_quantile <= 1:
            raise ValueError("plateau_fraction must be in (0, 1]; stability_quantile in [0, 1]")
        if not np.isfinite(self.tail_margin_seconds) or self.tail_margin_seconds < 0:
            raise ValueError("tail_margin_seconds must be finite and nonnegative")
        if not np.isfinite(self.max_relative_slope) or self.max_relative_slope < 0:
            raise ValueError("max_relative_slope must be finite and nonnegative")
        if self.min_window_samples < 2 or self.max_windows_per_phase < 1:
            raise ValueError("window sample count must be >= 2 and phase cap >= 1")


def _canonical_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.Series, Dict[str, str]]:
    """Normalize spelling and merge names that share an explicit numeric class id."""
    labels = df[label_col].astype(str).str.strip().str.lower()
    aliases: Dict[str, str] = {}
    class_col = next((c for c in ("Gas label", "Gas class") if c in df.columns), None)
    if class_col is None:
        return labels, aliases
    # Missing numeric class ids are common after combining historical exports;
    # pandas cannot always materialize a categorical group for NaN.
    valid_class = df[class_col].notna()
    for _, idx in df.loc[valid_class].groupby(class_col).groups.items():
        names = labels.loc[idx]
        canonical = str(names.value_counts().index[0])
        for name in names.unique():
            if str(name) != canonical:
                aliases[str(name)] = canonical
    return labels.replace(aliases), aliases


def _window_slices(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PlateauConfig,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    phase: str = "",
) -> List[pd.DataFrame]:
    windows: List[pd.DataFrame] = []
    cursor = start
    width = pd.Timedelta(seconds=cfg.window_seconds)
    stride = pd.Timedelta(seconds=cfg.stride_seconds)
    while cursor + width <= end:
        win = frame[(frame["_timestamp"] >= cursor) & (frame["_timestamp"] < cursor + width)]
        if len(win) >= cfg.min_window_samples:
            windows.append(win)
        elif diagnostics is not None:
            diagnostics.append({"phase": phase, "start": str(cursor), "end": str(cursor + width),
                                "kept": False, "reason": "too few valid samples", "score": None})
        cursor += stride
    return windows


def relative_slope_score(window: pd.DataFrame, quantile: float = 0.80) -> float:
    """Robust aggregate of per-sensor ``abs((dR/dt) / mean(R))``.

    The quantile prevents one dead/noisy channel from rejecting an otherwise
    stable array, unlike an all-sensors maximum gate.
    """
    t = window["_timestamp"]
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    if len(seconds) < 2 or float(np.ptp(seconds)) <= 0.0:
        return float("inf")
    centered_t = seconds - seconds.mean()
    denom = float(centered_t @ centered_t)
    numeric = window[RESISTANCE_SENSORS].apply(pd.to_numeric, errors="coerce")
    # Older recorder versions emitted 0 when a multiplexed channel was not
    # sampled on that frame. Carry the last real reading through the short
    # window instead of treating "not updated" as zero resistance.
    numeric = numeric.mask(numeric <= 0.0).ffill().bfill()
    values = numeric.to_numpy(float)
    means = numeric.mean(axis=0, skipna=True).to_numpy(float)
    filled = np.where(np.isfinite(values), values, means)
    slopes = centered_t @ filled / max(denom, 1e-12)
    relative = np.abs(slopes) / np.maximum(np.abs(means), 1e-9)
    relative = relative[np.isfinite(relative)]
    return float(np.quantile(relative, quantile)) if len(relative) else float("inf")


def _summarize_window(
    window: pd.DataFrame,
    label: str,
    phase: str,
    session_id: str,
    exposure_id: str,
    segment_id: int,
    score: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for sensor in ALL_SENSORS:
        if sensor in window.columns:
            values = pd.to_numeric(window[sensor], errors="coerce")
            if sensor in RESISTANCE_SENSORS:
                values = values.where(values > 0.0)
            finite = values.dropna()
            value = finite.median() if len(finite) else float("nan")
            row[sensor] = float(value) if pd.notna(value) else ENV_DEFAULTS.get(sensor, 0.0)
        else:
            row[sensor] = ENV_DEFAULTS.get(sensor, 0.0) if sensor in ENVIRONMENTAL_SENSORS else 0.0
    row.update(
        {
            "Gas name": label,
            "phase": phase,
            "session_id": session_id,
            "exposure_id": exposure_id,
            "segment_id": int(segment_id),
            "n_frames": int(len(window)),
            "stability_score": float(score),
            "t_start": str(window["_timestamp"].iloc[0]),
            "t_end": str(window["_timestamp"].iloc[-1]),
        }
    )
    return row


def build_plateau_training_frame(
    df: pd.DataFrame,
    label_col: str = "Gas name",
    timestamp_col: str = "Timestamp",
    config: Optional[PlateauConfig] = None,
    allow_empty: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(stable_windows, audit_report)`` for a continuous recording."""
    cfg = config or PlateauConfig()
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} not found")
    if timestamp_col not in df.columns:
        raise ValueError(
            f"plateau training requires timestamp column {timestamp_col!r}; "
            "choose raw-row training for an already aggregated CSV"
        )
    missing = [c for c in RESISTANCE_SENSORS if c not in df.columns]
    if missing:
        raise ValueError(f"plateau training requires R1-R17; missing {missing}")

    work = df.copy()
    try:
        work["_timestamp"] = pd.to_datetime(
            work[timestamp_col], errors="coerce", format="mixed"
        )
    except TypeError:  # pandas < 2.0
        work["_timestamp"] = pd.to_datetime(work[timestamp_col], errors="coerce")
    invalid_time = int(work["_timestamp"].isna().sum())
    work = work.dropna(subset=["_timestamp"]).sort_values("_timestamp").reset_index(drop=True)
    work["_label"], aliases = _canonical_labels(work, label_col)
    gap_seconds = work["_timestamp"].diff().dt.total_seconds()
    new_segment = work["_label"].ne(work["_label"].shift()) | gap_seconds.gt(cfg.max_gap_seconds)
    for name in ("session_id", "recording_id", "_label_block"):
        if name in work:
            new_segment |= work[name].ne(work[name].shift())
    work["_segment"] = new_segment.cumsum().astype(int)
    segments = [g.reset_index(drop=True) for _, g in work.groupby("_segment", sort=False)]

    rows: List[Dict[str, Any]] = []
    exposures: List[Dict[str, Any]] = []
    air_label = cfg.air_label.strip().lower()
    for index, smell in enumerate(segments):
        label = str(smell["_label"].iloc[0])
        if label == air_label:
            continue
        preceding = segments[index - 1] if index else None
        paired = preceding is not None and str(preceding["_label"].iloc[0]) == air_label
        if paired:
            for name in ("session_id", "recording_id", "_label_block"):
                if name in smell and smell[name].iloc[0] != preceding[name].iloc[0]:
                    paired = False
            if (smell["_timestamp"].iloc[0] - preceding["_timestamp"].iloc[-1]).total_seconds() > cfg.max_gap_seconds:
                paired = False
        if not paired:
            exposures.append({
                "label": label, "start": str(smell["_timestamp"].iloc[0]),
                "end": str(smell["_timestamp"].iloc[-1]), "windows": [],
                "rejection_reasons": ["no preceding clean-air segment in this recording/session"],
                "baseline_candidates": 0, "baseline_kept": 0,
                "plateau_candidates": 0, "plateau_kept": 0,
            })
            continue

        smell_start, smell_end = smell["_timestamp"].iloc[[0, -1]]
        duration_s = float((smell_end - smell_start).total_seconds())
        plateau_start = smell_start + pd.Timedelta(seconds=duration_s * (1.0 - cfg.plateau_fraction))
        plateau_end = smell_end - pd.Timedelta(seconds=cfg.tail_margin_seconds)
        baseline_start = max(
            preceding["_timestamp"].iloc[0],
            preceding["_timestamp"].iloc[-1] - pd.Timedelta(seconds=cfg.baseline_tail_seconds),
        )
        baseline_end = preceding["_timestamp"].iloc[-1]
        session_id = str(smell["session_id"].iloc[0]) if "session_id" in smell else str(smell_start.date())
        segment_id = int(smell["_segment"].iloc[0])
        recording_id = str(smell["recording_id"].iloc[0]) if "recording_id" in smell else ""
        recording_prefix = f"{recording_id}:" if recording_id else ""
        exposure_id = f"{session_id}:{recording_prefix}{segment_id}:{label}"

        counts = {"baseline_candidates": 0, "baseline_kept": 0, "plateau_candidates": 0, "plateau_kept": 0}
        diagnostics = []
        reasons = []
        for phase, source, start, end, out_label in (
            ("baseline_air", preceding, baseline_start, baseline_end, air_label),
            ("exposure_plateau", smell, plateau_start, plateau_end, label),
        ):
            windows = _window_slices(source, start, end, cfg, diagnostics, phase)
            key = "baseline" if phase == "baseline_air" else "plateau"
            counts[f"{key}_candidates"] = len(windows)
            if not windows:
                reasons.append(f"{phase}: selected interval is too short or contains too few samples")
            stable_windows: List[Tuple[pd.DataFrame, float]] = []
            for window in windows:
                score = relative_slope_score(window, cfg.stability_quantile)
                if score > cfg.max_relative_slope:
                    diagnostics.append({"phase": phase, "start": str(window["_timestamp"].iloc[0]), "end": str(window["_timestamp"].iloc[-1]), "kept": False, "reason": "relative slope exceeds threshold", "score": float(score) if np.isfinite(score) else None})
                    continue
                stable_windows.append((window, score))
            if len(stable_windows) > cfg.max_windows_per_phase:
                chosen = np.linspace(
                    0, len(stable_windows) - 1, cfg.max_windows_per_phase, dtype=int
                )
                selected = set(np.unique(chosen))
                for i, (window, score) in enumerate(stable_windows):
                    if i not in selected:
                        diagnostics.append({"phase": phase, "start": str(window["_timestamp"].iloc[0]), "end": str(window["_timestamp"].iloc[-1]), "kept": False, "reason": "per-phase window cap", "score": score})
                stable_windows = [stable_windows[i] for i in sorted(selected)]
            for window, score in stable_windows:
                rows.append(
                    _summarize_window(
                        window, out_label, phase, session_id, exposure_id, segment_id, score
                    )
                )
                counts[f"{key}_kept"] += 1
                if recording_id:
                    rows[-1]["recording_id"] = recording_id
                if "resistance_unit" in source:
                    rows[-1]["resistance_unit"] = str(source["resistance_unit"].iloc[0])
                diagnostics.append({"phase": phase, "start": str(window["_timestamp"].iloc[0]), "end": str(window["_timestamp"].iloc[-1]), "kept": True, "reason": "stable", "score": score})
        exposures.append(
            {
                "exposure_id": exposure_id,
                "label": label,
                "duration_seconds": duration_s,
                "start": str(smell_start), "end": str(smell_end),
                "plateau_start": str(plateau_start), "plateau_end": str(plateau_end),
                "windows": diagnostics, "rejection_reasons": reasons,
                **counts,
            }
        )

    if not rows and not allow_empty:
        raise ValueError(
            "plateau extraction produced no stable windows; relax max_relative_slope "
            "or inspect the timestamp/label transitions"
        )
    result = pd.DataFrame(rows, columns=None if rows else [*ALL_SENSORS, "Gas name", "session_id", "exposure_id"])
    classes = {str(k): int(v) for k, v in result["Gas name"].value_counts().items()}
    report: Dict[str, Any] = {
        "profile": "plateau",
        "config": asdict(cfg),
        "input_rows": int(len(df)),
        "invalid_timestamp_rows": invalid_time,
        "segments": int(len(segments)),
        "exposures": sum("exposure_id" in exposure for exposure in exposures),
        "unpaired_exposures": sum("exposure_id" not in exposure for exposure in exposures),
        "sessions": int(result["session_id"].nunique()),
        "output_windows": int(len(result)),
        "class_windows": classes,
        "label_aliases": aliases,
        "exposure_details": exposures,
    }
    return result, report
