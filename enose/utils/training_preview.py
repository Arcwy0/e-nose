"""Shared, read-only CSV preparation for preview and training."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from enose.config import RESISTANCE_SENSORS
from enose.utils.csv_io import parse_uploaded_enose_csv
from enose.utils.plateau import PlateauConfig, build_plateau_training_frame
from enose.utils.quality import analyze_recording_quality
from enose.utils.acquisition import acquisition_provenance


def prepare_csv_training(data, allow_unlabelled=False):
    """Return source, selected training frame, target, report; never fit or write."""
    if not data.csv_data.strip():
        raise ValueError("CSV data required")
    recording_date = data.recording_date or None
    if recording_date:
        date.fromisoformat(recording_date)
    source, parse_report = parse_uploaded_enose_csv(
        data.csv_data, label_column=data.target_column,
        pcb_resistance_unit=data.resistance_unit, recording_date=recording_date,
    )
    if parse_report["parser"] == "standard" or "_source_row" not in source:
        source["_source_row"] = range(len(source))
    if data.annotations:
        if data.target_column not in source:
            source[data.target_column] = None
        intervals = []
        for annotation in data.annotations:
            start, end, label = annotation.start_row, annotation.end_row, annotation.label.strip()
            if start >= end or not label:
                raise ValueError("annotations require start_row < end_row and a verified nonempty label")
            if any(start < right and end > left for left, right in intervals):
                raise ValueError("annotation row ranges overlap; provide unambiguous verified labels")
            if end > parse_report.get("source_rows", int(source["_source_row"].max()) + 1):
                raise ValueError("annotation extends beyond the source recording")
            intervals.append((start, end))
            mask = source["_source_row"].ge(start) & source["_source_row"].lt(end)
            if not mask.any():
                raise ValueError("annotation contains no valid measurements")
            source.loc[mask, data.target_column] = label
    if data.target_column not in source:
        if not allow_unlabelled:
            raise ValueError(f"Target column '{data.target_column}' not found. Add verified labels or explicit row annotations.")
        source[data.target_column] = None
    missing = [sensor for sensor in RESISTANCE_SENSORS if sensor not in source]
    if missing:
        raise ValueError(f"CSV requires R1-R17; missing {missing}")
    labels = source[data.target_column]
    normalized_labels = labels.astype(str).str.strip().str.lower()
    labelled = labels.notna() & ~normalized_labels.isin({"", "unknown", "nan", "none"})
    unlabelled = int((~labelled).sum())
    source["_label_block"] = labelled.ne(labelled.shift()).cumsum()
    if not labelled.any() and not allow_unlabelled:
        raise ValueError("No verified labelled rows remain")
    if labelled.any():
        source.loc[labelled, data.target_column] = source.loc[labelled, data.target_column].astype(str).str.strip()
        if data.lowercase_labels:
            source.loc[labelled, data.target_column] = source.loc[labelled, data.target_column].str.lower()
    timestamp_column = next((name for name in ("Timestamp", "timestamp") if name in source), None)
    if timestamp_column and timestamp_column != "Timestamp":
        source["Timestamp"] = source[timestamp_column]
    if "Timestamp" in source and parse_report["parser"] != "pcb_resistance_csv":
        time_only = source["Timestamp"].astype(str).str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?").all()
        if time_only:
            anchor = pd.Timestamp(recording_date or "1970-01-01")
            offsets = pd.to_timedelta(source["Timestamp"].astype(str))
            resets = offsets.diff().dt.total_seconds()
            if resets.between(-12 * 3600, 0, inclusive="neither").any():
                raise ValueError("CSV time moves backwards within a recording")
            days = resets.lt(-12 * 3600).cumsum()
            source["Timestamp"] = anchor + offsets + pd.to_timedelta(days, unit="D")
            parse_report["timestamp_date_known"] = recording_date is not None
    if data.recording_id:
        source["recording_id"] = data.recording_id
    elif "recording_id" not in source:
        source["recording_id"] = hashlib.sha256(data.csv_data.encode()).hexdigest()[:16]
    if data.session_id:
        source["session_id"] = data.session_id
    labelled_source = source.loc[labelled].copy()
    quality_source = labelled_source if not labelled_source.empty else source.drop(columns=[data.target_column])
    quality = analyze_recording_quality(quality_source, data.target_column)
    quality["environment_reliable"] = bool(parse_report.get("environment_reliable", False))
    warnings = quality["warnings"]
    if not quality["environment_reliable"]:
        warnings.append("environmental fields are absent or ambiguous; training uses R1-R17 only")
    if unlabelled:
        warnings.append(f"{unlabelled} rows without verified labels were excluded")
    if labelled_source.empty:
        warnings.append("Preview only: add verified smell labels or explicit row annotations before training. No smell class has been inferred.")
    if parse_report.get("timestamp_date_known") is False:
        warnings.append("recording date is unknown; timestamps use a synthetic 1970-01-01 anchor. Supply recording_date and independent session IDs before day-wise evaluation")
    profile = data.training_profile.strip().lower()
    if profile not in {"raw", "plateau"}:
        raise ValueError("training_profile must be 'raw' or 'plateau'")
    if labelled_source.empty:
        selected, target = labelled_source.copy(), data.target_column
        report = {"profile": profile, "input_rows": len(source), "output_windows": 0, "exposure_details": []}
    elif profile == "plateau":
        config = PlateauConfig(**data.plateau_options.model_dump()) if data.plateau_options else PlateauConfig()
        selected, report = build_plateau_training_frame(
            labelled_source, label_col=data.target_column, config=config, allow_empty=True,
        )
        target = "Gas name"
    elif profile == "raw":
        selected, target = labelled_source.copy(), data.target_column
        report = {"profile": "raw", "input_rows": len(source), "output_windows": len(selected), "exposure_details": []}
    else:
        raise ValueError("training_profile must be 'raw' or 'plateau'")
    if report.get("invalid_timestamp_rows"):
        warnings.append(f"{report['invalid_timestamp_rows']} rows with invalid timestamps were excluded from plateau extraction")
    report.update({"csv_parser": parse_report, "input_quality": quality,
                   "label_column": data.target_column, "source_rows": len(source),
                   "requires_verified_labels": labelled_source.empty,
                   "unlabelled_rows_excluded": unlabelled, "annotations_applied": len(data.annotations)})
    metadata = []
    if "acquisition_json" in source:
        for value in source["acquisition_json"]:
            if pd.isna(value) or not str(value).strip():
                metadata.append(None)
                continue
            try:
                item = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid acquisition_json provenance; expected a JSON object per row") from exc
            if not isinstance(item, dict):
                raise ValueError("acquisition_json provenance must contain JSON objects")
            metadata.append(item)
    report["acquisition_provenance"] = acquisition_provenance(metadata)
    return source, selected, target, report


def csv_training_preview(source, selected, target, report):
    """Bound browser plot and audit size independently of the uploaded file."""
    classes = {str(key): int(value) for key, value in selected[target].value_counts().items()}
    count = len(source)
    indices = np.unique(np.linspace(0, count - 1, min(count, 1200), dtype=int))
    points = []
    for position in indices:
        row = source.iloc[int(position)]
        values = {}
        for sensor in RESISTANCE_SENSORS:
            try:
                number = float(row[sensor])
            except (TypeError, ValueError):
                number = float("nan")
            values[sensor] = number if np.isfinite(number) else None
        points.append({"row": int(row["_source_row"]), "timestamp": str(row.get("Timestamp", "")),
                       "label": str(row.get(report["label_column"], "")) if pd.notna(row.get(report["label_column"])) else "", **values})
    details = report.get("exposure_details", [])
    bounded = []
    windows_left = 2000
    for exposure in details[:200]:
        item = dict(exposure)
        windows = item.get("windows", [])
        item["windows"] = windows[:windows_left]
        windows_left -= len(item["windows"])
        item["windows_omitted"] = len(windows) - len(item["windows"])
        bounded.append(item)
    ready = len(classes) >= 2 and not selected.empty
    return {
        "success": True, "training_ready": ready,
        "message": "Ready to train" if ready else (
            "Preview only: add verified labels or explicit row annotations before training."
            if report.get("requires_verified_labels") else
            "No valid labelled training windows from at least two classes. Inspect labels, clean-air pairing and extraction settings."
        ),
        "input_rows": report.get("input_rows", count), "selected_rows": len(selected),
        "class_counts": classes, "profile_report": {**report, "exposure_details": bounded},
        "plot": {"points": points, "source_rows": count, "downsampled": count > len(points),
                 "sensors": RESISTANCE_SENSORS, "unit": report["csv_parser"].get("resistance_unit", "unverified")},
        "exposures_omitted": max(0, len(details) - len(bounded)),
    }
