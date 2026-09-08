"""CSV I/O helpers for e-nose training data.

The historical dataset uses semicolon-separated single-column CSVs (R1;...;R17;T;H;CO2;H2S;CH2O;smell);
newer dumps are standard comma CSVs. `load_training_csv` transparently handles both and
returns a DataFrame with canonical column names (`R1..R17`, env sensors, `smell_label`).
"""

from __future__ import annotations

import csv
import os
from io import StringIO
from typing import Iterable, Optional

import pandas as pd

from enose.config import ALL_SENSORS, ENV_DEFAULTS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS


LABEL_CANDIDATES = ("smell_label", "smell", "smell;", "class", "label", "Gas name")


def _as_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_uploaded_enose_csv(
    text: str,
    label_column: str = "Gas name",
) -> tuple[pd.DataFrame, dict]:
    """Parse a normal CSV or recover the recorder's variable-width format.

    Some 2025 exports have 19 empty header columns because repeated UART
    environmental packets were appended before the gas label.  Pandas accepts
    the file but silently places numeric packet fragments in ``Gas name``.
    R1-R17 remain at fixed positions, so for those files we recover the last
    non-numeric field after R17 as the label and discard ambiguous environment
    packets.  Ordinary CSVs remain on the standard pandas path.
    """
    reader = csv.reader(StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV is empty")
    canonical_prefix = header[:18] == ["Timestamp", *RESISTANCE_SENSORS]
    variable_width = canonical_prefix and any(not name.strip() for name in header)
    if not variable_width:
        frame = pd.read_csv(StringIO(text), low_memory=False)
        return frame, {
            "parser": "standard",
            "recovered_rows": 0,
            "dropped_rows": 0,
            "environment_reliable": all(sensor in frame.columns for sensor in ENVIRONMENTAL_SENSORS),
        }

    records = []
    dropped = 0
    repaired = 0
    for fields in reader:
        if len(fields) < 20:
            dropped += 1
            continue
        resistances = [_as_float(value) for value in fields[1:18]]
        if len(resistances) != 17 or any(value is None for value in resistances):
            dropped += 1
            continue
        label_index = None
        for index in range(len(fields) - 1, 17, -1):
            token = fields[index].strip()
            if token and _as_float(token) is None:
                label_index = index
                break
        if label_index is None:
            dropped += 1
            continue
        label = fields[label_index].strip()
        row = {"Timestamp": fields[0].strip(), label_column: label}
        row.update({sensor: float(value) for sensor, value in zip(RESISTANCE_SENSORS, resistances)})
        # Only trust environment fields when the label is in its canonical
        # position. Variable-length UART fragments cannot be disambiguated.
        if label_index == 23:
            for sensor, value in zip(ENVIRONMENTAL_SENSORS, fields[18:23]):
                parsed = _as_float(value)
                row[sensor] = parsed if parsed is not None else ENV_DEFAULTS[sensor]
        else:
            row.update(ENV_DEFAULTS)
            repaired += 1
        if label_index + 1 < len(fields):
            gas_class = _as_float(fields[label_index + 1])
            if gas_class is not None:
                row["Gas label"] = gas_class
        records.append(row)
    if not records:
        raise ValueError("variable-width CSV recovery produced no valid rows")
    return pd.DataFrame(records), {
        "parser": "variable_width_recovery",
        "recovered_rows": repaired,
        "dropped_rows": dropped,
        "environment_reliable": False,
    }


def parse_semicolon_enose_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a single-column semicolon CSV into the canonical 22-feature frame.

    Expects each row to be ``R1;R2;...;R17;T;H;CO2;H2S;CH2O;label[;]``.
    Rows with fewer than 23 values are skipped silently.
    """
    col = df.columns[0]
    rows = []
    for raw in df[col].astype(str):
        parts = [p for p in raw.split(";") if p.strip() != ""]
        if len(parts) < 23:
            continue
        row: dict = {}
        for i, sensor in enumerate(RESISTANCE_SENSORS):
            try:
                row[sensor] = float(parts[i])
            except (ValueError, IndexError):
                row[sensor] = 0.0
        for i, sensor in enumerate(ENVIRONMENTAL_SENSORS):
            try:
                row[sensor] = float(parts[17 + i])
            except (ValueError, IndexError):
                row[sensor] = 0.0
        row["smell_label"] = str(parts[22]).strip()
        row["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        rows.append(row)
    return pd.DataFrame(rows)


def _coerce_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever label-like column exists to `smell_label`. Drop if none found."""
    if "smell_label" in df.columns:
        return df
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            df = df.copy()
            df["smell_label"] = df[cand]
            return df
    return df  # caller decides what to do with a label-less frame


def load_training_csv(candidates: Iterable[str]) -> Optional[pd.DataFrame]:
    """Try each path in order; return the first successfully parsed frame or None.

    Accepts both the legacy semicolon format and standard comma CSVs. Label column
    is normalized to `smell_label` when possible.
    """
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_csv(path)
        except Exception as e:
            print(f"[csv_io] skip {path}: {e}")
            continue

        df = parse_semicolon_enose_csv(raw) if len(raw.columns) == 1 else raw
        df = _coerce_label_column(df)
        if "smell_label" not in df.columns:
            print(f"[csv_io] {path} has no recognizable label column; skipping")
            continue
        print(f"[csv_io] loaded {len(df)} rows from {path}")
        return df
    return None


_GROUP_COLUMNS_PASSTHROUGH = (
    "segment_id", "window_idx", "session_id", "exposure_id",
    "recording_id", "source", "source_file",
)


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing sensor columns so the frame carries the full 22 features + label + timestamp.

    Known grouping columns (``session_id``, ``sniff_id``, ``recording_id``,
    ``source``, ``source_file``) are preserved when present so that the
    downstream training path can feed them to ``GroupShuffleSplit``. Any
    other extra columns are dropped — keep the schema tight so aggregated
    CSVs still flow through ``/smell/learn_from_csv`` without surprises.
    """
    df = df.copy()
    for col in ALL_SENSORS:
        if col not in df.columns:
            df[col] = 0.0 if col in RESISTANCE_SENSORS else ENV_DEFAULTS.get(col, 0.0)
    if "smell_label" not in df.columns:
        df["smell_label"] = "unknown"
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for col in ALL_SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["smell_label"] = df["smell_label"].astype(str)
    keep_extra = [c for c in _GROUP_COLUMNS_PASSTHROUGH if c in df.columns]
    return df[ALL_SENSORS + ["smell_label", "timestamp"] + keep_extra]


def save_training_samples(df: pd.DataFrame, path: str) -> str:
    """Write a training frame to CSV, creating parent directory as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path
