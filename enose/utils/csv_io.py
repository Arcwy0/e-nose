"""CSV I/O helpers for e-nose training data.

The historical dataset uses semicolon-separated single-column CSVs (R1;...;R17;T;H;CO2;H2S;CH2O;smell);
newer dumps are standard comma CSVs. `load_training_csv` transparently handles both and
returns a DataFrame with canonical column names (`R1..R17`, env sensors, `smell_label`).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd

from enose.config import ALL_SENSORS, ENV_DEFAULTS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS


LABEL_CANDIDATES = ("smell_label", "smell", "smell;", "class", "label", "Gas name")


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
    "segment_id", "window_idx", "session_id",
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
