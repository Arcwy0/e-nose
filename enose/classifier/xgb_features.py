"""Windowed feature extraction for the XGBoost backend.

Pure functions: take arrays/frames, return arrays. No state. Edit here when
changing what temporal statistics (mean/std/slope/percentiles/env-mean) are
computed per window, or how windows are cut from a recording.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from enose.config import ALL_SENSORS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS

from .xgb_config import XGBClassifierConfig


def resolve_device(config: XGBClassifierConfig) -> str:
    """Return "cuda" if nvidia-smi responds, else "cpu"; honors explicit override."""
    if config.device != "auto":
        return config.device
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def remap_labels(y: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Map labels to contiguous 0..N-1 — XGBoost requires no gaps in class indices."""
    unique_labels = sorted(np.unique(y))
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([mapping[v] for v in y]), mapping


def extract_window_features(window: np.ndarray, config: XGBClassifierConfig) -> np.ndarray:
    """Summarize a (window_size, 22) slice into a 1D feature vector.

    Resistance sensors get temporal stats (mean/std/min/max, optional slope and p25/p75);
    environmental sensors get a mean only. Slope falls back to zeros on degenerate windows.
    """
    n_samples, _ = window.shape
    n_resistance = len(RESISTANCE_SENSORS)
    resistance = window[:, :n_resistance]
    env = window[:, n_resistance:]

    feats: List[np.ndarray] = [
        np.mean(resistance, axis=0),
        np.std(resistance, axis=0),
        np.min(resistance, axis=0),
        np.max(resistance, axis=0),
    ]

    if config.use_slope and n_samples >= 3:
        x = np.arange(n_samples, dtype=np.float64)
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered ** 2))
        slopes = (x_centered @ resistance) / denom if denom > 0 else np.zeros(n_resistance)
        feats.append(slopes)

    if config.use_percentiles:
        feats.append(np.percentile(resistance, 25, axis=0))
        feats.append(np.percentile(resistance, 75, axis=0))

    feats.append(np.mean(env, axis=0))
    return np.concatenate(feats)


def feature_names(config: XGBClassifierConfig) -> List[str]:
    """Human-readable names for the columns produced by extract_window_features."""
    names: List[str] = []
    for prefix in ("mean", "std", "min", "max"):
        names += [f"{prefix}_{c}" for c in RESISTANCE_SENSORS]
    if config.use_slope:
        names += [f"slope_{c}" for c in RESISTANCE_SENSORS]
    if config.use_percentiles:
        names += [f"p25_{c}" for c in RESISTANCE_SENSORS]
        names += [f"p75_{c}" for c in RESISTANCE_SENSORS]
    names += [f"env_mean_{c}" for c in ENVIRONMENTAL_SENSORS]
    return names


def assign_sessions(df: pd.DataFrame, gap_seconds: float = 60.0) -> np.ndarray:
    """Tag each row with a session id based on timestamp gaps. Prevents train/test leakage."""
    timestamps = pd.to_datetime(df["Timestamp"])
    diffs = timestamps.diff().dt.total_seconds().abs().fillna(0)
    return (diffs > gap_seconds).cumsum().values


def create_windows(
    df: pd.DataFrame,
    config: XGBClassifierConfig,
    session_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows within each session; return (features, labels, session_group_ids)."""
    raw = df[ALL_SENSORS].values
    labels = df["Gas label"].values
    if session_ids is None:
        session_ids = np.zeros(len(df), dtype=int)

    X_list, y_list, g_list = [], [], []
    for sid in np.unique(session_ids):
        mask = session_ids == sid
        sess_raw = raw[mask]
        sess_labels = labels[mask]
        if len(sess_raw) < config.min_window_samples:
            continue
        start = 0
        while start + config.min_window_samples <= len(sess_raw):
            end = min(start + config.window_size, len(sess_raw))
            window = sess_raw[start:end]
            win_labels = sess_labels[start:end]
            unique, counts = np.unique(win_labels, return_counts=True)
            majority = unique[np.argmax(counts)]
            X_list.append(extract_window_features(window, config))
            y_list.append(majority)
            g_list.append(sid)
            start += config.window_stride

    if not X_list:
        raise ValueError("No windows created. Check data size / window config.")
    return np.array(X_list), np.array(y_list), np.array(g_list)
