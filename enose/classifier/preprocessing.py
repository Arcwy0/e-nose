"""Feature preprocessing: sanitization, ordering, scaling, augmentation, cleaning.

These functions are state-light — they take a DataFrame and return a DataFrame.
The classifier holds the fitted scaler and clipping bounds; this module holds
the transformations themselves.

Edit here if you want to change how resistance/environmental sensors are
treated before hitting the model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from enose.config import (
    ALL_SENSORS,
    ENVIRONMENTAL_SENSORS,
    RESISTANCE_SENSORS,
)


def ensure_dataframe(X: Union[Dict[str, float], pd.DataFrame, list]) -> pd.DataFrame:
    """Coerce input into a DataFrame. Accepts dict, list-of-dicts, or DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, dict):
        return pd.DataFrame([X])
    if isinstance(X, list):
        return pd.DataFrame(X)
    raise TypeError(f"X must be dict, list, or DataFrame (got {type(X).__name__})")


def sanitize_environmentals(
    df: pd.DataFrame,
    env_ranges: Dict[str, Tuple[float, float]],
    env_medians: Dict[str, float],
    verbose: bool = False,
) -> pd.DataFrame:
    """Replace out-of-range env values with per-sensor medians. H2S is clipped at 0.

    When ``verbose`` is True, prints a per-column count of how many rows
    were out of range. Useful during training to confirm that sensor-fault
    readings (e.g. T=222222 seen on the historical CSV) are being scrubbed
    before the model ever sees them.
    """
    df = df.copy()
    n_rows = len(df)
    for col in ENVIRONMENTAL_SENSORS:
        if col not in df.columns:
            continue
        low, high = env_ranges[col]
        med = env_medians[col]
        vals = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((vals < low) | (vals > high) | vals.isna())
        n_replaced = int(out_of_range.sum())
        vals = vals.where((vals >= low) & (vals <= high), med)
        if col == "H2S":
            vals = vals.clip(lower=0.0)
        df[col] = vals.fillna(med)
        if verbose and n_replaced > 0:
            pct = 100.0 * n_replaced / max(n_rows, 1)
            print(
                f"[sanitize_env] {col}: replaced {n_replaced}/{n_rows} "
                f"({pct:.1f}%) out-of-range values with median={med}"
            )
    return df


def order_and_fill_features(df: pd.DataFrame, env_medians: Dict[str, float]) -> pd.DataFrame:
    """Ensure all 22 columns exist in the canonical order. Missing R* = 0, missing env = median."""
    df = df.copy()
    for col in ALL_SENSORS:
        if col not in df.columns:
            if col.startswith("R"):
                df[col] = 0.0
            else:
                df[col] = env_medians.get(col, 0.0)
    return df[ALL_SENSORS]


def clean_resistances(
    df: pd.DataFrame,
    r_median: Dict[str, float],
    r_clip: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Replace zeros/NaNs with per-sensor median, winsorize to [p1, p99]."""
    df = df.copy()
    for c in RESISTANCE_SENSORS:
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").fillna(r_median.get(c, 0.0))
        v = v.mask(v == 0.0, r_median.get(c, 0.0))
        lo, hi = r_clip.get(c, (-np.inf, np.inf))
        v = v.clip(lower=lo, upper=hi)
        df[c] = v
    return df


def scale_resistances(
    df: pd.DataFrame,
    scaler,
    fit: bool = False,
) -> pd.DataFrame:
    """Apply a fitted scaler (StandardScaler / RobustScaler / …) to R1–R17.

    Environmental columns are untouched. The scaler is caller-owned so we can
    swap StandardScaler ↔ RobustScaler without changing this module.
    """
    df = df.copy()
    R = df[RESISTANCE_SENSORS].astype(float)
    if fit:
        R_scaled = scaler.fit_transform(R)
    else:
        R_scaled = scaler.transform(R)
    df[RESISTANCE_SENSORS] = R_scaled
    return df


def log1p_resistances(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to R1–R17 (env sensors untouched).

    Resistances span several orders of magnitude (tens to tens of thousands);
    log1p compresses that range so the downstream scaler is not dominated by
    rare high-magnitude samples. Safe for non-negative inputs; any residual
    negatives are clipped at 0 before the transform.
    """
    df = df.copy()
    R = df[RESISTANCE_SENSORS].astype(float).clip(lower=0.0)
    df[RESISTANCE_SENSORS] = np.log1p(R.values)
    return df


def augment_resistances(
    X: pd.DataFrame,
    n_aug: int,
    noise_max: float,
) -> pd.DataFrame:
    """Append n_aug noisy copies of X. Uniform noise in [-noise_max, +noise_max] on R1–R17 only.

    Environmental columns are preserved across copies (they represent context, not signal).
    """
    if n_aug <= 0:
        return X
    X_aug = [X]
    for _ in range(n_aug):
        noise = np.random.uniform(
            -noise_max, noise_max,
            size=(len(X), len(RESISTANCE_SENSORS)),
        )
        Xn = X.copy()
        Xn.loc[:, RESISTANCE_SENSORS] = X[RESISTANCE_SENSORS].values + noise
        X_aug.append(Xn)
    return pd.concat(X_aug, ignore_index=True)


def compute_resistance_clip_bounds(
    X: pd.DataFrame,
    ignore_zeros: bool = True,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """Return (r_clip {col: (p1, p99)}, r_median {col: median}) for R1–R17.

    IMPORTANT: bounds must be computed in the *same space* they will be applied
    in. The classifier computes these on RAW resistances (before scaling) so
    that `clean_resistances` — which also runs on raw data at inference time —
    can use them consistently.

    When ``ignore_zeros`` is True, per-sensor zero readings (which typically
    mean a dead / saturated channel rather than a real value) are excluded
    from the quantile and median computation so one bad channel doesn't poison
    the bounds for that sensor.
    """
    R = X[RESISTANCE_SENSORS].astype(float)
    r_clip: Dict[str, Tuple[float, float]] = {}
    r_median: Dict[str, float] = {}
    for c in RESISTANCE_SENSORS:
        col = R[c]
        if ignore_zeros:
            col = col[col > 0.0]
        if len(col) == 0:
            r_clip[c] = (0.0, 0.0)
            r_median[c] = 0.0
            continue
        r_clip[c] = (float(col.quantile(0.01)), float(col.quantile(0.99)))
        r_median[c] = float(col.median())
    return r_clip, r_median
