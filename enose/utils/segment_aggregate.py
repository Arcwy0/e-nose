"""Aggregate a long frame-level e-nose CSV into windowed training rows.

Data reality
------------
The sensors run continuously for hours. A label changes when the operator
swaps the gas bottle; there are no gaps between recordings. So the
fundamental unit is a **label segment** — a contiguous block of rows with
the same ``Gas name``. Segments are long (minutes to hours); classifying
single frames out of them conflates the rising transient after a bottle
swap, the stable plateau, and any decay.

What this module does
---------------------
1. Segment the recording by label (a new segment starts whenever ``Gas name``
   changes). Optionally split across ``session_id`` when long time gaps are
   present (e.g., the operator stopped recording for a day).
2. Within each segment, drop the head and tail (equilibration after the
   bottle swap, and any instability at the end) to keep only the stable
   plateau.
3. Slide fixed-duration windows across the plateau, emitting one feature
   row per window. ``segment_id`` tags the window so the training splitter
   can keep all windows from a single segment together (no leakage).

Two feature modes
-----------------
- ``summary`` — 22 columns (per-window median for each sensor). Drop-in
  replacement for the existing ``/smell/learn_from_csv`` pipeline.
- ``rich`` — 6 shape features per resistance sensor + env medians
  (~107 columns). Needs a classifier that doesn't hard-code the 22-feature
  schema; use once summary-mode accuracy has a ceiling.

CLI
---
    python -m enose.utils.segment_aggregate \
        --in data/new_data.csv \
        --out data/new_data_windows.csv \
        --mode summary \
        --window-seconds 120 --stride-seconds 60 \
        --skip-head-seconds 60 --skip-tail-seconds 30

On a 3-hour recording with 6 label segments averaging 30 min each, that
produces ~180 training rows, grouped by segment_id so GroupShuffleSplit
yields an honest held-out set.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from enose.config import (
    ALL_SENSORS,
    ENVIRONMENTAL_SENSORS,
    RESISTANCE_SENSORS,
)


# ── Segmentation ────────────────────────────────────────────────────────────

def segment_by_label(
    df: pd.DataFrame,
    label_col: str = "Gas name",
    timestamp_col: str = "Timestamp",
    session_gap_s: float = 3600.0,
) -> pd.DataFrame:
    """Add ``segment_id`` and ``session_id`` columns.

    A new *segment* starts at every label change (= bottle swap). A new
    *session* starts when the time gap exceeds ``session_gap_s`` — default
    1 h, which won't split a continuous multi-hour recording but will split
    two recordings that happened on different days. ``session_id`` lets you
    also do inter-session splits later if you accumulate many runs.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"CSV must have a '{timestamp_col}' column")
    if label_col not in df.columns:
        raise ValueError(f"CSV must have a '{label_col}' column")

    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    bad = ts.isna()
    if bad.any():
        print(f"[segment_aggregate] WARN: dropping {int(bad.sum())} rows with bad timestamps")
        out = out.loc[~bad].reset_index(drop=True)
        ts = ts.loc[~bad].reset_index(drop=True)

    order = np.argsort(ts.values)
    out = out.iloc[order].reset_index(drop=True)
    ts = ts.iloc[order].reset_index(drop=True)

    dt_s = ts.diff().dt.total_seconds().fillna(0.0)
    label_change = out[label_col].astype(str) != out[label_col].astype(str).shift(1)
    label_change.iloc[0] = True

    new_session = dt_s > session_gap_s
    new_session.iloc[0] = True

    out["segment_id"] = label_change.cumsum().astype(int)
    out["session_id"] = new_session.cumsum().astype(int)
    out["_ts_parsed"] = ts.values
    return out


# ── Window slicing within a segment ─────────────────────────────────────────

def _window_bounds(
    ts_ns: np.ndarray,
    window_s: float,
    stride_s: float,
    skip_head_s: float,
    skip_tail_s: float,
    min_window_samples: int,
) -> List[Tuple[int, int]]:
    """Time-based window bounds ``[(lo, hi), ...]`` into the segment.

    Using wall-clock seconds rather than frame counts keeps the window
    semantics stable even if the device briefly sampled faster or slower.
    """
    if len(ts_ns) == 0:
        return []
    t0 = ts_ns[0]
    t_last = ts_ns[-1]
    to_ns = lambda s: np.int64(s * 1_000_000_000)
    plateau_start = t0 + to_ns(skip_head_s)
    plateau_end = t_last - to_ns(skip_tail_s)
    if plateau_end <= plateau_start:
        # Segment too short for the chosen margins — use whatever we have.
        return [(0, len(ts_ns))] if len(ts_ns) >= min_window_samples else []

    w_ns = to_ns(window_s)
    stride_ns = to_ns(stride_s) if stride_s > 0 else w_ns

    bounds: List[Tuple[int, int]] = []
    cur = plateau_start
    while cur + w_ns <= plateau_end + 1:
        lo = int(np.searchsorted(ts_ns, cur, side="left"))
        hi = int(np.searchsorted(ts_ns, cur + w_ns, side="right"))
        if hi - lo >= min_window_samples:
            bounds.append((lo, hi))
        cur = cur + stride_ns
    if not bounds:
        lo = int(np.searchsorted(ts_ns, plateau_start, side="left"))
        hi = int(np.searchsorted(ts_ns, plateau_end, side="right"))
        if hi - lo >= min_window_samples:
            bounds.append((lo, hi))
    return bounds


# ── Per-window feature extractors ───────────────────────────────────────────

def summary_window(window: pd.DataFrame) -> Dict[str, float]:
    """22-column median — plateau-robust, drop-in for the 22-feature pipeline."""
    out: Dict[str, float] = {}
    for c in ALL_SENSORS:
        if c in window.columns:
            out[c] = float(pd.to_numeric(window[c], errors="coerce").median())
        else:
            out[c] = 0.0
    return out


def rich_window(window: pd.DataFrame) -> Dict[str, float]:
    """Six shape features per resistance sensor + median per env sensor.

    Per resistance sensor:
        _mean      mean over the window
        _std       stdev over the window
        _delta     peak - baseline (baseline = first 10% median)
        _argmax_f  time-to-peak as fraction of window length
        _slope     mean rise from baseline to peak
        _auc       baseline-subtracted integral / length

    These are cheap to compute, scale-free once log1p'd upstream, and
    disambiguate curves with similar plateaus but different kinetics
    (ipa vs ethanol, coffee vs ethanol).
    """
    out: Dict[str, float] = {}
    for c in RESISTANCE_SENSORS:
        if c not in window.columns:
            for suffix in ("mean", "std", "delta", "argmax_f", "slope", "auc"):
                out[f"{c}_{suffix}"] = 0.0
            continue
        v = pd.to_numeric(window[c], errors="coerce").fillna(0.0).values.astype(float)
        n = max(1, len(v))
        base_n = max(1, n // 10)
        baseline = float(np.median(v[:base_n]))
        peak_idx = int(np.argmax(v))
        peak = float(v[peak_idx])
        out[f"{c}_mean"] = float(v.mean())
        out[f"{c}_std"] = float(v.std(ddof=0))
        out[f"{c}_delta"] = peak - baseline
        out[f"{c}_argmax_f"] = peak_idx / max(1, n - 1)
        out[f"{c}_slope"] = (peak - baseline) / max(1, peak_idx)
        out[f"{c}_auc"] = float(np.trapz(v - baseline)) / n
    for c in ENVIRONMENTAL_SENSORS:
        if c in window.columns:
            out[c] = float(pd.to_numeric(window[c], errors="coerce").median())
        else:
            out[c] = 0.0
    return out


# ── Aggregation driver ──────────────────────────────────────────────────────

def aggregate(
    df: pd.DataFrame,
    mode: str = "summary",
    label_col: str = "Gas name",
    timestamp_col: str = "Timestamp",
    window_s: float = 120.0,
    stride_s: float = 60.0,
    skip_head_s: float = 60.0,
    skip_tail_s: float = 30.0,
    min_window_samples: int = 5,
    min_segment_samples: int = 20,
    session_gap_s: float = 3600.0,
) -> pd.DataFrame:
    """Segment -> trim head/tail -> window -> feature row per window.

    ``skip_head_s`` matters the most: the first minute after a bottle swap
    is the rising transient. Keeping it in would reintroduce the exact
    problem (mid-transient frames look like the wrong gas) we're trying to
    eliminate.
    """
    seg = segment_by_label(
        df,
        label_col=label_col,
        timestamp_col=timestamp_col,
        session_gap_s=session_gap_s,
    )

    rows: List[Dict[str, float]] = []
    dropped_short = 0
    n_segments_total = 0
    for (sid, gas), grp in seg.groupby(["segment_id", label_col], sort=False):
        n_segments_total += 1
        if len(grp) < min_segment_samples:
            dropped_short += 1
            continue

        grp = grp.reset_index(drop=True)
        ts_ns = grp["_ts_parsed"].astype("datetime64[ns]").astype(np.int64).values
        bounds = _window_bounds(
            ts_ns,
            window_s=window_s,
            stride_s=stride_s,
            skip_head_s=skip_head_s,
            skip_tail_s=skip_tail_s,
            min_window_samples=min_window_samples,
        )
        if not bounds:
            dropped_short += 1
            continue

        for wi, (lo, hi) in enumerate(bounds):
            window = grp.iloc[lo:hi]
            if mode == "summary":
                feat = summary_window(window)
            elif mode == "rich":
                feat = rich_window(window)
            else:
                raise ValueError(f"Unknown mode '{mode}' — use 'summary' or 'rich'")
            feat["segment_id"] = int(sid)
            feat["window_idx"] = int(wi)
            feat["session_id"] = int(grp["session_id"].iloc[0])
            feat[label_col] = str(gas)
            feat["n_frames"] = int(hi - lo)
            feat["t_start"] = str(grp[timestamp_col].iloc[lo])
            feat["t_end"] = str(grp[timestamp_col].iloc[hi - 1])
            rows.append(feat)

    if not rows:
        raise RuntimeError(
            "aggregate: produced 0 windows. Common causes: "
            "window_s longer than every segment, skip_head_s + skip_tail_s "
            "swallows the whole segment, or min_segment_samples too high."
        )

    result = pd.DataFrame(rows)
    print(f"[segment_aggregate] {n_segments_total} segments -> "
          f"{len(result)} windows ({dropped_short} segments too short to window)")

    # In summary mode, preserve exact 22-feature schema for drop-in retrain.
    if mode == "summary":
        extras = [label_col, "segment_id", "window_idx", "session_id",
                  "n_frames", "t_start", "t_end"]
        result = result[ALL_SENSORS + extras]
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segment a continuous e-nose recording by label and emit windowed training rows.",
    )
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out", dest="output_path", required=True)
    p.add_argument("--mode", choices=["summary", "rich"], default="summary",
                   help="summary (22-col, drop-in) or rich (~107 features)")
    p.add_argument("--label-col", default="Gas name")
    p.add_argument("--timestamp-col", default="Timestamp")
    p.add_argument("--window-seconds", type=float, default=120.0,
                   help="Window length in seconds (default 120)")
    p.add_argument("--stride-seconds", type=float, default=60.0,
                   help="Stride between windows in seconds (default 60; set equal to --window-seconds for non-overlapping)")
    p.add_argument("--skip-head-seconds", type=float, default=60.0,
                   help="Seconds to skip at the start of each segment — equilibration after bottle swap (default 60)")
    p.add_argument("--skip-tail-seconds", type=float, default=30.0,
                   help="Seconds to skip at the end of each segment (default 30)")
    p.add_argument("--min-window-samples", type=int, default=5,
                   help="Skip windows with fewer than this many frames")
    p.add_argument("--min-segment-samples", type=int, default=20,
                   help="Skip segments shorter than this (likely mislabels)")
    p.add_argument("--session-gap-seconds", type=float, default=3600.0,
                   help="Time gap that starts a new session_id (default 1 h)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"[segment_aggregate] reading {args.input_path}")
    df = pd.read_csv(args.input_path)
    print(f"[segment_aggregate] loaded {len(df)} frames, {len(df.columns)} cols")
    result = aggregate(
        df,
        mode=args.mode,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_s=args.window_seconds,
        stride_s=args.stride_seconds,
        skip_head_s=args.skip_head_seconds,
        skip_tail_s=args.skip_tail_seconds,
        min_window_samples=args.min_window_samples,
        min_segment_samples=args.min_segment_samples,
        session_gap_s=args.session_gap_seconds,
    )
    result.to_csv(args.output_path, index=False)
    print(f"[segment_aggregate] wrote {len(result)} windows × {len(result.columns)} cols "
          f"-> {args.output_path}")
    print(f"[segment_aggregate] segments: {result['segment_id'].nunique()}, "
          f"sessions: {result['session_id'].nunique()}")
    print("[segment_aggregate] per-class window counts:")
    print(result[args.label_col].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
