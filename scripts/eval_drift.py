#!/usr/bin/env python3
"""Cross-session drift evaluation harness.

Answers the question this whole feature exists for: *does classifying a
baseline-relative representation survive day-to-day sensor drift, where the
absolute pipeline does not?*

It does an honest **leave-one-session-out** (LOSO) evaluation: train on all but
one session, then predict the held-out session — the closest proxy to
"trained yesterday, run today" that the data allows. For the drift modes, the
held-out session is referenced to *its own* clean-air baseline before
prediction (exactly what happens in production via POST /smell/baseline).

It compares, side by side:
  (0) none      — current absolute features (the baseline to beat)
  (1) delta     — R - R0
  (2) ratio     — R / R0
  (3) logratio  — log1p(R) - log1p(R0)
  (4) logratio+snv

Read-only: trains in-memory, saves nothing, never touches trained_models/ or
the live server. Run inside the enose image, e.g.:

    docker exec -i enose-srv python scripts/eval_drift.py \
        --csv data/database_robodog_time_windows.csv --session-col session_id
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from enose.classifier import BalancedRFClassifier
from enose.classifier.config import SmellClassifierConfig
from enose.config import RESISTANCE_SENSORS

LABEL_CANDIDATES = ("smell_label", "Gas name", "gas_name", "label")
SESSION_CANDIDATES = ("session_id", "segment_id", "session", "recording_id", "source")

MODES = [
    ("none", False),
    ("delta", False),
    ("ratio", False),
    ("logratio", False),
    ("logratio", True),
]


def _pick(cols, candidates, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    for c in candidates:
        if c in cols:
            return c
    return None


def _subsample(df: pd.DataFrame, label_col: str, session_col: str, cap: int, seed: int) -> pd.DataFrame:
    """Cap rows per (session, class) so LOSO over a 200k-row file stays fast."""
    if cap <= 0:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, block in df.groupby([session_col, label_col]):
        if len(block) > cap:
            keep.append(block.iloc[rng.choice(len(block), cap, replace=False)])
        else:
            keep.append(block)
    return pd.concat(keep, ignore_index=True)


def _build(mode: str, snv: bool) -> BalancedRFClassifier:
    cfg = SmellClassifierConfig()
    cfg.baseline_mode = mode
    cfg.snv = snv
    cfg.use_augmentation = False   # honest signal, no inflation
    cfg.calibrated = False         # speed; calibration doesn't change argmax much
    return BalancedRFClassifier(config=cfg)


def _loso_score(df, label_col, session_col, air_label, mode, snv) -> Optional[float]:
    """Mean balanced accuracy across leave-one-session-out folds (non-air classes)."""
    sessions = sorted(df[session_col].astype(str).unique())
    if len(sessions) < 2:
        return None
    fold_scores: List[float] = []
    for held in sessions:
        tr = df[df[session_col].astype(str) != held]
        te = df[df[session_col].astype(str) == held]
        # only classes seen in BOTH sides can be scored fairly
        common = set(tr[label_col]) & set(te[label_col])
        if len(common) < 2:
            continue
        tr = tr[tr[label_col].isin(common)]
        te = te[te[label_col].isin(common)]

        clf = _build(mode, snv)
        groups = tr[session_col].astype(str) if mode != "none" else None
        clf.train(
            tr.drop(columns=[c for c in (label_col, session_col) if c in tr.columns]),
            y=tr[label_col],
            groups=groups,
        )
        # Reference the held-out session to ITS OWN clean-air baseline.
        if mode != "none":
            air = te[te[label_col].astype(str).str.lower() == air_label]
            if len(air):
                clf.update_baseline(
                    air.drop(columns=[c for c in (label_col, session_col) if c in air.columns])
                       .to_dict("records"),
                    ema=False,
                )
        X = te.drop(columns=[c for c in (label_col, session_col) if c in te.columns])
        y_true = te[label_col].astype(str).str.lower().tolist()
        preds = [clf.predict(X.iloc[[i]].to_dict("records")[0])[0] for i in range(len(X))]
        # score only the non-air classes (the smells we actually care to identify)
        idx = [i for i, y in enumerate(y_true) if y != air_label]
        if not idx:
            continue
        fold_scores.append(
            balanced_accuracy_score([y_true[i] for i in idx], [preds[i] for i in idx])
        )
    return float(np.mean(fold_scores)) if fold_scores else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/database_robodog_time_windows.csv")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--session-col", default=None)
    ap.add_argument("--air-label", default="air")
    ap.add_argument("--cap-per-class-session", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    label_col = _pick(df.columns, LABEL_CANDIDATES, args.label_col)
    session_col = _pick(df.columns, SESSION_CANDIDATES, args.session_col)
    if label_col is None:
        print(f"[eval_drift] no label column found in {list(df.columns)[:12]}…", file=sys.stderr)
        return 2
    if session_col is None:
        print(
            "[eval_drift] no session/segment column — cross-session eval needs one "
            f"of {SESSION_CANDIDATES}. Pass --session-col.",
            file=sys.stderr,
        )
        return 2

    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    air = args.air_label.lower()
    df = _subsample(df, label_col, session_col, args.cap_per_class_session, args.seed)

    n_sessions = df[session_col].nunique()
    smells = sorted(set(df[label_col]) - {air})
    print(f"\n[eval_drift] {args.csv}")
    print(f"  rows={len(df)}  sessions={n_sessions} ({session_col})  "
          f"non-air classes={smells}\n")
    if n_sessions < 2:
        print("  ⚠ only one session present — cannot measure cross-session drift.")
        print("    Collect the SAME smells on ≥2 days (see docs/DRIFT_DATA_PROTOCOL.md).")
        return 1

    print(f"  {'representation':<18}{'cross-session bal.acc':>22}")
    print(f"  {'-'*18}{'-'*22:>22}")
    baseline = None
    for mode, snv in MODES:
        name = f"{mode}+snv" if snv else mode
        score = _loso_score(df, label_col, session_col, air, mode, snv)
        if mode == "none" and not snv:
            baseline = score
        if score is None:
            print(f"  {name:<18}{'n/a':>22}")
            continue
        delta = "" if baseline is None else f"  ({score - baseline:+.3f} vs absolute)"
        print(f"  {name:<18}{score:>21.3f}{delta}")
    print("\n  Higher = more drift-robust. 'none' is the current pipeline.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
