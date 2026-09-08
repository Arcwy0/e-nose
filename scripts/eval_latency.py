#!/usr/bin/env python3
"""Evaluate early-response accuracy versus measurement latency.

The script accepts all historical CSVs, recovers variable-width recorder rows,
deduplicates overlapping cumulative exports, extracts one response per exposure
and latency, then performs leave-one-exposure-out validation.  It is intended
for model selection, not as a replacement for a final held-out acquisition day.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from enose.config import RESISTANCE_SENSORS
from enose.utils.csv_io import parse_uploaded_enose_csv
from enose.utils.episode import EarlyResponseConfig, build_early_response_training_frame


def _load(paths: List[str]) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    frames = []
    reports: Dict[str, dict] = {}
    for order, path in enumerate(paths):
        frame, report = parse_uploaded_enose_csv(Path(path).read_text(), "Gas name")
        frame["_source_order"] = order
        reports[path] = report
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["Gas name"] = (
        combined["Gas name"].astype(str).str.lower().str.strip()
        .replace({"amyl alcohol": "amyl"})
    )
    # Numeric class ids were reused for different analytes in different years;
    # textual names are the only safe cross-export identity.
    combined = combined.drop(columns=["Gas label", "Gas class"], errors="ignore")
    # Cumulative recorder exports overlap. Ignore environment columns during
    # deduplication because the malformed UART rows cannot recover them.
    keys = ["Timestamp", *RESISTANCE_SENSORS, "Gas name"]
    key_frame = combined[keys].copy()
    for sensor in RESISTANCE_SENSORS:
        key_frame[sensor] = pd.to_numeric(key_frame[sensor], errors="coerce").round(7)
    hashes = pd.util.hash_pandas_object(key_frame, index=False)
    keep = ~hashes.duplicated()
    combined = combined.loc[keep].drop(columns=["_source_order"]).reset_index(drop=True)
    reports["deduplication"] = {
        "input_rows": int(sum(len(frame) for frame in frames)),
        "unique_rows": int(len(combined)),
        "overlap_rows_removed": int((~keep).sum()),
    }
    return combined, reports


def _response_features(frame: pd.DataFrame, latency: float):
    X, y, groups = [], [], []
    for exposure_id, episode in frame.groupby("exposure_id", sort=False):
        baseline = episode[episode["phase"] == "baseline_air"]
        response = episode[
            (episode["phase"] == "early_response")
            & np.isclose(episode["latency_seconds"].astype(float), latency)
        ]
        if len(baseline) != 1 or len(response) != 1:
            continue
        r0 = baseline.iloc[0][RESISTANCE_SENSORS].to_numpy(float)
        rt = response.iloc[0][RESISTANCE_SENSORS].to_numpy(float)
        valid = np.isfinite(r0) & np.isfinite(rt) & (r0 > 0.0) & (rt > 0.0)
        if int(valid.sum()) < 12:
            continue
        log_response = np.zeros(len(RESISTANCE_SENSORS), dtype=float)
        log_response[valid] = np.log(rt[valid] / r0[valid])
        center = float(np.median(log_response[valid]))
        scale = float(np.median(np.abs(log_response[valid] - center))) + 1e-4
        shape = (log_response - center) / scale
        summary = np.asarray([
            np.linalg.norm(log_response), center,
            np.quantile(log_response[valid], 0.1),
            np.quantile(log_response[valid], 0.9),
        ])
        X.append(np.r_[log_response, shape, valid.astype(float), summary])
        y.append(str(response.iloc[0]["Gas name"]))
        groups.append(str(exposure_id))
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def _models():
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=150, class_weight="balanced", max_features=0.8,
            random_state=42, n_jobs=-1,
        ),
        "rbf_svm": make_pipeline(
            RobustScaler(),
            SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        ),
    }


def _leave_one_exposure_out(model, X, y, groups):
    predicted, truth = [], []
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if not set(y[test]).issubset(set(y[train])):
            continue
        model.fit(X[train], y[train])
        predicted.extend(model.predict(X[test]).tolist())
        truth.extend(y[test].tolist())
    return np.asarray(truth), np.asarray(predicted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Historical CSV files; cumulative overlap is removed")
    parser.add_argument("--latencies", nargs="+", type=float, default=[20, 30, 60, 120])
    parser.add_argument("--classes", nargs="+", help="Optional label subset")
    args = parser.parse_args()

    source, parse_reports = _load(args.csv)
    early, report = build_early_response_training_frame(
        source, config=EarlyResponseConfig(latencies_seconds=tuple(args.latencies))
    )
    dedup = parse_reports["deduplication"]
    print(
        f"{dedup['input_rows']} input rows -> {dedup['unique_rows']} unique rows "
        f"({dedup['overlap_rows_removed']} overlaps removed)"
    )
    print(
        f"{report['exposures']} usable exposures across {report['sessions']} days; "
        f"baselines={report['baseline_kinds']}"
    )
    for latency in args.latencies:
        X, y, groups = _response_features(early, float(latency))
        if args.classes:
            requested = {label.lower().strip() for label in args.classes}
            requested_mask = np.asarray([label in requested for label in y])
            X, y, groups = X[requested_mask], y[requested_mask], groups[requested_mask]
        counts = pd.Series(y).value_counts()
        eligible = set(counts[counts >= 2].index)
        mask = np.asarray([label in eligible for label in y])
        X, y, groups = X[mask], y[mask], groups[mask]
        labels = sorted(eligible)
        print(f"\nlatency={latency:g}s; exposures={len(y)}; classes={counts.to_dict()}")
        for name, model in _models().items():
            truth, predicted = _leave_one_exposure_out(model, X, y, groups)
            accuracy = float(np.mean(truth == predicted)) if len(truth) else 0.0
            balanced = balanced_accuracy_score(truth, predicted) if len(truth) else 0.0
            print(f"{name}: accuracy={accuracy:.4f} balanced_accuracy={balanced:.4f} n={len(truth)}")
            print(pd.DataFrame(
                confusion_matrix(truth, predicted, labels=labels),
                index=labels, columns=labels,
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
