#!/usr/bin/env python3
"""Leakage-free plateau evaluation on a later, independent recording.

Both CSVs are reduced to stable one-minute windows.  Models are trained on
complete exposure cycles from ``--train`` and evaluated on complete cycles
from ``--test``; no adjacent rows leak across the boundary.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from enose.classifier import BalancedRFClassifier, TwoStageResponseClassifier
from enose.classifier.config import SmellClassifierConfig
from enose.config import ALL_SENSORS
from enose.utils.plateau import build_plateau_training_frame


def _model(name: str):
    cfg = SmellClassifierConfig()
    cfg.use_augmentation = False
    cfg.n_augmentations = 0
    cfg.calibrated = False
    cfg.n_estimators = 300
    if name == "absolute_rf":
        cfg.baseline_mode = "none"
        return BalancedRFClassifier(config=cfg)
    if name == "logratio_rf":
        cfg.baseline_mode = "logratio"
        return BalancedRFClassifier(config=cfg)
    return TwoStageResponseClassifier(config=cfg)


def _evaluate(clf, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[List[str], List[str]]:
    clf.train(
        train[ALL_SENSORS],
        train["Gas name"],
        groups=train["session_id"],
        baseline_groups=train["exposure_id"],
        test_size=0.2,
    )
    truth: List[str] = []
    predicted: List[str] = []
    for _, episode in test.groupby("exposure_id", sort=False):
        if getattr(clf.config, "baseline_mode", "none") != "none":
            air = episode[episode["phase"] == "baseline_air"]
            clf.update_baseline(air[ALL_SENSORS].to_dict("records"), ema=False)
        for _, row in episode.iterrows():
            truth.append(str(row["Gas name"]))
            predicted.append(clf.predict(row[ALL_SENSORS].to_dict())[0])
    return truth, predicted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Earlier continuous labelled CSV")
    parser.add_argument("--test", required=True, help="Later independent continuous labelled CSV")
    parser.add_argument("--label-col", default="Gas name")
    args = parser.parse_args()

    train, train_report = build_plateau_training_frame(
        pd.read_csv(args.train, low_memory=False), label_col=args.label_col
    )
    test, test_report = build_plateau_training_frame(
        pd.read_csv(args.test, low_memory=False), label_col=args.label_col
    )
    train_classes = sorted(set(train["Gas name"]))
    common = sorted(set(train_classes) & set(test["Gas name"]))
    if len(common) < 2:
        raise SystemExit(f"need at least two shared classes; found {common}")
    # Train exactly as deployment would: keep every class known from the
    # earlier dataset, even if the later recording does not exercise it.
    test = test[test["Gas name"].isin(common)].reset_index(drop=True)
    print(
        f"train: {train_report['input_rows']} rows -> {len(train)} stable windows; "
        f"test: {test_report['input_rows']} rows -> {len(test)} stable windows"
    )
    print(f"training classes: {train_classes}; test classes: {common}\n")

    for name in ("absolute_rf", "logratio_rf", "two_stage"):
        clf = _model(name)
        truth, predicted = _evaluate(clf, train, test)
        accuracy = float(np.mean(np.asarray(truth) == np.asarray(predicted)))
        balanced = balanced_accuracy_score(truth, predicted)
        print(f"{name}: accuracy={accuracy:.4f} balanced_accuracy={balanced:.4f}")
        labels = sorted(set(train_classes) | set(predicted))
        print(pd.DataFrame(confusion_matrix(truth, predicted, labels=labels), index=labels, columns=labels))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
