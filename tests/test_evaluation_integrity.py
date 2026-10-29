"""Critical exposure isolation and deployment-history integrity checks."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imblearn")

from enose.classifier import preprocessing
from enose.classifier.balanced_rf import BalancedRFClassifier, GroupCalibratedEstimator
from enose.classifier.config import SmellClassifierConfig
from enose.classifier.registry import create_candidate, promote_candidate, rollback
from enose.classifier import training
from enose.config import RESISTANCE_SENSORS


def _recording(groups=6):
    rows, labels, identifiers = [], [], []
    for group in range(groups):
        for label, level in (("air", 100), ("acetone", 30)):
            for repeat in range(3):
                rows.append({sensor: level + group * 10 + repeat for sensor in RESISTANCE_SENSORS})
                labels.append(label)
                identifiers.append(str(group))
    return pd.DataFrame(rows), pd.Series(labels), pd.Series(identifiers)


def _classifier(calibrated=False):
    return BalancedRFClassifier(config=SmellClassifierConfig(
        n_estimators=4, n_jobs=1, calibrated=calibrated,
        use_augmentation=False, n_augmentations=0,
    ))


def test_holdout_does_not_fit_resistance_statistics_on_test_rows(monkeypatch):
    frame, labels, groups = _recording()
    observed = []
    original = preprocessing.compute_resistance_clip_bounds

    def capture(raw, *args, **kwargs):
        observed.append(raw.copy())
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(preprocessing, "compute_resistance_clip_bounds", capture)
    classifier = _classifier()
    classifier.train(frame, labels, groups=groups)
    train_indices = classifier._split_train_indices_
    assert observed[0]["R1"].tolist() == frame.iloc[train_indices]["R1"].tolist()
    assert len(observed[0]) < len(frame)
    assert len(observed[-1]) == len(frame)  # deployment refit, after metrics


def test_calibration_keeps_exposures_and_learned_preprocessing_separate(tmp_path):
    frame, labels, groups = _recording()
    classifier = _classifier(calibrated=True)
    classifier.train(frame, labels, groups=groups)
    assert isinstance(classifier.model, GroupCalibratedEstimator)
    for (worker, _), identifiers in zip(classifier.model.folds, classifier.calibration_fold_groups_):
        assert not set(identifiers["fit"]) & set(identifiers["calibration"])
        expected = frame.loc[groups.isin(identifiers["fit"]), "R1"].median()
        assert worker.r_median_["R1"] == expected
    assert np.allclose(classifier.predict_proba(frame[:2]).sum(axis=1), 1)
    path = classifier.save_model(str(tmp_path / "model"))
    restored = BalancedRFClassifier.load_model(path)
    assert np.allclose(restored.predict_proba(frame[:2]), classifier.predict_proba(frame[:2]))


def test_single_exposure_per_class_reports_no_generalization_accuracy():
    frame, labels, _ = _recording(groups=1)
    classifier = _classifier(calibrated=True)
    score = classifier.train(frame, labels, groups=labels)
    assert score is None
    assert classifier.is_fitted
    assert classifier.training_history["balanced_accuracy"][-1] is None
    assert classifier.validation_strategy_ == "not_estimable"
    assert classifier.last_test_size_ == 0
    assert classifier.confusion_matrix_ is None
    assert classifier.calibration_strategy_ == "disabled_insufficient_groups"


def test_live_baseline_changes_calibrated_predictions_and_reset_restores_fallback():
    frame, labels, groups = _recording()
    # Identical repeat exposures isolate baseline adaptation from session drift.
    for sensor in RESISTANCE_SENSORS:
        frame[sensor] = np.where(labels == "air", 100.0, 30.0)
    classifier = _classifier(calibrated=True)
    classifier.config.baseline_mode = "logratio"
    classifier.train(frame, labels, groups=groups)
    assert isinstance(classifier.model, GroupCalibratedEstimator)
    reading = pd.DataFrame([{sensor: 100.0 for sensor in RESISTANCE_SENSORS}])
    original = classifier.predict_proba(reading)
    original_diagnostics = classifier.process_sensor_data(reading)
    assert classifier.predict(reading) == ["air"]

    classifier.update_baseline([{sensor: 1000.0 for sensor in RESISTANCE_SENSORS}], ema=False)
    adapted = classifier.predict_proba(reading)
    assert not np.allclose(adapted, original)
    assert classifier.predict(reading) == ["acetone"]
    assert all(worker.live_baseline_captured_ for worker, _ in classifier.model.folds)

    # Identity invalidation resets the captured flag; the next prediction must
    # restore each fold's training reference, not retain the discarded live R0.
    classifier.live_baseline_captured_ = False
    assert np.allclose(classifier.process_sensor_data(reading), original_diagnostics)
    assert np.allclose(classifier.predict_proba(reading), original)
    assert classifier.predict(reading) == ["air"]
    assert not any(worker.live_baseline_captured_ for worker, _ in classifier.model.folds)


def test_candidate_promotion_requires_explicit_choice_and_restores_history(tmp_path):
    frame, labels, groups = _recording()
    classifier = _classifier()
    classifier.train(frame, labels, groups=groups)
    classifier.canonical_training_data_ = frame.assign(smell_label=labels)
    root = tmp_path / "candidates"
    identifier = create_candidate(classifier, root)
    model_path = tmp_path / "active.joblib"
    history_path = tmp_path / "history.csv"
    model_path.write_bytes(b"original-model")
    history_path.write_text("original-history", encoding="utf-8")
    with pytest.raises(ValueError, match="paired independent"):
        promote_candidate(identifier, root, model_path, history_path)
    assert model_path.read_bytes() == b"original-model"
    snapshot = promote_candidate(identifier, root, model_path, history_path, allow_unevaluated=True)
    assert history_path.read_text(encoding="utf-8") != "original-history"
    rollback(snapshot, root, model_path, history_path)
    assert model_path.read_bytes() == b"original-model"
    assert history_path.read_text(encoding="utf-8") == "original-history"


def test_candidate_retrain_does_not_write_active_artifacts(tmp_path, monkeypatch):
    frame, labels, groups = _recording()
    frame["exposure_id"] = groups
    active = _classifier()
    model_directory = tmp_path / "active_models"
    history = tmp_path / "history.csv"
    history.write_text("unchanged-history", encoding="utf-8")

    class SmallClassifier(BalancedRFClassifier):
        def __init__(self, *args, config=None, **kwargs):
            config.n_estimators = 4
            config.n_jobs = 1
            config.calibrated = False
            super().__init__(*args, config=config, **kwargs)

    monkeypatch.setattr(training, "get_classifier_backend", lambda _: SmallClassifier)
    ok, _, candidate = training.retrain_with_all_data(
        active, frame, labels, use_augmentation=False, merge_history=False,
        combined_save_path=str(history), model_out_dir=str(model_directory), persist=False,
    )
    assert ok and candidate.is_fitted
    assert not active.is_fitted
    assert candidate.canonical_training_data_ is not None
    assert history.read_text(encoding="utf-8") == "unchanged-history"
    assert not model_directory.exists()
