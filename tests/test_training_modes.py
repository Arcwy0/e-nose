import numpy as np
import pandas as pd
import pytest

from enose.config import ALL_SENSORS


def test_csv_training_merges_history_by_default():
    pytest.importorskip("fastapi")
    from enose.server.schemas import CSVLearningData

    request = CSVLearningData(csv_data="R1,Gas name\n1,air")
    assert request.merge_history is True


def test_csv_training_can_request_replacement():
    pytest.importorskip("fastapi")
    from enose.server.schemas import CSVLearningData

    request = CSVLearningData(csv_data="R1,Gas name\n1,air", merge_history=False)
    assert request.merge_history is False


def test_split_falls_back_when_group_holdout_would_hide_classes():
    """One contiguous group per class cannot form a class-complete group split."""
    pytest.importorskip("imblearn")
    from enose.classifier.balanced_rf import BalancedRFClassifier

    rng = np.random.default_rng(42)
    labels = pd.Series(np.repeat(["air", "acetone", "ethanol"], 20))
    groups = pd.Series(np.repeat(["run-air", "run-acetone", "run-ethanol"], 20))
    X = pd.DataFrame(rng.normal(size=(len(labels), len(ALL_SENSORS))), columns=ALL_SENSORS)
    clf = BalancedRFClassifier()

    _, _, y_train, y_test = clf._split(X, labels, groups=groups, test_size=0.1)

    expected = {"air", "acetone", "ethanol"}
    assert set(y_train) == expected
    assert set(y_test) == expected
