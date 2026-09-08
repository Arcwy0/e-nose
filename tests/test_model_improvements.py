import numpy as np
import pandas as pd
import pytest

from enose.config import RESISTANCE_SENSORS
from enose.utils.csv_io import parse_uploaded_enose_csv
from enose.utils.episode import EarlyResponseConfig, build_early_response_training_frame
from enose.utils.plateau import PlateauConfig, build_plateau_training_frame
from enose.utils.quality import analyze_recording_quality, summarize_online_capture


def _block(start, seconds, label, value, class_id):
    timestamp = pd.date_range(start, periods=seconds + 1, freq="s")
    frame = pd.DataFrame({"Timestamp": timestamp, "Gas name": label, "Gas label": class_id})
    for index, sensor in enumerate(RESISTANCE_SENSORS):
        frame[sensor] = value + index * 0.1
    return frame


def test_plateau_profile_uses_pre_exposure_air_and_excludes_recovery():
    before = _block("2026-01-01 10:00:00", 180, "air", 100.0, 0)
    odor = _block("2026-01-01 10:03:01", 180, "acetone", 30.0, 1)
    # Recovery deliberately still looks like acetone. It must never be emitted
    # as clean-air training data for this exposure.
    recovery = _block("2026-01-01 10:06:02", 180, "air", 35.0, 0)
    source = pd.concat([before, odor, recovery], ignore_index=True)
    cfg = PlateauConfig(
        window_seconds=30,
        stride_seconds=30,
        baseline_tail_seconds=60,
        plateau_fraction=0.5,
        tail_margin_seconds=0,
        max_relative_slope=0.01,
    )

    result, report = build_plateau_training_frame(source, config=cfg)

    assert report["exposures"] == 1
    assert set(result["phase"]) == {"baseline_air", "exposure_plateau"}
    assert result.loc[result["Gas name"] == "air", "R1"].min() == pytest.approx(100.0)
    assert result.loc[result["Gas name"] == "acetone", "R1"].max() == pytest.approx(30.0)


def test_plateau_profile_merges_names_with_the_same_explicit_class_id():
    a = _block("2026-01-01 10:00:00", 60, "air", 100.0, 0)
    b = _block("2026-01-01 10:01:01", 60, "amyl alcohol", 40.0, 3)
    c = _block("2026-01-01 10:02:02", 60, "air", 100.0, 0)
    d = _block("2026-01-01 10:03:03", 60, "amyl", 35.0, 3)
    source = pd.concat([a, b, c, d], ignore_index=True)
    cfg = PlateauConfig(
        window_seconds=10,
        stride_seconds=10,
        baseline_tail_seconds=30,
        plateau_fraction=0.5,
        tail_margin_seconds=0,
        max_relative_slope=0.01,
    )

    result, report = build_plateau_training_frame(source, config=cfg)

    odor_names = set(result["Gas name"]) - {"air"}
    assert len(odor_names) == 1
    assert report["label_aliases"] in (
        {"amyl alcohol": "amyl"},
        {"amyl": "amyl alcohol"},
    )


def test_plateau_profile_splits_recorder_gaps():
    before = _block("2026-01-01 10:00:00", 120, "air", 100.0, 0)
    exposure = _block("2026-01-01 10:02:01", 180, "acetone", 30.0, 1)
    # The logger resumes next day with a stale analyte label. It is a separate
    # segment with no preceding clean-air baseline and must not be included.
    stale = _block("2026-01-02 10:00:00", 180, "acetone", 25.0, 1)
    cfg = PlateauConfig(
        window_seconds=30,
        stride_seconds=30,
        baseline_tail_seconds=60,
        plateau_fraction=0.5,
        tail_margin_seconds=0,
        max_relative_slope=0.01,
    )

    result, report = build_plateau_training_frame(
        pd.concat([before, exposure, stale], ignore_index=True), config=cfg
    )

    assert report["segments"] == 3
    assert report["exposures"] == 1
    assert not result["t_start"].str.startswith("2026-01-02").any()


def test_variable_width_recorder_csv_recovers_label_and_ignores_uart_fragments():
    header = ",".join(
        ["Timestamp", *RESISTANCE_SENSORS, "T", "H", "CO2", "H2S", "CH2O", "Gas name", "Gas label", "", ""]
    )
    resistance = ",".join(str(i) for i in range(1, 18))
    normal = f"2025-01-01 10:00:00,{resistance},21,40,500,0,5,air,1,,"
    corrupt = f"2025-01-01 10:00:02.500,{resistance},21,40,500,21,40,500,0,5,acetone,2"

    result, report = parse_uploaded_enose_csv(
        "\n".join((header, normal, corrupt)), label_column="Gas name"
    )

    assert result["Gas name"].tolist() == ["air", "acetone"]
    assert report["parser"] == "variable_width_recovery"
    assert report["recovered_rows"] == 1
    assert result.loc[1, "T"] == 21.0  # safe default, not a shifted UART field


def test_plateau_median_ignores_multiplexed_zero_placeholders():
    before = _block("2026-01-01 10:00:00", 60, "air", 100.0, 0)
    exposure = _block("2026-01-01 10:01:01", 60, "acetone", 30.0, 1)
    # More than half of the individual frames omit R1, but the real positive
    # readings still define its window value.
    before.loc[before.index % 3 != 0, "R1"] = 0.0
    exposure.loc[exposure.index % 3 != 0, "R1"] = 0.0
    cfg = PlateauConfig(
        window_seconds=20, stride_seconds=20, baseline_tail_seconds=40,
        plateau_fraction=0.5, tail_margin_seconds=0, max_relative_slope=0.01,
    )

    result, _ = build_plateau_training_frame(
        pd.concat([before, exposure], ignore_index=True), config=cfg
    )

    assert result.loc[result["Gas name"] == "air", "R1"].min() == pytest.approx(100.0)
    assert result.loc[result["Gas name"] == "acetone", "R1"].max() == pytest.approx(30.0)


def test_early_response_profile_supports_onset_baseline_and_short_latency():
    exposure = _block("2026-01-01 10:00:00", 90, "acetone", 30.0, 1)
    # Simulate a response beginning from air at 100 and moving toward 30.
    for sensor in RESISTANCE_SENSORS:
        exposure[sensor] = np.linspace(100.0, 30.0, len(exposure))
    cfg = EarlyResponseConfig(
        latencies_seconds=(20.0, 30.0), onset_baseline_seconds=10.0,
        response_window_seconds=10.0, min_window_samples=3,
    )

    result, report = build_early_response_training_frame(exposure, config=cfg)

    assert report["exposures"] == 1
    assert report["baseline_kinds"] == {"onset_proxy": 1}
    assert result["latency_seconds"].tolist() == [0.0, 20.0, 30.0]
    assert result.iloc[0]["Gas name"] == "air"
    assert result.iloc[-1]["R1"] < result.iloc[0]["R1"]


def test_recording_quality_treats_zero_placeholders_as_missing():
    frame = _block("2026-01-01 10:00:00", 9, "air", 100.0, 0)
    frame.loc[:7, "R1"] = 0.0
    frame.loc[9, "R1"] = 101.0
    frame["R2"] = 0.0

    report = analyze_recording_quality(frame)

    assert report["resistance_sensors"]["R1"]["status"] == "sparse"
    assert report["resistance_sensors"]["R2"]["status"] == "dead"
    assert "R2" in report["dead_resistance_sensors"]


def test_online_capture_uses_zero_aware_stable_median_windows():
    frame = pd.DataFrame({
        sensor: np.repeat(100.0 + index, 20)
        for index, sensor in enumerate(RESISTANCE_SENSORS)
    })
    frame.loc[frame.index % 2 == 0, "R1"] = 0.0

    result, report = summarize_online_capture(frame, max_windows=4)

    assert len(result) == 4
    assert (result["R1"] == 100.0).all()
    assert report["input_frames"] == 20
    assert report["rejected_windows"] == 0


def test_online_capture_rejects_strong_transients():
    frame = pd.DataFrame({
        sensor: np.linspace(100.0 + index, 20.0 + index, 20)
        for index, sensor in enumerate(RESISTANCE_SENSORS)
    })

    with pytest.raises(ValueError, match="stability"):
        summarize_online_capture(frame, max_windows=1, max_relative_mad=0.05)


def test_two_stage_estimator_separates_air_detection_from_odor_shape():
    pytest.importorskip("imblearn")
    from enose.classifier.two_stage import TwoStageEstimator

    rng = np.random.default_rng(42)
    air = rng.normal(0.0, 0.01, size=(30, 17))
    shape_a = np.linspace(-0.2, -1.0, 17)
    shape_b = np.r_[np.linspace(-1.0, -0.2, 8), np.linspace(-0.2, -1.0, 9)]
    acetone = shape_a + rng.normal(0.0, 0.01, size=(30, 17))
    ethanol = 3.0 * shape_b + rng.normal(0.0, 0.01, size=(30, 17))
    X = np.vstack([air, acetone, ethanol])
    y = np.asarray(["air"] * 30 + ["acetone"] * 30 + ["ethanol"] * 30)

    model = TwoStageEstimator().fit(X, y)
    predicted = model.predict(np.vstack([air[:2], acetone[:2] * 2.0, ethanol[:2] * 0.5]))

    assert predicted.tolist() == ["air", "air", "acetone", "acetone", "ethanol", "ethanol"]
    assert np.allclose(model.predict_proba(X[:4]).sum(axis=1), 1.0)


def test_deployable_model_is_refit_on_all_rows_after_holdout_metrics():
    pytest.importorskip("imblearn")
    from enose.classifier.balanced_rf import BalancedRFClassifier
    from enose.classifier.config import SmellClassifierConfig

    class CountingEstimator:
        def fit(self, X, y):
            self.n_fit_rows = len(X)
            self.classes_ = np.asarray(sorted(np.unique(y)))
            return self

        def predict(self, X):
            return np.repeat(self.classes_[0], len(X))

    class CountingClassifier(BalancedRFClassifier):
        def _build_model(self, y_fit=None):
            return CountingEstimator()

    rows = []
    labels = []
    groups = []
    for group in range(6):
        for class_index, label in enumerate(("acetone", "air", "ethanol")):
            rows.append({sensor: 10.0 + group + class_index for sensor in RESISTANCE_SENSORS})
            labels.append(label)
            groups.append(group)
    cfg = SmellClassifierConfig()
    cfg.calibrated = False
    cfg.use_env_sensors = False
    cfg.use_augmentation = False
    cfg.n_augmentations = 0
    clf = CountingClassifier(config=cfg)

    clf.train(pd.DataFrame(rows), labels, groups=pd.Series(groups), test_size=0.2)

    assert clf.model.n_fit_rows == len(rows)
