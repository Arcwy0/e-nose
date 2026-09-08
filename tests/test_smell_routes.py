import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from enose.config import ENV_DEFAULTS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS


def test_console_accepts_resistance_only_input(monkeypatch):
    pytest.importorskip("fastapi")

    from enose.server import state
    from enose.server.routes.smell import test_console_input
    from enose.server.schemas import ConsoleSensorData

    class DummyClassifier:
        is_fitted = True
        classes_ = np.asarray(["air", "ethanol"])

        def __init__(self):
            self.inputs = []

        def predict(self, sample):
            self.inputs.append(dict(sample))
            return ["air"]

        def predict_proba(self, sample):
            self.inputs.append(dict(sample))
            return np.asarray([[0.9, 0.1]])

    classifier = DummyClassifier()
    monkeypatch.setattr(state, "smell_classifier", classifier)
    values = ",".join(str(i) for i in range(1, 18))

    result = asyncio.run(test_console_input(ConsoleSensorData(values=values)))

    assert result["predicted_smell"] == "air"
    assert result["sensor_input"]["resistance_sensors"] == {
        name: float(i) for i, name in enumerate(RESISTANCE_SENSORS, start=1)
    }
    assert result["sensor_input"]["environmental_sensors"] == ENV_DEFAULTS
    assert all(sample[name] == ENV_DEFAULTS[name]
               for sample in classifier.inputs
               for name in ENVIRONMENTAL_SENSORS)


def test_live_baseline_and_stable_window_classification(monkeypatch):
    pytest.importorskip("fastapi")

    from enose.server import state
    from enose.server.live_buffer import buffer
    from enose.server.routes.smell import classify_stable_live_window, set_baseline_from_live

    class DummyClassifier:
        is_fitted = True
        classes_ = np.asarray(["air", "acetone"])
        config = SimpleNamespace(baseline_mode="logratio")

        def update_baseline(self, samples, ema=False):
            self.baseline_samples = samples
            return {name: float(np.mean([row[name] for row in samples])) for name in RESISTANCE_SENSORS}

        @staticmethod
        def predict(sample):
            return ["air"]

        @staticmethod
        def predict_proba(sample):
            return np.asarray([[0.95, 0.05]])

    classifier = DummyClassifier()
    monkeypatch.setattr(state, "smell_classifier", classifier)
    buffer.clear()
    now = time.time()
    for index in range(31):
        sample = {name: 100.0 + sensor_index for sensor_index, name in enumerate(RESISTANCE_SENSORS)}
        buffer.push({"t": now - 60 + index * 2, "sample": sample, "session_id": "test"})

    kwargs = {"window": 60.0, "max_relative_slope": 0.002, "min_samples": 10, "session_id": "test"}
    baseline = asyncio.run(set_baseline_from_live(**kwargs))
    result = asyncio.run(classify_stable_live_window(**kwargs))

    assert baseline["applied"] is True
    assert baseline["n_air_samples"] == 31
    assert result["stable"] is True
    assert result["predicted_smell"] == "air"
    assert result["n_samples"] == 31
    buffer.clear()


def test_fast_live_window_abstains_when_probabilities_are_tied(monkeypatch):
    pytest.importorskip("fastapi")

    from enose.server import state
    from enose.server.live_buffer import buffer
    from enose.server.routes.smell import classify_recent_window

    class DummyClassifier:
        is_fitted = True
        live_baseline_captured_ = True
        classes_ = np.asarray(["air", "acetone"])
        config = SimpleNamespace(baseline_mode="logratio")

        @staticmethod
        def predict_proba(sample):
            return np.asarray([[0.52, 0.48]])

    monkeypatch.setattr(state, "smell_classifier", DummyClassifier())
    buffer.clear()
    now = time.time()
    for index in range(16):
        sample = {
            name: 100.0 + sensor_index
            for sensor_index, name in enumerate(RESISTANCE_SENSORS)
        }
        buffer.push({"t": now - 15 + index, "sample": sample, "session_id": "fast"})

    result = asyncio.run(classify_recent_window(
        window=15.0, bin_seconds=5.0, min_samples=5,
        max_relative_slope=0.002, min_confidence=0.45,
        min_margin=0.10, session_id="fast",
    ))

    assert result["abstained"] is True
    assert result["predicted_smell"] == "unknown"
    assert result["candidate_smell"] == "air"
    assert result["bins_used"] == 3
    assert result["inference_compute_ms"] >= 0.0
    buffer.clear()


def test_recovery_compares_live_window_to_session_baseline(monkeypatch):
    pytest.importorskip("fastapi")

    from enose.server import state
    from enose.server.live_buffer import buffer
    from enose.server.routes.smell import recovery_status

    class DummyClassifier:
        is_fitted = True
        live_baseline_captured_ = True
        config = SimpleNamespace(baseline_mode="logratio")
        sensor_baseline_ = {name: 100.0 for name in RESISTANCE_SENSORS}

    monkeypatch.setattr(state, "smell_classifier", DummyClassifier())
    buffer.clear()
    now = time.time()
    for index in range(16):
        buffer.push({
            "t": now - 15 + index,
            "sample": {name: 100.0 for name in RESISTANCE_SENSORS},
            "session_id": "recovery",
        })

    result = asyncio.run(recovery_status(
        window=15.0, response_threshold=0.12, max_relative_slope=0.002,
        min_samples=5, session_id="recovery",
    ))

    assert result["recovered"] is True
    assert result["response_score"] == pytest.approx(0.0)
    buffer.clear()


def test_fast_live_window_requires_a_current_session_baseline(monkeypatch):
    pytest.importorskip("fastapi")

    from fastapi import HTTPException
    from enose.server import state
    from enose.server.routes.smell import classify_recent_window

    classifier = SimpleNamespace(
        is_fitted=True,
        live_baseline_captured_=False,
        classes_=np.asarray(["air", "acetone"]),
        config=SimpleNamespace(baseline_mode="logratio"),
    )
    monkeypatch.setattr(state, "smell_classifier", classifier)

    with pytest.raises(HTTPException) as error:
        asyncio.run(classify_recent_window(
            window=15.0, bin_seconds=5.0, min_samples=5,
            max_relative_slope=0.002, min_confidence=0.45,
            min_margin=0.10, session_id="new-session",
        ))

    assert error.value.status_code == 409
    assert "clean-air baseline" in error.value.detail
