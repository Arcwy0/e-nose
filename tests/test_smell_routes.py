import asyncio

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
