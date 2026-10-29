"""Critical guards against stale baselines and silent resistance-unit errors."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from enose.utils.acquisition import (
    acquisition_fingerprint, acquisition_provenance, bind_live_baseline,
    live_baseline_error,
)


def test_baseline_binding_rejects_identity_configuration_or_session_changes():
    acquisition = {
        "backend": "pcb", "device_id": "board-A", "resistance_unit": "ohm",
        "config": {"tia_map": list(range(17)), "temperature_setpoint_c": 285},
        "segment_id": 0, "frame_index": 10, "temp0_c": 284,
    }
    for changed in ("device", "configuration", "session", "segment", "metadata_missing"):
        classifier = SimpleNamespace(
            config=SimpleNamespace(baseline_mode="logratio"), live_baseline_captured_=True,
        )
        bind_live_baseline(classifier, acquisition, "session-A")
        incoming = deepcopy(acquisition)
        incoming.update(frame_index=123, temp0_c=286, received_at=123456.0)
        assert acquisition_fingerprint(incoming) == acquisition_fingerprint(acquisition)
        assert live_baseline_error(classifier, incoming, "session-A") is None
        session = "session-A"
        if changed == "device":
            incoming["device_id"] = "board-B"
        elif changed == "configuration":
            incoming["config"]["temperature_setpoint_c"] = 290
        elif changed == "session":
            session = "session-B"
        elif changed == "segment":
            incoming["segment_id"] = 1
        else:
            incoming = None
        assert live_baseline_error(classifier, incoming, session)
        assert classifier.live_baseline_captured_ is False
        assert live_baseline_error(classifier, acquisition, "session-A")

    legacy = SimpleNamespace(
        config=SimpleNamespace(baseline_mode="logratio"), live_baseline_captured_=True,
    )
    assert live_baseline_error(legacy) is None
    replay = acquisition_provenance([{"backend": "replay", "device_id": "file-A", "resistance_unit": "ohm"}])
    assert replay["fingerprints"] and replay["verified"] is False


def test_console_verified_units_and_finite_positive_resistances(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from enose.server import state
    from enose.server.routes.smell import test_console_input
    from enose.server.schemas import ConsoleSensorData

    classifier = SimpleNamespace(
        is_fitted=True, acquisition_resistance_unit_="ohm",
        classes_=np.asarray(["air", "acetone"]),
        predict=lambda sample: ["air"],
        predict_proba=lambda sample: np.asarray([[0.9, 0.1]]),
    )
    monkeypatch.setattr(state, "smell_classifier", classifier)
    values = ",".join(["100"] * 17)
    for unit, first in ((None, "100"), ("mohm", "100"), ("ohm", "0"), ("ohm", "nan"), ("ohm", "inf")):
        with pytest.raises(HTTPException):
            asyncio.run(test_console_input(ConsoleSensorData(
                values=",".join([first] + ["100"] * 16), resistance_unit=unit,
            )))
    result = asyncio.run(test_console_input(ConsoleSensorData(values=values, resistance_unit="ohm")))
    assert result["predicted_smell"] == "air"
