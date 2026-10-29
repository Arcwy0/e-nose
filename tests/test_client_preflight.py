"""Backend probing must never send binary control queries to legacy boards."""

import json

import pytest

from enose.client.preflight import collect_preflight


@pytest.mark.parametrize("backend", ["legacy", "simulation", "replay"])
def test_preflight_never_probes_non_pcb_protocols(backend):
    class Sensor:
        offline_mode = backend == "simulation"
        port = None
        baud_rate_enose = 9600
        last_error = None
        index = 0

        def refresh_status(self):
            raise AssertionError("Non-PCB backend must never receive STATUS")

        def _request(self, *args):
            raise AssertionError("Non-PCB backend must never receive commands")

        def get_acquisition_metadata(self):
            return {"backend": backend, "resistance_unit": "ohm", "valid": True}

        def read_single_measurement(self):
            self.index += 1
            return {f"R{i}": 100.0 for i in range(1, 18)} if self.index == 1 else None

    sensor = Sensor()
    sensor.backend = backend
    report = collect_preflight(sensor, duration=0.025)
    assert report["valid_unique_samples"] == 1
    assert report["idle_polls"] > 0
    assert report["read_errors"] == 0
    assert report["polls_with_error_status"] == 0
    assert report["backend"] == backend
    assert report["protocol"]["status"] in {"unsupported_not_probed", "not_applicable_not_probed"}
    json.dumps(report, allow_nan=False)
