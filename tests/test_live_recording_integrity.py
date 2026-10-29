"""Network backpressure must not discard locally acquired measurements."""
import threading
import time

from enose.client.live import LivePublisher
from enose.client.session import SessionRecorder


def test_slow_network_keeps_local_recording_independent(tmp_path):
    class Sensor:
        def __init__(self):
            self.index = 0

        def read_single_measurement(self):
            self.index += 1
            return {f"R{i}": float(self.index + i) for i in range(1, 18)}

        def get_acquisition_metadata(self):
            return {"backend": "pcb", "resistance_unit": "ohm", "frame_index": self.index,
                    "valid": True, "sample_age_seconds": 0, "stale_after_seconds": 1}

    sensor = Sensor()
    recorder = SessionRecorder(str(tmp_path / "recording.npz"))
    started, release = threading.Event(), threading.Event()
    publisher = LivePublisher(sensor, "http://unused", rate_hz=100,
                              on_sample=recorder.submit, on_complete=recorder.complete)

    def slow_push(t, sample, entry, pose=None):
        started.set()
        release.wait(1)
        entry["predicted"] = "synthetic_test"

    publisher._push = slow_push
    publisher.start()
    try:
        assert started.wait(1)
        time.sleep(0.08)
        assert len(recorder) >= 3
        assert publisher.n_network_dropped >= 1
    finally:
        release.set()
        publisher.stop()
    assert len(recorder) == sensor.index
    assert [row["sample"]["R1"] for row in recorder._rows] == list(range(2, sensor.index + 2))
