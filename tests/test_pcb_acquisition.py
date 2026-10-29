"""Critical PCB data-integrity contracts. All USB traffic is simulated."""

import asyncio
import json
import struct
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from enose.client import pcb_protocol as protocol
from enose.client.pcb import PCBSensor
from enose.client.pcb_replay import PCBReplaySensor, create_sensor
from enose.config import ENV_DEFAULTS
from enose.utils.pcb_csv import normalize_pcb_recording


def measurement(index=1, mode=0, flags=0x38):
    return struct.pack("<Ihhh17iBhBBh", index, 285, 284, 297,
                       *[1_000_000 + i for i in range(17)], flags, 285, mode, 0, 0)


class FakeUSB:
    """Fragmented/coalesced binary traffic; never opens a physical port."""

    def __init__(self):
        self.is_open = False
        self.commands = []
        self.buffer = bytearray()
        self.lock = threading.Lock()

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        pass

    @property
    def in_waiting(self):
        with self.lock:
            return len(self.buffer)

    def read(self, length):
        # Intentionally split even the seven-byte frame header.
        with self.lock:
            chunk = bytes(self.buffer[:min(length, 3)])
            del self.buffer[:len(chunk)]
        if not chunk:
            time.sleep(0.001)
        return chunk

    def write(self, frame):
        version, command, sequence, length = struct.unpack_from("<BBBH", frame, 2)
        self.commands.append(command)
        response = protocol.pack_frame(protocol.ACK, sequence, bytes([command, 0]))
        if command == protocol.GET_INFO:
            info = bytearray(228)
            struct.pack_into("<3I", info, 0, 1, 2, 3)
            info[14:16] = bytes([17, 6])
            info[120:125] = b"0.6.9"
            struct.pack_into("<17f", info, 160, *[1.0] * 17)
            response = protocol.pack_frame(protocol.INFO, sequence, info) + response
        elif command == protocol.GET_STATUS:
            status = bytearray(49)
            status[8] = 0x10
            response += protocol.pack_frame(protocol.STATUS, sequence, status)
        elif command == protocol.STREAM_START:
            response += protocol.pack_frame(protocol.STREAM, 45, measurement())
        with self.lock:
            self.buffer.extend(response)
        return len(frame)


class PCBIntegrityTests(unittest.TestCase):
    def test_parser_recovers_noisy_split_and_coalesced_frames(self):
        parser = protocol.StreamParser()
        valid = protocol.pack_frame(protocol.STREAM, 3, measurement())
        malformed = b"\xa5\x5a\x01\x90\x00\xff\xff"
        frames = []
        data = b"noise\xa5" + malformed + valid + valid
        for start in range(0, len(data), 5):
            frames.extend(parser.feed(data[start:start + 5]))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0], (protocol.STREAM, 3, measurement()))
        self.assertGreater(parser.rejected_headers, 0)

    def test_handshake_handles_ack_order_without_heater_or_flash_commands(self):
        usb = FakeUSB()
        sensor = PCBSensor(port=("SIMULATED_PCB", None), command_timeout=0.5)
        with patch("enose.client.pcb.serial.Serial", return_value=usb):
            try:
                self.assertTrue(sensor.connect())
                deadline = time.monotonic() + 1
                sample = None
                while sample is None and time.monotonic() < deadline:
                    sample = sensor.read_single_measurement()
                    time.sleep(0.001)
                self.assertEqual(sample["R1"], 1_000_000.0)
                self.assertEqual(sample["T"], ENV_DEFAULTS["T"])
                self.assertEqual(sensor.get_acquisition_metadata()["temp0_c"], 285)
            finally:
                sensor.disconnect()
        self.assertEqual(usb.commands, [1, 2, 0x13, 3, 4])
        self.assertFalse(usb.is_open)

    def test_resistance_is_scaled_once_not_passed_through_rlow(self):
        sensor = PCBSensor(resistance_unit="mohm")
        sensor._on_frame(protocol.STREAM, 0, measurement())
        self.assertEqual(sensor.read_single_measurement()["R1"], 1.0)
        self.assertEqual(sensor.get_acquisition_metadata()["resistance_unit"], "mohm")
        with self.assertRaises(ValueError):
            sensor.set_rlow(0.1)

    def test_raw_faults_and_invalid_resistance_never_enter_model_samples(self):
        sensor = PCBSensor()
        for index, kwargs in enumerate([{"mode": 1}, {"flags": 0x39}, {"flags": 0x3C}, {"flags": 0xF8}]):
            sensor._on_frame(protocol.STREAM, index, measurement(index=index, **kwargs))
            self.assertIsNone(sensor.read_single_measurement())
            self.assertIn("fault" if "flags" in kwargs else "resistance mode", sensor.last_error)
        for value in [0, -1, float("nan"), float("inf"), 2147483647]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.canonical_resistances([value] * 17)

    def test_counter_gaps_wrap_and_freshness_do_not_duplicate_samples(self):
        sensor = PCBSensor()
        sensor._on_frame(protocol.STREAM, 0, measurement(index=0xFFFFFFFF))
        self.assertIsNotNone(sensor.read_single_measurement())
        self.assertIsNone(sensor.read_single_measurement())
        sensor._on_frame(protocol.STREAM, 0, measurement(index=0))
        self.assertIsNotNone(sensor.read_single_measurement())
        self.assertEqual(sensor.dropped_frames, 0)
        sensor._on_frame(protocol.STREAM, 0, measurement(index=3))
        self.assertEqual(sensor.dropped_frames, 2)
        sensor._received_mono -= 5
        self.assertIsNone(sensor.read_single_measurement())

    def test_csv_preserves_units_elapsed_time_and_never_invents_ambient_or_labels(self):
        frame = pd.DataFrame({**{f"rs_ch{i}_ohm": [1_000_000, 2_000_000] for i in range(17)},
                              "timestamp_iso": ["23:59:59.0", "00:00:01.0"], "temp_actual_c": [285, 286]})
        result, report = normalize_pcb_recording(frame, "mohm")
        self.assertEqual(result["R1"].tolist(), [1, 2])
        self.assertEqual((result["Timestamp"].iloc[1] - result["Timestamp"].iloc[0]).total_seconds(), 2)
        self.assertEqual(result["pcb_temp_actual_c"].tolist(), [285, 286])
        self.assertNotIn("T", result.columns)
        self.assertNotIn("smell_label", result.columns)
        self.assertFalse(report["timestamp_date_known"])
        with self.assertRaises(ValueError):
            normalize_pcb_recording(frame.rename(columns={"rs_ch0_ohm": "raw_ch0"}))

    def test_csv_replay_never_opens_serial_or_repeats_a_row(self):
        frame = pd.DataFrame({**{f"rs_ch{i}_ohm": [100, 200] for i in range(17)},
                              "timestamp_iso": ["18:00:00.0", "18:00:02.0"], "temp_actual_c": [285, 285]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.csv"
            frame.to_csv(path, index=False)
            sensor = PCBReplaySensor(path, speed=0)
            with patch("enose.client.pcb.serial.Serial", side_effect=AssertionError("No hardware allowed")):
                self.assertTrue(sensor.connect())
                self.assertEqual(sensor.read_single_measurement()["R1"], 100)
                self.assertEqual(sensor.read_single_measurement()["R1"], 200)
                self.assertIsNone(sensor.read_single_measurement())
                sensor.disconnect()

    def test_backend_default_and_legacy_conversion_are_separate(self):
        self.assertIsInstance(create_sensor(), PCBSensor)
        legacy = create_sensor("legacy", rlow=0.1)
        self.assertEqual(legacy.baud_rate_enose, 9600)
        self.assertEqual(legacy.rlow, 0.1)
        self.assertEqual(create_sensor().baud_rate_enose, 250000)
        self.assertTrue(create_sensor(offline=True).offline_mode)

    def test_disk_history_units_are_checked_before_training_or_writes(self):
        from enose.classifier.balanced_rf import BalancedRFClassifier
        from enose.classifier.training import retrain_with_all_data
        incoming = pd.DataFrame({"R1": [1.0, 2.0]})
        for model_unit, history_unit, incoming_unit in [("ohm", None, "ohm"), (None, "ohm", None)]:
            classifier = BalancedRFClassifier()
            classifier.acquisition_resistance_unit_ = model_unit
            classifier.is_fitted = model_unit is not None
            history = pd.DataFrame({"R1": [1, 2], "smell_label": ["air", "test"]})
            history.attrs["acquisition_history_source"] = "disk"
            if history_unit:
                history["resistance_unit"] = history_unit
            with self.subTest(model=model_unit, history=history_unit):
                with patch("enose.classifier.training._load_history", return_value=history), \
                     patch.object(BalancedRFClassifier, "train", side_effect=AssertionError("Must not fit")), \
                     self.assertRaisesRegex(ValueError, "unit"):
                    retrain_with_all_data(classifier, incoming, pd.Series(["air", "test"]),
                                          acquisition_resistance_unit=incoming_unit)

    def test_prediction_requires_valid_matching_explicit_units(self):
        from enose.utils.acquisition import prediction_unit_error
        for model_unit, acquired_unit in [(None, "ohm"), ("mohm", "ohm"), (None, "invalid")]:
            with self.subTest(model=model_unit, acquired=acquired_unit):
                model = SimpleNamespace(acquisition_resistance_unit_=model_unit)
                self.assertIsNotNone(prediction_unit_error(model, {
                    "backend": "pcb", "resistance_unit": acquired_unit,
                }))
        model = SimpleNamespace(acquisition_resistance_unit_="ohm")
        self.assertIsNone(prediction_unit_error(model, {"backend": "pcb", "resistance_unit": "ohm"}))
        self.assertIsNotNone(prediction_unit_error(model, {
            "backend": "pcb", "resistance_unit": "ohm", "valid": False,
        }))

    def test_unit_provenance_survives_supported_classifier_persistence(self):
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from enose.classifier.balanced_rf import BalancedRFClassifier
        from enose.classifier.two_stage import TwoStageResponseClassifier
        from enose.classifier.xgb_tabular import XGBTabularClassifier
        for kind in (BalancedRFClassifier, TwoStageResponseClassifier, XGBTabularClassifier):
            with self.subTest(backend=kind.__name__), tempfile.TemporaryDirectory() as directory:
                classifier = kind()
                classifier.model = DummyClassifier().fit(np.ones((2, 22)), ["air", "test"])
                classifier.classes_ = np.asarray(["air", "test"])
                classifier.acquisition_resistance_unit_ = "ohm"
                loaded = kind.load_model(classifier.save_model(directory))
                self.assertEqual(loaded.acquisition_resistance_unit_, "ohm")

    def test_live_windows_refuse_stale_or_mixed_unit_provenance(self):
        from fastapi import HTTPException
        from enose.server import state
        from enose.server.live_buffer import buffer
        from enose.server.routes.smell import _recent_live_frame
        model = SimpleNamespace(acquisition_resistance_unit_="ohm")
        now = time.time()
        try:
            with patch.object(state, "smell_classifier", model):
                buffer.clear()
                buffer.push({"t": now - 100, "sample": {"R1": 1}})
                with self.assertRaises(HTTPException) as stale:
                    _recent_live_frame(15, None)
                self.assertEqual(stale.exception.status_code, 409)
                buffer.clear()
                buffer.push({"t": now, "sample": {"R1": 1}})
                buffer.push({"t": now, "sample": {"R1": 1_000_000}, "acquisition": {
                    "backend": "pcb", "resistance_unit": "ohm", "device_id": "test", "valid": True,
                }})
                with self.assertRaises(HTTPException) as mixed:
                    _recent_live_frame(15, None)
                self.assertEqual(mixed.exception.status_code, 409)
        finally:
            buffer.clear()

    def test_invalid_acquisition_metadata_cannot_bypass_live_unit_checks(self):
        from unittest.mock import Mock
        from enose.client.live import LivePublisher
        for invalid in ({"backend": "pcb", "resistance_unit": "ohm", "temp0_c": float("nan")}, None, {}):
            with self.subTest(metadata=invalid):
                sensor = SimpleNamespace(read_single_measurement=lambda: {"R1": 1.0})
                on_sample, on_complete = Mock(), Mock()
                publisher = LivePublisher(sensor, "http://unused", classify=True,
                                          on_sample=on_sample, on_complete=on_complete)

                def metadata():
                    publisher._stop.set()
                    return invalid

                sensor.get_acquisition_metadata = metadata
                with patch.object(publisher, "_push") as push:
                    publisher._loop()
                push.assert_not_called()
                on_sample.assert_not_called()
                on_complete.assert_not_called()
                self.assertEqual(publisher.n_read_failed, 1)
                self.assertIn("sample skipped", publisher.last_error)

    def test_local_pcb_logs_cannot_append_ohms_to_a_legacy_file(self):
        from enose.client.pipeline import TrainingPipeline
        from enose.client.sensors import ENoseSensor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.csv"
            legacy = ENoseSensor(offline_mode=True)
            sample = legacy.read_single_measurement()
            TrainingPipeline(None, None, legacy, local_log_path=str(path)).save_local_data([sample], "test")
            original = path.read_bytes()
            pcb = PCBSensor()
            pcb._on_frame(protocol.STREAM, 0, measurement())
            sample = pcb.read_single_measurement()
            TrainingPipeline(None, None, pcb, local_log_path=str(path)).save_local_data([sample], "test")
            self.assertEqual(path.read_bytes(), original)
            saved = pd.read_csv(path.with_name("recording.pcb-ohm.csv"))
            self.assertEqual(saved["resistance_unit"].tolist(), ["ohm"])
            self.assertEqual(saved["R1"].tolist(), [1_000_000.0])


if __name__ == "__main__":
    unittest.main()
