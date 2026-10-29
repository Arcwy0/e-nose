"""Read recorded Rs-mode CSVs through the sensor interface, without hardware.

Rows are timed from the file, not synthesized at 10 Hz. No UART, heater, VLM,
or robot commands are issued. A CSV row index is not a hardware frame counter.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from enose.config import ENV_DEFAULTS, RESISTANCE_SENSORS, RLOW
from enose.utils.pcb_csv import normalize_pcb_recording
from .pcb import PCBSensor


class PCBReplaySensor(PCBSensor):
    backend = "replay"

    def __init__(self, path, resistance_unit="ohm", speed=1.0):
        super().__init__(resistance_unit=resistance_unit)
        if not 0 <= speed < float("inf"):
            raise ValueError("Replay speed must be finite and >= 0 (0 = unthrottled)")
        self.path = Path(path)
        self.speed = speed
        self._rows = None
        self._row_index = 0
        self._replay_started = None
        self._connected = False
        self.report = {}

    def connect(self):
        try:
            name_date = re.search(r"_(\d{8})_\d{6}_", self.path.name)
            date = pd.to_datetime(name_date[1], format="%Y%m%d") if name_date else None
            self._rows, self.report = normalize_pcb_recording(
                pd.read_csv(self.path), self.resistance_unit, recording_date=date,
            )
            self._row_index = 0
            self._replay_started = time.monotonic()
            self._connected = True
            cadence = self._rows["Timestamp"].diff().dt.total_seconds().median()
            self.stale_after = max(1.0, 3 * float(cadence) / self.speed) if self.speed and pd.notna(cadence) else 1.0
            print(f"PCB CSV replay: {len(self._rows)} valid rows in {self.resistance_unit}; no hardware. {self.report}")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            print(f"Replay failed: {exc}")
            return False

    def read_single_measurement(self):
        if not self._connected or self._row_index >= len(self._rows):
            return None
        row = self._rows.iloc[self._row_index]
        elapsed = (row["Timestamp"] - self._rows.iloc[0]["Timestamp"]).total_seconds()
        if self.speed and time.monotonic() - self._replay_started < elapsed / self.speed:
            return None
        sample = {name: float(row[name]) for name in RESISTANCE_SENSORS}
        sample.update(ENV_DEFAULTS)
        self._delivered_metadata = {
            "backend": "replay", "resistance_unit": self.resistance_unit, "environment_source": "defaults",
            "device_id": self.path.stem, "valid": True, "frame_index": self._row_index,
            "received_at": time.time(), "sample_age_seconds": 0.0, "stale_after_seconds": self.stale_after,
            "dropped_frames": 0, "segment_id": 0,
            "temp0_c": float(row.get("pcb_temp_actual_c", 0)),
            "duty_permille": float(row.get("pcb_duty_permille", 0)), "flags": 0,
            "source_timestamp": str(row["Timestamp"]),
            "config": {"synthetic_frame_index": True,
                       "timestamp_date_known": self.report["timestamp_date_known"],
                       "dropped_csv_rows": self.report["dropped_rows"]},
        }
        self._row_index += 1
        return sample

    def prepare_recording(self):
        self._recording_ready = self._connected and not self._is_recording
        return self._recording_ready

    def disconnect(self):
        self.stop_recording()
        self._connected = False


def create_sensor(backend="pcb", port=None, baud_rate_enose=None, baud_rate_UART=9600,
                  offline=False, rlow=RLOW, resistance_unit="ohm", csv_path=None, replay_speed=1.0):
    """Shared CLI/mission factory. Direct ENoseSensor construction stays legacy."""
    from .sensors import ENoseSensor

    if baud_rate_enose is not None and baud_rate_enose <= 0:
        raise ValueError("Serial baud rate must be positive")
    if offline:
        if csv_path:
            raise ValueError("--offline and --sensor-csv are mutually exclusive")
        return ENoseSensor(port, baud_rate_enose or 9600, baud_rate_UART, True, rlow=rlow)
    if backend == "legacy":
        return ENoseSensor(port, baud_rate_enose or 9600, baud_rate_UART, False, rlow=rlow)
    if backend == "pcb":
        if csv_path:
            raise ValueError("--sensor-csv requires --sensor-backend replay")
        return PCBSensor(port, baud_rate_enose or 250000, baud_rate_UART, resistance_unit=resistance_unit)
    if backend == "replay":
        if not csv_path:
            raise ValueError("--sensor-backend replay requires --sensor-csv PATH")
        if port and any(port):
            raise ValueError("Replay does not use serial ports")
        return PCBReplaySensor(csv_path, resistance_unit=resistance_unit, speed=replay_speed)
    raise ValueError(f"Unknown sensor backend: {backend}")
