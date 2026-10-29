"""Continuous USB CDC acquisition with opt-in monitored PI heater control.

No flash, calibration, gain, fixed-duty, or robot commands are sent. Heater
disable is attempted and verified on disconnect; STREAM_STOP alone is not off.
"""

from __future__ import annotations

import copy
import math
import threading
import time

import serial

from enose.config import ENV_DEFAULTS, ENVIRONMENTAL_SENSORS
from . import pcb_protocol as protocol
from .sensors import ENoseSensor, sanitize_environmentals_inplace


class PCBSensor(ENoseSensor):
    backend = "pcb"

    def __init__(self, port=None, baud_rate_enose=250000, baud_rate_UART=9600,
                 resistance_unit="ohm", stale_after=1.0, command_timeout=3.0):
        super().__init__(port, baud_rate_enose, baud_rate_UART)
        if resistance_unit not in protocol.RESISTANCE_SCALES:
            raise ValueError("Resistance unit must be ohm, kohm, or mohm")
        if not all(math.isfinite(value) and value > 0 for value in (stale_after, command_timeout)):
            raise ValueError("PCB timeouts must be positive")
        self.resistance_unit = resistance_unit
        self.stale_after = float(stale_after)
        self.command_timeout = float(command_timeout)
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = None
        self._env_reader = None
        self._parser = protocol.StreamParser()
        self._info = {}
        self._status = {}
        self._pending = {}
        self._sequence = 0
        self._latest = None
        self._latest_metadata = {}
        self._delivered_metadata = {}
        self._version = 0
        self._returned_version = 0
        self._last_frame = None
        self._segment = 0
        self._received_mono = None
        self._env = dict(ENV_DEFAULTS)
        self._env_received = None
        self._io_error = None
        self.last_error = None
        self.dropped_frames = 0
        self._status_version = 0
        from .heater import HeaterController
        self.heater = HeaterController(self)

    def set_rlow(self, value):
        raise ValueError("RLOW is legacy-only; PCB firmware already converts ADC to resistance")

    def connect(self):
        if self._reader and self._reader.is_alive():
            return self._io_error is None
        if not self.port or not self.port[0]:
            self.port = self.select_port()
        if not self.port or not self.port[0]:
            return False
        if self.port[1] and self.port[1] == self.port[0]:
            self.last_error = "PCB and environmental UART must use different ports"
            print(self.last_error)
            return False
        try:
            self._stop.clear()
            self.heater.shutdown_verified = False
            self._io_error = None
            self._info, self._status = {}, {}
            self._parser = protocol.StreamParser()
            self._latest = None
            self._received_mono = None
            self._delivered_metadata = {}
            self._last_frame = None
            self._env_received = None
            self._segment += 1
            connection = serial.Serial()
            self.serial_conn_enose = connection
            connection.port = self.port[0]
            connection.baudrate = self.baud_rate_enose
            connection.timeout = 0.05
            connection.write_timeout = 1.0
            connection.dtr = False
            connection.rts = False
            connection.open()
            connection.reset_input_buffer()
            if self.port[1]:
                self.serial_conn_UART = serial.Serial(self.port[1], self.baud_rate_UART, timeout=0.2)
                self._env_reader = threading.Thread(target=self._environment_loop, daemon=True)
                self._env_reader.start()
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            self._request(protocol.GET_INFO)
            self._request(protocol.GET_STATUS)
            with self._condition:
                ready = self._condition.wait_for(
                    lambda: bool(self._info and self._status) or self._io_error,
                    timeout=self.command_timeout,
                )
                if not ready or self._io_error:
                    raise RuntimeError(self._io_error or "PCB INFO/STATUS handshake timed out")
                if self._info["channel_count"] != 17:
                    raise ValueError("Expected a 17-channel PCB")
                if self._status["flags"] & protocol.FAULT_MASK:
                    raise ValueError(f"PCB fault/calibration flags: 0x{self._status['flags']:02x}")
            self._request(protocol.SET_VALUE_MODE, b"\0")  # Runtime only; never SAVE_SETTINGS.
            self._request(protocol.STREAM_START)
            with self._condition:
                # Old binary variants lacking an in-frame mode are accepted
                # only after the firmware acknowledged the Rs-mode command.
                self._status["stream_value_mode"] = 0
            print(f"PCB connected: {self.port[0]} @ {self.baud_rate_enose}; output {self.resistance_unit}.")
            print("Heating is opt-in; heater disable is attempted on exit. No firmware host-loss watchdog is documented.")
            return True
        except (Exception, KeyboardInterrupt) as exc:
            self.last_error = str(exc)
            print(f"PCB connection failed: {exc}. For the old board use --sensor-backend legacy.")
            self.disconnect()
            return False

    def _request(self, command, payload=b"", response=None):
        with self._write_lock:
            self._sequence = self._sequence % 255 + 1
            sequence = self._sequence
            with self._condition:
                self._pending[sequence] = [command, None, None]
            frame = protocol.pack_frame(command, sequence, payload)
            try:
                if self.serial_conn_enose.write(frame) != len(frame):
                    raise IOError("PCB short serial write")
                with self._condition:
                    ready = self._condition.wait_for(
                        lambda: (self._pending[sequence][1] is not None and (
                            self._pending[sequence][1] != 0 or response is None or
                            self._pending[sequence][2] is not None)) or self._io_error,
                        timeout=self.command_timeout,
                    )
                    status = self._pending[sequence][1]
                    if not ready or self._io_error or status != 0:
                        raise IOError(self._io_error or f"PCB command 0x{command:02x}: ACK {status}")
                    return copy.deepcopy(self._pending[sequence][2])
            finally:
                with self._condition:
                    self._pending.pop(sequence, None)

    def _read_loop(self):
        try:
            while not self._stop.is_set():
                connection = self.serial_conn_enose
                chunk = connection.read(max(1, min(connection.in_waiting, 4096)))
                for message, sequence, payload in self._parser.feed(chunk):
                    self._on_frame(message, sequence, payload)
        except Exception as exc:
            with self._condition:
                self._io_error = self.last_error = str(exc)
                self._latest = None
                self._condition.notify_all()

    def _on_frame(self, message, sequence, payload):
        with self._condition:
            try:
                if message == protocol.ACK:
                    pending = self._pending.get(sequence)
                    if pending and pending[0] == payload[0]:
                        pending[1] = payload[1]
                elif message == protocol.INFO:
                    self._info = protocol.parse_info(payload)
                    pending = self._pending.get(sequence)
                    if pending and pending[0] == protocol.GET_INFO:
                        pending[2] = self._info
                elif message == protocol.STATUS:
                    self._status = protocol.parse_status(payload)
                    self._status_version += 1
                    pending = self._pending.get(sequence)
                    if pending and pending[0] == protocol.GET_STATUS:
                        pending[2] = self._status
                elif message == protocol.STREAM:
                    measurement = protocol.parse_measurement(payload, self._status.get("stream_value_mode"))
                    self._accept_measurement(measurement)
            except (ValueError, IndexError) as exc:
                if self.last_error != str(exc):
                    print(f"PCB measurement rejected: {exc}")
                self.last_error = str(exc)
                self._latest = None
            self._condition.notify_all()

    def _accept_measurement(self, measurement):
        index = measurement["frame_index"]
        if self._last_frame is not None:
            delta = (index - self._last_frame) & 0xFFFFFFFF
            if delta == 0:
                return
            if delta >= 0x80000000:
                self._segment += 1  # Reset, not uint32 rollover.
            else:
                self.dropped_frames += max(0, delta - 1)
        self._last_frame = index
        if measurement["stream_value_mode"] != 0:
            raise ValueError("PCB is not in resistance mode; raw ADC cannot be classified")
        if measurement["flags"] & protocol.FAULT_MASK:
            raise ValueError(f"PCB fault/calibration flags: 0x{measurement['flags']:02x}")
        sample = protocol.canonical_resistances(measurement["channels_ohm"], self.resistance_unit)
        env_fresh = self._env_received is not None and time.monotonic() - self._env_received <= 3.0
        if self.port and self.port[1] and not env_fresh:
            raise ValueError("Configured environmental UART has no fresh readings")
        sample.update(self._env if env_fresh else ENV_DEFAULTS)
        now = time.time()
        self._latest = sample
        self._received_mono = time.monotonic()
        self._latest_metadata = {
            **{key: measurement[key] for key in ("frame_index", "temp0_c", "temp1_c", "duty_permille", "flags")},
            "backend": self.backend, "resistance_unit": self.resistance_unit,
            "environment_source": "uart" if env_fresh else "defaults",
            "device_id": self._info.get("device_id", "unknown"),
            "firmware_version": self._info.get("firmware_version", "unknown"),
            "received_at": now, "sample_age_seconds": 0.0,
            "stale_after_seconds": self.stale_after, "valid": True,
            "dropped_frames": self.dropped_frames, "segment_id": self._segment,
            "config": {key: value for key, value in self._info.items() if key not in ("device_id", "device_name")},
        }
        self._latest_metadata["heater"] = self.heater.diagnostics()
        self._version += 1
        self.last_error = None

    def _environment_loop(self):
        while not self._stop.is_set():
            try:
                line = self.serial_conn_UART.readline().decode("utf-8").strip()
                if not line:
                    continue
                values = [float(value) for value in line.split()]
                if len(values) != 5:
                    continue
                env = dict(zip(ENVIRONMENTAL_SENSORS, values))
                sanitize_environmentals_inplace(env)
                with self._condition:
                    self._env = env
                    self._env_received = time.monotonic()
            except Exception as exc:
                self.last_error = f"Environmental UART: {exc}"
                self._stop.wait(0.1)

    def read_single_measurement(self):
        with self._condition:
            if (self._latest is None or self._received_mono is None or self._io_error
                    or time.monotonic() - self._received_mono > self.stale_after
                    or self._version == self._returned_version):
                return None
            self._returned_version = self._version
            self._delivered_metadata = copy.deepcopy(self._latest_metadata)
            self._delivered_metadata["sample_age_seconds"] = time.monotonic() - self._received_mono
            return dict(self._latest)

    def get_acquisition_metadata(self):
        with self._condition:
            return copy.deepcopy(self._delivered_metadata or {
                "backend": self.backend, "resistance_unit": self.resistance_unit, "valid": False,
                "environment_source": "uart" if self.port and self.port[1] else "defaults",
            })

    def refresh_status(self):
        # Unsolicited SEQ=0 telemetry and another request's status never count
        # as confirmation of heater control. Wait for matching data AND ACK.
        return self._request(protocol.GET_STATUS, response=protocol.STATUS)

    def start_recording_after_confirmation(self, duration_seconds=60, target_samples=100):
        if duration_seconds <= 0 or target_samples <= 0:
            raise ValueError("Recording duration and sample count must be positive")
        # Discard pre-confirmation snapshots. Keep the acquisition thread alive
        # while the user waits, so USB buffers never accumulate old readings.
        with self._condition:
            self._returned_version = self._version
        # The legacy maxsize=200 queue silently truncates long recordings.
        import queue
        self._data_buffer = queue.Queue(maxsize=target_samples)
        return super().start_recording_after_confirmation(duration_seconds, target_samples)

    def disconnect(self):
        self.stop_recording()
        connection = self.serial_conn_enose
        try:
            if connection and getattr(connection, "is_open", False):
                self.heater.stop()
            if connection and getattr(connection, "is_open", False) and not self._io_error:
                try:
                    self._request(protocol.STREAM_STOP)
                except (OSError, AttributeError) as exc:
                    self.last_error = str(exc)
        finally:
            self._stop.set()
            for thread in (self._reader, self._env_reader):
                if thread and thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            for connection in (self.serial_conn_enose, self.serial_conn_UART):
                if connection:
                    connection.close()
            self.serial_conn_enose = self.serial_conn_UART = None
            with self._condition:
                self._latest = None
                self._pending.clear()
