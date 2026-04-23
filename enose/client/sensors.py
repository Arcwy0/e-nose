"""E-nose sensor handler: dual-serial I/O, ADC→resistance transform, offline simulation.

Raw e-nose UART gives ADC counts for R1–R17; we convert each count to resistance via
`(RLOW·COEF·VCC·EG)/(VREF·v) - RLOW`. Environmentals (T/H/CO2/H2S/CH2O) arrive from
a separate UART already in engineering units and are sanitized to plausible ranges.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports

from enose.config import (
    ALL_SENSORS,
    COEF,
    DEFAULT_BAUD_RATE,
    DEFAULT_RECORDING_TIME,
    DEFAULT_TARGET_SAMPLES,
    EG,
    ENV_DEFAULTS,
    ENV_SANITY_MEDIANS,
    ENV_SANITY_RANGES,
    ENVIRONMENTAL_SENSORS,
    RLOW,
    VCC,
    VREF,
)


# Reference smell profiles for offline simulation (R1–R17 resistance patterns).
_SMELL_PROFILES: Dict[str, List[float]] = {
    "air":       [0.1, 0.05, 0.08, 0.02, 0.12, 0.09, 0.06, 0.04, 0.11, 0.07, 0.08, 0.05, 0.03, 0.09, 0.06, 0.04, 0.08],
    "coffee":    [15.2, 8.3, 12.1, 3.4, 18.9, 11.2, 9.8, 6.7, 14.3, 10.5, 13.2, 7.8, 5.6, 12.4, 8.9, 6.3, 11.7],
    "rose":      [8.7, 12.4, 6.8, 9.2, 5.3, 14.7, 11.8, 13.2, 7.9, 15.1, 6.4, 12.8, 10.9, 8.5, 13.6, 9.7, 7.2],
    "lemon":     [6.3, 4.8, 9.7, 12.1, 7.5, 6.9, 8.2, 5.4, 10.8, 4.6, 11.3, 7.7, 9.1, 6.8, 5.2, 8.9, 10.4],
    "vanilla":   [11.8, 9.2, 7.4, 5.8, 13.6, 8.7, 10.3, 12.1, 6.9, 9.8, 8.5, 11.4, 7.2, 10.6, 9.1, 7.8, 8.3],
    "apple":     [4.2, 7.8, 5.6, 8.9, 6.3, 9.4, 7.1, 5.8, 8.7, 6.5, 9.2, 5.4, 7.6, 8.1, 6.8, 9.3, 5.9],
    "mint":      [9.4, 6.2, 11.8, 7.3, 8.9, 5.7, 12.4, 8.6, 7.1, 10.2, 6.8, 9.5, 11.1, 7.4, 8.2, 6.9, 10.7],
    "chocolate": [13.7, 11.4, 9.8, 8.2, 15.3, 10.6, 12.9, 9.5, 11.7, 8.8, 14.2, 10.1, 9.3, 12.6, 8.4, 11.9, 10.8],
}


def sanitize_environmentals_inplace(d: dict) -> None:
    """Clamp env sensors to plausible ranges; replace NaN/out-of-range with median."""
    for k in ENVIRONMENTAL_SENSORS:
        if k not in d:
            continue
        low, high = ENV_SANITY_RANGES[k]
        med = ENV_SANITY_MEDIANS[k]
        try:
            v = float(d[k])
        except (TypeError, ValueError):
            v = med
        if k == "H2S":
            v = max(0.0, v)
        if (v < low) or (v > high) or np.isnan(v):
            v = med
        d[k] = float(v)


def transform_sensor_values(values: List[float]) -> List[float]:
    """ADC → resistance for R1–R17 only. Non-positive counts map to 0."""
    transformed: List[float] = []
    for i, value in enumerate(values[:17]):
        if value == 0.0:
            transformed.append(0.0)
            continue
        try:
            r = (RLOW * COEF * VCC * EG / (VREF * value)) - RLOW
            transformed.append(max(0.0, r))
        except ZeroDivisionError:
            print(f"Warning: div-by-zero for R{i + 1}")
            transformed.append(0.0)
        except Exception as e:
            print(f"Warning: transform error R{i + 1}: {e}")
            transformed.append(0.0)
    return transformed


class ENoseSensor:
    """Dual-serial sensor handler with threaded recording and offline simulation."""

    def __init__(
        self,
        port: Optional[Tuple[str, str]] = None,
        baud_rate_enose: int = DEFAULT_BAUD_RATE,
        baud_rate_UART: int = DEFAULT_BAUD_RATE,
        offline_mode: bool = False,
    ) -> None:
        self.port = port
        self.baud_rate_enose = baud_rate_enose
        self.baud_rate_UART = baud_rate_UART
        self.offline_mode = offline_mode
        self.serial_conn_enose = None
        self.serial_conn_UART = None
        self._is_recording = False
        self._recording_ready = False
        self._lock = threading.Lock()
        self._data_buffer: queue.Queue = queue.Queue(maxsize=200)
        self._recording_thread: Optional[threading.Thread] = None
        self._current_smell_name = "air"
        self._base_environmental = dict(ENV_DEFAULTS)

    # ── Offline simulation ─────────────────────────────────────────────────
    def generate_realistic_sensor_data(self, smell_name: str = "air") -> Dict[str, float]:
        """22-feature simulated sample with Gaussian noise around a reference profile."""
        base_profile = _SMELL_PROFILES.get(
            smell_name.lower(),
            [random.uniform(0.5, 20.0) for _ in range(17)],
        )
        resistance_values = [max(0.0, v + random.gauss(0, v * 0.1)) for v in base_profile]

        environmental_values: List[float] = []
        for sensor in ENVIRONMENTAL_SENSORS:
            base = self._base_environmental[sensor]
            if sensor == "T":
                v = max(15.0, min(30.0, base + random.gauss(0, 1.5)))
            elif sensor == "H":
                v = max(20.0, min(80.0, base + random.gauss(0, 8.0)))
            elif sensor == "CO2":
                v = max(250.0, min(1000.0, base + random.gauss(0, 50.0)))
            elif sensor == "H2S":
                v = max(0.0, min(5.0, base + random.gauss(0, 0.3)))
            else:  # CH2O
                v = max(0.0, min(30.0, base + random.gauss(0, 2.0)))
            environmental_values.append(v)

        result = dict(zip(ALL_SENSORS, resistance_values + environmental_values))
        sanitize_environmentals_inplace(result)
        return result

    def set_simulation_smell(self, smell_name: str) -> None:
        self._current_smell_name = smell_name.lower()
        print(f"Simulation smell set to '{smell_name}'")

    # ── Serial port management ────────────────────────────────────────────
    def list_available_ports(self):
        if self.offline_mode:
            print("Offline mode: skipping serial port detection")
            return []
        ports = serial.tools.list_ports.comports()
        print("\nAvailable serial ports:")
        if not ports:
            print("  (none)")
            return []
        for i, port in enumerate(ports):
            print(f"  [{i + 1}] {port.device} - {port.description}")
        return ports

    def select_port(self) -> Tuple[Optional[str], Optional[str]]:
        """Interactively pick e-nose + UART ports. Returns (enose_dev, uart_dev)."""
        if self.offline_mode:
            return ("OFFLINE_ENOSE", "OFFLINE_UART")

        ports = self.list_available_ports()
        if not ports:
            return None, None
        devices = [p.device for p in ports]

        while True:
            try:
                sel_enose = input(f"Select e-nose port (1-{len(ports)} or device): ").strip()
                sel_uart = input(f"Select UART port (1-{len(ports)} or device): ").strip()
                if not sel_enose or not sel_uart:
                    continue
                try:
                    ei = int(sel_enose) - 1
                    ui = int(sel_uart) - 1
                    if 0 <= ei < len(devices) and 0 <= ui < len(devices):
                        return devices[ei], devices[ui]
                except ValueError:
                    if sel_enose in devices and sel_uart in devices:
                        return sel_enose, sel_uart
                print("Invalid selection.")
            except KeyboardInterrupt:
                print("\nCancelled.")
                return None, None

    def connect(self) -> bool:
        """Open both serial connections (or simulate)."""
        if self.offline_mode:
            self.serial_conn_enose = "SIMULATED_ENOSE"
            self.serial_conn_UART = "SIMULATED_UART"
            print("Offline: simulated e-nose + UART connected")
            return True

        if not self.port:
            self.port = self.select_port()
            if not self.port or len(self.port) != 2 or not self.port[0] or not self.port[1]:
                return False

        try:
            with self._lock:
                if (self.serial_conn_enose and self.serial_conn_UART
                        and getattr(self.serial_conn_enose, "is_open", False)
                        and getattr(self.serial_conn_UART, "is_open", False)):
                    print(f"Already connected to {self.port[0]} and {self.port[1]}")
                    return True

                try:
                    self.serial_conn_enose = serial.Serial(self.port[0], self.baud_rate_enose, timeout=1)
                except Exception as e:
                    print(f"E-nose connect failed on {self.port[0]}: {e}")
                    return False

                try:
                    self.serial_conn_UART = serial.Serial(self.port[1], self.baud_rate_UART, timeout=1)
                except Exception as e:
                    print(f"UART connect failed on {self.port[1]}: {e}")
                    self.serial_conn_enose.close()
                    return False

                time.sleep(1)
                if self.serial_conn_enose.in_waiting:
                    self.serial_conn_enose.read(self.serial_conn_enose.in_waiting)
                if self.serial_conn_UART.in_waiting:
                    self.serial_conn_UART.read(self.serial_conn_UART.in_waiting)

                print(f"Connected: e-nose @ {self.port[0]} ({self.baud_rate_enose}), "
                      f"UART @ {self.port[1]} ({self.baud_rate_UART})")
                return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self) -> None:
        if self.offline_mode:
            self.serial_conn_enose = None
            self.serial_conn_UART = None
            return

        with self._lock:
            for name, conn in (("e-nose", self.serial_conn_enose), ("UART", self.serial_conn_UART)):
                if conn and getattr(conn, "is_open", False):
                    try:
                        conn.close()
                        print(f"Disconnected {name}")
                    except Exception as e:
                        print(f"{name} disconnect error: {e}")
            self.serial_conn_enose = None
            self.serial_conn_UART = None

    # ── Reading ────────────────────────────────────────────────────────────
    def _parse_and_transform_line(self, line: str, line_uart: str) -> Optional[Dict[str, float]]:
        """Parse one e-nose line + one UART line → 22-feature dict."""
        try:
            raw_values = [float(x) for x in line.strip().split()]
            if len(raw_values) < 17:
                print(f"Insufficient e-nose values: {len(raw_values)}/17")
                return None

            raw_uart = [float(str(i).strip()) for i in line_uart.strip().split("\t")]
            if len(raw_uart) < 5:
                print(f"Insufficient UART values: {len(raw_uart)}/5")
                return None

            transformed = transform_sensor_values(raw_values)
            all_values = transformed + raw_uart[:5]
            result = dict(zip(ALL_SENSORS, all_values))
            sanitize_environmentals_inplace(result)
            return result
        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def read_single_measurement(self) -> Optional[Dict[str, float]]:
        """One 22-feature sample from hardware, or simulated."""
        if self.offline_mode:
            return self.generate_realistic_sensor_data(self._current_smell_name)

        with self._lock:
            ce, cu = self.serial_conn_enose, self.serial_conn_UART
            if not (ce and cu and getattr(ce, "is_open", False) and getattr(cu, "is_open", False)):
                print("Serial connections not available")
                return None

            try:
                if ce.in_waiting == 0:
                    time.sleep(0.1)
                    if ce.in_waiting == 0:
                        return None
                if cu.in_waiting == 0:
                    time.sleep(0.1)
                    if cu.in_waiting == 0:
                        return None

                line_e = ce.readline().decode("utf-8")
                line_u = cu.readline().decode("utf-8")
                if not line_e.strip() or not line_u.strip():
                    return None
                return self._parse_and_transform_line(line_e, line_u)
            except Exception as e:
                print(f"Read error: {e}")
                return None

    # ── Threaded recording ─────────────────────────────────────────────────
    def _recording_worker(self, duration_seconds: int, target_samples: int) -> None:
        mode = "SIMULATED" if self.offline_mode else "HARDWARE"
        print(f"\n{mode} recording: {target_samples} samples / {duration_seconds}s "
              f"({target_samples / duration_seconds:.2f} Hz)")
        print("-" * 60)

        while not self._data_buffer.empty():
            try:
                self._data_buffer.get_nowait()
            except queue.Empty:
                break

        self._is_recording = True
        start = time.time()
        end = start + duration_seconds
        interval = duration_seconds / target_samples
        next_at = start
        count = 0
        last_print = start

        while time.time() < end and self._is_recording and count < target_samples:
            now = time.time()
            if now >= next_at:
                sample = self.read_single_measurement()
                if sample and len(sample) == 22:
                    try:
                        self._data_buffer.put_nowait(sample)
                    except queue.Full:
                        try:
                            self._data_buffer.get_nowait()
                            self._data_buffer.put_nowait(sample)
                        except queue.Empty:
                            pass
                    count += 1
                    next_at += interval

            if now - last_print >= 1.0:
                pct = min(100, round(count / target_samples * 100))
                remaining = max(0, round(duration_seconds - (now - start)))
                print(f"Recording: {pct:3d}% | {count:3d}/{target_samples} | {remaining:2d}s left",
                      end="\r")
                last_print = now

            time.sleep(0.05 if self.offline_mode else 0.01)

        self._is_recording = False
        actual = time.time() - start
        print(f"\n{mode} done: {count} samples in {actual:.1f}s ({count / actual:.2f} Hz)")
        if count < target_samples:
            print(f"Warning: only {count}/{target_samples} collected")

    def prepare_recording(self) -> bool:
        if self.offline_mode:
            self._recording_ready = True
            print("Simulated recording prepared")
            return True

        with self._lock:
            ce, cu = self.serial_conn_enose, self.serial_conn_UART
            if not (ce and getattr(ce, "is_open", False)):
                print("E-nose not connected")
                return False
            if not (cu and getattr(cu, "is_open", False)):
                print("UART not connected")
                return False
            if self._is_recording:
                print("Recording already running")
                return False

        self._recording_ready = True
        print("Recording prepared")
        return True

    def start_recording_after_confirmation(
        self,
        duration_seconds: int = DEFAULT_RECORDING_TIME,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
    ) -> bool:
        if not self._recording_ready:
            print("Recording not prepared. Call prepare_recording() first.")
            return False
        self._recording_ready = False
        self._recording_thread = threading.Thread(
            target=self._recording_worker,
            args=(duration_seconds, target_samples),
            daemon=True,
        )
        self._recording_thread.start()
        return True

    def stop_recording(self) -> List[Dict[str, float]]:
        """Signal stop, drain buffer, return collected 22-feature samples."""
        if self._is_recording:
            self._is_recording = False
            if self._recording_thread and self._recording_thread.is_alive():
                self._recording_thread.join(timeout=3.0)

        collected: List[Dict[str, float]] = []
        while not self._data_buffer.empty():
            try:
                sample = self._data_buffer.get_nowait()
                if len(sample) == 22:
                    collected.append(sample)
            except queue.Empty:
                break

        mode = "simulated" if self.offline_mode else "hardware"
        print(f"Collected {len(collected)} valid 22-feature {mode} samples")
        return collected

    @staticmethod
    def get_average_reading(data: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Mean across N samples → one 22-feature dict."""
        if not data:
            return None
        try:
            df = pd.DataFrame(data)
            if len(df.columns) != 22:
                print(f"Warning: expected 22 features, got {len(df.columns)}")
            return df.mean().to_dict()
        except Exception as e:
            print(f"Average error: {e}")
            return None
