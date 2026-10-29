"""Opt-in PI heater control. Host cleanup is not a firmware safety watchdog."""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass

from . import pcb_protocol as protocol


@dataclass(frozen=True)
class HeaterPolicy:
    target_c: float
    maximum_c: float
    maximum_seconds: float = 1800.0
    shutdown_seconds: float = 240.0

    def __post_init__(self):
        values = (self.target_c, self.maximum_c, self.maximum_seconds, self.shutdown_seconds)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Heater limits must be finite")
        if not 0 < self.target_c < self.maximum_c <= 32767:
            raise ValueError("Require 0 < approved target < approved maximum temperature (deg C)")
        if int(self.target_c) != self.target_c:
            raise ValueError("Firmware setpoint must be an integer number of deg C")
        if min(self.maximum_seconds, self.shutdown_seconds) <= 0:
            raise ValueError("Heater runtime and shutdown deadlines must be positive seconds")


class HeaterController:
    """Monitor locally; never set gains, fixed duty, calibration, or flash."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.policy = None
        self.state = "not_started"
        self.reason = None
        self.shutdown_verified = False
        self._lock = threading.RLock()
        self._monitor_stop = threading.Event()
        self._monitor = None
        self._started = None
        self._locked_out = False

    def configure(self, policy):
        if not isinstance(policy, HeaterPolicy):
            raise ValueError("An explicit HeaterPolicy is required")
        if self.state in {"heating", "external", "cooling"}:
            raise ValueError("Stop heating before changing limits")
        self.policy = policy

    def observe_external(self):
        """Monitor existing supplier-configured PI operation without enabling it."""
        if self.policy is None:
            raise ValueError("An explicit monitoring policy is required")
        status = self.sensor.refresh_status()
        if status.get("heater_enabled") or status.get("heater_duty_permille", 0) > 0:
            self._started = time.monotonic()
            self.state = "external"
            self.shutdown_verified = False
            self._monitor_stop.clear()
            self._monitor = threading.Thread(target=self._watch, daemon=True, name="enose-heater-monitor")
            self._monitor.start()

    def start(self):
        with self._lock:
            if self.policy is None:
                raise ValueError("Provide approved heater target and maximum temperature first")
            if self._locked_out:
                raise RuntimeError("Heater safety trip: inspect hardware and reconnect before enabling")
            if self.state == "heating":
                return
            if self.state == "cooling":
                raise RuntimeError("Wait for shutdown to finish before re-enabling heating")
            # Invalidate BEFORE writing anything: an ACK/status failure after
            # enable must never reuse a previous verified-off result.
            self.shutdown_verified = False
            try:
                status = self.sensor.refresh_status()
                self._validate_status(status)
                self.sensor._request(protocol.SET_TEMP_SETPOINT, struct.pack("<h", int(self.policy.target_c)))
                status = self.sensor.refresh_status()
                self._validate_status(status)
                if status.get("temp_setpoint_c") != int(self.policy.target_c):
                    raise RuntimeError("Firmware did not confirm the requested heater setpoint")
                self.sensor._request(protocol.SET_HEATER_ENABLE, b"\x01")
                status = self.sensor.refresh_status()
                self._validate_status(status)
                if status.get("heater_enabled") is not True:
                    raise RuntimeError("Firmware did not confirm heater enable")
                self.sensor._info["temp_setpoint_c"] = int(self.policy.target_c)
                self._started = time.monotonic()
                self.state, self.reason = "heating", None
                self.shutdown_verified = False
                self._monitor_stop.clear()
                self._monitor = threading.Thread(target=self._watch, daemon=True, name="enose-heater-monitor")
                self._monitor.start()
            except BaseException:
                self.stop()
                raise

    def _temperature(self, status):
        active = status.get("active_temp_sensor")
        if active not in (0, 1):
            return None
        value = status.get(f"temp{active}_c")
        return float(value) if value is not None and math.isfinite(float(value)) and float(value) > 0 else None

    def _validate_status(self, status):
        if status["flags"] & protocol.FAULT_MASK or not status["flags"] & 0x10:
            raise RuntimeError("Heating requires fault-free, temperature-calibrated firmware")
        if status.get("fixed_duty_enabled") is not False:
            raise RuntimeError("Verified normal PI mode is required; fixed-duty mode is unsupported")
        temperature = self._temperature(status)
        if temperature is None or temperature >= self.policy.maximum_c:
            raise RuntimeError("Active temperature feedback is missing or above the approved maximum")

    def _watch(self):
        while not self._monitor_stop.wait(0.25):
            if self.state not in {"heating", "external"}:
                return
            reason = None
            try:
                if self.sensor._io_error:
                    raise RuntimeError("USB communication lost")
                if self._started is not None and time.monotonic() - self._started > self.policy.maximum_seconds:
                    raise RuntimeError("maximum heater session duration reached")
                if self.sensor._received_mono is None or time.monotonic() - self.sensor._received_mono > self.sensor.stale_after:
                    raise RuntimeError("measurement stream is stale")
                status = self.sensor.refresh_status()
                temperature = self._temperature(status)
                if status["flags"] & protocol.FAULT_MASK:
                    raise RuntimeError("firmware heater/temperature/calibration fault")
                if not status["flags"] & 0x10:
                    raise RuntimeError("temperature calibration is not valid")
                if temperature is None or temperature >= self.policy.maximum_c:
                    raise RuntimeError("missing temperature feedback or overtemperature")
                if status.get("fixed_duty_enabled") is not False:
                    raise RuntimeError("unexpected fixed-duty mode")
            except (OSError, RuntimeError, ValueError) as exc:
                reason = str(exc)
            if reason:
                self._locked_out = True
                self.reason = reason
                print(f"[heater] SAFETY TRIP: {reason}; requesting controlled shutdown")
                self.stop()
                return

    def stop(self, timeout=None):
        """Request disable and verify both disabled flag AND zero actual duty.

        Firmware may ramp down. A timeout/USB error is NOT proof of heater-off.
        """
        self._monitor_stop.set()
        deadline = time.monotonic() + (timeout if timeout is not None else (
            self.policy.shutdown_seconds if self.policy else 240.0))
        acquired = self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            print("[heater] WARNING: another shutdown attempt is still running; OFF is not confirmed here")
            return False
        try:
            self.shutdown_verified = False
            connection = self.sensor.serial_conn_enose
            if not connection or not getattr(connection, "is_open", False):
                self.state = "off_unverified"
                return False
            self.state = "cooling"
            try:
                self.sensor._request(protocol.SET_HEATER_ENABLE, b"\x00")
                last_notice = 0.0
                while time.monotonic() < deadline:
                    status = self.sensor.refresh_status()
                    if status.get("heater_enabled") is False and status.get("heater_duty_permille") == 0:
                        self.state, self.shutdown_verified = "off_verified", True
                        print("[heater] Power-off verified (disabled, duty=0). Sensor may still be hot.")
                        return True
                    now = time.monotonic()
                    if now - last_notice >= 5:
                        print(f"[heater] Cooling: enabled={status.get('heater_enabled')}, "
                              f"duty={status.get('heater_duty_permille')} permille; awaiting zero")
                        last_notice = now
                    time.sleep(min(0.2, max(0, deadline - now)))
                self.reason = "controlled cooldown did not reach verified zero duty before deadline"
            except (OSError, RuntimeError, ValueError) as exc:
                self.reason = str(exc)
            self.state = "off_unverified"
            print(f"[heater] WARNING: OFF NOT VERIFIED: {self.reason}. "
                  "Check the board and disconnect heater power using the approved procedure.")
            return False
        finally:
            self._lock.release()

    def diagnostics(self):
        return {"state": self.state, "reason": self.reason, "off_verified": self.shutdown_verified,
                "target_c": self.policy.target_c if self.policy else None,
                "maximum_c": self.policy.maximum_c if self.policy else None,
                "host_loss_watchdog": False}
