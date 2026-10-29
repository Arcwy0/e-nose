"""Critical heater contracts. All commands target fake boards only."""
import struct
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from enose.client.heater import HeaterController, HeaterPolicy
from enose.client.lifecycle import sensor_lifecycle
from enose.client.pcb import PCBSensor
from enose.client import pcb_protocol as protocol


class Board:
    def __init__(self):
        self.serial_conn_enose = SimpleNamespace(is_open=True)
        self.commands = []
        self._info = {}
        self._io_error = None
        self._received_mono = time.monotonic()
        self.stale_after = 1
        self.status = dict(flags=0x10, active_temp_sensor=0, temp0_c=25, temp1_c=25,
                           temp_setpoint_c=300, fixed_duty_enabled=False,
                           heater_enabled=False, heater_duty_permille=0)
        self.fail_after_enable = False

    def _request(self, command, payload=b""):
        self.commands.append((command, payload))
        if command == protocol.SET_TEMP_SETPOINT:
            self.status["temp_setpoint_c"] = struct.unpack("<h", payload)[0]
        if command == protocol.SET_HEATER_ENABLE:
            self.status["heater_enabled"] = payload == b"\x01"

    def refresh_status(self):
        if self.fail_after_enable and self.status["heater_enabled"]:
            self.fail_after_enable = False
            raise RuntimeError("Confirmation lost")
        return dict(self.status)


def test_enable_confirmation_failure_cannot_reuse_verified_off_state():
    board = Board()
    heater = HeaterController(board)
    heater.configure(HeaterPolicy(300, 350, shutdown_seconds=0.01))
    assert heater.stop()
    board.fail_after_enable = True
    with pytest.raises(RuntimeError, match="Confirmation lost"):
        heater.start()
    assert board.commands[-1] == (protocol.SET_HEATER_ENABLE, b"\x00")
    assert board.status["heater_enabled"] is False


def test_temperature_is_rechecked_before_enable():
    board = Board()
    original = board._request
    def request(command, payload=b""):
        original(command, payload)
        if command == protocol.SET_TEMP_SETPOINT:
            board.status["temp0_c"] = 350
    board._request = request
    heater = HeaterController(board)
    heater.configure(HeaterPolicy(300, 350, shutdown_seconds=0.01))
    with pytest.raises(RuntimeError, match="maximum"):
        heater.start()
    assert (protocol.SET_HEATER_ENABLE, b"\x01") not in board.commands
    assert board.commands[-1] == (protocol.SET_HEATER_ENABLE, b"\x00")


def test_unsolicited_status_cannot_confirm_heater_off():
    sensor = PCBSensor(command_timeout=0.01)
    payload = bytes(49)
    def write(frame):
        _, command, sequence, _ = struct.unpack_from("<BBBH", frame, 2)
        sensor._on_frame(protocol.STATUS, 0, payload)
        sensor._on_frame(protocol.ACK, sequence, bytes([command, 0]))
        return len(frame)
    sensor.serial_conn_enose = SimpleNamespace(write=write)
    with pytest.raises(OSError):
        sensor.refresh_status()


def test_graceful_interrupt_always_disconnects_sensor():
    sensor = SimpleNamespace(disconnect=Mock())
    with sensor_lifecycle(sensor):
        raise KeyboardInterrupt()
    sensor.disconnect.assert_called_once()


def test_zero_duty_without_disable_is_not_verified_off():
    board = Board()
    board._request = Mock()  # Simulate ignored disable while status stays on.
    board.status["heater_enabled"] = True
    heater = HeaterController(board)
    assert not heater.stop(timeout=0.001)
    assert not heater.shutdown_verified
    assert heater.state == "off_unverified"


def test_runtime_limit_trips_and_requests_disable():
    board = Board()
    heater = HeaterController(board)
    heater.configure(HeaterPolicy(300, 350, maximum_seconds=0.01, shutdown_seconds=0.01))
    heater.start()
    deadline = time.monotonic() + 2
    while not heater.shutdown_verified and time.monotonic() < deadline:
        time.sleep(0.01)
    assert heater.shutdown_verified
    assert board.commands[-1] == (protocol.SET_HEATER_ENABLE, b"\x00")
    with pytest.raises(RuntimeError, match="safety trip"):
        heater.start()
