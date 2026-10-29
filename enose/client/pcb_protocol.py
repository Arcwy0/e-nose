"""Headless ENOSE USB CDC codec (spec v2; frame VERSION remains 0x01).

No serial ports or hardware commands are opened/sent by this module.
Resistance-mode values are already calibrated integer ohms, not legacy ADC.
"""

from __future__ import annotations

import math
import struct
import time

SOF = b"\xa5\x5a"
VERSION = 1
GET_INFO, GET_STATUS, STREAM_START, STREAM_STOP = 1, 2, 3, 4
SET_VALUE_MODE = 0x13
ACK, INFO, STATUS, STREAM = 0x80, 0x81, 0x82, 0x90
RESISTANCE_SCALES = {"ohm": 1.0, "kohm": 1e-3, "mohm": 1e-6}
FAULT_MASK = 0xC7  # Heater faults, calibration mode, temperature open circuits.


def pack_frame(command: int, sequence: int, payload: bytes = b"") -> bytes:
    if len(payload) > 768:
        raise ValueError("PCB command payload exceeds firmware RX limit")
    return struct.pack("<2sBBBH", SOF, VERSION, command, sequence & 255, len(payload)) + payload


class StreamParser:
    """Bounded incremental parser; tolerate split/coalesced/noisy serial reads.

    There is no application CRC in this protocol. Length/version validation
    cannot certify electrical integrity. Discard an incomplete frame after 1 s.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.incomplete_since = None
        self.rejected_headers = 0

    def feed(self, data: bytes) -> list:
        now = time.monotonic()
        if self.incomplete_since is not None and now - self.incomplete_since > 1.0:
            del self.buffer[:1]
            self.incomplete_since = None
        self.buffer.extend(data)
        frames = []
        while self.buffer:
            start = self.buffer.find(SOF)
            if start < 0:
                self.buffer[:] = self.buffer[-1:] if self.buffer[-1:] == SOF[:1] else b""
                break
            del self.buffer[:start]
            if len(self.buffer) < 7:
                break
            version, message, sequence, length = struct.unpack_from("<BBBH", self.buffer, 2)
            valid_length = length <= 768
            if message == STREAM:
                valid_length = length in (79, 80, 81, 82, 85)
            elif message == ACK:
                valid_length = length == 2
            elif message == INFO:
                valid_length = 76 <= length <= 768
            elif message == STATUS:
                valid_length = 35 <= length <= 768
            if version != VERSION or not valid_length:
                del self.buffer[:1]
                self.rejected_headers += 1
                self.incomplete_since = None
                continue
            if len(self.buffer) < length + 7:
                self.incomplete_since = self.incomplete_since or now
                break
            frames.append((message, sequence, bytes(self.buffer[7:7 + length])))
            del self.buffer[:7 + length]
            self.incomplete_since = None
        return frames


def parse_info(payload: bytes) -> dict:
    if len(payload) < 76:
        raise ValueError("PCB INFO payload is too short")
    uid = struct.unpack_from("<3I", payload)
    result = {
        "device_id": "-".join(f"{value:08X}" for value in uid),
        "channel_count": payload[14], "gain_count": payload[15],
        "flags": payload[18], "active_temp_sensor": payload[21],
        "rf_table_ohm": list(struct.unpack_from("<6i", payload, 31)),
        "tia_map": list(payload[59:76]),
    }
    if len(payload) >= 97:
        result["device_name"] = payload[76:97].split(b"\0", 1)[0].decode("utf-8", errors="replace")
    if len(payload) >= 100:
        result["stream_value_mode"] = payload[99]
    if len(payload) >= 120:
        result["analog_voltages"] = list(struct.unpack_from("<5f", payload, 100))
    if len(payload) >= 136:
        result["firmware_version"] = payload[120:136].split(b"\0", 1)[0].decode("ascii", errors="replace")
    if len(payload) >= 138:
        result["gas_lpf_coeff_q12"] = struct.unpack_from("<H", payload, 136)[0]
    if len(payload) >= 160:
        result["sdadc_gain_cal"] = list(struct.unpack_from("<3f", payload, 141))
    if len(payload) >= 228:
        result["channel_calib_coeff"] = list(struct.unpack_from("<17f", payload, 160))
    # Corrupt metadata must not turn into NaN/Infinity in HTTP/NPZ JSON.
    for value in result.values():
        if isinstance(value, list) and any(isinstance(x, float) and not math.isfinite(x) for x in value):
            raise ValueError("PCB INFO contains non-finite calibration values")
    return result


def parse_status(payload: bytes) -> dict:
    if len(payload) < 35:
        raise ValueError("PCB STATUS payload is too short")
    result = {"flags": payload[8], "uptime_ms": struct.unpack_from("<I", payload)[0]}
    if len(payload) >= 44:
        result["stream_value_mode"] = payload[43]
    return result


def parse_measurement(payload: bytes, fallback_mode=None) -> dict:
    if len(payload) not in (79, 80, 81, 82, 85):
        raise ValueError("Unsupported PCB measurement payload length")
    values = struct.unpack_from("<Ihhh17iB", payload)
    mode = payload[81] if len(payload) >= 82 else (payload[79] if len(payload) == 80 else fallback_mode)
    return {
        "frame_index": values[0], "temp0_c": values[1], "temp1_c": values[2],
        "duty_permille": values[3], "channels_ohm": list(values[4:21]),
        "flags": values[21], "stream_value_mode": mode,
    }


def canonical_resistances(channels, unit: str = "ohm") -> dict:
    if unit not in RESISTANCE_SCALES:
        raise ValueError("Resistance unit must be ohm, kohm, or mohm")
    if len(channels) != 17:
        raise ValueError("PCB must provide exactly 17 channels")
    values = [float(value) for value in channels]
    if any(not math.isfinite(value) or value <= 0 or value >= 2147483647 for value in values):
        raise ValueError("PCB resistance is non-positive, non-finite, or at the i32 limit")
    return {f"R{index + 1}": value * RESISTANCE_SCALES[unit] for index, value in enumerate(values)}
