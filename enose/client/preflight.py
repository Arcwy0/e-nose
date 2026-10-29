"""Read-only diagnostics for an already-connected acquisition backend.

Only the new PCB backend receives GET_STATUS. Legacy boards have no INFO or
STATUS protocol. This module never connects, enables heating, or actuates robots.
"""

from __future__ import annotations

from importlib import metadata
import math
import platform
import re
import time

from enose import __version__
from enose.config import ALL_SENSORS
from enose.utils.acquisition import normalize_resistance_unit, resistance_sample_error

from .pcb_protocol import FAULT_MASK


def _message(value):
    # Request exceptions can contain authenticated URLs. Never export them.
    return re.sub(r"https?://\S+", "<server-url>", str(value))[:500]


def _json_value(value, depth=0):
    """Bound optional diagnostics and replace non-finite values with null."""
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _message(value)[:160]
    if isinstance(value, dict):
        return {str(key)[:80]: _json_value(item, depth + 1) for key, item in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, depth + 1) for item in value[:32]]
    return None


def _versions():
    versions = {"enose": __version__, "python": platform.python_version()}
    for distribution in ("pyserial", "requests", "numpy", "pandas", "scikit-learn",
                         "fastapi", "torch", "transformers", "opencv-python", "opencv-python-headless"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _metadata(sensor):
    getter = getattr(sensor, "get_acquisition_metadata", None)
    if not callable(getter):
        return {}
    result = getter()
    return result if isinstance(result, dict) else {}


def _server_report(server_api, acquisition):
    if server_api is None:
        return {"checked": False}
    result = {"checked": True}
    for method_name, destination in (("test_connection", "build"), ("get_model_info", "model")):
        method = getattr(server_api, method_name, None)
        if not callable(method):
            result[destination] = {"available": False}
            continue
        try:
            data, error = method()
            if error:
                result[destination] = {"available": False, "error": _message(error)}
            elif isinstance(data, dict):
                if destination == "build":
                    keys = ("version", "build", "build_id", "models", "status")
                else:
                    keys = ("model_loaded", "is_fitted", "classifier_backend", "classes", "resistance_unit", "drift")
                result[destination] = {"available": True, **{key: data[key] for key in keys if key in data}}
            else:
                result[destination] = {"available": False, "error": "Unexpected diagnostic response"}
        except Exception as exc:
            result[destination] = {"available": False, "error": _message(exc)}
    model = result.get("model", {})
    incoming = normalize_resistance_unit(acquisition.get("resistance_unit"))
    trained = normalize_resistance_unit(model.get("resistance_unit"))
    if not model.get("available") or not model.get("is_fitted", model.get("model_loaded", False)):
        compatibility, reason = "unknown", "No fitted model information available"
    elif trained and incoming:
        compatibility = "matching_units" if trained == incoming else "incompatible_units"
        reason = "Matching units do not establish board/calibration compatibility" if trained == incoming else "Acquisition and model resistance units differ"
    else:
        compatibility, reason = "unverified_units", "Acquisition or historical model units are unverified"
    result["model_compatibility"] = {"status": compatibility, "reason": reason}
    drift = model.get("drift") or {}
    mode = drift.get("baseline_mode", "unknown")
    result["baseline_readiness"] = {
        "mode": mode,
        "captured": drift.get("baseline_captured"),
        "ready": True if mode == "none" else (False if drift.get("baseline_captured") is False else None),
        "note": "A captured baseline must also match the active board/configuration and publisher session",
    }
    return _json_value(result)


def collect_preflight(sensor, server_api=None, duration=3.0):
    """Observe valid unique samples without changing backend configuration.

    ``duration`` is the observation time in seconds, not a hard timeout for an
    existing blocking serial read or HTTP wrapper. Delivery Hz is measured on
    this host; it is not a declaration of the board's configured sample rate.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Preflight duration must be finite and positive seconds")
    backend = "simulation" if getattr(sensor, "offline_mode", False) else getattr(sensor, "backend", "legacy")
    acquisition, diagnostic_errors = {}, []
    try:
        acquisition = _metadata(sensor)
    except Exception as exc:
        diagnostic_errors.append(_message(exc))
    status = None
    protocol = {"info": "unsupported_not_probed", "status": "unsupported_not_probed"}
    if backend == "pcb":
        protocol["info"] = "from_connection_handshake_not_requeried"
        refresh = getattr(sensor, "refresh_status", None)
        if callable(refresh):
            try:
                status = refresh()
                protocol["status"] = "read_only_query_succeeded"
            except Exception as exc:
                protocol["status"] = "query_failed"
                diagnostic_errors.append(_message(exc))
        else:
            protocol["status"] = "unavailable"
        # INFO was already obtained during connection. Never issue another
        # handshake or expose unrelated arbitrary sensor attributes.
        info = getattr(sensor, "_info", {}) or {}
        for key in ("device_id", "firmware_version", "resistance_unit"):
            if key not in acquisition and key in info:
                acquisition[key] = info[key]
    elif backend != "legacy":
        protocol = {"info": "not_applicable_not_probed", "status": "not_applicable_not_probed"}
    started = time.monotonic()
    arrivals, seen, latest_sample = [], set(), None
    idle_polls = duplicate_samples = rejected_samples = read_errors = polls_with_error_status = 0
    rejection_reasons = set()
    while time.monotonic() - started < duration:
        try:
            sample = sensor.read_single_measurement()
            acquisition = {**acquisition, **_metadata(sensor)}
        except Exception as exc:
            read_errors += 1
            rejection_reasons.add(_message(exc))
            sample = None
        if sample is None:
            error = getattr(sensor, "last_error", None)
            if error:
                polls_with_error_status += 1
                rejection_reasons.add(_message(error))
            else:
                idle_polls += 1  # Expected when polling faster than acquisition.
        else:
            error = resistance_sample_error(sample, acquisition)
            if acquisition.get("valid") is False:
                error = "Acquisition marks the measurement invalid"
            clean = {}
            try:
                clean = {key: float(sample[key]) for key in ALL_SENSORS if key in sample}
                if not clean or not all(math.isfinite(value) for value in clean.values()):
                    error = "Empty or non-finite sensor measurement"
            except (TypeError, ValueError):
                error = "Invalid sensor measurement format"
            if error:
                rejected_samples += 1
                rejection_reasons.add(error)
            else:
                frame = acquisition.get("frame_index")
                identity = (acquisition.get("device_id"), acquisition.get("segment_id"), frame)
                if frame is not None and identity in seen:
                    duplicate_samples += 1
                else:
                    if frame is not None:
                        seen.add(identity)
                    arrivals.append(time.monotonic())
                    latest_sample = clean
        time.sleep(min(0.01, max(0.0, duration - (time.monotonic() - started))))
    elapsed = time.monotonic() - started
    span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
    age = acquisition.get("sample_age_seconds")
    received = getattr(sensor, "_received_mono", None)
    if isinstance(received, (int, float)):
        age = max(0.0, time.monotonic() - received)
    elif acquisition.get("received_at") is not None:
        try:
            age = max(0.0, time.time() - float(acquisition["received_at"]))
        except (TypeError, ValueError):
            pass
    heater = None
    if backend == "pcb":
        diagnostics = getattr(getattr(sensor, "heater", None), "diagnostics", None)
        if callable(diagnostics):
            heater = diagnostics()
    flags = (status or {}).get("flags", acquisition.get("flags"))
    faults = None if flags is None else {
        "any_fault_or_calibration_mode": bool(int(flags) & FAULT_MASK),
        "heater1_fault": bool(int(flags) & 0x01),
        "heater2_fault": bool(int(flags) & 0x02),
        "calibration_mode": bool(int(flags) & 0x04),
        "temperature0_open": bool(int(flags) & 0x40),
        "temperature1_open": bool(int(flags) & 0x80),
    }
    return _json_value({
        "versions": _versions(), "backend": backend, "protocol": protocol,
        "port": getattr(sensor, "port", None),
        "baud_rate": getattr(sensor, "baud_rate_enose", None),
        "resistance_unit": acquisition.get("resistance_unit", getattr(sensor, "resistance_unit", None)),
        "observed_seconds": round(elapsed, 3), "valid_unique_samples": len(arrivals),
        "delivery_rate_hz": (len(arrivals) - 1) / span if span > 0 else None,
        "sample_age_seconds": age, "idle_polls": idle_polls,
        "duplicate_samples": duplicate_samples, "rejected_samples": rejected_samples,
        "read_errors": read_errors, "polls_with_error_status": polls_with_error_status,
        "rejection_reasons": sorted(rejection_reasons)[:16],
        "diagnostic_errors": diagnostic_errors[:8], "sample": latest_sample,
        "device_id": acquisition.get("device_id"),
        "firmware_version": acquisition.get("firmware_version"),
        "flags": flags, "faults": faults,
        "dropped_frames": acquisition.get("dropped_frames"),
        "rejected_headers": getattr(getattr(sensor, "_parser", None), "rejected_headers", None),
        "status": status, "heater": heater,
        "server": _server_report(server_api, acquisition),
        "limitations": "Observation only; no chemical accuracy, calibration, or physical heater-off validation",
    })
