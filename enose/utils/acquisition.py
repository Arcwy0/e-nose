"""Resistance-unit provenance guards; never infer units from signal magnitude."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Optional


RESISTANCE_UNITS = frozenset({"ohm", "kohm", "mohm"})

# Measurement values, counters and timestamps deliberately do not participate.
_IDENTITY_CONFIG_KEYS = frozenset({
    "rlow", "channel_count", "channel_map", "tia_map", "rf_table_ohm",
    "analog_voltages", "sdadc_gain_cal", "channel_calib_coeff", "gas_lpf_coeff_q12",
    "active_temp_sensor", "stream_value_mode", "heater_mode", "temperature_setpoint_c",
    "target_temperature_c", "temperature_sensor", "gain_count", "gain_indices",
    "temp_setpoint_c", "heater_max_temp_c",
})


def acquisition_identity(acquisition):
    """Stable declared acquisition configuration, not proof of calibration."""
    if not acquisition:
        return None
    config = acquisition.get("config") or {}
    return {
        "backend": acquisition.get("backend"),
        "device_id": acquisition.get("device_id"),
        "firmware_version": acquisition.get("firmware_version"),
        "resistance_unit": normalize_resistance_unit(acquisition.get("resistance_unit")),
        "config": {key: config[key] for key in sorted(_IDENTITY_CONFIG_KEYS) if key in config},
    }


def acquisition_fingerprint(acquisition):
    identity = acquisition_identity(acquisition)
    if identity is None:
        return None
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def acquisition_provenance(acquisitions):
    """Summarize declarations; a replay filename is never a physical board ID."""
    metadata = list(acquisitions)
    fingerprints = sorted({value for item in metadata if (value := acquisition_fingerprint(item))})
    verified = bool(metadata) and all(
        item and item.get("backend") == "pcb"
        and item.get("device_id") not in (None, "", "unknown")
        and normalize_resistance_unit(item.get("resistance_unit")) is not None
        for item in metadata
    )
    return {
        "fingerprints": fingerprints,
        "verified": verified,
        "warning": None if verified else (
            "Physical acquisition provenance is unknown or incomplete; replay filenames "
            "do not establish board identity or calibration compatibility."
        ),
    }


def bind_live_baseline(classifier, acquisition=None, session_id=None):
    """Bind a just-captured baseline to its acquisition and publisher session."""
    classifier.live_baseline_context_ = {
        "fingerprint": acquisition_fingerprint(acquisition),
        "session_id": session_id,
        "segment_id": (acquisition or {}).get("segment_id"),
    }


def live_baseline_error(classifier, acquisition=None, session_id=None):
    """Block stale relative baselines, retaining historical metadata-free APIs."""
    mode = getattr(getattr(classifier, "config", None), "baseline_mode", "none") or "none"
    if mode == "none":
        return None
    if not bool(getattr(classifier, "live_baseline_captured_", False)):
        return "Capture a clean-air baseline for this live session before classifying."
    context = getattr(classifier, "live_baseline_context_", None)
    fingerprint = acquisition_fingerprint(acquisition)
    if context is None:
        if not acquisition:
            return None  # Historical callers stored only the captured boolean.
    elif (context.get("fingerprint") == fingerprint
          and context.get("session_id") == session_id
          and context.get("segment_id") == (acquisition or {}).get("segment_id")):
        return None
    classifier.live_baseline_captured_ = False
    return (
        "Acquisition identity, configuration, or session changed; capture a new "
        "clean-air baseline before classifying or learning."
    )


def resistance_sample_error(sample, acquisition=None, *, strict=False):
    """PCB values must be finite positive resistances, never ADC sentinels."""
    if not strict and (acquisition or {}).get("backend") not in {"pcb", "replay"}:
        return None
    for index in range(1, 18):
        name = f"R{index}"
        try:
            value = float(sample[name])
        except (KeyError, TypeError, ValueError):
            return f"Invalid resistance {name}: all 17 finite positive values are required."
        if not math.isfinite(value) or value <= 0:
            return f"Invalid resistance {name}: values must be finite and positive."
    return None


def prediction_error_code(message):
    """Machine-readable companion to existing human-readable error strings."""
    text = message.lower()
    if "baseline" in text:
        return "baseline_required"
    if "unit" in text or "scaling" in text:
        return "resistance_unit_mismatch"
    if "resistance" in text or "non-finite" in text:
        return "invalid_resistance"
    if "invalid" in text:
        return "invalid_acquisition"
    return "prediction_unavailable"


def normalize_resistance_unit(value) -> Optional[str]:
    unit = str(value).strip().lower() if value is not None else ""
    return unit if unit in RESISTANCE_UNITS else None


def model_resistance_unit(classifier) -> Optional[str]:
    return normalize_resistance_unit(getattr(classifier, "acquisition_resistance_unit_", None))


def frame_resistance_unit(frame) -> Optional[str]:
    """Read explicit per-row units; absent columns retain legacy unknown units."""
    if "resistance_unit" not in frame.columns:
        return None
    values = frame["resistance_unit"]
    if not hasattr(values, "tolist"):
        raise ValueError("CSV must contain only one resistance_unit column")
    raw = values.tolist()
    units = {normalize_resistance_unit(value) for value in raw}
    if not raw:
        return None
    if len(units) != 1 or None in units:
        raise ValueError("resistance_unit must explicitly identify one of ohm/kohm/mohm on every row")
    return units.pop()


def training_resistance_unit(classifier, incoming_unit, merge_history=True, has_history=False) -> Optional[str]:
    """Resolve units before any training/history mutation or reject ambiguity."""
    incoming = normalize_resistance_unit(incoming_unit)
    current = model_resistance_unit(classifier)
    if not merge_history:
        return incoming
    history_present = bool(getattr(classifier, "is_fitted", False)) or has_history
    if history_present and incoming and not current:
        raise ValueError(
            "Historical resistance units are unverified; refusing to merge PCB/annotated values. "
            "Back up the model and history, then use replacement training (merge_history=false) "
            "with correctly annotated data, or verify and annotate the historical units first."
        )
    if current and not incoming:
        raise ValueError(
            f"The model/history uses {current}, but incoming resistance units are unverified. "
            "Provide matching acquisition metadata or an explicit resistance_unit CSV column; "
            "values are not rescaled automatically."
        )
    if current and incoming and current != incoming:
        raise ValueError(
            f"Resistance unit mismatch: model/history uses {current}; incoming data uses {incoming}. "
            "Select matching acquisition/import units, or back up and use replacement training "
            "(merge_history=false). Values are not rescaled automatically."
        )
    return incoming or current


def prediction_unit_error(classifier, acquisition) -> Optional[str]:
    """Return a diagnostic instead of classifying incompatible acquired values."""
    if not acquisition:
        return None
    if acquisition.get("valid") is False:
        return "Prediction disabled: acquisition marked this measurement invalid."
    incoming = normalize_resistance_unit(acquisition.get("resistance_unit"))
    current = model_resistance_unit(classifier)
    if incoming is None:
        if acquisition.get("backend") in {"pcb", "replay"}:
            return "Prediction disabled: PCB/replay resistance units must explicitly be ohm/kohm/mohm."
        if current and acquisition.get("backend") in {"pcb", "replay", "legacy", "simulation"}:
            return f"Prediction disabled: model uses {current}, but acquisition resistance units are unverified. Select matching verified units."
        return None
    if current is None:
        return (
            f"Prediction disabled: acquisition uses {incoming}, but model resistance units are unverified. "
            "Back up the model/history and use replacement training on correctly annotated data."
        )
    if incoming != current:
        return f"Prediction disabled: model uses {current}, acquisition uses {incoming}. Select matching acquisition units; no automatic scaling is performed."
    return None
