"""Resistance-unit provenance guards; never infer units from signal magnitude."""

from __future__ import annotations

from typing import Optional


RESISTANCE_UNITS = frozenset({"ohm", "kohm", "mohm"})


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
