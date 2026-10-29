"""Normalize resistance-mode desktop recordings without inventing labels.

Time-only exports use a fixed 1970-01-01 anchor unless a recording date is
provided. The anchor preserves elapsed time; it is NOT a measured date.
"""

from __future__ import annotations

import re

import pandas as pd

from enose.client.pcb_protocol import canonical_resistances

PCB_COLUMNS = [f"rs_ch{index}_ohm" for index in range(17)]


def normalize_pcb_recording(frame, resistance_unit="ohm", recording_date=None):
    if any(re.fullmatch(r"raw_ch\d+", str(name)) for name in frame.columns):
        raise ValueError("RAW_LPF CSV contains ADC codes, not resistances; export Rs mode instead")
    missing = [name for name in PCB_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"PCB CSV is missing resistance channels: {missing}")
    if any(f"R{index}" in frame.columns for index in range(1, 18)):
        raise ValueError("CSV contains both PCB and canonical resistance columns; refusing ambiguous conversion")
    if "timestamp_iso" not in frame.columns:
        raise ValueError("PCB CSV requires timestamp_iso")
    work = frame.copy()
    work["_source_row"] = range(len(work))
    measurements = []
    valid_indices = []
    for position, row in frame.iterrows():
        try:
            sample = canonical_resistances([row[name] for name in PCB_COLUMNS], resistance_unit)
            faults = [float(row.get(name, 0)) for name in ("heater1_fault", "heater2_fault")]
            if any(value != 0 for value in faults):
                continue
        except (ValueError, TypeError):
            continue
        valid_indices.append(position)
        measurements.append(sample)
    if not measurements:
        raise ValueError("PCB CSV contains no valid fault-free resistance measurements")
    # Compute time on ALL source rows before dropping faulty rows, so the
    # surviving samples retain real gaps and midnight crossings.
    timestamps = []
    day_offset = 0
    previous = None
    time_only = frame["timestamp_iso"].astype(str).str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?").all()
    if time_only:
        anchor = pd.Timestamp(recording_date or "1970-01-01").normalize()
        for token in frame["timestamp_iso"].astype(str):
            value = pd.to_timedelta(token)
            if previous is not None and value < previous:
                if (previous - value).total_seconds() < 12 * 3600:
                    raise ValueError("PCB CSV time moves backwards within a recording")
                day_offset += 1
            timestamps.append(anchor + pd.Timedelta(days=day_offset) + value)
            previous = value
    else:
        timestamps = list(pd.to_datetime(frame["timestamp_iso"], errors="raise", format="mixed"))
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("PCB CSV timestamps are not chronological")
    work["Timestamp"] = timestamps
    work = work.loc[valid_indices].copy()
    work = work.drop(columns=PCB_COLUMNS)
    for name in measurements[0]:
        work[name] = [sample[name] for sample in measurements]
    work = work.rename(columns={"temp_actual_c": "pcb_temp_actual_c", "duty_permille": "pcb_duty_permille"})
    work["resistance_unit"] = resistance_unit
    report = {
        "parser": "pcb_resistance_csv", "recovered_rows": 0,
        "source_rows": len(frame),
        "dropped_rows": len(frame) - len(work), "environment_reliable": False,
        "resistance_unit": resistance_unit, "input_resistance_unit": "ohm",
        "timestamp_date_known": not time_only or recording_date is not None,
        "warning": "CH0..CH16 mapped to R1..R17 by index; physical equivalence to the old array is unverified. "
                   "PCB temperature is not ambient T. Smell labels are never inferred.",
    }
    return work.reset_index(drop=True), report
