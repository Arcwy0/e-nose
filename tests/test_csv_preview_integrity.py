"""Data-integrity checks for operator annotations and shared CSV selection."""

import json
import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from enose.server.schemas import CSVLearningData
from enose.utils.training_preview import prepare_csv_training, csv_training_preview


def _recording(count=601):
    frame = pd.DataFrame({
        "timestamp_iso": pd.date_range("2026-10-28 18:00:00", periods=count, freq="2s").strftime("%H:%M:%S"),
        "temp_actual_c": 285,
    })
    for channel in range(17):
        frame[f"rs_ch{channel}_ohm"] = [1_000_000 + channel if index < 300 else 500_000 + channel for index in range(count)]
    return frame


def test_preview_and_training_share_windows_labels_dates_units_and_identity():
    frame = _recording()
    data = CSVLearningData(
        csv_data=frame.to_csv(index=False), training_profile="plateau",
        resistance_unit="mohm", recording_date="2026-10-28",
        session_id="board8-day1", recording_id="verified-experiment",
        annotations=[{"start_row": 0, "end_row": 300, "label": "air"},
                     {"start_row": 300, "end_row": 601, "label": "acetone"}],
    )
    source, training, target, report = prepare_csv_training(data)
    preview = csv_training_preview(source, training, target, report)
    source_again, training_again, target_again, _ = prepare_csv_training(data)
    pd.testing.assert_frame_equal(training, training_again)
    assert target == target_again == "Gas name"
    assert preview["training_ready"]
    assert sum(preview["class_counts"].values()) == len(training)
    assert set(training["resistance_unit"]) == {"mohm"}
    assert set(training["session_id"]) == {"board8-day1"}
    assert set(training["recording_id"]) == {"verified-experiment"}
    assert training["t_start"].str.startswith("2026-10-28").all()
    assert source.loc[0, "R1"] == 1.0
    assert "T" not in source and "pcb_temp_actual_c" in source
    json.dumps(preview, allow_nan=False)


def test_annotations_reference_source_rows_after_fault_filtering_and_never_invent_labels():
    frame = _recording(4)
    frame.loc[1, "rs_ch0_ohm"] = -1
    csv_data = frame.to_csv(index=False)
    with pytest.raises(ValueError, match="verified labels"):
        prepare_csv_training(CSVLearningData(csv_data=csv_data))
    source_preview, selected_preview, target_preview, report_preview = prepare_csv_training(
        CSVLearningData(csv_data=csv_data, training_profile="plateau"), allow_unlabelled=True,
    )
    preview = csv_training_preview(source_preview, selected_preview, target_preview, report_preview)
    assert not preview["training_ready"] and preview["class_counts"] == {}
    assert len(preview["plot"]["points"]) == 3 and selected_preview.empty
    assert source_preview["smell_label"].isna().all()
    assert "verified labels" in preview["message"]
    assert report_preview["input_quality"]["label_counts"] == {}
    data = CSVLearningData(csv_data=csv_data, annotations=[
        {"start_row": 0, "end_row": 1, "label": "air"},
        {"start_row": 2, "end_row": 4, "label": "acetone"},
    ])
    source, _, _, report = prepare_csv_training(data)
    assert source["_source_row"].tolist() == [0, 2, 3]
    assert source["smell_label"].tolist() == ["air", "acetone", "acetone"]
    assert report["csv_parser"]["dropped_rows"] == 1
    assert report["csv_parser"]["timestamp_date_known"] is False
    overlapping = data.model_copy(update={"annotations": [
        *data.annotations,
        data.annotations[0].model_copy(update={"end_row": 3}),
    ]})
    with pytest.raises(ValueError, match="overlap"):
        prepare_csv_training(overlapping)


def test_preview_reports_empty_extraction_without_fitting_or_cross_session_baselines():
    frame = _recording(601)
    frame["smell_label"] = ["air"] * 300 + ["acetone"] * 301
    frame["session_id"] = ["first"] * 300 + ["second"] * 301
    source, selected, target, report = prepare_csv_training(CSVLearningData(
        csv_data=frame.to_csv(index=False), training_profile="plateau",
    ))
    preview = csv_training_preview(source, selected, target, report)
    assert selected.empty and not preview["training_ready"]
    assert any("no preceding clean-air" in reason
               for exposure in report["exposure_details"] for reason in exposure["rejection_reasons"])


def test_csv_candidate_staging_never_swaps_active_model_or_writes_active_history(monkeypatch):
    from enose.classifier import registry
    from enose.server.routes import training

    original = SimpleNamespace(is_fitted=False, classes_=np.array([]))
    candidate = SimpleNamespace(
        classes_=np.array(["air", "acetone"]), config=SimpleNamespace(baseline_mode="none", snv=False),
        get_model_info=lambda: {"validation": {"strategy": "not_estimable", "estimable": False}},
    )
    calls = []

    def retrain(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["persist"] is False
        return True, None, candidate

    def forbidden(*args, **kwargs):
        raise AssertionError("staging must not replace or reload the active classifier")

    monkeypatch.setattr(training.state, "require_classifier", lambda: original)
    monkeypatch.setattr(training.state, "set_classifier", forbidden)
    monkeypatch.setattr(training, "reload_smell_classifier", forbidden)
    monkeypatch.setattr(training, "_retrain_with_units", retrain)
    monkeypatch.setattr(registry, "create_candidate", lambda fitted, root: "a" * 32)
    frame = _recording(4)
    frame["smell_label"] = ["air", "air", "acetone", "acetone"]
    response = asyncio.run(training.learn_from_csv(CSVLearningData(
        csv_data=frame.to_csv(index=False), merge_history=False, stage_candidate=True,
    )))
    assert response["staged"] and response["candidate_id"] == "a" * 32
    assert response["current_accuracy"] is None and len(calls) == 1
    assert candidate.acquisition_provenance_["verified"] is False
