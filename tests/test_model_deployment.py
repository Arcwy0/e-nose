"""Deployment checks operate exclusively on temporary model/history files."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import pandas as pd

pytest.importorskip("imblearn")
from fastapi import HTTPException
from pydantic import ValidationError

from enose.server import state
from enose.server.routes import models


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    root = tmp_path / "candidates"
    identifier = "1" * 32
    directory = root / identifier
    directory.mkdir(parents=True)
    (directory / "model.joblib").write_bytes(b"candidate-model")
    (directory / "history.csv").write_text("candidate-history", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"id": identifier, "external_comparison": None}), encoding="utf-8")
    active_model = tmp_path / "active.joblib"
    active_history = tmp_path / "active.csv"
    active_model.write_bytes(b"original-model")
    active_history.write_text("original-history", encoding="utf-8")
    old = SimpleNamespace(name="old", is_fitted=True)
    monkeypatch.setattr(models, "candidate_root", lambda: str(root))
    monkeypatch.setattr(models, "SMELL_MODEL_PATH", str(active_model))
    monkeypatch.setattr(models, "TRAINING_DATA_PATH", str(active_history))
    monkeypatch.setattr(models, "_load_classifier", lambda path: SimpleNamespace(name=path.read_bytes()))
    monkeypatch.setattr(state, "smell_classifier", old)
    return identifier, active_model, active_history, old


def test_promotion_requires_override_and_rollback_restores_both_files(deployment):
    identifier, model, history, old = deployment
    with pytest.raises(HTTPException) as blocked:
        asyncio.run(models.promote(models.PromoteCandidate(candidate_id=identifier)))
    assert blocked.value.status_code == 409
    assert state.smell_classifier is old
    assert model.read_bytes() == b"original-model"
    response = asyncio.run(models.promote(models.PromoteCandidate(candidate_id=identifier, allow_unevaluated=True)))
    assert response["status"] == "promoted"
    assert model.read_bytes() == b"candidate-model"
    assert history.read_text() == "candidate-history"
    result = asyncio.run(models.rollback(models.RollbackCandidate(snapshot_id=response["rollback_id"])))
    assert result["status"] == "rolled_back"
    assert model.read_bytes() == b"original-model"
    assert history.read_text() == "original-history"
    with pytest.raises(ValidationError):
        models.PromoteCandidate(candidate_id="../active")


def test_failed_in_memory_swap_restores_model_and_history(deployment, monkeypatch):
    identifier, model, history, old = deployment

    def fail(_):
        raise RuntimeError("injected swap failure")

    monkeypatch.setattr(state, "set_classifier", fail)
    with pytest.raises(RuntimeError, match="injected swap failure"):
        asyncio.run(models.promote(models.PromoteCandidate(candidate_id=identifier, allow_unevaluated=True)))
    assert state.smell_classifier is old
    assert model.read_bytes() == b"original-model"
    assert history.read_text() == "original-history"


def test_external_evaluation_rejects_overlap_and_never_changes_active_state(deployment, monkeypatch):
    identifier, model, history, old = deployment
    candidate_history = models._snapshot_directory(identifier) / "history.csv"

    class Predictor:
        classes_ = ["air", "acetone"]
        acquisition_resistance_unit_ = "ohm"
        _original_baseline_ = {}

        def predict(self, frame):
            return ["air" if value > 50 else "acetone" for value in frame["R1"]]

    monkeypatch.setattr(models, "_load_classifier", lambda _: Predictor())
    training = pd.DataFrame([{**{f"R{i}": 10.0 for i in range(1, 18)}, "smell_label": "acetone", "resistance_unit": "ohm"}])
    training.to_csv(history, index=False)
    training.to_csv(candidate_history, index=False)
    request = models.EvaluateCandidate(candidate_id=identifier, csv_data=training.to_csv(index=False), independent_confirmed=True)
    with pytest.raises(HTTPException) as blocked:
        asyncio.run(models.evaluate(request))
    assert blocked.value.status_code == 409 and "overlaps" in blocked.value.detail

    independent = pd.DataFrame([
        {**{f"R{i}": 100.0 for i in range(1, 18)}, "smell_label": "air"},
        {**{f"R{i}": 30.0 for i in range(1, 18)}, "smell_label": "acetone"},
    ])
    before = history.read_bytes()
    result = asyncio.run(models.evaluate(models.EvaluateCandidate(
        candidate_id=identifier, csv_data=independent.to_csv(index=False), independent_confirmed=True,
    )))
    assert result["comparison"]["rows"] == 2
    assert result["comparison"]["active"]["accuracy"] == 1.0
    assert result["comparison"]["candidate"]["accuracy"] == 1.0
    assert model.read_bytes() == b"original-model" and history.read_bytes() == before
    assert state.smell_classifier is old
    manifest = json.loads((models._snapshot_directory(identifier) / "manifest.json").read_text())
    assert manifest["external_comparison"]["active_model_sha256"]
    result = asyncio.run(models.promote(models.PromoteCandidate(candidate_id=identifier)))
    assert result["status"] == "promoted"  # no unevaluated override after paired check
