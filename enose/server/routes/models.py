"""Explicit candidate promotion and recoverable rollback of model/history pairs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from enose.classifier.balanced_rf import BalancedRFClassifier
from enose.classifier import registry
from enose.config import RESISTANCE_SENSORS, SMELL_MODEL_PATH, TRAINED_MODELS_DIR, TRAINING_DATA_PATH
from enose.utils.acquisition import frame_resistance_unit
from enose.utils.csv_io import load_training_csv, parse_uploaded_enose_csv
from .. import state


router = APIRouter()
# Shared with active-training routes. Deployment requires one server worker;
# this lock deliberately does not claim coordination across OS processes.
deployment_lock = asyncio.Lock()


class PromoteCandidate(BaseModel):
    candidate_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    allow_unevaluated: bool = False
    allow_regression: bool = False


class RollbackCandidate(BaseModel):
    snapshot_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")


class EvaluateCandidate(BaseModel):
    candidate_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    csv_data: str
    target_column: str = "smell_label"
    resistance_unit: Literal["ohm", "kohm", "mohm"] = "ohm"
    independent_confirmed: bool = False


def candidate_root():
    return os.path.join(TRAINED_MODELS_DIR, "candidates")


def _snapshot_directory(identifier, rollback=False):
    root = Path(candidate_root())
    if rollback:
        root = root / "rollbacks"
    directory = registry._directory(root, identifier)
    if directory.is_symlink() or not directory.is_dir():
        raise HTTPException(status_code=404, detail="Registry snapshot not found")
    if directory.resolve().parent != root.resolve():
        raise HTTPException(status_code=400, detail="Invalid registry snapshot path")
    for name in ("model.joblib", "history.csv", "manifest.json"):
        if (directory / name).is_symlink():
            raise HTTPException(status_code=400, detail="Registry snapshots cannot contain symlinks")
    return directory


def _load_classifier(path):
    # Loading trusted server-produced registry artifacts only, never uploads.
    classifier = BalancedRFClassifier.load_model(str(path))
    classifier.live_baseline_captured_ = False
    classifier.live_baseline_context_ = None
    return classifier


@router.get("/smell/models/candidates")
async def candidates():
    return {"candidates": registry.list_candidates(candidate_root())}


def _row_fingerprints(frame, target):
    """Exact raw R1-R17 + normalized label check, not proof of independence."""
    if target not in frame or any(sensor not in frame for sensor in RESISTANCE_SENSORS):
        return set()
    values = frame[RESISTANCE_SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="<f8")
    labels = frame[target].astype(str).str.strip().str.lower()
    return {
        hashlib.sha256(row.tobytes() + label.encode("utf-8")).hexdigest()
        for row, label in zip(values, labels)
        if np.isfinite(row).all()
    }


@router.post("/smell/models/evaluate")
async def evaluate(data: EvaluateCandidate):
    """Evaluate the SAME independent rows without training or active-state writes."""
    if not data.independent_confirmed:
        raise HTTPException(status_code=409, detail="Confirm independent exposures/sessions excluded from BOTH training histories")
    async with deployment_lock:
        directory = _snapshot_directory(data.candidate_id)
        state.require_fitted_classifier()
        try:
            frame, report = parse_uploaded_enose_csv(
                data.csv_data, label_column=data.target_column,
                pcb_resistance_unit=data.resistance_unit,
            )
            if frame.empty or report.get("dropped_rows", 0):
                raise ValueError("Evaluation CSV must contain valid rows; malformed/faulty rows cannot be silently dropped")
            if data.target_column not in frame:
                raise ValueError("Evaluation requires the explicitly selected verified label column")
            missing = [sensor for sensor in RESISTANCE_SENSORS if sensor not in frame]
            if missing:
                raise ValueError(f"Evaluation requires all R1-R17 channels; missing {missing}")
            values = frame[RESISTANCE_SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Evaluation resistances must all be finite and positive")
            labels = frame[data.target_column]
            if labels.isna().any():
                raise ValueError("Every evaluation row needs a verified smell label")
            labels = labels.astype(str).str.strip().str.lower()
            if labels.isin({"", "unknown", "nan", "none"}).any():
                raise ValueError("Every evaluation row needs a verified smell label")
            declared_unit = frame_resistance_unit(frame)
            if declared_unit and declared_unit != data.resistance_unit:
                raise HTTPException(status_code=409, detail="CSV resistance units conflict with the explicitly selected evaluation units; no automatic rescaling")
            frame["resistance_unit"] = data.resistance_unit
            frame[data.target_column] = labels
        except HTTPException:
            raise
        except (ValueError, TypeError, pd.errors.ParserError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        try:
            # Evaluate persisted deployment artifacts, never today's unrelated
            # live baseline. No baseline is inferred from evaluation labels.
            active = _load_classifier(Path(SMELL_MODEL_PATH))
            candidate = _load_classifier(directory / "model.joblib")
            for classifier in (active, candidate):
                classifier.sensor_baseline_ = dict(getattr(classifier, "_original_baseline_", {}) or {})
            fingerprints = _row_fingerprints(frame, data.target_column)
            for path in (Path(TRAINING_DATA_PATH), directory / "history.csv"):
                if not path.is_file():
                    raise HTTPException(status_code=409, detail="Both persisted training histories are required for overlap screening")
                history = load_training_csv([str(path)])
                if history is None:
                    raise HTTPException(status_code=409, detail="Cannot verify the persisted training history for overlap")
                if fingerprints & _row_fingerprints(history, "smell_label"):
                    raise HTTPException(status_code=409, detail="Evaluation overlaps training history on exact raw resistance values and labels; provide independent exposures")
            comparison = registry.compare_on_external_holdout(
                active, candidate, frame, labels, independent_confirmed=True,
            )
            comparison.update({
                "active_model_sha256": registry.artifact_sha256(SMELL_MODEL_PATH),
                "candidate_model_sha256": registry.artifact_sha256(directory / "model.joblib"),
                "active_history_sha256": registry.artifact_sha256(TRAINING_DATA_PATH),
                "candidate_history_sha256": registry.artifact_sha256(directory / "history.csv"),
                "overlap_check": "no_exact_R_plus_label_overlap_detected",
                "baseline_reference": "persisted training fallback; no test-label-derived baseline or session adaptation",
                "limitation": "Exact-row checks do not establish exposure/session independence; operator confirmation remains required. Rows are evaluated as supplied, without plateau extraction.",
            })
            registry.record_external_comparison(data.candidate_id, candidate_root(), comparison)
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=409, detail="Persisted active/candidate model and history are required") from error
        return {"status": "evaluated", "candidate_id": data.candidate_id, "comparison": comparison, "parse_report": report}


@router.post("/smell/models/promote")
async def promote(data: PromoteCandidate):
    async with deployment_lock:
        directory = _snapshot_directory(data.candidate_id)
        try:
            # Validate deserialization BEFORE changing either active artifact.
            classifier = _load_classifier(directory / "model.joblib")
            previous = state.smell_classifier
            snapshot_id = registry.promote_candidate(
                data.candidate_id, candidate_root(), SMELL_MODEL_PATH, TRAINING_DATA_PATH,
                allow_unevaluated=data.allow_unevaluated,
                allow_regression=data.allow_regression,
            )
            try:
                state.set_classifier(classifier)
            except Exception:
                registry.rollback(snapshot_id, candidate_root(), SMELL_MODEL_PATH, TRAINING_DATA_PATH)
                state.smell_classifier = previous
                raise
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Candidate snapshot is incomplete") from error
        return {
            "status": "promoted", "candidate_id": data.candidate_id,
            "rollback_id": snapshot_id, "baseline_required": True,
        }


@router.post("/smell/models/rollback")
async def rollback(data: RollbackCandidate):
    async with deployment_lock:
        directory = _snapshot_directory(data.snapshot_id, rollback=True)
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            classifier = _load_classifier(directory / "model.joblib") if manifest["existed"]["model.joblib"] else None
            previous = state.smell_classifier
            recovery_id = registry.snapshot_active_pair(candidate_root(), SMELL_MODEL_PATH, TRAINING_DATA_PATH)
            try:
                registry.rollback(data.snapshot_id, candidate_root(), SMELL_MODEL_PATH, TRAINING_DATA_PATH)
                state.set_classifier(classifier)
            except Exception:
                registry.rollback(recovery_id, candidate_root(), SMELL_MODEL_PATH, TRAINING_DATA_PATH)
                state.smell_classifier = previous
                raise
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Rollback snapshot is incomplete") from error
        return {"status": "rolled_back", "snapshot_id": data.snapshot_id, "baseline_required": True}
