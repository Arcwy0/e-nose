"""Isolated model candidates and recoverable, explicitly requested promotion.

Only load registry artifacts produced by this trusted server: joblib is not an
untrusted upload format. Candidate holdouts are not comparative benchmarks.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import uuid

import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from enose.utils.csv_io import save_training_samples
from enose.utils.acquisition import frame_resistance_unit, model_resistance_unit
from .persistence import build_save_payload


def _directory(root, identifier):
    if not isinstance(identifier, str) or len(identifier) != 32:
        raise ValueError("Invalid registry identifier")
    try:
        if uuid.UUID(hex=identifier).hex != identifier:
            raise ValueError("Invalid registry identifier")
    except (ValueError, AttributeError) as error:
        raise ValueError("Invalid registry identifier") from error
    return Path(root) / identifier


def _write_json(path, value):
    temporary = path.with_suffix(".json.pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def create_candidate(classifier, root="trained_models/candidates", comparison=None):
    """Snapshot an already-fitted candidate and canonical history; return its ID."""
    history = getattr(classifier, "canonical_training_data_", None)
    if not getattr(classifier, "is_fitted", False) or history is None:
        raise ValueError("Candidate must be fitted with retained canonical training data")
    identifier = uuid.uuid4().hex
    directory = _directory(root, identifier)
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(build_save_payload(classifier), directory / "model.joblib")
    save_training_samples(history, str(directory / "history.csv"))
    manifest = {
        "id": identifier,
        "status": "candidate",
        "validation": classifier.get_model_info().get("validation", {}),
        "calibration": classifier.get_model_info().get("calibration", {}),
        "classes": [str(label) for label in classifier.classes_],
        "training_rows": len(history),
        "external_comparison": comparison,
        "note": "Training holdout metrics from different fits are not a paired comparison.",
    }
    _write_json(directory / "manifest.json", manifest)
    return identifier


def list_candidates(root="trained_models/candidates"):
    """Return complete candidate manifests only; ignore incomplete writes."""
    directory = Path(root)
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*/manifest.json"))]


def artifact_sha256(path):
    """Bind a comparison to the exact persisted artifact, not a mutable name."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_external_comparison(candidate_id, root, comparison):
    directory = _directory(root, candidate_id)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["external_comparison"] = comparison
    _write_json(path, manifest)


def compare_on_external_holdout(active, candidate, X, y, *, independent_confirmed=False):
    """Compare both models on the SAME operator-confirmed independent rows.

    The caller must establish non-overlap with BOTH training histories. This
    cannot be inferred from labels or ordinary row-wise train/test scores.
    """
    if not independent_confirmed:
        raise ValueError("Confirm this evaluation set is independent of both training histories")
    if not len(X) or len(X) != len(y):
        raise ValueError("External holdout needs aligned, nonempty features and labels")
    metrics = {}
    units = frame_resistance_unit(X)
    for name, classifier in (("active", active), ("candidate", candidate)):
        if model_resistance_unit(classifier) != units:
            raise ValueError("External evaluation units must match both classifiers; unverified units are not inferred")
        if not set(map(str, y)).issubset(set(map(str, classifier.classes_))):
            raise ValueError("Both classifiers must know every external evaluation label")
        prediction = classifier.predict(X)
        metrics[name] = {
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        }
    return {"rows": len(X), "independent_confirmed": True, **metrics}


def _archive_existing(path, directory, key):
    if path.exists():
        shutil.copy2(path, directory / key)
        return True
    return False


def _replace_from(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + "." + uuid.uuid4().hex + ".pending")
    shutil.copy2(source, pending)
    os.replace(pending, target)


def snapshot_active_pair(root, model_path, history_path, *, candidate=None):
    """Archive current active artifacts before any deployment mutation."""
    if Path(model_path).resolve() == Path(history_path).resolve():
        raise ValueError("Model and history paths must be different")
    identifier = uuid.uuid4().hex
    directory = _directory(Path(root) / "rollbacks", identifier)
    directory.mkdir(parents=True, exist_ok=False)
    targets = {"model.joblib": Path(model_path), "history.csv": Path(history_path)}
    _write_json(directory / "manifest.json", {
        "candidate": candidate,
        "targets": {key: str(path.resolve()) for key, path in targets.items()},
        "existed": {key: _archive_existing(path, directory, key) for key, path in targets.items()},
    })
    return identifier


def rollback(snapshot_id, root, model_path, history_path):
    """Restore the model/history pair saved before promotion; preserve displaced files."""
    directory = _directory(Path(root) / "rollbacks", snapshot_id)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for key, target in (("model.joblib", Path(model_path)), ("history.csv", Path(history_path))):
        expected = manifest["targets"][key]
        if str(target.resolve()) != expected:
            raise ValueError("Rollback targets differ from the saved promotion targets")
    displaced = directory / ("displaced_" + uuid.uuid4().hex)
    displaced.mkdir()
    for key, target in (("model.joblib", Path(model_path)), ("history.csv", Path(history_path))):
        if target.exists():
            shutil.copy2(target, displaced / key)
        if manifest["existed"][key]:
            _replace_from(directory / key, target)
        elif target.exists():
            # Preserve rather than delete a newly introduced active artifact.
            shutil.move(str(target), str(displaced / ("introduced_" + key)))
    return snapshot_id


def promote_candidate(candidate_id, root, model_path, history_path, *, allow_unevaluated=False, allow_regression=False):
    """Explicit promotion with rollback snapshot; never choose a candidate automatically.

    Callers must serialize promotion/training and swap the in-memory classifier
    only after success. Two-file replacement is recoverable, not crash-atomic.
    """
    directory = _directory(root, candidate_id)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    comparison = manifest.get("external_comparison")
    if not comparison and not allow_unevaluated:
        raise ValueError("No paired independent evaluation; explicit allow_unevaluated is required")
    if comparison and (
        not comparison.get("independent_confirmed")
        or comparison.get("active_model_sha256") != artifact_sha256(model_path)
        or comparison.get("candidate_model_sha256") != artifact_sha256(directory / "model.joblib")
        or comparison.get("active_history_sha256") != artifact_sha256(history_path)
        or comparison.get("candidate_history_sha256") != artifact_sha256(directory / "history.csv")
    ):
        raise ValueError("Paired evaluation no longer matches the active/candidate artifacts; evaluate again")
    if comparison and comparison["candidate"]["balanced_accuracy"] < comparison["active"]["balanced_accuracy"] and not allow_regression:
        raise ValueError("Candidate regressed on the shared external holdout")
    if not (directory / "model.joblib").is_file() or not (directory / "history.csv").is_file():
        raise ValueError("Candidate snapshot is incomplete")
    targets = {"model.joblib": Path(model_path), "history.csv": Path(history_path)}
    snapshot_id = snapshot_active_pair(root, model_path, history_path, candidate=candidate_id)
    try:
        for key, target in targets.items():
            _replace_from(directory / key, target)
    except Exception:
        rollback(snapshot_id, root, model_path, history_path)
        raise
    return snapshot_id
