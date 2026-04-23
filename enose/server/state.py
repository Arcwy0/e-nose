"""Shared runtime state for the FastAPI server.

Routes import `state` and read `state.vlm_model`, `state.smell_classifier`, etc.
Mutations should flow through the setters here so logging stays in one place.
"""

from __future__ import annotations

from typing import Any, Optional

from enose.classifier import BalancedRFClassifier


vlm_model: Any = None
vlm_processor: Any = None
smell_classifier: Optional[BalancedRFClassifier] = None


def set_vlm(model: Any, processor: Any) -> None:
    global vlm_model, vlm_processor
    vlm_model = model
    vlm_processor = processor


def set_classifier(clf: Optional[BalancedRFClassifier]) -> None:
    global smell_classifier
    smell_classifier = clf


def require_classifier() -> BalancedRFClassifier:
    """Raise HTTP 503 if the classifier is not available. Routes should prefer this over None-checks."""
    from fastapi import HTTPException

    if smell_classifier is None:
        raise HTTPException(status_code=503, detail="Smell classifier not available")
    return smell_classifier


def require_fitted_classifier() -> BalancedRFClassifier:
    """Raise HTTP 503 if classifier is missing or not yet trained."""
    from fastapi import HTTPException

    clf = require_classifier()
    if not clf.is_fitted:
        raise HTTPException(status_code=503, detail="Smell classifier not trained")
    return clf
