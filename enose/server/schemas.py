"""Pydantic request/response models for the FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SensorData(BaseModel):
    """Single 22-feature reading. Defaults are neutral room conditions."""

    R1: float = 0.0
    R2: float = 0.0
    R3: float = 0.0
    R4: float = 0.0
    R5: float = 0.0
    R6: float = 0.0
    R7: float = 0.0
    R8: float = 0.0
    R9: float = 0.0
    R10: float = 0.0
    R11: float = 0.0
    R12: float = 0.0
    R13: float = 0.0
    R14: float = 0.0
    R15: float = 0.0
    R16: float = 0.0
    R17: float = 0.0
    T: float = 21.0
    H: float = 49.0
    CO2: float = 400.0
    H2S: float = 0.0
    CH2O: float = 5.0
    acquisition: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class CommitProvenance(BaseModel):
    """Optional audit trail attached to an autonomous-training commit.

    The robot mission policy fills this in so a human can later inspect:
    *what image the VLM saw*, *with what grounding score*, *with what robot
    pose at sample time*. Without it, you can't tell why the classifier
    learned a wrong label weeks after the mission.
    """

    grounding_score: float
    bbox: Optional[List[float]] = None              # [x1, y1, x2, y2] pixels
    pose: Optional[Dict[str, Any]] = None           # {x, y, theta, frame_id}
    session_id: Optional[str] = None
    detector_task: Optional[str] = None             # e.g. "<CAPTION_TO_PHRASE_GROUNDING>"
    image_b64: Optional[str] = None                 # base64-encoded image bytes
    image_filename: Optional[str] = None            # original filename hint
    extras: Optional[Dict[str, Any]] = None         # free-form (notes, target label aliases, …)


class OnlineLearningData(BaseModel):
    """Batch of labelled samples for incremental training."""

    sensor_data: List[Dict[str, float]]
    labels: List[str]
    provenance: Optional[CommitProvenance] = None
    acquisition: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class BaselineData(BaseModel):
    """Clean-air readings that define the current session's drift baseline.

    Sent at the start of a session (~30–60 s of clean air). Only used when the
    active model was trained with a baseline_mode; a no-op for absolute models.
    """

    sensor_data: List[Dict[str, float]]
    ema: bool = False
    acquisition: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class PlateauOptions(BaseModel):
    """Optional extraction overrides; defaults preserve the existing profile."""

    window_seconds: float = Field(60.0, ge=1, le=86400)
    stride_seconds: float = Field(60.0, ge=1, le=86400)
    baseline_tail_seconds: float = Field(600.0, ge=1, le=86400)
    plateau_fraction: float = Field(0.25, gt=0, le=1)
    tail_margin_seconds: float = Field(30.0, ge=0, le=86400)
    max_relative_slope: float = Field(0.002, ge=0, le=1)


class ExposureAnnotation(BaseModel):
    """Operator-verified label on zero-based source rows [start_row, end_row)."""

    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)
    label: str = Field(min_length=1, max_length=128)


class CSVLearningData(BaseModel):
    """CSV (as a string) + training hyperparameters."""

    csv_data: str
    target_column: str = "smell_label"
    # Only desktop rs_ch*_ohm exports are converted; canonical R columns are
    # never rescaled. Smell labels still must be supplied by the experimenter.
    resistance_unit: str = "ohm"
    use_augmentation: bool = True
    n_augmentations: int = 5
    noise_std: float = 0.0015
    lowercase_labels: bool = True
    # True is the Idea-1 default: retain prior classes/samples and add this CSV.
    # False trains a fresh model and replaces the canonical training history.
    merge_history: bool = True
    # API keeps immediate training for compatibility; browser defaults to staging.
    stage_candidate: bool = False
    # ``plateau`` converts a continuous timestamped recording into stable
    # one-minute windows; ``raw`` preserves the historical row-wise behavior.
    training_profile: str = "raw"
    # ``two_stage`` is the drift/intensity-robust air detector + odor-shape
    # classifier. Existing API callers remain on balanced_rf unless requested.
    classifier_backend: str = "balanced_rf"
    baseline_mode: str = "none"
    snv: bool = False
    plateau_options: Optional[PlateauOptions] = None
    recording_date: Optional[str] = None
    session_id: Optional[str] = Field(None, max_length=128)
    recording_id: Optional[str] = Field(None, max_length=128)
    annotations: List[ExposureAnnotation] = Field(default_factory=list)


class ConsoleSensorData(BaseModel):
    """Comma-separated 17 resistance values, optionally followed by 5 env values."""

    values: str
    resistance_unit: Optional[str] = None
