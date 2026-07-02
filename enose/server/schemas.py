"""Pydantic request/response models for the FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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


class CSVLearningData(BaseModel):
    """CSV (as a string) + training hyperparameters."""

    csv_data: str
    target_column: str = "smell_label"
    use_augmentation: bool = True
    n_augmentations: int = 5
    noise_std: float = 0.0015
    lowercase_labels: bool = True


class ConsoleSensorData(BaseModel):
    """Comma-separated 22 numbers, R1,...,R17,T,H,CO2,H2S,CH2O."""

    values: str
