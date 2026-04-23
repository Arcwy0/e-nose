"""Pydantic request/response models for the FastAPI endpoints."""

from __future__ import annotations

from typing import Dict, List

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


class OnlineLearningData(BaseModel):
    """Batch of labelled samples for incremental training."""

    sensor_data: List[Dict[str, float]]
    labels: List[str]


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
