"""Common interface for smell classifiers.

Any classifier used by the server should implement these methods so routes
don't have to special-case the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class SmellClassifierBase(ABC):
    """Minimal interface the FastAPI server expects from a smell classifier."""

    is_fitted: bool
    classes_: np.ndarray

    @abstractmethod
    def train(
        self,
        X: Union[pd.DataFrame, Dict[str, Any]],
        y: Optional[Union[pd.Series, List, np.ndarray]] = None,
        use_augmentation: Optional[bool] = None,
        n_augmentations: Optional[int] = None,
        noise_std: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> float:
        """Fit from scratch. Returns a holdout accuracy score."""

    @abstractmethod
    def online_update(
        self,
        X: Union[pd.DataFrame, Dict[str, Any]],
        y: Union[pd.Series, List, np.ndarray],
        use_augmentation: Optional[bool] = None,
        n_augmentations: Optional[int] = None,
    ) -> bool:
        """Append samples + refit (new classes allowed)."""

    @abstractmethod
    def predict(self, X) -> List[str]: ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    @abstractmethod
    def process_sensor_data(self, X) -> pd.DataFrame:
        """Normalize/sanitize/fill a raw sample into the model's feature frame."""

    @abstractmethod
    def save_model(self, out_dir: str) -> str: ...

    @classmethod
    @abstractmethod
    def load_model(cls, path: str) -> "SmellClassifierBase": ...
