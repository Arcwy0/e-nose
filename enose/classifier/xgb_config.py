"""Tunables for the XGBoost odor classifier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class XGBClassifierConfig:
    """Hyperparameters for XGBOdorClassifier. Edit here, not the class body."""

    # Windowing
    window_size: int = 20
    window_stride: int = 10
    min_window_samples: int = 10

    # Feature extraction toggles
    use_slope: bool = True
    use_percentiles: bool = True

    # XGBoost
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # Online learning
    exemplar_memory_per_class: int = 200
    online_retrain_epochs: int = 50

    # Drift mitigation
    drift_ema_alpha: float = 0.01

    # Paths
    model_dir: str = "./odor_model"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "XGBClassifierConfig":
        with open(path) as f:
            return cls(**json.load(f))
