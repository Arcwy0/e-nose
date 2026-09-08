"""Baseline-relative two-stage classifier for stable e-nose windows.

Stage 1 detects whether the response differs from clean air.  Stage 2 removes
overall response magnitude and identifies the odor from the 17-sensor pattern.
This prevents concentration/intensity from becoming a proxy for class (the
observed acetone→air/ethanol failure).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .balanced_rf import BalancedRFClassifier
from .config import SmellClassifierConfig


class IdentityResistanceScaler:
    """Sklearn-like no-op scaler; the estimator needs unscaled log responses."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        self.mean_ = np.zeros(self.n_features_in_, dtype=float)  # fitted marker
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class TwoStageEstimator:
    """Air detector + magnitude-invariant conditional odor classifier."""

    def __init__(self, air_label: str = "air", random_state: int = 42):
        self.air_label = str(air_label)
        self.random_state = int(random_state)
        self.classes_ = np.array([])
        self.n_features_in_: Optional[int] = None
        self.air_detector = None
        self.odor_classifier = None
        self.odor_classes_: np.ndarray = np.array([])

    @staticmethod
    def _shape(X) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        mean = array.mean(axis=1, keepdims=True)
        std = array.std(axis=1, keepdims=True)
        return (array - mean) / np.maximum(std, 1e-9)

    def fit(self, X, y, sample_weight=None):
        array = np.asarray(X, dtype=float)
        labels = np.asarray(y).astype(str)
        self.n_features_in_ = int(array.shape[1])
        self.classes_ = np.asarray(sorted(np.unique(labels)))
        is_odor = labels != self.air_label
        if not np.any(~is_odor) or not np.any(is_odor):
            raise ValueError("two-stage training requires both clean-air and odor samples")

        self.air_detector = make_pipeline(
            RobustScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=self.random_state,
            ),
        )
        self.air_detector.fit(array, is_odor, **(
            {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
        ))

        self.odor_classes_ = np.asarray(sorted(np.unique(labels[is_odor])))
        if len(self.odor_classes_) > 1:
            self.odor_classifier = make_pipeline(
                RobustScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=self.random_state,
                ),
            )
            fit_args = {}
            if sample_weight is not None:
                fit_args["logisticregression__sample_weight"] = np.asarray(sample_weight)[is_odor]
            self.odor_classifier.fit(self._shape(array[is_odor]), labels[is_odor], **fit_args)
        else:
            self.odor_classifier = None
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.air_detector is None:
            raise RuntimeError("two-stage estimator not fitted")
        array = np.asarray(X, dtype=float)
        detector_classes = list(self.air_detector.classes_)
        odor_probability = self.air_detector.predict_proba(array)[:, detector_classes.index(True)]
        if self.odor_classifier is None:
            conditional = np.ones((len(array), 1), dtype=float)
            conditional_classes = list(self.odor_classes_)
        else:
            conditional = self.odor_classifier.predict_proba(self._shape(array))
            conditional_classes = list(self.odor_classifier.classes_)

        result = np.zeros((len(array), len(self.classes_)), dtype=float)
        class_index = {name: i for i, name in enumerate(self.classes_)}
        result[:, class_index[self.air_label]] = 1.0 - odor_probability
        for j, name in enumerate(conditional_classes):
            result[:, class_index[name]] = odor_probability * conditional[:, j]
        return result

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class TwoStageResponseClassifier(BalancedRFClassifier):
    """Drop-in classifier backend using log-ratio response-shape features."""

    backend_name = "two_stage"

    def __init__(
        self,
        model_type: str = "two_stage",
        online_learning: bool = True,
        config: Optional[SmellClassifierConfig] = None,
    ):
        cfg = config or SmellClassifierConfig()
        cfg.baseline_mode = "logratio"
        cfg.snv = False
        cfg.use_env_sensors = False
        cfg.calibrated = False
        cfg.use_augmentation = False
        cfg.n_augmentations = 0
        super().__init__(model_type=model_type, online_learning=online_learning, config=cfg)

    def _build_scaler(self):
        return IdentityResistanceScaler()

    def _build_model(self, y_fit=None):
        return TwoStageEstimator(
            air_label=getattr(self.config, "air_label", "air"),
            random_state=self.config.random_state,
        )

    def train(self, X, y=None, **kwargs):
        # Noise augmentation in already-normalized log-response space degrades
        # the air detector and turns correlated frames into fake evidence.
        kwargs["use_augmentation"] = False
        kwargs["n_augmentations"] = 0
        return super().train(X, y, **kwargs)
