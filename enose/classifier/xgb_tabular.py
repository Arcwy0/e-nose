"""XGBoost drop-in for BalancedRFClassifier — same 22-feature tabular pipeline.

Use this when you want to compare XGBoost against Balanced RF on identical
preprocessing (clean_resistances, log1p, RobustScaler, dedup, group-aware
split). The only thing that changes is the final estimator.

For temporal windowing / drift correction, use ``XGBOdorClassifier`` instead —
that's a different model that operates on windows, not single frames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("Install xgboost: pip install xgboost") from e

from .balanced_rf import BalancedRFClassifier
from .config import SmellClassifierConfig


class _XGBStringAdapter:
    """Wrap XGBClassifier so it accepts string labels and reweights for imbalance.

    Mirrors the small slice of the sklearn estimator API that
    BalancedRFClassifier actually uses: ``fit(X, y)``, ``predict(X)``,
    ``predict_proba(X)``, ``classes_``. XGBoost wants integer targets, so we
    hold a LabelEncoder internally and translate both ways. Class imbalance
    is handled via per-sample weights (``n_total / (n_classes * class_count)``)
    rather than resampling, which is how XGBoost is conventionally balanced.
    """

    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)
        self._encoder = LabelEncoder()
        self._model: Optional[XGBClassifier] = None
        self.classes_: np.ndarray = np.array([])

    def _build(self, n_classes: int) -> XGBClassifier:
        params = dict(self._params)
        params.update(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist",
        )
        # XGBClassifier rejects num_class=None for binary; drop it.
        params = {k: v for k, v in params.items() if v is not None}
        return XGBClassifier(**params)

    def fit(self, X, y, sample_weight=None):
        y_str = np.asarray(y).astype(str)
        y_int = self._encoder.fit_transform(y_str)
        self.classes_ = np.asarray(self._encoder.classes_)

        if sample_weight is None:
            # Balanced weighting — same spirit as BalancedRF's resampling, just
            # via weights so XGBoost sees every sample.
            counts = pd.Series(y_int).value_counts().to_dict()
            n_total = len(y_int)
            n_classes = len(self.classes_)
            w = np.array([n_total / (n_classes * counts[int(v)]) for v in y_int])
        else:
            w = np.asarray(sample_weight)

        self._model = self._build(len(self.classes_))
        self._model.fit(np.asarray(X), y_int, sample_weight=w, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGB adapter not fitted")
        pred_int = np.asarray(self._model.predict(np.asarray(X))).astype(int).ravel()
        return self._encoder.inverse_transform(pred_int)

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGB adapter not fitted")
        return np.asarray(self._model.predict_proba(np.asarray(X)))

    @property
    def n_features_in_(self):
        """Expose the fitted XGBClassifier's feature count so
        BalancedRFClassifier._model_features can detect shape mismatches
        (wrapper-level attribute; sklearn normally sets this at the top-level
        estimator, but we don't inherit from BaseEstimator)."""
        if self._model is None:
            raise AttributeError("n_features_in_")
        return getattr(self._model, "n_features_in_", None)


class XGBTabularClassifier(BalancedRFClassifier):
    """BalancedRFClassifier with XGBoost swapped in for the final estimator.

    All preprocessing (sanitize env, order cols, clean_resistances on raw
    bounds, optional log1p, RobustScaler, dedup-or-group split, per-class
    report) is inherited unchanged so the A/B against BalancedRF is apples-
    to-apples.

    Calibration is off by default — XGBoost with logloss is already fairly
    well-calibrated, and wrapping it in CalibratedClassifierCV(cv=3) on
    leaky time-series data distorts more than it helps. Flip
    ``config.calibrated=True`` to re-enable.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        online_learning: bool = True,
        config: Optional[SmellClassifierConfig] = None,
    ):
        cfg = config or SmellClassifierConfig()
        # Calibration off for XGB unless explicitly requested — avoid double
        # hit of leakage (cv folds on a time-series set) + Platt distortion.
        cfg.calibrated = False
        super().__init__(model_type=model_type, online_learning=online_learning, config=cfg)

    def _build_model(self):
        """Hyperparameters kept conservative — trees + depth tuned for 22
        features × 6 classes × ~10⁵ rows. Edit SmellClassifierConfig if you
        want to expose these; for now they're local to the backend."""
        params: Dict[str, Any] = dict(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
        base = _XGBStringAdapter(params)
        if not self.config.calibrated:
            return base
        return self._make_calibrator(base, method=self.config.calibration_method, cv=3)


def get_classifier_backend(name: str) -> type:
    """Resolve a backend name to a classifier class.

    Unknown names fall back to BalancedRF so a typo in ``ENOSE_CLASSIFIER``
    doesn't crash startup — it just logs the miss and uses the default.
    """
    key = (name or "").strip().lower()
    if key in ("xgb", "xgboost", "xgbtabular"):
        return XGBTabularClassifier
    if key in ("", "rf", "balanced_rf", "brf"):
        return BalancedRFClassifier
    print(f"[classifier] unknown backend '{name}'; falling back to balanced_rf")
    return BalancedRFClassifier
