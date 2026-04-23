"""Smell classifiers.

Two implementations are available:
    BalancedRFClassifier - 22 features, Balanced RF + Platt calibration.
                           This is the one the FastAPI server loads by default.
    XGBClassifier        - 22 features with temporal windowing, drift correction,
                           exemplar replay for online learning.

Both conform to the SmellClassifierBase interface (train / predict / predict_proba /
online_update / save_model / load_model).
"""

from .base import SmellClassifierBase
from .config import SmellClassifierConfig
from .balanced_rf import BalancedRFClassifier
from .xgb import XGBOdorClassifier, XGBClassifierConfig
from .xgb_tabular import XGBTabularClassifier, get_classifier_backend

# Back-compat alias: existing code imports `SmellClassifier` expecting the Balanced RF
SmellClassifier = BalancedRFClassifier

__all__ = [
    "SmellClassifierBase",
    "SmellClassifierConfig",
    "BalancedRFClassifier",
    "SmellClassifier",
    "XGBTabularClassifier",
    "XGBOdorClassifier",
    "XGBClassifierConfig",
    "get_classifier_backend",
]
