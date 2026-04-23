"""BalancedRFClassifier — the 22-feature smell classifier used by the server.

Composes preprocessing, diagnostics, and persistence submodules. The class
itself holds state (fitted scaler, clip bounds, centroids, model) and orchestrates
train / predict / online_update. Edit preprocessing.py for data-shaping changes
and diagnostics.py for OOD/centroid scoring; change this file only for control
flow or model wiring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError as e:
    raise ImportError("Install imbalanced-learn: pip install imbalanced-learn") from e

from enose.config import ALL_SENSORS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS

from .base import SmellClassifierBase
from .config import SmellClassifierConfig
from . import diagnostics, persistence, preprocessing


class BalancedRFClassifier(SmellClassifierBase):
    """Balanced Random Forest + Platt calibration on 22 sensor features."""

    RESISTANCE_SENSORS: List[str] = RESISTANCE_SENSORS
    ENVIRONMENTAL_SENSORS: List[str] = ENVIRONMENTAL_SENSORS
    ALL_SENSORS: List[str] = ALL_SENSORS

    _label_col: str = "Gas name"

    def __init__(
        self,
        model_type: str = "balanced_rf",
        online_learning: bool = True,
        config: Optional[SmellClassifierConfig] = None,
    ):
        self.config = config or SmellClassifierConfig()
        self.model = self._build_model()

        self.scaler_r = self._build_scaler()
        self.is_fitted: bool = False
        self.classes_: np.ndarray = np.array([])
        self.class_weights_: Dict[str, float] = {}
        self.selected_features: List[str] = list(self.ALL_SENSORS)
        self.original_features: List[str] = list(self.ALL_SENSORS)
        self.training_history: Dict[str, List[float]] = {"accuracy": [], "balanced_accuracy": []}
        self.last_training_data: Optional[pd.DataFrame] = None

        # legacy-artifact flags; load() flips these when a pre-dict model is restored
        self._legacy_pipeline: bool = False
        self._use_internal_scaler: bool = True

        # diagnostics state (populated during train)
        self.r_clip_: Dict[str, Tuple[float, float]] = {}
        self.r_median_: Dict[str, float] = {}
        self.feature_means_: Dict[str, float] = {}
        self.feature_stds_: Dict[str, float] = {}
        self.class_centroids_: Optional[pd.DataFrame] = None

        # Per-class test-set metrics (populated by _log_test_report on every
        # successful `train()`). Exposed via get_model_info() and the UI so the
        # chemist can see *which* smells are misclassified rather than just a
        # single conflated accuracy number.
        self.per_class_metrics_: Dict[str, Dict[str, float]] = {}
        self.confusion_matrix_: Optional[List[List[int]]] = None
        self.confusion_labels_: List[str] = []
        self.last_test_size_: int = 0

    # ── model construction ──────────────────────────────────────────────────
    def _build_model(self):
        base = BalancedRandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        if not self.config.calibrated:
            return base
        return self._make_calibrator(base, method=self.config.calibration_method, cv=3)

    def _build_scaler(self):
        """RobustScaler by default — less sensitive to the outliers and
        saturated zero readings that plague raw resistance data."""
        kind = getattr(self.config, "scaler_kind", "robust").lower()
        if kind == "standard":
            return StandardScaler()
        return RobustScaler()

    @property
    def _model_features(self) -> List[str]:
        """Columns actually fed to the estimator at fit/predict time.

        When ``config.use_env_sensors`` is False the 5 environmental channels
        (T/H/CO2/H2S/CH2O) are excluded. The full 22-column frame still flows
        through the preprocessing pipeline so that env sanitation runs if the
        columns are present — they're dropped only at the model boundary.

        Safety net: if a loaded model's ``n_features_in_`` disagrees with the
        config flag (e.g. an old 22-feature model loaded after the flag was
        flipped to False), we trust the fitted model. Predictions routed
        through a feature set that doesn't match training would blow up with
        an imblearn "Feature shape mismatch" error; falling back to the
        model's actual arity keeps inference working until the user retrains.
        """
        use_env = getattr(self.config, "use_env_sensors", True)
        candidate = list(self.ALL_SENSORS) if use_env else list(self.RESISTANCE_SENSORS)

        # Strip explicitly-dead env features (H2S in our data — constant 0).
        # Harmless when use_env=False since they weren't in the candidate
        # list to begin with.
        dead = set(getattr(self.config, "dead_env_features", ()) or ())
        if dead:
            candidate = [c for c in candidate if c not in dead]

        n_fit = getattr(self.model, "n_features_in_", None)
        if n_fit is None or n_fit == len(candidate):
            return candidate
        if n_fit == len(self.ALL_SENSORS):
            if not getattr(self, "_warned_feature_mismatch", False):
                print(
                    f"[classifier] loaded model was fit on {n_fit} features but "
                    f"config says use_env_sensors={use_env} ({len(candidate)} features). "
                    "Falling back to 22-feature inference. Retrain to fix."
                )
                self._warned_feature_mismatch = True
            return list(self.ALL_SENSORS)
        if n_fit == len(self.RESISTANCE_SENSORS):
            if not getattr(self, "_warned_feature_mismatch", False):
                print(
                    f"[classifier] loaded model was fit on {n_fit} features but "
                    f"config says use_env_sensors={use_env} ({len(candidate)} features). "
                    "Falling back to 17-feature inference. Retrain to fix."
                )
                self._warned_feature_mismatch = True
            return list(self.RESISTANCE_SENSORS)
        return candidate

    @staticmethod
    def _make_calibrator(base_clf, method: str = "sigmoid", cv: int = 3):
        """CalibratedClassifierCV — sklearn >=1.2 uses `estimator=`, older uses `base_estimator=`."""
        try:
            return CalibratedClassifierCV(estimator=base_clf, method=method, cv=cv)
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base_clf, method=method, cv=cv)

    # ── preprocessing wrapper used by server ────────────────────────────────
    def process_sensor_data(self, X) -> pd.DataFrame:
        """Sanitize env, fill missing cols, clean resistances (raw-space bounds),
        optional log1p, then scale if fitted.

        This is the *inference* path. It must mirror the training path in
        `train()` step-for-step; see that method for the full pipeline.
        """
        df = preprocessing.ensure_dataframe(X)
        df = preprocessing.sanitize_environmentals(df, self.config.env_ranges, self.config.env_medians)
        df = preprocessing.order_and_fill_features(df, self.config.env_medians)
        df = preprocessing.clean_resistances(df, self.r_median_, self.r_clip_)
        if getattr(self.config, "use_log1p", False) and getattr(self, "_log1p_applied", True):
            df = preprocessing.log1p_resistances(df)
        if self.is_fitted and self._use_internal_scaler and self._scaler_is_fitted():
            df = preprocessing.scale_resistances(df, self.scaler_r, fit=False)
        return df

    def _scaler_is_fitted(self) -> bool:
        """RobustScaler exposes ``center_``, StandardScaler exposes ``mean_``."""
        return hasattr(self.scaler_r, "center_") or hasattr(self.scaler_r, "mean_")

    def _prepare_xy(self, X, y, verbose: bool = False):
        """Split X into features frame + label series; normalize labels to lowercase strings.

        ``verbose`` flips on a per-column report for ``sanitize_environmentals``
        so training logs show how many T/H/CO2/... readings were out of range.
        Inference callers leave it False to avoid a log line per predict.
        """
        df = preprocessing.ensure_dataframe(X)
        if y is None and self._label_col in df.columns:
            y = df[self._label_col]
            df = df.drop(columns=[self._label_col], errors="ignore")
        if y is not None:
            y = pd.Series([str(v).strip().lower() for v in list(y)])
        df = preprocessing.sanitize_environmentals(
            df, self.config.env_ranges, self.config.env_medians, verbose=verbose,
        )
        df = preprocessing.order_and_fill_features(df, self.config.env_medians)
        return df, y

    # ── train / online_update ───────────────────────────────────────────────
    def train(
        self,
        X,
        y=None,
        use_augmentation: Optional[bool] = None,
        n_augmentations: Optional[int] = None,
        noise_std: Optional[float] = None,  # accepted for API compat
        test_size: Optional[float] = None,
        groups: Optional[pd.Series] = None,
    ) -> float:
        """Fit-from-scratch on (X, y).

        Pipeline (training and inference now agree on every step):
            sanitize env → order cols → clean_resistances (raw bounds) →
            [log1p] → split → augment (train only) → fit scaler → transform →
            fit model.

        ``groups`` is an optional per-row session / recording id. If provided,
        a `GroupShuffleSplit` is used so no single recording appears in both
        train and test — this is the only reliable way to measure
        generalization when the dataset is a time series of consecutive frames.
        When ``groups`` is not provided, rows are de-duplicated (rounded to
        ``config.dedup_round_decimals``) before the stratified split, which
        removes the worst of the near-neighbor leakage.
        """
        if use_augmentation is not None:
            self.config.use_augmentation = bool(use_augmentation)
        if n_augmentations is not None:
            self.config.n_augmentations = int(n_augmentations)
        if test_size is None:
            test_size = self.config.test_size

        X_df, y_series = self._prepare_xy(X, y, verbose=True)
        if y_series is None:
            raise ValueError("Labels required (column 'Gas name' or pass y).")

        # Remember the raw (but sanitized/ordered) training frame for online_update.
        self.last_training_data = pd.concat(
            [X_df.copy(), y_series.rename(self._label_col)], axis=1
        )

        # Clip bounds and medians are computed on RAW resistances — the same
        # space `clean_resistances` sees at inference time.
        self.r_clip_, self.r_median_ = preprocessing.compute_resistance_clip_bounds(
            X_df, ignore_zeros=True
        )
        X_df = preprocessing.clean_resistances(X_df, self.r_median_, self.r_clip_)
        if self.config.use_log1p:
            X_df = preprocessing.log1p_resistances(X_df)
        self._log1p_applied = bool(self.config.use_log1p)

        # Train/test split: group-aware when possible, else dedup + stratified.
        X_train, X_test, y_train, y_test = self._split(
            X_df, y_series, groups=groups, test_size=test_size
        )

        if self.config.use_augmentation and self.config.n_augmentations > 0:
            X_train = preprocessing.augment_resistances(
                X_train, self.config.n_augmentations, self.config.noise_max
            )
            y_train = pd.concat(
                [y_train] * (self.config.n_augmentations + 1), ignore_index=True
            )

        # Fit scaler on the (already-cleaned, possibly log1p'd) training split only.
        self.scaler_r = self._build_scaler()
        X_train_s = preprocessing.scale_resistances(X_train, self.scaler_r, fit=True)
        X_test_s = preprocessing.scale_resistances(X_test, self.scaler_r, fit=False)

        # Diagnostics state (z-scores, centroids) live in scaled space.
        feats = self._model_features
        self.feature_means_ = X_train_s[feats].mean().to_dict()
        stds = X_train_s[feats].std(ddof=0).replace(0, 1e-9)
        self.feature_stds_ = stds.to_dict()
        self.class_centroids_ = (
            X_train_s.assign(_y=y_train.values)
            .groupby("_y")[feats]
            .mean()
        )

        self.model = self._build_model()
        print(
            f"[train] fitting {type(self.model).__name__} on "
            f"{len(feats)} features (use_env_sensors="
            f"{getattr(self.config, 'use_env_sensors', True)}); "
            f"X_train shape={X_train_s[feats].values.shape}"
        )
        self.model.fit(X_train_s[feats].values, y_train.values)

        y_pred = self.model.predict(X_test_s[feats].values)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        self.classes_ = getattr(self.model, "classes_", np.unique(y_train.values))
        self.is_fitted = True
        self.selected_features = list(feats)
        self.training_history["accuracy"].append(float(acc))
        self.training_history["balanced_accuracy"].append(float(bal_acc))
        self._log_test_report(y_test, y_pred, acc, bal_acc)
        return float(bal_acc)

    # ── split + reporting helpers ───────────────────────────────────────────
    def _split(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        groups: Optional[pd.Series],
        test_size: float,
    ):
        """Pick the right split strategy for the available structure.

        Preference order:
          1. GroupShuffleSplit on ``groups`` (one recording stays in one side).
          2. Deduplicate near-identical rows, then stratified split.
          3. Plain stratified split.
        """
        if (
            groups is not None
            and self.config.group_shuffle_when_available
            and len(pd.Series(groups).unique()) >= 2
        ):
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=self.config.random_state,
            )
            train_idx, test_idx = next(gss.split(X_df, y_series, groups=groups))
            return (
                X_df.iloc[train_idx].reset_index(drop=True),
                X_df.iloc[test_idx].reset_index(drop=True),
                y_series.iloc[train_idx].reset_index(drop=True),
                y_series.iloc[test_idx].reset_index(drop=True),
            )

        X_use = X_df.reset_index(drop=True)
        y_use = y_series.reset_index(drop=True)
        if self.config.dedup_before_split:
            # Round-and-hash to catch frames that are near-duplicates of each
            # other — the main source of train/test leakage on a continuous
            # time-series recording.
            dec = self.config.dedup_round_decimals
            key_feats = X_use[self.ALL_SENSORS].round(dec).astype(str).agg("|".join, axis=1)
            key = key_feats.str.cat(y_use.astype(str), sep="||")
            mask = ~key.duplicated()
            n_dropped = int((~mask).sum())
            if n_dropped:
                print(
                    f"[train] dedup dropped {n_dropped}/{len(mask)} near-duplicate rows "
                    f"(round={dec} decimals) before split"
                )
            X_use = X_use.loc[mask].reset_index(drop=True)
            y_use = y_use.loc[mask].reset_index(drop=True)

        # Stratify only when every class has ≥2 samples after dedup.
        counts = y_use.value_counts()
        stratify = y_use if (counts.min() >= 2) else None
        if stratify is None:
            print("[train] skipping stratification — at least one class has <2 samples")

        return train_test_split(
            X_use,
            y_use,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )

    def _log_test_report(self, y_test, y_pred, acc: float, bal_acc: float) -> None:
        """Print confusion matrix + per-class precision/recall/F1 so we see
        what the model actually struggles with rather than one conflated
        accuracy number.

        Also stashes structured metrics on the classifier (``per_class_metrics_``,
        ``confusion_matrix_``, ``confusion_labels_``) so they can be surfaced
        via the API / UI without re-running the test set.
        """
        labels = sorted(pd.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)])))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report_txt = classification_report(
            y_test, y_pred, labels=labels, digits=4, zero_division=0
        )
        # output_dict=True gives us a per-label {precision, recall, f1-score, support}
        # map plus the accuracy / macro / weighted aggregates. Stored as plain
        # floats so JSON serialization in /smell/model_info just works.
        report_dict = classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        per_class: Dict[str, Dict[str, float]] = {}
        for lbl in labels:
            row = report_dict.get(lbl) or report_dict.get(str(lbl)) or {}
            per_class[str(lbl)] = {
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1": float(row.get("f1-score", 0.0)),
                "support": int(row.get("support", 0)),
            }
        self.per_class_metrics_ = per_class
        self.confusion_matrix_ = [[int(v) for v in row] for row in cm.tolist()]
        self.confusion_labels_ = [str(l) for l in labels]
        self.last_test_size_ = int(len(y_test))

        print("\n[train] ── test-set report ─────────────────────────────────")
        print(f"[train] accuracy          = {acc:.4f}")
        print(f"[train] balanced accuracy = {bal_acc:.4f}")
        print(f"[train] labels: {labels}")
        print("[train] confusion matrix (rows=true, cols=pred):")
        header = "              " + "  ".join(f"{l:>10}" for l in labels)
        print(header)
        for lbl, row in zip(labels, cm):
            print(f"{lbl:>12}  " + "  ".join(f"{v:>10d}" for v in row))
        print("[train] per-class metrics:")
        print(report_txt)
        print("[train] ────────────────────────────────────────────────────\n")

    def online_update(
        self,
        X,
        y,
        use_augmentation: Optional[bool] = None,
        n_augmentations: Optional[int] = None,
        groups: Optional[pd.Series] = None,
    ) -> bool:
        """Append to last_training_data and refit from scratch. Supports new classes."""
        if not self.is_fitted:
            self.train(
                X, y,
                use_augmentation=use_augmentation,
                n_augmentations=n_augmentations,
                groups=groups,
            )
            return True

        X_df, y_series = self._prepare_xy(X, y)
        new_block = pd.concat([X_df, y_series.rename(self._label_col)], axis=1)
        if self.last_training_data is not None and len(self.last_training_data):
            merged = pd.concat([self.last_training_data, new_block], ignore_index=True)
        else:
            merged = new_block

        self.train(
            merged.drop(columns=[self._label_col]),
            merged[self._label_col],
            use_augmentation=use_augmentation,
            n_augmentations=n_augmentations,
            groups=groups,
        )
        return True

    # ── predict / predict_proba ─────────────────────────────────────────────
    def _model_input(self, X):
        """Feed legacy pipelines a DataFrame (preserves column names); modern estimators get ndarray."""
        df = self.process_sensor_data(X)
        feats = self._model_features
        return df[feats] if self._legacy_pipeline else df[feats].values

    def predict(self, X) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return [str(p) for p in self.model.predict(self._model_input(X))]

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_input = self._model_input(X)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_input)
        elif hasattr(self.model, "decision_function"):
            dfc = np.array(self.model.decision_function(X_input))
            if dfc.ndim == 1:
                dfc = np.vstack([-dfc, dfc]).T
            expv = np.exp(dfc - dfc.max(axis=1, keepdims=True))
            proba = expv / expv.sum(axis=1, keepdims=True)
        else:
            preds = self.model.predict(X_input)
            classes = list(getattr(self.model, "classes_", self.classes_)) or sorted(set(preds))
            idx = {c: i for i, c in enumerate(classes)}
            proba = np.zeros((len(preds), len(classes)), dtype=float)
            for r, c in enumerate(preds):
                proba[r, idx[c]] = 1.0

        # Align columns to self.classes_ order when available
        if len(self.classes_) and hasattr(self.model, "classes_"):
            model_classes = list(self.model.classes_)
            order = [model_classes.index(c) for c in self.classes_ if c in model_classes]
            if len(order) == proba.shape[1]:
                proba = proba[:, order]
        return proba

    # ── diagnostics ─────────────────────────────────────────────────────────
    def diagnose_sample(self, X) -> Dict[str, Any]:
        """Per-feature z-scores, OOD score, nearest-centroid L2 for one sample."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        scaled = self.process_sensor_data(X)
        probs = self.predict_proba(X)
        return diagnostics.diagnose(
            scaled,
            self.feature_means_,
            self.feature_stds_,
            self.class_centroids_,
            probs[0],
            self.classes_,
            features=self._model_features,
        )

    def predict_with_ood(self, X):
        """Predict + OOD-gated confidence. Very-OOD samples get confidence damped toward uniform."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        scaled = self.process_sensor_data(X)
        proba = self.predict_proba(X)[0]

        row = scaled.iloc[0]
        feats = self._model_features
        zs = diagnostics.compute_z_scores(row, self.feature_means_, self.feature_stds_, features=feats)
        ood_score = diagnostics.compute_ood_score(zs, features=feats)

        dmin: Optional[float] = None
        if self.class_centroids_ is not None and len(self.class_centroids_):
            feats = self._model_features
            v = scaled[feats].values.reshape(1, -1)
            dists = [
                np.linalg.norm(
                    v - self.class_centroids_.loc[c][feats].values.reshape(1, -1),
                    ord=2,
                )
                for c in self.class_centroids_.index
            ]
            dmin = float(np.min(dists))

        if ood_score > 3.0 or (dmin is not None and dmin > 5.0):
            k = len(self.classes_)
            proba = 0.5 * proba + 0.5 * (np.ones_like(proba) / k)

        label_idx = int(np.argmax(proba))
        label = str(self.classes_[label_idx])
        conf = float(proba[label_idx])
        return label, conf, proba.tolist(), {"ood_score": ood_score, "min_centroid_L2": dmin}

    # ── persistence ─────────────────────────────────────────────────────────
    def save_model(self, out_dir: str = "trained_models") -> str:
        return persistence.save(self, out_dir=out_dir)

    @classmethod
    def load_model(cls, path: str) -> "BalancedRFClassifier":
        return persistence.load(path, cls)

    # ── visualization / analysis ────────────────────────────────────────────
    def generate_visualizations(self, out_dir: str = "data") -> Dict[str, str]:
        from enose.visualization import generate_all_plots
        return generate_all_plots(self, out_dir=out_dir)

    def get_model_info(self) -> Dict[str, Any]:
        """Summary for `/smell/model_info` — classes, feature lists, training history, env config."""
        return {
            "model_loaded": True,
            "is_fitted": self.is_fitted,
            "classes": [str(c) for c in self.classes_],
            "n_classes": int(len(self.classes_)),
            "features": {
                "all": list(self.ALL_SENSORS),
                "resistance": list(self.RESISTANCE_SENSORS),
                "environmental": list(self.ENVIRONMENTAL_SENSORS),
                "selected": list(self.selected_features or []),
                "model_input": list(self._model_features),
                "use_env_sensors": bool(getattr(self.config, "use_env_sensors", True)),
            },
            "training_history": self.training_history,
            "class_weights": self.class_weights_,
            "env_config": {
                "ranges": self.config.env_ranges,
                "medians": self.config.env_medians,
            },
            "calibration": {
                "enabled": self.config.calibrated,
                "method": self.config.calibration_method,
            },
            # Per-class precision/recall/F1 from the most recent test split.
            # Empty dict if this classifier was loaded from a pre-metrics
            # artifact — the UI renders a "retrain to populate" hint in that
            # case.
            "per_class_metrics": dict(self.per_class_metrics_ or {}),
            "confusion_matrix": self.confusion_matrix_,
            "confusion_labels": list(self.confusion_labels_ or []),
            "last_test_size": int(getattr(self, "last_test_size_", 0) or 0),
        }

    def analyze_data_quality(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """`output_dir` accepted for API compat; `analyze_data_quality` is stats-only."""
        from enose.visualization import analyze_data_quality as _adq
        result = _adq(self.last_training_data, label_col=self._label_col, sensors=self.ALL_SENSORS)
        result["env_ranges"] = self.config.env_ranges
        result["env_medians"] = self.config.env_medians
        return result

    def analyze_environmental_sensors(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        from enose.visualization.generate import generate_environmental_by_class
        return generate_environmental_by_class(self, out_dir=output_dir or "data")
