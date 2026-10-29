"""BalancedRFClassifier — the 22-feature smell classifier used by the server.

Composes preprocessing, diagnostics, and persistence submodules. The class
itself holds state (fitted scaler, clip bounds, centroids, model) and orchestrates
train / predict / online_update. Edit preprocessing.py for data-shaping changes
and diagnostics.py for OOD/centroid scoring; change this file only for control
flow or model wiring.
"""

from __future__ import annotations

from copy import deepcopy
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError as e:
    raise ImportError("Install imbalanced-learn: pip install imbalanced-learn") from e

from enose.config import ALL_SENSORS, ENVIRONMENTAL_SENSORS, RESISTANCE_SENSORS

from .base import SmellClassifierBase
from .config import SmellClassifierConfig
from . import diagnostics, persistence, preprocessing


class GroupCalibratedEstimator:
    """Average group-held-out calibrators with their fold-local preprocessing.

    Inputs are raw resistance/environmental values, not the outer classifier's
    scaled diagnostic features. Each pair owns a separately fitted pipeline.
    """

    accepts_raw_features = True

    def __init__(self, folds, features, classes):
        self.folds = folds
        self.features = list(features)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = len(self.features)

    def predict_proba(self, X):
        raw = pd.DataFrame(X, columns=self.features)
        probabilities = []
        for worker, calibrator in self.folds:
            transformed = worker.process_sensor_data(raw)[self.features].values
            predicted = calibrator.predict_proba(transformed)
            order = [list(calibrator.classes_).index(label) for label in self.classes_]
            probabilities.append(predicted[:, order])
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BalancedRFClassifier(SmellClassifierBase):
    """Balanced Random Forest + Platt calibration on 22 sensor features."""

    RESISTANCE_SENSORS: List[str] = RESISTANCE_SENSORS
    ENVIRONMENTAL_SENSORS: List[str] = ENVIRONMENTAL_SENSORS
    ALL_SENSORS: List[str] = ALL_SENSORS

    _label_col: str = "Gas name"
    backend_name: str = "balanced_rf"

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
        self.training_history: Dict[str, List[Optional[float]]] = {"accuracy": [], "balanced_accuracy": []}
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
        self.validation_strategy_: str = "not_evaluated"
        self.validation_warning_: Optional[str] = None

        # Compact training-set summaries computed at train() time and persisted,
        # so /smell/model_info and /smell/class_examples keep working after the
        # server reloads the model from disk (which drops last_training_data).
        self.class_distribution_: Dict[str, int] = {}
        self.class_examples_: Dict[str, List[float]] = {}

        # Drift-invariant representation state (opt-in; empty in absolute mode).
        # sensor_baseline_ = the ACTIVE clean-air baseline used for baseline-relative
        # features (updated per session at inference via /smell/baseline);
        # _original_baseline_ = the training-time fallback. Both map R1-R17 → R0.
        self.sensor_baseline_: Dict[str, float] = {}
        self._original_baseline_: Dict[str, float] = {}
        # Training fallback and a baseline captured from today's hardware are
        # intentionally distinct. This flag resets after every restart/train.
        self.live_baseline_captured_: bool = False

    # ── model construction ──────────────────────────────────────────────────
    def _build_model(self, y_fit=None):
        base = BalancedRandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        if not self.config.calibrated:
            return base

        # Adaptive calibration CV. CalibratedClassifierCV runs an internal
        # StratifiedKFold, which raises "y needs more than 1 class" (HTTP 500
        # at the API) whenever any class has fewer samples than the fold count.
        # Small new-scent batches from Idea-1 online learning hit this
        # routinely. Clamp the fold count to the smallest per-class count, and
        # skip calibration entirely when a class is too small to split at all
        # (uncalibrated probabilities beat a crashed fit).
        cv = 3
        if y_fit is not None:
            _, counts = np.unique(np.asarray(y_fit), return_counts=True)
            min_class = int(counts.min()) if len(counts) else 0
            if len(counts) < 2:
                print("[train] only one class present — skipping probability calibration")
                return base
            if min_class < 2:
                print(
                    f"[train] smallest class has {min_class} sample(s) — "
                    "skipping probability calibration (needs ≥2 per class)"
                )
                return base
            cv = max(2, min(3, min_class))
            if cv < 3:
                print(f"[train] small classes (min={min_class}) — calibration cv={cv}")
        return self._make_calibrator(base, method=self.config.calibration_method, cv=cv)

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
    @staticmethod
    def _mean_baselines(per_group: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Average per-group clean-air baselines into one fallback baseline."""
        acc: Dict[str, List[float]] = {}
        for b in (per_group or {}).values():
            for k, v in (b or {}).items():
                acc.setdefault(k, []).append(float(v))
        return {k: float(np.mean(vs)) for k, vs in acc.items() if vs}

    def _effective_baseline_mode(self) -> str:
        """Configured baseline_mode, but 'none' when no baseline is available —
        so we never emit half-relative features."""
        mode = getattr(self.config, "baseline_mode", "none") or "none"
        if mode == "none":
            return "none"
        return mode if self.sensor_baseline_ else "none"

    def process_sensor_data(self, X) -> pd.DataFrame:
        """Sanitize env, fill missing cols, clean resistances (raw-space bounds),
        optional baseline-relative + log1p + SNV, then scale if fitted.

        This is the *inference* path. It must mirror the training path in
        `train()` step-for-step; see that method for the full pipeline. The
        baseline-relative / SNV steps are no-ops unless the model was trained
        with those options (absolute path stays identical).
        """
        self._sync_calibration_baselines()
        df = preprocessing.ensure_dataframe(X)
        df = preprocessing.sanitize_environmentals(df, self.config.env_ranges, self.config.env_medians)
        df = preprocessing.order_and_fill_features(df, self.config.env_medians)
        df = preprocessing.clean_resistances(df, self.r_median_, self.r_clip_)
        mode = self._effective_baseline_mode()
        if mode != "none":
            df = preprocessing.baseline_relative_resistances(df, self.sensor_baseline_, mode)
        # log1p decision was baked into _log1p_applied at train (skips for delta/logratio).
        if getattr(self.config, "use_log1p", False) and getattr(self, "_log1p_applied", True):
            df = preprocessing.log1p_resistances(df)
        if getattr(self.config, "snv", False):
            df = preprocessing.snv_normalize(df)
        if self.is_fitted and self._use_internal_scaler and self._scaler_is_fitted():
            df = preprocessing.scale_resistances(df, self.scaler_r, fit=False)
        return df

    def update_baseline(self, air_samples, ema: bool = True) -> Dict[str, float]:
        """Set/refresh the ACTIVE clean-air baseline from same-session air readings.

        ``air_samples`` is anything accepted by ``process_sensor_data`` input
        (dict, list-of-dicts, DataFrame). Only meaningful when the model was
        trained with a baseline_mode; otherwise it's stored but unused. Returns
        the new active baseline. EMA-smoothed against the existing baseline
        (like the XGB path) unless ``ema=False``.
        """
        df = preprocessing.ensure_dataframe(air_samples)
        df = preprocessing.sanitize_environmentals(df, self.config.env_ranges, self.config.env_medians)
        df = preprocessing.order_and_fill_features(df, self.config.env_medians)
        df = preprocessing.clean_resistances(df, self.r_median_, self.r_clip_)
        new = preprocessing.compute_air_baseline(df, label_col="__none__")  # treat all rows as air
        if not new:
            return dict(self.sensor_baseline_)
        alpha = float(getattr(self.config, "baseline_ema_alpha", 0.3))
        if ema and self.sensor_baseline_:
            merged = {}
            for k in set(self.sensor_baseline_) | set(new):
                old = float(self.sensor_baseline_.get(k, new.get(k, 0.0)))
                cur = float(new.get(k, old))
                merged[k] = alpha * cur + (1.0 - alpha) * old
            self.sensor_baseline_ = merged
        else:
            self.sensor_baseline_ = dict(new)
        self.live_baseline_captured_ = bool(self.sensor_baseline_)
        self._sync_calibration_baselines()
        return dict(self.sensor_baseline_)

    def _sync_calibration_baselines(self):
        """Share operator-captured air, restoring fold fallbacks after reset.

        Training/calibration folds retain their own fallback R0. A live capture
        replaces every fold's R0 with the same current physical reference.
        """
        if not getattr(self.model, "accepts_raw_features", False):
            return
        if not self.live_baseline_captured_:
            # Keep outer OOD diagnostics consistent with fold prediction after
            # identity invalidation or loading an artifact with capture reset.
            self.sensor_baseline_ = dict(self._original_baseline_)
        for worker, _ in self.model.folds:
            worker.sensor_baseline_ = dict(
                self.sensor_baseline_ if self.live_baseline_captured_ else worker._original_baseline_
            )
            worker.live_baseline_captured_ = bool(self.live_baseline_captured_)

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
    def _fit_preprocessing(self, raw, labels, baseline_groups=None):
        """Learn every data-dependent transform from these fit rows only."""
        frame = raw.reset_index(drop=True).copy()
        self.r_clip_, self.r_median_ = preprocessing.compute_resistance_clip_bounds(frame)
        mode = getattr(self.config, "baseline_mode", "none") or "none"
        if mode != "none":
            self.r_clip_ = {sensor: (-np.inf, np.inf) for sensor in self.RESISTANCE_SENSORS}
        frame = preprocessing.clean_resistances(frame, self.r_median_, self.r_clip_)
        self.sensor_baseline_ = {}
        self._original_baseline_ = {}
        self.live_baseline_captured_ = False
        if mode != "none":
            group = pd.Series(list(baseline_groups)) if baseline_groups is not None else pd.Series("all", index=frame.index)
            labelled = frame.assign(**{self._label_col: np.asarray(labels)})
            transformed, baselines = preprocessing.apply_baseline_relative_per_group(
                labelled, group, mode, air_label=self.config.air_label, label_col=self._label_col,
            )
            self.sensor_baseline_ = self._mean_baselines(baselines)
            self._original_baseline_ = dict(self.sensor_baseline_)
            frame = transformed[list(self.ALL_SENSORS)]
            if not self.sensor_baseline_:
                mode = "none"
        self._log1p_applied = bool(self.config.use_log1p) and mode not in ("delta", "logratio")
        if self._log1p_applied:
            frame = preprocessing.log1p_resistances(frame)
        if self.config.snv:
            frame = preprocessing.snv_normalize(frame)
        self.scaler_r = self._build_scaler()
        return preprocessing.scale_resistances(frame, self.scaler_r, fit=True)

    def _calibration_folds(self, raw, labels, groups):
        """Return group-disjoint folds only when every side contains all classes."""
        if groups is None or len(np.unique(labels)) < 2:
            return []
        group_values = np.asarray(groups).astype(str)
        labels = np.asarray(labels)
        all_classes = set(labels)
        for count in range(min(3, len(np.unique(group_values))), 1, -1):
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=self.config.random_state)
            try:
                folds = list(splitter.split(raw, labels, group_values))
            except ValueError:
                continue
            if all(set(labels[fit]) == all_classes and set(labels[cal]) == all_classes for fit, cal in folds):
                return folds
        return []

    def _fit_estimator(self, raw, labels, groups=None, baseline_groups=None):
        """Fit final/holdout estimator; calibration never splits an exposure."""
        frame = self._fit_preprocessing(raw, labels, baseline_groups)
        features = self._model_features
        folds = self._calibration_folds(raw, labels, groups) if self.config.calibrated else []
        self.calibration_strategy_ = "group_disjoint_fold_local" if folds else "disabled_insufficient_groups" if self.config.calibrated else "disabled"
        self.calibration_fold_groups_ = []
        if folds:
            calibrated = []
            for fit_indices, cal_indices in folds:
                cfg = deepcopy(self.config)
                cfg.calibrated = False
                worker = type(self)(config=cfg)
                fit_baselines = np.asarray(baseline_groups)[fit_indices] if baseline_groups is not None else None
                fitted = worker._fit_preprocessing(raw.iloc[fit_indices], np.asarray(labels)[fit_indices], fit_baselines)
                fit_labels = pd.Series(np.asarray(labels)[fit_indices])
                if cfg.use_augmentation and cfg.n_augmentations > 0:
                    fitted = preprocessing.augment_resistances(fitted, cfg.n_augmentations, cfg.noise_max)
                    fit_labels = pd.concat([fit_labels] * (cfg.n_augmentations + 1), ignore_index=True)
                estimator = worker._build_model(y_fit=fit_labels.values)
                estimator.fit(fitted[features].values, fit_labels.values)
                worker.is_fitted = True
                calibration_values = worker.process_sensor_data(raw.iloc[cal_indices])[features].values
                calibrator = self._make_calibrator(estimator, method=cfg.calibration_method, cv="prefit")
                calibrator.fit(calibration_values, np.asarray(labels)[cal_indices])
                calibrated.append((worker, calibrator))
                group_array = np.asarray(groups).astype(str)
                self.calibration_fold_groups_.append({
                    "fit": sorted(set(group_array[fit_indices])),
                    "calibration": sorted(set(group_array[cal_indices])),
                })
            self.model = GroupCalibratedEstimator(calibrated, features, np.unique(labels))
        else:
            # Do not quietly replace independent-group calibration with row CV.
            was_calibrated = self.config.calibrated
            self.config.calibrated = False
            try:
                self.model = self._build_model(y_fit=np.asarray(labels))
            finally:
                self.config.calibrated = was_calibrated
            fit_labels = pd.Series(labels).reset_index(drop=True)
            fitted = frame
            if self.config.use_augmentation and self.config.n_augmentations > 0:
                fitted = preprocessing.augment_resistances(frame, self.config.n_augmentations, self.config.noise_max)
                fit_labels = pd.concat([fit_labels] * (self.config.n_augmentations + 1), ignore_index=True)
            self.model.fit(fitted[features].values, fit_labels.values)
        self.is_fitted = True
        return frame

    def train(
        self,
        X,
        y=None,
        use_augmentation: Optional[bool] = None,
        n_augmentations: Optional[int] = None,
        noise_std: Optional[float] = None,  # accepted for API compat
        test_size: Optional[float] = None,
        groups: Optional[pd.Series] = None,
        baseline_groups: Optional[pd.Series] = None,
    ) -> Optional[float]:
        """Fit-from-scratch on (X, y).

        Split raw rows first; learn cleaning, baselines and scaling on the
        training partition only. Calibration owns additional group-disjoint
        fold-local preprocessing. Refit deployment state after freezing metrics.

        ``groups`` is an optional per-row session / recording id. If provided,
        a `GroupShuffleSplit` is used so no single recording appears in both
        train and test — this is the only reliable way to measure
        generalization when the dataset is a time series of consecutive frames.
        Without usable independent groups, fit deployment state but return
        None: a random row split cannot estimate unseen-exposure accuracy.
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
        X_df = X_df.reset_index(drop=True)
        if len(X_df) != len(y_series):
            raise ValueError("Feature rows and labels must have the same length")
        for values in (groups, baseline_groups):
            if values is not None and len(values) != len(X_df):
                raise ValueError("Group identifiers must align with every feature row")

        # Remember the raw (but sanitized/ordered) training frame for online_update.
        self.last_training_data = pd.concat(
            [X_df.copy(), y_series.rename(self._label_col)], axis=1
        )

        # Split RAW rows before any medians, clipping, baselines or scaling are learned.
        X_train, X_test, y_train, y_test = self._split(
            X_df, y_series, groups=groups, test_size=test_size,
        )
        group_values = np.asarray(groups).astype(str) if groups is not None else None
        baseline_values = np.asarray(baseline_groups if baseline_groups is not None else groups) if baseline_groups is not None or groups is not None else None
        bal_acc = None
        self.per_class_metrics_ = {}
        self.confusion_matrix_ = None
        self.confusion_labels_ = []
        self.last_test_size_ = 0
        if len(X_test):
            train_indices = self._split_train_indices_
            self._fit_estimator(
                X_train, y_train,
                group_values[train_indices] if group_values is not None else None,
                baseline_values[train_indices] if baseline_values is not None else None,
            )
            self.classes_ = np.unique(y_train)
            y_pred = self.model.predict(self._model_input(X_test))
            acc = float(accuracy_score(y_test, y_pred))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            self._log_test_report(y_test, y_pred, acc, bal_acc)
        else:
            acc = None
        self.training_history["accuracy"].append(acc)
        self.training_history["balanced_accuracy"].append(bal_acc)

        # Freeze holdout diagnostics, then refit deployment preprocessing/model.
        X_all_s = self._fit_estimator(X_df, y_series, group_values, baseline_values)
        features = self._model_features
        self.feature_means_ = X_all_s[features].mean().to_dict()
        self.feature_stds_ = X_all_s[features].std(ddof=0).replace(0, 1e-9).to_dict()
        self.class_centroids_ = X_all_s.assign(_y=y_series.values).groupby("_y")[features].mean()
        self.classes_ = getattr(self.model, "classes_", np.unique(y_series))
        self.selected_features = list(features)
        self._compute_training_summaries()
        return bal_acc

    def _compute_training_summaries(self) -> None:
        """Cache compact per-class training summaries (counts + mean feature
        vectors) so they survive save→reload. Uses the retained training frame,
        which is present right after ``train()``/``online_update()``.
        """
        df = self.last_training_data
        if df is None or not len(df) or self._label_col not in getattr(df, "columns", []):
            self.class_distribution_ = {}
            self.class_examples_ = {}
            return
        try:
            counts = df[self._label_col].value_counts()
            self.class_distribution_ = {str(k): int(v) for k, v in counts.items()}
        except Exception:
            self.class_distribution_ = {}
        present = [s for s in self.ALL_SENSORS if s in df.columns]
        examples: Dict[str, List[float]] = {}
        if present:
            try:
                means = df.groupby(self._label_col)[present].mean()
                for lbl, row in means.iterrows():
                    examples[str(lbl)] = [round(float(row.get(s, 0.0)), 4) for s in self.ALL_SENSORS]
            except Exception:
                examples = {}
        self.class_examples_ = examples

    # ── split + reporting helpers ───────────────────────────────────────────
    def _split(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        groups: Optional[pd.Series],
        test_size: float,
    ):
        """Use an all-class group holdout, or explicitly decline evaluation."""
        if (
            groups is not None
            and self.config.group_shuffle_when_available
            and len(pd.Series(groups).unique()) >= 2
        ):
            n_groups = int(pd.Series(groups).nunique())
            # A 10% holdout can mean only one exposure and therefore omit most
            # odors. Reserve enough complete groups for all-class diagnostics
            # when the recording actually contains such a split.
            requested_groups = int(np.ceil(float(test_size) * n_groups))
            group_test_size = min(
                max(requested_groups, int(y_series.nunique())),
                n_groups - 1,
            )
            gss = GroupShuffleSplit(
                n_splits=64,
                test_size=group_test_size,
                random_state=self.config.random_state,
            )
            all_labels = set(y_series.astype(str))
            for train_idx, test_idx in gss.split(X_df, y_series, groups=groups):
                train_labels = set(y_series.iloc[train_idx].astype(str))
                test_labels = set(y_series.iloc[test_idx].astype(str))
                if train_labels == all_labels and test_labels == all_labels:
                    print("[train] using group-disjoint split with all classes in train and test")
                    self.validation_strategy_ = "group_disjoint_holdout"
                    self.validation_warning_ = None
                    self._split_train_indices_ = train_idx
                    self._split_test_indices_ = test_idx
                    return (
                        X_df.iloc[train_idx].reset_index(drop=True),
                        X_df.iloc[test_idx].reset_index(drop=True),
                        y_series.iloc[train_idx].reset_index(drop=True),
                        y_series.iloc[test_idx].reset_index(drop=True),
                    )
        self.validation_strategy_ = "not_estimable"
        self.validation_warning_ = (
            "Unseen-exposure accuracy is not estimable: no group-disjoint holdout "
            "contains every class in both train and test. Collect independent "
            "labelled exposures per class. No row-wise substitute is reported."
        )
        self._split_train_indices_ = np.arange(len(X_df))
        self._split_test_indices_ = np.asarray([], dtype=int)
        return X_df.reset_index(drop=True), X_df.iloc[:0].copy(), y_series.reset_index(drop=True), y_series.iloc[:0].copy()

    def _log_test_report(self, y_test, y_pred, acc: float, bal_acc: float) -> None:
        """Print confusion matrix + per-class precision/recall/F1 so we see
        what the model actually struggles with rather than one conflated
        accuracy number.

        Also stashes structured metrics on the classifier (``per_class_metrics_``,
        ``confusion_matrix_``, ``confusion_labels_``) so they can be surfaced
        via the API / UI without re-running the test set.
        """
        # Always render the complete learned label set. If a very small dataset
        # cannot place a class in the holdout, its row has support 0 instead of
        # silently disappearing from the confusion matrix.
        labels = sorted(
            set(str(v) for v in self.classes_)
            | set(str(v) for v in np.asarray(y_test))
            | set(str(v) for v in np.asarray(y_pred))
        )
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
        if getattr(self.model, "accepts_raw_features", False):
            self._sync_calibration_baselines()
            frame, _ = self._prepare_xy(X, None)
            return frame[self._model_features].values
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
            "classifier_backend": getattr(self, "backend_name", "balanced_rf"),
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
                "enabled": isinstance(self.model, (CalibratedClassifierCV, GroupCalibratedEstimator)),
                "method": self.config.calibration_method,
                "strategy": getattr(self, "calibration_strategy_", "legacy_or_unknown"),
            },
            # Drift-invariant feature representation this model uses (so the UI /
            # operator can see whether it's the absolute or a baseline-relative
            # model, and whether a live session baseline has been captured).
            "drift": {
                "baseline_mode": getattr(self.config, "baseline_mode", "none"),
                "snv": bool(getattr(self.config, "snv", False)),
                "baseline_captured": bool(self.live_baseline_captured_),
                "training_fallback_available": bool(self._original_baseline_),
            },
            # Per-class precision/recall/F1 from the most recent test split.
            # Empty dict if this classifier was loaded from a pre-metrics
            # artifact — the UI renders a "retrain to populate" hint in that
            # case.
            "per_class_metrics": dict(self.per_class_metrics_ or {}),
            "confusion_matrix": self.confusion_matrix_,
            "confusion_labels": list(self.confusion_labels_ or []),
            "last_test_size": int(getattr(self, "last_test_size_", 0) or 0),
            "validation": {
                "strategy": str(getattr(self, "validation_strategy_", "unknown")),
                "warning": getattr(self, "validation_warning_", None),
                "estimable": getattr(self, "validation_strategy_", "unknown") == "group_disjoint_holdout",
            },
            # Full-dataset class balance (not just the last test split's
            # support) so the UI can show how many samples back each class.
            "class_distribution": self._class_distribution(),
            # Prefer the live frame; fall back to the persisted per-class counts
            # so the total stays consistent with class_distribution after a
            # reload (which drops last_training_data).
            "total_training_samples": (
                int(len(self.last_training_data))
                if self.last_training_data is not None
                else int(sum(self.class_distribution_.values()))
            ),
        }

    def _class_distribution(self) -> Dict[str, int]:
        """Per-class training-sample counts.

        Prefers the cached summary (persisted, survives reload); falls back to
        recomputing from the retained training frame when present. Empty dict
        for legacy artifacts that carry neither.
        """
        if self.class_distribution_:
            return dict(self.class_distribution_)
        df = self.last_training_data
        if df is None or self._label_col not in getattr(df, "columns", []):
            return {}
        try:
            counts = df[self._label_col].value_counts()
            return {str(k): int(v) for k, v in counts.items()}
        except Exception:
            return {}

    def class_example_vectors(self) -> Dict[str, List[float]]:
        """Representative 22-feature vector per class (mean of training data),
        ordered to ``ALL_SENSORS``.

        Powers the UI's dynamic "Try: <class>" buttons so every known scent
        gets a one-click example. Prefers the cached summary (persisted); falls
        back to recomputing from the retained training frame. Empty when
        neither is available.
        """
        if self.class_examples_:
            return {k: list(v) for k, v in self.class_examples_.items()}
        df = self.last_training_data
        if df is None or not len(df) or self._label_col not in getattr(df, "columns", []):
            return {}
        present = [s for s in self.ALL_SENSORS if s in df.columns]
        if not present:
            return {}
        try:
            means = df.groupby(self._label_col)[present].mean()
        except Exception:
            return {}
        out: Dict[str, List[float]] = {}
        for lbl, row in means.iterrows():
            out[str(lbl)] = [round(float(row.get(s, 0.0)), 4) for s in self.ALL_SENSORS]
        return out

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
