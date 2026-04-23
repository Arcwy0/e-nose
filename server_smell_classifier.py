# server_smell_classifier_c.py
# Production-ready smell classifier for 22-feature e-nose (R1-R17 + T,H,CO2,H2S,CH2O)
# - Offline training from CSV or DataFrame
# - "Online" updates by appending data and refitting (robustly supports adding new classes)
# - Environmental sensor sanity filtering with your specified ranges + medians
# - Data augmentation on R1–R17 with uniform noise in [-noise_max, +noise_max]
# - Handles class imbalance via Balanced Random Forest
# - Reports balanced accuracy and generates key visualizations

from __future__ import annotations

import os
import json
import time
import joblib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception as e:
    raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn") from e


@dataclass
class SmellClassifierConfig:
    model_type: str = "balanced_rf"   # accepted for compatibility; we use Balanced RF
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 400
    max_depth: Optional[int] = None
    n_jobs: int = -1
    # augmentation
    use_augmentation: bool = True
    n_augmentations: int = 3
    noise_max: float = 0.005  # uniform noise amplitude for R1-R17
    # environmental sanity ranges + fixed medians (per your spec)
    env_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "T":   (5.0, 30.0),
        "H":   (25.0, 65.0),
        "CO2": (300.0, 2500.0),
        "H2S": (0.0, float("inf")),
        "CH2O": (0.0, 200.0),
    })
    env_medians: Dict[str, float] = field(default_factory=lambda: {
        "T": 20.0,
        "H": 45.0,
        "CO2": 200.0,
        "H2S": 0.0,
        "CH2O": 10.0,
    })
    figure_dpi: int = 140
    n_estimators = 800
    calibrated: bool = True          # ⚙️ NEW: turn on probability calibration
    calibration_method: str = "sigmoid"  # "isotonic" or "sigmoid" (Platt)


class SmellClassifier:
    # Canonical feature lists
    RESISTANCE_SENSORS: List[str] = [f"R{i}" for i in range(1, 18)]
    ENVIRONMENTAL_SENSORS: List[str] = ["T", "H", "CO2", "H2S", "CH2O"]
    ALL_SENSORS: List[str] = RESISTANCE_SENSORS + ENVIRONMENTAL_SENSORS

    def __init__(self, model_type: str = "balanced_rf", online_learning: bool = True,
                 config: Optional[SmellClassifierConfig] = None):
        self.config = config or SmellClassifierConfig()
        # base model
        base = BalancedRandomForestClassifier(
        n_estimators=self.config.n_estimators,
        max_depth=self.config.max_depth,
        random_state=self.config.random_state,
        n_jobs=self.config.n_jobs,
)

        self.model = (
            self._make_calibrator(base, method=self.config.calibration_method, cv=3)
            if self.config.calibrated
            else base
        )

        # state
        self.scaler_r = StandardScaler()
        self.is_fitted: bool = False
        self.classes_: np.ndarray = np.array([])
        self.class_weights_: Dict[str, float] = {}
        self.selected_features: List[str] = list(self.ALL_SENSORS)
        self.original_features: List[str] = list(self.ALL_SENSORS)  # server may overwrite
        self.training_history: Dict[str, List[float]] = {"accuracy": [], "balanced_accuracy": []}
        self.last_training_data: Optional[pd.DataFrame] = None  # raw features + "Gas name"
        self._label_col: str = "Gas name"
        self._label_normalizer = lambda s: str(s).strip().lower()
        self._legacy_pipeline: bool = False         # True if we loaded an older full sklearn pipeline
        self._use_internal_scaler: bool = True      # False for legacy so we don't double-scale
        self.r_clip_: dict[str, tuple[float,float]] = {}  # per R-sensor (p1, p99)
        self.r_median_: dict[str, float] = {}
        self.feature_means_: Dict[str, float] = {}
        self.feature_stds_: Dict[str, float] = {}
        self.class_centroids_: Optional[pd.DataFrame] = None  # in scaled-R + raw-env space

    # ---------- utils ----------
    @staticmethod
    def _make_calibrator(base_clf, method="isotonic", cv=3):
        """
        Build CalibratedClassifierCV in a way that works on sklearn >=1.2 (estimator=)
        and older versions (base_estimator= fallback).
        """
        try:
            # sklearn >= 1.2 (documented in 1.7.2 API)
            return CalibratedClassifierCV(estimator=base_clf, method=method, cv=cv)
        except TypeError:
            # older sklearn (pre-1.2) used base_estimator=
            return CalibratedClassifierCV(base_estimator=base_clf, method=method, cv=cv)
    def _ensure_dataframe(self, X: Union[Dict[str, float], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("X must be a dict or pandas DataFrame")

    def _sanitize_environmentals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.ENVIRONMENTAL_SENSORS:
            if col not in df.columns:
                continue
            low, high = self.config.env_ranges[col]
            med = self.config.env_medians[col]
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where((vals >= low) & (vals <= high), med)
            if col == "H2S":
                vals = vals.clip(lower=0.0)
            df[col] = vals.fillna(med)
        return df

    def _order_and_fill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.ALL_SENSORS:
            if col not in df.columns:
                if col.startswith("R"):
                    df[col] = 0.0
                else:
                    df[col] = self.config.env_medians.get(col, 0.0)
        return df[self.ALL_SENSORS]

    def _scale_resistances(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        R = df[self.RESISTANCE_SENSORS].astype(float)
        if fit:
            R_scaled = self.scaler_r.fit_transform(R)
        else:
            R_scaled = self.scaler_r.transform(R)
        df[self.RESISTANCE_SENSORS] = R_scaled
        return df

    def process_sensor_data(self, X: Union[Dict[str, float], pd.DataFrame]) -> pd.DataFrame:
        """
        Used by the server for both single-sample predict and batch train.
        """
        df = self._ensure_dataframe(X)
        df = self._sanitize_environmentals(df)
        df = self._order_and_fill_features(df)
        df = self._clean_resistances(df)
        if self.is_fitted and self._use_internal_scaler and hasattr(self, "scaler_r") and hasattr(self.scaler_r, "mean_"):
            df = self._scale_resistances(df, fit=False)
        return df

    def _augment(self, X: pd.DataFrame, n_aug: int) -> pd.DataFrame:
        if n_aug <= 0 or not self.config.use_augmentation:
            return X
        X_aug = [X]
        for _ in range(n_aug):
            noise = np.random.uniform(
                -self.config.noise_max, self.config.noise_max,
                size=(len(X), len(self.RESISTANCE_SENSORS))
            )
            Xn = X.copy()
            Xn.loc[:, self.RESISTANCE_SENSORS] = X[self.RESISTANCE_SENSORS].values + noise
            X_aug.append(Xn)
        return pd.concat(X_aug, ignore_index=True)

    def _prepare_xy(self, X: Union[pd.DataFrame, Dict[str, Any]],
                    y: Optional[Union[pd.Series, List, np.ndarray]] = None):
        dfX = self._ensure_dataframe(X)
        # y can be inside X as "Gas name"
        if y is None and self._label_col in dfX.columns:
            y = dfX[self._label_col]
            dfX = dfX.drop(columns=[self._label_col], errors="ignore")
        if y is not None:
            y = pd.Series([self._label_normalizer(v) for v in list(y)])
        dfX = self._sanitize_environmentals(dfX)
        dfX = self._order_and_fill_features(dfX)
        return dfX, y

    # ---------- train / update / predict ----------
    def _clean_resistances(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.RESISTANCE_SENSORS:
            if c not in df: continue
            v = pd.to_numeric(df[c], errors="coerce").fillna(self.r_median_.get(c, 0.0))
            # replace exact zeros with median
            v = v.mask(v == 0.0, self.r_median_.get(c, 0.0))
            # winsorize to [p1,p99]
            lo, hi = self.r_clip_.get(c, (-np.inf, np.inf))
            v = v.clip(lower=lo, upper=hi)
            df[c] = v
        return df
    
    def train(self,
              X: Union[pd.DataFrame, Dict[str, Any]],
              y: Optional[Union[pd.Series, List, np.ndarray]] = None,
              use_augmentation: Optional[bool] = None,
              n_augmentations: Optional[int] = None,
              noise_std: Optional[float] = None,   # kept for API compatibility (ignored)
              test_size: Optional[float] = None) -> float:
        """
        Fit model from scratch. Returns balanced accuracy on a holdout set.
        """
        if use_augmentation is not None:
            self.config.use_augmentation = bool(use_augmentation)
        if n_augmentations is not None:
            self.config.n_augmentations = int(n_augmentations)
        if test_size is None:
            test_size = self.config.test_size

        X_df, y_series = self._prepare_xy(X, y)
        if y_series is None:
            raise ValueError("Labels required (column 'Gas name' or pass y).")

        # for reference in visualizations/analysis
        self.last_training_data = pd.concat([X_df.copy(), y_series.rename(self._label_col)], axis=1)

        # stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=test_size,
            random_state=self.config.random_state, stratify=y_series
        )

        # optional augmentation
        if self.config.use_augmentation and self.config.n_augmentations > 0:
            X_train = self._augment(X_train, self.config.n_augmentations)
            y_train = pd.concat([y_train] * (self.config.n_augmentations + 1), ignore_index=True)

        # fit scaler on train, transform both
        X_train_s = self._scale_resistances(X_train, fit=True)
        X_test_s  = self._scale_resistances(X_test,  fit=False)
        R = X_train_s[self.RESISTANCE_SENSORS]
        p1 = R.quantile(0.01)
        p99 = R.quantile(0.99)
        med = R.median()
        self.r_clip_ = {c: (float(p1[c]), float(p99[c])) for c in self.RESISTANCE_SENSORS}
        self.r_median_ = {c: float(med[c]) for c in self.RESISTANCE_SENSORS}

        # ⚙️ NEW: store stats & centroids for diagnostics
        self.feature_means_ = X_train_s[self.ALL_SENSORS].mean().to_dict()
        # avoid 0 std
        stds = X_train_s[self.ALL_SENSORS].std(ddof=0).replace(0, 1e-9)
        self.feature_stds_ = stds.to_dict()
        self.class_centroids_ = X_train_s.assign(_y=y_train.values).groupby("_y")[self.ALL_SENSORS].mean()

        # fit model
        if self.config.calibrated:
            base = BalancedRandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            self.model = self._make_calibrator(base, method=self.config.calibration_method, cv=3)
        else:
            self.model = BalancedRandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        self.model.fit(X_train_s[self.ALL_SENSORS].values, y_train.values)

        # evaluate
        y_pred = self.model.predict(X_test_s[self.ALL_SENSORS].values)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        # classes_
        self.classes_ = getattr(self.model, "classes_", np.unique(y_train.values))

        self.is_fitted = True
        # ensure classes_ order matches model
        self.classes_ = getattr(self.model, "classes_", np.unique(y_train.values))
        self.selected_features = list(self.ALL_SENSORS)
        self.training_history["accuracy"].append(float(acc))
        self.training_history["balanced_accuracy"].append(float(bal_acc))

        return float(bal_acc)
    
    def diagnose_sample(self, X: Union[pd.DataFrame, Dict[str, float]]) -> Dict[str, Any]:
        """
        Return per-feature z-scores, global OOD score, and nearest class centroids (L2)
        in the same feature space used by the model (scaled R + raw env).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        df = self.process_sensor_data(X)
        Xs = self._scale_resistances(df, fit=False)

        # per-feature z = (x - mean)/std (use train-set stats we cached)
        zs = {}
        for f in self.ALL_SENSORS:
            mu = self.feature_means_.get(f, 0.0)
            sd = max(self.feature_stds_.get(f, 1.0), 1e-9)
            zs[f] = float((Xs.iloc[0][f] - mu) / sd)

        # simple global OOD score
        ood = float(np.mean(np.abs([zs[f] for f in self.ALL_SENSORS])))

        # nearest centroid distances
        dists = {}
        if self.class_centroids_ is not None and len(self.class_centroids_) > 0:
            v = Xs[self.ALL_SENSORS].values.reshape(1, -1)
            for cls_name, row in self.class_centroids_.iterrows():
                c = row.values.reshape(1, -1)
                d = float(np.linalg.norm(v - c, ord=2))
                dists[str(cls_name)] = d

        # predicted probs (already calibrated if enabled)
        probs = self.predict_proba(X)[0].tolist()
        return {
            "z_scores": zs,
            "ood_score": ood,
            "nearest_centroid_L2": dict(sorted(dists.items(), key=lambda kv: kv[1])[:5]),
            "classes": list(map(str, self.classes_)),
            "probs": probs
        }
        
    def online_update(self,
                      X: Union[pd.DataFrame, Dict[str, Any]],
                      y: Union[pd.Series, List, np.ndarray],
                      use_augmentation: Optional[bool] = None,
                      n_augmentations: Optional[int] = None) -> bool:
        """
        Append data and refit (keeps a single, strong classifier for both modes).
        Safe with new classes because retraining learns the expanded label set.
        """
        if not self.is_fitted:
            _ = self.train(X, y, use_augmentation=use_augmentation, n_augmentations=n_augmentations)
            return True

        X_df, y_series = self._prepare_xy(X, y)
        if self.last_training_data is not None and len(self.last_training_data):
            new_all = pd.concat(
                [self.last_training_data, pd.concat([X_df, y_series.rename(self._label_col)], axis=1)],
                ignore_index=True
            )
        else:
            new_all = pd.concat([X_df, y_series.rename(self._label_col)], axis=1)

        _ = self.train(
            new_all.drop(columns=[self._label_col]),
            new_all[self._label_col],
            use_augmentation=use_augmentation,
            n_augmentations=n_augmentations
        )
        return True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        df = self.process_sensor_data(X)
        Xs = df if not self._use_internal_scaler else self._scale_resistances(df, fit=False)
        # Give legacy pipelines a DataFrame (keeps column names if their pipeline uses them)
        X_input = Xs[self.ALL_SENSORS] if self._legacy_pipeline else Xs[self.ALL_SENSORS].values
        preds = self.model.predict(X_input)
        return [str(p) for p in preds]

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        df = self.process_sensor_data(X)
        Xs = df if not self._use_internal_scaler else self._scale_resistances(df, fit=False)
        X_input = Xs[self.ALL_SENSORS] if self._legacy_pipeline else Xs[self.ALL_SENSORS].values

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_input)
        elif hasattr(self.model, "decision_function"):
            dfc = self.model.decision_function(X_input)
            # softmax fallback for decision_function
            dfc = np.array(dfc)
            if dfc.ndim == 1:
                # binary clf returns (n,), expand to (n,2)
                dfc = np.vstack([-dfc, dfc]).T
            expv = np.exp(dfc - dfc.max(axis=1, keepdims=True))
            proba = expv / expv.sum(axis=1, keepdims=True)
        else:
            # last resort: 1.0 on predicted class, 0 elsewhere
            preds = self.model.predict(X_input)
            classes = list(getattr(self.model, "classes_", self.classes_))
            if not classes:
                classes = sorted(set(preds))
            proba = np.zeros((len(preds), len(classes)), dtype=float)
            idx = {c: i for i, c in enumerate(classes)}
            for r, c in enumerate(preds):
                proba[r, idx[c]] = 1.0

        # align columns to self.classes_ if we have them
        if len(self.classes_) and hasattr(self.model, "classes_"):
            # map from model order -> self order
            model_classes = list(getattr(self.model, "classes_", self.classes_))
            order = [model_classes.index(c) for c in self.classes_ if c in model_classes]
            proba = proba[:, order] if len(order) == proba.shape[1] else proba
        return proba

    def predict_with_ood(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        df = self.process_sensor_data(X)
        Xs = self._scale_resistances(df, fit=False) if (self._use_internal_scaler and hasattr(self.scaler_r, "mean_")) else df
        proba = self.predict_proba(df)[0]  # calibrated if enabled
        label_idx = int(np.argmax(proba))
        label = str(self.classes_[label_idx])
        conf = float(proba[label_idx])

        # OOD score: mean |z| over all features (using train μ/σ you stored)
        z = []
        for f in self.ALL_SENSORS:
            mu = self.feature_means_.get(f, 0.0); sd = max(self.feature_stds_.get(f, 1.0), 1e-9)
            z.append(abs((Xs.iloc[0][f] - mu) / sd))
        ood_score = float(np.mean(z))

        # centroid distance check
        dmin = None
        if self.class_centroids_ is not None and len(self.class_centroids_):
            v = Xs[self.ALL_SENSORS].values.reshape(1,-1)
            dists = [np.linalg.norm(v - self.class_centroids_.loc[c][self.ALL_SENSORS].values.reshape(1,-1), ord=2)
                    for c in self.class_centroids_.index]
            dmin = float(np.min(dists))

        # simple gating: if very OOD, down-weight confidence or abstain
        if ood_score > 3.0 or (dmin is not None and dmin > 5.0):
            # damp confidence (mix with uniform)
            k = len(self.classes_)
            proba = 0.5 * proba + 0.5 * (np.ones_like(proba) / k)
            label_idx = int(np.argmax(proba))
            label = str(self.classes_[label_idx])
            conf = float(proba[label_idx])

        return label, conf, proba.tolist(), {"ood_score": ood_score, "min_centroid_L2": dmin}
    # ---------- persistence ----------

    def save_model(self, out_dir: str = "trained_models") -> str:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"smell_classifier_22f_{ts}.joblib")
        payload = {
            "model": self.model,
            "scaler_r": self.scaler_r,
            "classes_": self.classes_,
            "class_weights_": self.class_weights_,
            "selected_features": self.selected_features,
            "original_features": self.original_features,
            "training_history": self.training_history,
            "env_ranges": self.config.env_ranges,
            "env_medians": self.config.env_medians,
        }
        joblib.dump(payload, path)
        # manifest for quick inspection
        with open(os.path.join(out_dir, f"smell_classifier_22f_{ts}.json"), "w") as f:
            json.dump({
                "path": path,
                "n_features": len(self.ALL_SENSORS),
                "features": self.ALL_SENSORS,
                "classes": list(map(str, self.classes_)),
                "balanced_accuracy_last": self.training_history["balanced_accuracy"][-1]
                    if self.training_history["balanced_accuracy"] else None,
                "saved_at": ts
            }, f, indent=2)
        return path

    @classmethod
    def load_model(cls, path: str) -> "SmellClassifier":
        """
        Backward-compatible loader.
        - If `path` contains the NEW format (dict with model + scaler_r), restore as usual.
        - If it contains a LEGACY artifact (sklearn Pipeline or dict without scaler_r),
          load it, mark as legacy, and skip internal scaling in predict().
        """
        payload = joblib.load(path)

        obj = cls()
        obj.is_fitted = True
        obj._legacy_pipeline = False
        obj._use_internal_scaler = True

        # Case A: payload is a plain sklearn estimator/pipeline (legacy)
        if hasattr(payload, "predict"):
            obj.model = payload
            obj.classes_ = getattr(payload, "classes_", np.array([]))
            obj._legacy_pipeline = True
            obj._use_internal_scaler = False
            return obj

        # Case B: payload is a dict-like container
        if isinstance(payload, dict):
            # Model
            if "model" in payload and hasattr(payload["model"], "predict"):
                obj.model = payload["model"]
            else:
                # sometimes legacy dumps use 'pipeline' or another key
                pipe = None
                if "pipeline" in payload and hasattr(payload["pipeline"], "predict"):
                    pipe = payload["pipeline"]
                else:
                    for k, v in payload.items():
                        if hasattr(v, "predict"):
                            pipe = v
                            break
                if pipe is not None:
                    obj.model = pipe
                    obj._legacy_pipeline = True
                    obj._use_internal_scaler = False
                else:
                    raise ValueError("Unrecognized model payload: no estimator found")

            # Classes
            obj.classes_ = np.array(
                payload.get("classes_", getattr(obj.model, "classes_", np.array([])))
            )

            # Scaler
            if "scaler_r" in payload:
                obj.scaler_r = payload["scaler_r"]
                obj._use_internal_scaler = True
            elif "scaler" in payload:
                # Legacy scaler (likely inside a Pipeline anyway) — avoid double-scaling.
                obj.scaler_r = payload["scaler"]
                obj._use_internal_scaler = False
            else:
                # No saved scaler => assume pipeline handles scaling
                obj._use_internal_scaler = False

            # Other metadata (optional)
            obj.class_weights_ = payload.get("class_weights_", {})
            obj.selected_features = payload.get("selected_features", list(obj.ALL_SENSORS))
            obj.original_features = payload.get("original_features", list(obj.ALL_SENSORS))
            obj.training_history = payload.get("training_history", {"accuracy": [], "balanced_accuracy": []})
            env_ranges = payload.get("env_ranges")
            env_medians = payload.get("env_medians")
            if env_ranges: obj.config.env_ranges = env_ranges
            if env_medians: obj.config.env_medians = env_medians
            return obj

        # Unknown payload type
        raise ValueError(f"Unsupported model payload type: {type(payload)}")

    # ---------- visualizations & analysis ----------

    def _safe_plot_path(self, out_dir: str, name: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, name)

    def generate_visualizations(self, out_dir: str = "data") -> Dict[str, str]:
        """
        Generates confusion matrix, class counts, feature importances and env histograms.
        Returns dict of {plot_name: relative_path_from_out_dir}.
        """
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out = {}

        if self.last_training_data is None or not self.is_fitted:
            return out

        X = self.last_training_data[self.ALL_SENSORS]
        y = self.last_training_data[self._label_col]

        # fresh holdout for plots
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )
        X_trs = self._scale_resistances(X_tr, fit=True)
        X_tes = self._scale_resistances(X_te, fit=False)
        self.model.fit(X_trs[self.ALL_SENSORS].values, y_tr.values)
        y_pred = self.model.predict(X_tes[self.ALL_SENSORS].values)

        # confusion matrix
        cm = confusion_matrix(y_te, y_pred, labels=np.unique(y))
        fig = plt.figure(dpi=self.config.figure_dpi)
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick = np.arange(len(np.unique(y)))
        plt.xticks(tick, np.unique(y), rotation=45, ha="right")
        plt.yticks(tick, np.unique(y))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        fp = self._safe_plot_path(plots_dir, "confusion_matrix.png")
        fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
        out["confusion_matrix"] = os.path.relpath(fp, out_dir)

        # class counts
        fig = plt.figure(dpi=self.config.figure_dpi)
        counts = y.value_counts().sort_values(ascending=False)
        counts.plot(kind="bar")
        plt.ylabel("Samples")
        plt.title("Class Counts")
        plt.tight_layout()
        fp = self._safe_plot_path(plots_dir, "class_counts.png")
        fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
        out["class_counts"] = os.path.relpath(fp, out_dir)

        # feature importances
        if hasattr(self.model, "feature_importances_"):
            fig = plt.figure(dpi=self.config.figure_dpi)
            importances = self.model.feature_importances_
            idx = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[idx])
            plt.xticks(range(len(importances)), np.array(self.ALL_SENSORS)[idx], rotation=45, ha="right")
            plt.title("Feature Importances (Balanced RF)")
            plt.tight_layout()
            fp = self._safe_plot_path(plots_dir, "feature_importances.png")
            fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
            out["feature_importances"] = os.path.relpath(fp, out_dir)

        # environmental histograms
        fig = plt.figure(dpi=self.config.figure_dpi)
        for i, col in enumerate(self.ENVIRONMENTAL_SENSORS):
            plt.subplot(3, 2, i + 1)
            X[col].hist(bins=30)
            plt.title(col)
        plt.tight_layout()
        fp = self._safe_plot_path(plots_dir, "environmental_hist.png")
        fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
        out["environmental_hist"] = os.path.relpath(fp, out_dir)

        return out

    def analyze_data_quality(self) -> Dict[str, Any]:
        if self.last_training_data is None:
            return {"summary": "No training data available."}
        df = self.last_training_data.copy()
        return {
            "n_samples": int(len(df)),
            "classes": sorted(list(df["Gas name"].unique())),
            "class_counts": df["Gas name"].value_counts().to_dict(),
            "sensor_stats": df[self.ALL_SENSORS].describe().to_dict(),
            "env_ranges": self.config.env_ranges,
            "env_medians": self.config.env_medians,
        }

    def analyze_environmental_sensors(self) -> Dict[str, Any]:
        if self.last_training_data is None:
            return {"summary": "No training data available."}

        out = {}
        base = "data/plots"
        os.makedirs(base, exist_ok=True)
        df = self.last_training_data.copy()

        for col in self.ENVIRONMENTAL_SENSORS:
            fig = plt.figure(dpi=self.config.figure_dpi)
            classes = sorted(df["Gas name"].unique())
            data = [df[df["Gas name"] == c][col].values for c in classes]
            plt.boxplot(data, labels=classes, vert=True, showmeans=True)
            plt.title(f"{col} by class")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fp = self._safe_plot_path(base, f"{col}_by_class.png")
            fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
            out[f"{col}_by_class"] = os.path.relpath(fp, "data")

        out["summary"] = "Environmental sensor distributions saved."
        return out
