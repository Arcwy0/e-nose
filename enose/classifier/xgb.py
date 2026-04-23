"""XGBoost-based odor classifier with windowing, exemplar replay, and drift EMA.

Alternative to BalancedRFClassifier. Better when:
    - You have ordered time-series recordings (sessions) and want temporal features.
    - You want online class addition without losing prior classes (exemplar replay).
    - You want drift correction from a clean-air baseline.

Conforms to SmellClassifierBase but note: predict() here accepts a window frame
(n_samples × 22). For single-sample streaming, use `predict_single` which buffers
until a window is ready.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("Install xgboost: pip install xgboost") from e

from enose.config import ALL_SENSORS, RESISTANCE_SENSORS

from .base import SmellClassifierBase
from .xgb_config import XGBClassifierConfig
from .xgb_features import (
    assign_sessions,
    create_windows,
    extract_window_features,
    remap_labels,
    resolve_device,
)

logger = logging.getLogger(__name__)


class XGBOdorClassifier(SmellClassifierBase):
    """Windowed XGBoost odor classifier with online learning + drift correction."""

    def __init__(self, config: Optional[XGBClassifierConfig] = None):
        self.config = config or XGBClassifierConfig()
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[XGBClassifier] = None
        self.exemplar_memory: Dict[int, deque] = {}
        self.sensor_baseline: Optional[np.ndarray] = None
        self._original_baseline: Optional[np.ndarray] = None
        self._inference_buffer: List[np.ndarray] = []
        self._label_map: Dict[int, int] = {}
        self._inverse_map: Dict[int, int] = {}

        # SmellClassifierBase attrs
        self.is_fitted: bool = False
        self.classes_: np.ndarray = np.array([])

    # ── SmellClassifierBase interface ───────────────────────────────────────
    def process_sensor_data(self, X) -> pd.DataFrame:
        """For XGB, "processing" is identity — windowing happens inside predict/train."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, list):
            return pd.DataFrame(X)
        raise TypeError(f"Unsupported X type: {type(X).__name__}")

    def train(
        self,
        X: Union[pd.DataFrame, Dict[str, Any]],
        y=None,
        use_augmentation: Optional[bool] = None,  # unused; XGB has no noise aug path
        n_augmentations: Optional[int] = None,
        noise_std: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> float:
        """Fit from a frame containing `Timestamp`, 22 sensor cols, and `Gas name`."""
        df = self.process_sensor_data(X)
        if "Gas name" not in df.columns and y is not None:
            df = df.copy()
            df["Gas name"] = list(y)
        metrics = self._train_from_frame(df)
        return float(metrics["accuracy"])

    def online_update(
        self,
        X: Union[pd.DataFrame, Dict[str, Any]],
        y,
        use_augmentation: Optional[bool] = None,
        n_augmentations: Optional[int] = None,
    ) -> bool:
        """Treat all provided samples as belonging to a single class (majority of y)."""
        df = self.process_sensor_data(X)
        y_list = list(y)
        if not y_list:
            return False
        class_name = max(set(y_list), key=y_list.count)
        self.learn_new_class(df, str(class_name))
        return True

    def predict(self, X) -> List[str]:
        """Classify one window. X is a (n_samples, 22) frame or array."""
        return [self._predict_window(X)["label"]]

    def predict_proba(self, X) -> np.ndarray:
        """Probabilities across known classes for one window input."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        samples = self._as_window_array(X)
        if self.sensor_baseline is not None:
            samples = np.array([self._apply_drift_correction(s) for s in samples])
        features = extract_window_features(samples, self.config).reshape(1, -1)
        return self.model.predict_proba(self.scaler.transform(features))

    def save_model(self, out_dir: str) -> str:
        self.save(out_dir)
        return str(Path(out_dir))

    @classmethod
    def load_model(cls, path: str) -> "XGBOdorClassifier":
        return cls.load(path)

    # ── Core training ───────────────────────────────────────────────────────
    def _train_from_frame(self, df: pd.DataFrame, temporal_split: bool = True) -> Dict[str, Any]:
        logger.info(f"Training on {len(df)} samples, {df['Gas name'].nunique()} classes")

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(sorted(df["Gas name"].unique()))
        df = df.copy()
        df["Gas label"] = self.label_encoder.transform(df["Gas name"])

        session_ids = assign_sessions(df) if "Timestamp" in df.columns else np.zeros(len(df), dtype=int)
        n_sessions = len(np.unique(session_ids))
        logger.info(f"Sessions: {n_sessions}")

        X, y, groups = create_windows(df, self.config, session_ids)
        logger.info(f"Windows: {len(X)}, feature dim: {X.shape[1]}")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        y_mapped, label_map = remap_labels(y)
        inverse_map = {v: k for k, v in label_map.items()}
        n_classes = len(label_map)

        weights = self._class_weights(y_mapped, n_classes)
        sample_weights = np.array([weights[int(v)] for v in y_mapped])

        train_idx, test_idx = self._pick_split(X_scaled, y_mapped, groups, n_sessions, temporal_split)
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
        w_train = sample_weights[train_idx]

        self.model = self._build_xgb(n_classes, n_estimators=self.config.n_estimators)
        eval_set = [(X_test, y_test)] if len(X_test) else None
        self.model.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_set, verbose=False)

        self._label_map = label_map
        self._inverse_map = inverse_map
        self.classes_ = np.array(self.label_encoder.classes_)
        self.is_fitted = True

        return self._evaluate(X_train, y_train, X_test, y_test, inverse_map)

    def _pick_split(self, X_scaled, y_mapped, groups, n_sessions: int, temporal_split: bool):
        if len(X_scaled) < 10:
            logger.info("Dataset too small for split — training on all windows")
            return np.arange(len(X_scaled)), np.array([], dtype=int)
        if temporal_split and n_sessions >= 4:
            gkf = GroupKFold(n_splits=min(5, n_sessions))
            return list(gkf.split(X_scaled, y_mapped, groups))[-1]
        split = int(0.8 * len(X_scaled))
        return np.arange(split), np.arange(split, len(X_scaled))

    def _class_weights(self, y_mapped, n_classes: int) -> Dict[int, float]:
        unique, counts = np.unique(y_mapped, return_counts=True)
        total = len(y_mapped)
        return {int(c): total / (n_classes * cnt) for c, cnt in zip(unique, counts)}

    def _build_xgb(self, n_classes: int, n_estimators: int, learning_rate: Optional[float] = None) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=learning_rate if learning_rate is not None else self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            device=resolve_device(self.config),
            random_state=42,
            verbosity=0,
        )

    def _evaluate(self, X_train, y_train, X_test, y_test, inverse_map) -> Dict[str, Any]:
        if len(X_test) == 0:
            y_pred_train = self.model.predict(X_train).astype(int).flatten()
            acc = float(np.mean(y_pred_train == y_train))
            return {"accuracy": acc, "per_class": {}, "n_train_windows": len(X_train),
                    "n_test_windows": 0, "n_classes": len(self.label_encoder.classes_),
                    "class_names": list(self.label_encoder.classes_), "confusion_matrix": []}

        y_pred = self.model.predict(X_test).astype(int).flatten()
        present = sorted(set(int(v) for v in y_test) | set(int(v) for v in y_pred))
        present_names = [self.label_encoder.inverse_transform([inverse_map[l]])[0] for l in present]
        report = classification_report(y_test, y_pred, labels=present, target_names=present_names,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=present)
        return {
            "accuracy": report.get("accuracy", 0.0),
            "per_class": {n: report[n] for n in self.label_encoder.classes_ if n in report},
            "n_train_windows": len(X_train),
            "n_test_windows": len(X_test),
            "n_classes": len(self.label_encoder.classes_),
            "class_names": list(self.label_encoder.classes_),
            "confusion_matrix": cm.tolist(),
        }

    # ── Inference (streaming) ──────────────────────────────────────────────
    def predict_single(self, sample: np.ndarray) -> Optional[Dict[str, Any]]:
        """Buffer raw 22-feature samples; emit a prediction once the window is full."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        if self.sensor_baseline is not None:
            sample = self._apply_drift_correction(sample)
        self._inference_buffer.append(sample)
        if len(self._inference_buffer) >= self.config.min_window_samples:
            window = np.array(self._inference_buffer[-self.config.window_size:])
            self._inference_buffer.clear()
            return self._predict_window(window)
        return None

    def clear_buffer(self) -> None:
        self._inference_buffer.clear()

    def _as_window_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[ALL_SENSORS].values
        if isinstance(X, np.ndarray):
            return X
        if isinstance(X, list):
            return np.array([[row[c] for c in ALL_SENSORS] for row in X])
        raise TypeError(f"Window input must be DataFrame/ndarray/list, got {type(X).__name__}")

    def _predict_window(self, X) -> Dict[str, Any]:
        samples = self._as_window_array(X)
        if self.sensor_baseline is not None:
            samples = np.array([self._apply_drift_correction(s) for s in samples])
        features = extract_window_features(samples, self.config).reshape(1, -1)
        probas = self.model.predict_proba(self.scaler.transform(features))[0]
        pred_mapped = int(np.argmax(probas))
        name, idx = self._decode(pred_mapped)
        return {
            "label": name,
            "label_idx": int(idx),
            "confidence": float(probas[pred_mapped]),
            "probabilities": {self._decode(i)[0]: float(p) for i, p in enumerate(probas)},
        }

    def _decode(self, mapped_idx: int):
        original_label = self._inverse_map[mapped_idx]
        name = self.label_encoder.inverse_transform([original_label])[0]
        return name, original_label

    # ── Online learning ────────────────────────────────────────────────────
    def learn_new_class(self, samples_df: pd.DataFrame, class_name: str) -> Dict[str, Any]:
        """Add a class from a small batch using exemplar replay to avoid catastrophic forgetting."""
        logger.info(f"Learning '{class_name}' from {len(samples_df)} samples")

        current = list(self.label_encoder.classes_) if self.label_encoder is not None else []
        if class_name not in current:
            current.append(class_name)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(sorted(current))
        new_label = int(self.label_encoder.transform([class_name])[0])

        new_raw = samples_df[ALL_SENSORS].values
        self.exemplar_memory.setdefault(
            new_label, deque(maxlen=self.config.exemplar_memory_per_class)
        )
        for row in new_raw:
            self.exemplar_memory[new_label].append(row)

        new_X, new_y = self._window_raw(new_raw, new_label)
        replay_X, replay_y = self._build_replay_dataset()

        parts_X = [replay_X] + ([np.array(new_X)] if new_X else [])
        parts_y = [replay_y] + ([np.array(new_y)] if new_y else [])
        all_X = np.vstack(parts_X)
        all_y = np.concatenate(parts_y)

        self.scaler = StandardScaler()
        all_X_scaled = self.scaler.fit_transform(all_X)

        all_y_mapped, label_map = remap_labels(all_y)
        inverse_map = {v: k for k, v in label_map.items()}
        n_classes = len(label_map)

        weights = self._class_weights(all_y_mapped, n_classes)
        sample_weights = np.array([weights[int(v)] for v in all_y_mapped])

        self.model = self._build_xgb(
            n_classes,
            n_estimators=self.config.n_estimators + self.config.online_retrain_epochs,
            learning_rate=self.config.learning_rate * 0.5,
        )
        self.model.fit(all_X_scaled, all_y_mapped, sample_weight=sample_weights, verbose=False)

        self._label_map = label_map
        self._inverse_map = inverse_map
        self.classes_ = np.array(self.label_encoder.classes_)
        self.is_fitted = True

        y_pred = self.model.predict(all_X_scaled).astype(int).flatten()
        acc = float(np.mean(y_pred == all_y_mapped))
        return {
            "class_name": class_name,
            "class_label": new_label,
            "new_samples": len(new_raw),
            "new_windows": len(new_X),
            "total_replay_windows": len(replay_X),
            "retrain_accuracy": acc,
            "all_classes": list(self.label_encoder.classes_),
        }

    def _window_raw(self, raw: np.ndarray, label: int):
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        start = 0
        while start + self.config.min_window_samples <= len(raw):
            end = min(start + self.config.window_size, len(raw))
            X_list.append(extract_window_features(raw[start:end], self.config))
            y_list.append(label)
            start += self.config.window_stride
        return X_list, y_list

    def _build_replay_dataset(self):
        X_list, y_list = [], []
        for label, samples in self.exemplar_memory.items():
            raw = np.array(list(samples))
            xs, ys = self._window_raw(raw, label)
            X_list.extend(xs); y_list.extend(ys)
        if not X_list:
            raise ValueError("Exemplar memory too small for replay.")
        return np.array(X_list), np.array(y_list)

    # ── Drift correction ───────────────────────────────────────────────────
    def update_drift_baseline(self, air_sample: np.ndarray) -> None:
        """EMA-update the resistance baseline from a clean-air reading. Shape: (17,)."""
        if self.sensor_baseline is None:
            self.sensor_baseline = air_sample.copy()
            self._original_baseline = air_sample.copy()
            return
        a = self.config.drift_ema_alpha
        self.sensor_baseline = a * air_sample + (1 - a) * self.sensor_baseline

    def _apply_drift_correction(self, sample: np.ndarray) -> np.ndarray:
        corrected = sample.copy()
        n_r = len(RESISTANCE_SENSORS)
        if self._original_baseline is not None and self.sensor_baseline is not None:
            ratio = np.where(self.sensor_baseline > 0,
                             self._original_baseline / self.sensor_baseline, 1.0)
            corrected[:n_r] = sample[:n_r] * ratio
        return corrected

    def _init_drift_baseline_from_training(self, df: pd.DataFrame) -> None:
        air_mask = df["Gas name"] == "air"
        if not air_mask.any():
            return
        air = df.loc[air_mask, RESISTANCE_SENSORS].values
        self.sensor_baseline = np.mean(air, axis=0)
        self._original_baseline = self.sensor_baseline.copy()
        logger.info("Drift baseline initialized from 'air' class")

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, model_dir: Optional[str] = None) -> None:
        out = Path(model_dir or self.config.model_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.config.save(str(out / "config.json"))
        self.model.save_model(str(out / "xgb_model.json"))
        state = {
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "exemplar_memory": {k: list(v) for k, v in self.exemplar_memory.items()},
            "sensor_baseline": self.sensor_baseline,
            "_original_baseline": self._original_baseline,
            "_label_map": self._label_map,
            "_inverse_map": self._inverse_map,
        }
        with open(out / "state.pkl", "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved XGB odor classifier → {out}")

    @classmethod
    def load(cls, model_dir: str) -> "XGBOdorClassifier":
        path = Path(model_dir)
        config = XGBClassifierConfig.load(str(path / "config.json"))
        obj = cls(config)
        obj.model = XGBClassifier()
        obj.model.load_model(str(path / "xgb_model.json"))
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        obj.scaler = state["scaler"]
        obj.label_encoder = state["label_encoder"]
        obj.exemplar_memory = {
            k: deque(v, maxlen=config.exemplar_memory_per_class)
            for k, v in state["exemplar_memory"].items()
        }
        obj.sensor_baseline = state["sensor_baseline"]
        obj._original_baseline = state.get("_original_baseline")
        obj._label_map = state["_label_map"]
        obj._inverse_map = state["_inverse_map"]
        obj.classes_ = np.array(obj.label_encoder.classes_) if obj.label_encoder else np.array([])
        obj.is_fitted = obj.model is not None
        logger.info(f"Loaded XGB odor classifier ← {path} | classes: {list(obj.classes_)}")
        return obj
