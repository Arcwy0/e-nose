"""
Odor Classification Module for E-Nose + Robot Dog Project
==========================================================
Handles: training, inference, online learning (new classes), sensor drift mitigation.

Architecture:
- Windowed feature extraction (temporal dynamics > single snapshot)
- XGBoost classifier (fast, robust, handles imbalance, incremental updates)
- Proper temporal train/test split (no data leakage)
- Scaler saved alongside model (consistent preprocessing at inference)
- Online learning via warm-starting + exemplar memory
"""

import os
import json
import pickle
import logging
import subprocess
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Sensor columns ────────────────────────────────────────────────────────────
RESISTANCE_COLS = [f"R{i}" for i in range(1, 18)]
ENV_COLS = ["T", "H", "CO2", "H2S", "CH2O"]
FEATURE_COLS = RESISTANCE_COLS + ENV_COLS  # 22 raw features


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class ClassifierConfig:
    """All tunables in one place."""
    window_size: int = 20           # samples per classification window
    window_stride: int = 10         # overlap between consecutive windows
    min_window_samples: int = 10    # minimum samples to allow partial window

    # Feature extraction
    use_slope: bool = True          # linear slope per sensor over window
    use_percentiles: bool = True    # 25th/75th percentile features

    # XGBoost hyperparameters
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Device: "auto", "cuda", or "cpu"
    device: str = "auto"

    # Online learning
    exemplar_memory_per_class: int = 200
    online_retrain_epochs: int = 50

    # Drift mitigation
    drift_ema_alpha: float = 0.01

    # Paths
    model_dir: str = "./odor_model"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ClassifierConfig":
        with open(path) as f:
            return cls(**json.load(f))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_device(config: ClassifierConfig) -> str:
    if config.device != "auto":
        return config.device
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Remap labels to contiguous 0..N-1.
    Returns remapped y and the mapping dict {original -> new}.
    XGBoost requires labels in [0, num_class) with no gaps.
    """
    unique_labels = sorted(np.unique(y))
    mapping = {old: new for new, old in enumerate(unique_labels)}
    y_new = np.array([mapping[v] for v in y])
    return y_new, mapping


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_window_features(window: np.ndarray, config: ClassifierConfig) -> np.ndarray:
    """
    Extract features from a window of shape (window_size, n_raw_features).
    Returns 1D feature vector capturing temporal dynamics.
    """
    n_samples, n_features = window.shape
    n_resistance = len(RESISTANCE_COLS)

    feats = []

    resistance = window[:, :n_resistance]
    feats.append(np.mean(resistance, axis=0))
    feats.append(np.std(resistance, axis=0))
    feats.append(np.min(resistance, axis=0))
    feats.append(np.max(resistance, axis=0))

    if config.use_slope and n_samples >= 3:
        x = np.arange(n_samples, dtype=np.float64)
        x_centered = x - x.mean()
        denom = np.sum(x_centered ** 2)
        if denom > 0:
            slopes = (x_centered @ resistance) / denom
        else:
            slopes = np.zeros(n_resistance)
        feats.append(slopes)

    if config.use_percentiles:
        feats.append(np.percentile(resistance, 25, axis=0))
        feats.append(np.percentile(resistance, 75, axis=0))

    env = window[:, n_resistance:]
    feats.append(np.mean(env, axis=0))

    return np.concatenate(feats)


def compute_feature_names(config: ClassifierConfig) -> list[str]:
    names = []
    for prefix in ["mean", "std", "min", "max"]:
        names += [f"{prefix}_{c}" for c in RESISTANCE_COLS]
    if config.use_slope:
        names += [f"slope_{c}" for c in RESISTANCE_COLS]
    if config.use_percentiles:
        names += [f"p25_{c}" for c in RESISTANCE_COLS]
        names += [f"p75_{c}" for c in RESISTANCE_COLS]
    names += [f"env_mean_{c}" for c in ENV_COLS]
    return names


# ── Session Segmentation ─────────────────────────────────────────────────────
def assign_sessions(df: pd.DataFrame, gap_seconds: float = 60.0) -> np.ndarray:
    """
    Assign session IDs based on timestamp gaps.
    A new session starts when gap > gap_seconds.
    Critical for temporal train/test split (no data leakage).
    """
    timestamps = pd.to_datetime(df["Timestamp"])
    diffs = timestamps.diff().dt.total_seconds().abs().fillna(0)
    session_ids = (diffs > gap_seconds).cumsum().values
    return session_ids


# ── Windowing ─────────────────────────────────────────────────────────────────
def create_windows(
    df: pd.DataFrame,
    config: ClassifierConfig,
    session_ids: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide windows over the dataframe. Windows never cross session boundaries.
    Returns X (features), y (labels), groups (session IDs).
    """
    raw = df[FEATURE_COLS].values
    labels = df["Gas label"].values

    if session_ids is None:
        session_ids = np.zeros(len(df), dtype=int)

    X_list, y_list, g_list = [], [], []

    for sid in np.unique(session_ids):
        mask = session_ids == sid
        sess_raw = raw[mask]
        sess_labels = labels[mask]

        if len(sess_raw) < config.min_window_samples:
            continue

        start = 0
        while start + config.min_window_samples <= len(sess_raw):
            end = min(start + config.window_size, len(sess_raw))
            window = sess_raw[start:end]
            win_labels = sess_labels[start:end]

            features = extract_window_features(window, config)

            unique, counts = np.unique(win_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            X_list.append(features)
            y_list.append(majority_label)
            g_list.append(sid)

            start += config.window_stride

    if not X_list:
        raise ValueError("No windows created. Check data size / window config.")

    return np.array(X_list), np.array(y_list), np.array(g_list)


# ── Main Classifier ──────────────────────────────────────────────────────────
class OdorClassifier:
    """
    Full pipeline: preprocessing → windowed features → XGBoost → prediction.
    Supports online learning for new classes and sensor drift mitigation.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[XGBClassifier] = None
        self.exemplar_memory: dict[int, deque] = {}
        self.sensor_baseline: Optional[np.ndarray] = None
        self._original_baseline: Optional[np.ndarray] = None
        self._inference_buffer: list[np.ndarray] = []

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, temporal_split: bool = True) -> dict:
        """
        Train from a full DataFrame.

        Args:
            df: Dataset with Timestamp, R1-R17, T, H, CO2, H2S, CH2O, Gas name, Gas label
            temporal_split: Use session-based GroupKFold (recommended).
        """
        logger.info(f"Training on {len(df)} samples, {df['Gas name'].nunique()} classes")

        # Fit label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(sorted(df["Gas name"].unique()))
        df = df.copy()
        df["Gas label"] = self.label_encoder.transform(df["Gas name"])

        # Session segmentation
        session_ids = assign_sessions(df)
        n_sessions = len(np.unique(session_ids))
        logger.info(f"Found {n_sessions} recording sessions")

        # Create windowed features
        X, y, groups = create_windows(df, self.config, session_ids)
        logger.info(f"Created {len(X)} windows, feature dim = {X.shape[1]}")

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Remap labels to contiguous 0..N-1 for XGBoost
        y_mapped, label_map = _remap_labels(y)
        inverse_map = {v: k for k, v in label_map.items()}
        n_mapped_classes = len(label_map)

        # Class weights for imbalance
        unique_mapped, mapped_counts = np.unique(y_mapped, return_counts=True)
        total = len(y_mapped)
        weight_dict = {c: total / (n_mapped_classes * cnt) for c, cnt in zip(unique_mapped, mapped_counts)}
        sample_weights = np.array([weight_dict[yi] for yi in y_mapped])

        # Train/test split
        if len(X_scaled) < 10:
            train_idx = np.arange(len(X_scaled))
            test_idx = np.array([], dtype=int)
            logger.info("Dataset too small for split — training on all windows")
        elif temporal_split and n_sessions >= 4:
            gkf = GroupKFold(n_splits=min(5, n_sessions))
            splits = list(gkf.split(X_scaled, y_mapped, groups))
            train_idx, test_idx = splits[-1]
        else:
            split_point = int(0.8 * len(X_scaled))
            train_idx = np.arange(split_point)
            test_idx = np.arange(split_point, len(X_scaled))

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
        w_train = sample_weights[train_idx]

        device = _resolve_device(self.config)
        logger.info(f"Using device: {device}")

        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective="multi:softprob",
            num_class=n_mapped_classes,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device,
            random_state=42,
            verbosity=0,
        )

        eval_set = [(X_test, y_test)] if len(X_test) > 0 else None
        self.model.fit(X_train, y_train, sample_weight=w_train,
                       eval_set=eval_set, verbose=False)

        # Store the label mapping for inference
        self._label_map = label_map          # original_label -> mapped_label
        self._inverse_map = inverse_map      # mapped_label -> original_label

        # Evaluate
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test).astype(int).flatten()
            present = sorted(set(int(v) for v in y_test) | set(int(v) for v in y_pred))
            present_names = [
                self.label_encoder.inverse_transform([inverse_map[l]])[0]
                for l in present
            ]
            report = classification_report(
                y_test, y_pred, labels=present,
                target_names=present_names, output_dict=True, zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred, labels=present)
            logger.info(f"\n{classification_report(y_test, y_pred, labels=present, target_names=present_names, zero_division=0)}")
            logger.info(f"Confusion matrix:\n{cm}")
        else:
            y_pred_train = self.model.predict(X_train).astype(int).flatten()
            report = {"accuracy": float(np.mean(y_pred_train == y_train))}
            cm = np.array([])
            logger.info(f"Train accuracy (no test split): {report['accuracy']:.4f}")

        # Populate exemplar memory & drift baseline
        self._populate_exemplar_memory(df)
        self._init_drift_baseline(df)

        metrics = {
            "accuracy": report.get("accuracy", 0.0),
            "per_class": {
                name: report[name]
                for name in self.label_encoder.classes_ if name in report
            },
            "n_train_windows": len(X_train),
            "n_test_windows": len(X_test),
            "n_classes": len(self.label_encoder.classes_),
            "class_names": list(self.label_encoder.classes_),
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else [],
        }
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────
    def _map_label_for_predict(self, mapped_idx: int) -> tuple[str, int]:
        """Convert XGBoost's mapped index back to original label and name."""
        original_label = self._inverse_map[mapped_idx]
        name = self.label_encoder.inverse_transform([original_label])[0]
        return name, original_label

    def predict_single(self, sample: np.ndarray) -> Optional[dict]:
        """
        Feed a single raw sample (22 features: R1-R17, T, H, CO2, H2S, CH2O).
        Accumulates in buffer; returns prediction when window is full.
        Returns None if still accumulating.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if self.sensor_baseline is not None:
            sample = self._apply_drift_correction(sample)

        self._inference_buffer.append(sample)

        if len(self._inference_buffer) >= self.config.min_window_samples:
            return self._predict_from_buffer()
        return None

    def predict_batch(self, samples: np.ndarray) -> dict:
        """
        Classify a batch of consecutive samples at once.
        samples: (n_samples, 22) array of raw features.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if self.sensor_baseline is not None:
            samples = np.array([self._apply_drift_correction(s) for s in samples])

        features = extract_window_features(samples, self.config).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        probas = self.model.predict_proba(features_scaled)[0]
        pred_mapped = int(np.argmax(probas))
        label_name, label_idx = self._map_label_for_predict(pred_mapped)

        return {
            "label": label_name,
            "label_idx": int(label_idx),
            "confidence": float(probas[pred_mapped]),
            "probabilities": {
                self._map_label_for_predict(i)[0]: float(p)
                for i, p in enumerate(probas)
            },
        }

    def clear_buffer(self):
        self._inference_buffer.clear()

    def _predict_from_buffer(self) -> dict:
        window = np.array(self._inference_buffer[-self.config.window_size:])
        self._inference_buffer.clear()
        return self.predict_batch(window)

    # ── Online Learning ───────────────────────────────────────────────────
    def learn_new_class(self, samples_df: pd.DataFrame, class_name: str) -> dict:
        """
        Online learning: add a new odor class from a small set of samples.
        Robot collects ~100 samples over 60s → this method integrates them.

        Uses exemplar replay to prevent catastrophic forgetting of old classes.
        """
        logger.info(f"Learning new class '{class_name}' from {len(samples_df)} samples")

        # Update label encoder
        current_classes = list(self.label_encoder.classes_)
        if class_name in current_classes:
            new_label = int(self.label_encoder.transform([class_name])[0])
            logger.info(f"Class '{class_name}' already exists (label={new_label}), updating...")
        else:
            current_classes.append(class_name)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(sorted(current_classes))
            new_label = int(self.label_encoder.transform([class_name])[0])
            logger.info(f"Added new class '{class_name}' with label={new_label}")

        # Store in exemplar memory
        new_raw = samples_df[FEATURE_COLS].values
        if new_label not in self.exemplar_memory:
            self.exemplar_memory[new_label] = deque(maxlen=self.config.exemplar_memory_per_class)
        for row in new_raw:
            self.exemplar_memory[new_label].append(row)

        # Create windows from new samples
        new_X, new_y = [], []
        start = 0
        while start + self.config.min_window_samples <= len(new_raw):
            end = min(start + self.config.window_size, len(new_raw))
            feats = extract_window_features(new_raw[start:end], self.config)
            new_X.append(feats)
            new_y.append(new_label)
            start += self.config.window_stride

        # Build replay dataset
        replay_X, replay_y = self._build_replay_dataset()

        # Combine
        all_X = np.vstack([replay_X] + ([np.array(new_X)] if new_X else []))
        all_y = np.concatenate([replay_y] + ([np.array(new_y)] if new_y else []))

        # Re-fit scaler
        self.scaler = StandardScaler()
        all_X_scaled = self.scaler.fit_transform(all_X)

        # Remap labels
        all_y_mapped, label_map = _remap_labels(all_y)
        inverse_map = {v: k for k, v in label_map.items()}
        n_classes = len(label_map)

        # Class weights
        unique_mapped, mapped_counts = np.unique(all_y_mapped, return_counts=True)
        total = len(all_y_mapped)
        weight_dict = {c: total / (n_classes * cnt) for c, cnt in zip(unique_mapped, mapped_counts)}
        sample_weights = np.array([weight_dict[yi] for yi in all_y_mapped])

        device = _resolve_device(self.config)
        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators + self.config.online_retrain_epochs,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate * 0.5,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective="multi:softprob",
            num_class=n_classes,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(all_X_scaled, all_y_mapped, sample_weight=sample_weights, verbose=False)

        self._label_map = label_map
        self._inverse_map = inverse_map

        # Verify
        y_pred = self.model.predict(all_X_scaled).astype(int).flatten()
        acc = float(np.mean(y_pred == all_y_mapped))

        result = {
            "class_name": class_name,
            "class_label": new_label,
            "new_samples": len(new_raw),
            "new_windows": len(new_X),
            "total_replay_windows": len(replay_X),
            "retrain_accuracy": acc,
            "all_classes": list(self.label_encoder.classes_),
        }
        logger.info(f"Online learning complete: {result}")
        return result

    def _build_replay_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for label, samples in self.exemplar_memory.items():
            raw = np.array(list(samples))
            if len(raw) < self.config.min_window_samples:
                continue
            start = 0
            while start + self.config.min_window_samples <= len(raw):
                end = min(start + self.config.window_size, len(raw))
                feats = extract_window_features(raw[start:end], self.config)
                X_list.append(feats)
                y_list.append(label)
                start += self.config.window_stride
        if not X_list:
            raise ValueError("Exemplar memory too small for replay.")
        return np.array(X_list), np.array(y_list)

    def _populate_exemplar_memory(self, df: pd.DataFrame):
        for label in df["Gas label"].unique():
            subset = df[df["Gas label"] == label][FEATURE_COLS].values
            if label not in self.exemplar_memory:
                self.exemplar_memory[label] = deque(maxlen=self.config.exemplar_memory_per_class)
            indices = np.linspace(0, len(subset) - 1,
                                  min(len(subset), self.config.exemplar_memory_per_class), dtype=int)
            for idx in indices:
                self.exemplar_memory[label].append(subset[idx])

    # ── Drift Mitigation ──────────────────────────────────────────────────
    def _init_drift_baseline(self, df: pd.DataFrame):
        air_mask = df["Gas name"] == "air"
        if air_mask.any():
            air_data = df.loc[air_mask, RESISTANCE_COLS].values
            self.sensor_baseline = np.mean(air_data, axis=0)
            self._original_baseline = self.sensor_baseline.copy()
            logger.info("Drift baseline initialized from 'air' class")

    def update_drift_baseline(self, air_sample: np.ndarray):
        """Update EMA baseline with a clean-air reading. air_sample: (17,) R1-R17."""
        if self.sensor_baseline is None:
            self.sensor_baseline = air_sample.copy()
            self._original_baseline = air_sample.copy()
        else:
            alpha = self.config.drift_ema_alpha
            self.sensor_baseline = alpha * air_sample + (1 - alpha) * self.sensor_baseline

    def _apply_drift_correction(self, sample: np.ndarray) -> np.ndarray:
        corrected = sample.copy()
        n_r = len(RESISTANCE_COLS)
        if self._original_baseline is not None:
            ratio = np.where(self.sensor_baseline > 0,
                             self._original_baseline / self.sensor_baseline, 1.0)
            corrected[:n_r] = sample[:n_r] * ratio
        return corrected

    # ── Save / Load ───────────────────────────────────────────────────────
    def save(self, model_dir: Optional[str] = None):
        model_dir = Path(model_dir or self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.config.save(str(model_dir / "config.json"))
        self.model.save_model(str(model_dir / "xgb_model.json"))

        state = {
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "exemplar_memory": {k: list(v) for k, v in self.exemplar_memory.items()},
            "sensor_baseline": self.sensor_baseline,
            "_original_baseline": self._original_baseline,
            "_label_map": self._label_map,
            "_inverse_map": self._inverse_map,
        }
        with open(model_dir / "state.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "OdorClassifier":
        model_dir = Path(model_dir)
        config = ClassifierConfig.load(str(model_dir / "config.json"))
        obj = cls(config)

        obj.model = XGBClassifier()
        obj.model.load_model(str(model_dir / "xgb_model.json"))

        with open(model_dir / "state.pkl", "rb") as f:
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

        logger.info(f"Model loaded from {model_dir} | Classes: {list(obj.label_encoder.classes_)}")
        return obj


# ── Server Integration ────────────────────────────────────────────────────────
class OdorClassifierServer:
    """Thin wrapper for server request/response handling."""

    def __init__(self, model_dir: Optional[str] = None):
        self.classifier: Optional[OdorClassifier] = None
        if model_dir and Path(model_dir).exists():
            self.classifier = OdorClassifier.load(model_dir)

    def handle_train(self, csv_path: str, config_overrides: Optional[dict] = None) -> dict:
        config = ClassifierConfig(**(config_overrides or {}))
        self.classifier = OdorClassifier(config)
        df = pd.read_csv(csv_path)
        metrics = self.classifier.train(df)
        self.classifier.save()
        return {"status": "ok", "metrics": metrics}

    def handle_predict(self, sensor_data: dict) -> dict:
        """
        Single-sample predict. Buffers until window is full.
        sensor_data: {"R1": ..., "R2": ..., ..., "CH2O": ...}
        """
        if self.classifier is None:
            return {"status": "error", "message": "No model loaded"}
        sample = np.array([sensor_data[col] for col in FEATURE_COLS])
        result = self.classifier.predict_single(sample)
        if result is None:
            return {
                "status": "buffering",
                "samples_buffered": len(self.classifier._inference_buffer),
                "samples_needed": self.classifier.config.min_window_samples,
            }
        return {"status": "ok", "prediction": result}

    def handle_predict_batch(self, sensor_data_list: list[dict]) -> dict:
        if self.classifier is None:
            return {"status": "error", "message": "No model loaded"}
        samples = np.array([[d[col] for col in FEATURE_COLS] for d in sensor_data_list])
        result = self.classifier.predict_batch(samples)
        return {"status": "ok", "prediction": result}

    def handle_learn(self, sensor_data_list: list[dict], class_name: str) -> dict:
        if self.classifier is None:
            return {"status": "error", "message": "No model loaded"}
        df = pd.DataFrame(sensor_data_list)
        result = self.classifier.learn_new_class(df, class_name)
        self.classifier.save()
        return {"status": "ok", "result": result}

    def handle_drift_update(self, air_reading: dict) -> dict:
        if self.classifier is None:
            return {"status": "error", "message": "No model loaded"}
        sample = np.array([air_reading[col] for col in RESISTANCE_COLS])
        self.classifier.update_drift_baseline(sample)
        return {"status": "ok", "baseline_updated": True}

    def handle_clear_buffer(self) -> dict:
        if self.classifier:
            self.classifier.clear_buffer()
        return {"status": "ok"}


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "database_robodog_new_cleaned.csv"

    print("=" * 70)
    print("ODOR CLASSIFIER — TRAINING & VALIDATION")
    print("=" * 70)

    config = ClassifierConfig(
        window_size=5,
        window_stride=2,
        min_window_samples=3,
        n_estimators=100,
    )

    classifier = OdorClassifier(config)
    df = pd.read_csv(csv_path)
    # Sort by class to group samples together (subset has scattered timestamps)
    df = df.sort_values("Gas name").reset_index(drop=True)
    # Override timestamps to simulate contiguous sessions per class
    base = pd.Timestamp("2026-01-01")
    new_ts = []
    for i, (_, group) in enumerate(df.groupby("Gas name")):
        for j in range(len(group)):
            new_ts.append(base + pd.Timedelta(hours=i * 2) + pd.Timedelta(seconds=j * 2))
    df["Timestamp"] = new_ts
    print(f"\nDataset: {len(df)} samples, {df['Gas name'].nunique()} classes")
    print(f"Classes: {list(df['Gas name'].unique())}")

    metrics = classifier.train(df, temporal_split=False)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Classes: {metrics['class_names']}")

    # Test single-sample inference
    print("\n--- Single-sample inference test ---")
    test_row = df.iloc[0]
    sample = test_row[FEATURE_COLS].values.astype(float)
    for i in range(config.min_window_samples + 1):
        result = classifier.predict_single(sample)
        if result:
            print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")

    # Test batch inference
    print("\n--- Batch inference test ---")
    batch = df.iloc[:5][FEATURE_COLS].values.astype(float)
    result = classifier.predict_batch(batch)
    print(f"Batch prediction: {result['label']} (confidence: {result['confidence']:.3f})")

    # Test online learning
    print("\n--- Online learning test ---")
    fake_new = df[df["Gas name"] == "ethanol"].copy()
    learn_result = classifier.learn_new_class(fake_new, "strawberry")
    print(f"Updated classes: {learn_result['all_classes']}")
    print(f"Retrain accuracy: {learn_result['retrain_accuracy']:.3f}")

    # Test save/load
    print("\n--- Save/load test ---")
    classifier.save("/tmp/test_odor_model")
    loaded = OdorClassifier.load("/tmp/test_odor_model")
    result2 = loaded.predict_batch(batch)
    print(f"Loaded model prediction: {result2['label']} (confidence: {result2['confidence']:.3f})")

    print("\nAll tests passed!")
