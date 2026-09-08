"""HTTP client for the FastAPI server. One method per endpoint, uniform (result, error) return."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from enose.config import ALL_SENSORS, DEFAULT_SERVER_URL, N_FEATURES


Result = Tuple[Optional[Any], Optional[str]]


class ServerAPI:
    """Thin HTTP wrapper. Every call returns (payload, error) — exactly one is None."""

    def __init__(self, base_url: str = DEFAULT_SERVER_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _handle_response(self, response: requests.Response) -> Result:
        try:
            response.raise_for_status()
            try:
                return response.json(), None
            except json.JSONDecodeError:
                return response.text, None
        except requests.exceptions.HTTPError:
            try:
                detail = response.json()
                if isinstance(detail, dict) and "detail" in detail:
                    return None, f"HTTP {response.status_code}: {detail['detail']}"
                return None, f"HTTP {response.status_code}: {detail}"
            except Exception:
                return None, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return None, f"Response error: {e}"

    # ── Health / info ─────────────────────────────────────────────────────
    def test_connection(self) -> Result:
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            result, error = self._handle_response(r)
            if error:
                return None, error
            if isinstance(result, dict):
                cfg = result.get("sensor_configuration", {})
                total = cfg.get("total_features", 0)
                if total == N_FEATURES:
                    print(f"Server confirmed {N_FEATURES}-feature support")
                else:
                    print(f"Warning: server reports {total} features, expected {N_FEATURES}")
            return result, None
        except requests.exceptions.Timeout:
            return None, "Connection timeout"
        except Exception as e:
            return None, f"Connection error: {e}"

    def get_model_info(self) -> Result:
        try:
            r = requests.get(f"{self.base_url}/smell/model_info", timeout=5)
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Model info timeout"
        except Exception as e:
            return None, f"Model info error: {e}"

    def set_baseline(self, air_samples: List[Dict[str, float]], ema: bool = False) -> Result:
        """Register this session's clean-air baseline for drift-robust models.

        Call once at session start with ~30–60 s of clean-air readings. No-op
        server-side when the model was trained in absolute mode. See
        ``enose.server.schemas.BaselineData``.
        """
        try:
            r = requests.post(
                f"{self.base_url}/smell/baseline",
                json={"sensor_data": air_samples, "ema": bool(ema)},
                timeout=15,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Baseline timeout"
        except Exception as e:
            return None, f"Baseline error: {e}"

    def set_baseline_from_live(self, window: float = 60.0) -> Result:
        """Capture a baseline from the server's recent stable live buffer."""
        try:
            r = requests.post(
                f"{self.base_url}/smell/baseline/live",
                params={"window": window},
                timeout=15,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Live baseline timeout"
        except Exception as e:
            return None, f"Live baseline error: {e}"

    def classify_stable_window(self, window: float = 60.0) -> Result:
        """Classify a stable median window from the server live buffer."""
        try:
            r = requests.post(
                f"{self.base_url}/smell/classify_stable",
                params={"window": window},
                timeout=15,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Stable-window classification timeout"
        except Exception as e:
            return None, f"Stable-window classification error: {e}"

    def classify_fast_window(self, window: float = 15.0) -> Result:
        """Average short live-window probabilities for a low-latency result."""
        try:
            r = requests.post(
                f"{self.base_url}/smell/classify_window",
                params={"window": window},
                timeout=15,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Fast-window classification timeout"
        except Exception as e:
            return None, f"Fast-window classification error: {e}"

    def recovery_status(self, window: float = 15.0) -> Result:
        """Check whether the live array has recovered to the session baseline."""
        try:
            r = requests.get(
                f"{self.base_url}/smell/recovery",
                params={"window": window},
                timeout=15,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Recovery check timeout"
        except Exception as e:
            return None, f"Recovery check error: {e}"

    # ── Vision ────────────────────────────────────────────────────────────
    def detect_object(self, image_path: str, object_name: str) -> Result:
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                data = {"text": f"Find {object_name}"}
                r = requests.post(
                    f"{self.base_url}/predict/object",
                    files=files, data=data, timeout=30,
                )
            return self._handle_response(r)
        except FileNotFoundError:
            return None, f"Image not found: {image_path}"
        except requests.exceptions.Timeout:
            return None, "Detection timeout"
        except Exception as e:
            return None, f"Detection error: {e}"

    def detect_scene(self, image_path: str, labels: Sequence[str]) -> Result:
        """Ground a list of candidate labels in one Florence-2 call.

        Used by the robot mission policy at each SEARCH waypoint to score all
        candidate targets without paying N separate round-trips. Returns the
        ``{labels, detections, image_size, ...}`` payload from
        ``POST /predict/scene``.
        """
        if not labels:
            return None, "labels list cannot be empty"
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                data = {"labels": json.dumps(list(labels))}
                r = requests.post(
                    f"{self.base_url}/predict/scene",
                    files=files, data=data, timeout=60,
                )
            return self._handle_response(r)
        except FileNotFoundError:
            return None, f"Image not found: {image_path}"
        except requests.exceptions.Timeout:
            return None, "Scene detection timeout"
        except Exception as e:
            return None, f"Scene detection error: {e}"

    # ── Classification ────────────────────────────────────────────────────
    def classify_smell(self, sensor_data: Dict[str, float]) -> Result:
        try:
            if len(sensor_data) != N_FEATURES:
                print(f"Warning: expected {N_FEATURES} features, got {len(sensor_data)}")
            r = requests.post(f"{self.base_url}/smell/classify", json=sensor_data, timeout=5)
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Classification timeout"
        except Exception as e:
            return None, f"Classification error: {e}"

    def test_console_input(self, sensor_values: str) -> Result:
        try:
            values = [float(x.strip()) for x in sensor_values.split(",")]
            if len(values) not in (17, N_FEATURES):
                return None, (
                    f"Expected 17 or {N_FEATURES} values, got {len(values)}. "
                    "Format: R1,...,R17 OR R1,...,R17,T,H,CO2,H2S,CH2O"
                )
        except ValueError as e:
            return None, f"Invalid number format: {e}"
        try:
            r = requests.post(
                f"{self.base_url}/smell/test_console",
                json={"values": sensor_values}, timeout=10,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Console test timeout"
        except Exception as e:
            return None, f"Console test error: {e}"

    # ── Training ──────────────────────────────────────────────────────────
    def online_learning(
        self,
        sensor_data: List[Dict[str, float]],
        object_name: str,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """Commit a labelled batch. ``provenance`` is an optional audit-trail
        dict matching the server's ``CommitProvenance`` schema — see
        ``enose.server.schemas.CommitProvenance`` and the robot policy's
        ``_commit`` for the producer side."""
        try:
            if sensor_data and len(sensor_data[0]) != N_FEATURES:
                print(f"Warning: expected {N_FEATURES} features, got {len(sensor_data[0])}")
            payload: Dict[str, Any] = {
                "sensor_data": sensor_data,
                "labels": [object_name] * len(sensor_data),
            }
            if provenance is not None:
                payload["provenance"] = provenance
            r = requests.post(
                f"{self.base_url}/smell/online_learning",
                json=payload, timeout=30,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Online learning timeout"
        except Exception as e:
            return None, f"Online learning error: {e}"

    def list_provenance(self, limit: int = 100, label: Optional[str] = None) -> Result:
        """List recent autonomous-training commits. Used by the audit tool."""
        try:
            params: Dict[str, Any] = {"limit": limit}
            if label:
                params["label"] = label
            r = requests.get(
                f"{self.base_url}/smell/provenance", params=params, timeout=10,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Provenance list timeout"
        except Exception as e:
            return None, f"Provenance list error: {e}"

    def learn_from_csv(
        self,
        csv_file_path: str,
        target_column: str = "Gas name",
        use_augmentation: bool = True,
        n_augmentations: int = 5,
        lowercase_labels: bool = True,
        merge_history: bool = True,
        training_profile: str = "raw",
        classifier_backend: str = "balanced_rf",
        baseline_mode: str = "none",
        snv: bool = False,
    ) -> Result:
        try:
            df = pd.read_csv(csv_file_path)
            print(f"CSV: {len(df)} rows, {len(df.columns)} cols")
            if target_column not in df.columns:
                return None, f"Target column '{target_column}' not in {df.columns.tolist()}"
            missing = [s for s in ALL_SENSORS if s not in df.columns]
            if missing:
                print(f"Warning: missing sensors in CSV: {missing} (server fills defaults)")
        except Exception as e:
            return None, f"CSV validation failed: {e}"

        try:
            with open(csv_file_path, "r") as f:
                csv_content = f.read()
            payload = {
                "csv_data": csv_content,
                "target_column": target_column,
                "use_augmentation": use_augmentation,
                "n_augmentations": n_augmentations,
                "noise_std": 0.0015,
                "lowercase_labels": lowercase_labels,
                "merge_history": merge_history,
                "training_profile": training_profile,
                "classifier_backend": classifier_backend,
                "baseline_mode": baseline_mode,
                "snv": snv,
            }
            mode = "merge history" if merge_history else "replace history"
            print(f"Sending CSV learning request: {mode}, aug={use_augmentation}(n={n_augmentations})")
            r = requests.post(
                f"{self.base_url}/smell/learn_from_csv",
                json=payload, timeout=600,
            )
            return self._handle_response(r)
        except FileNotFoundError:
            return None, f"CSV file not found: {csv_file_path}"
        except requests.exceptions.Timeout:
            return None, "CSV learning timeout"
        except Exception as e:
            return None, f"CSV learning error: {e}"

    # ── Analytics ─────────────────────────────────────────────────────────
    def get_settling(
        self,
        window: float = 8.0,
        threshold: float = 0.02,
        session_id: Optional[str] = None,
    ) -> Result:
        """Poll ``/smell/settling`` — see ``enose.server.routes.analytics`` for the schema.

        Used by the robot's SETTLE state to decide when the sensor array
        has stabilized.
        """
        try:
            params: Dict[str, Any] = {"window": window, "threshold": threshold}
            if session_id:
                params["session_id"] = session_id
            r = requests.get(f"{self.base_url}/smell/settling", params=params, timeout=5)
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Settling check timeout"
        except Exception as e:
            return None, f"Settling check error: {e}"

    def visualize_data(self) -> Result:
        try:
            r = requests.get(f"{self.base_url}/smell/visualize_data", timeout=30)
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Visualization timeout"
        except Exception as e:
            return None, f"Visualization error: {e}"

    def analyze_data_quality(self) -> Result:
        try:
            r = requests.get(f"{self.base_url}/smell/analyze_data", timeout=30)
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Analysis timeout"
        except Exception as e:
            return None, f"Analysis error: {e}"

    def environmental_analysis(self) -> Result:
        try:
            r = requests.get(f"{self.base_url}/smell/environmental_analysis", timeout=30)
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Environmental analysis timeout"
        except Exception as e:
            return None, f"Environmental analysis error: {e}"
