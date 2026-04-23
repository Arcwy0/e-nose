"""HTTP client for the FastAPI server. One method per endpoint, uniform (result, error) return."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

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
    def online_learning(self, sensor_data: List[Dict[str, float]], object_name: str) -> Result:
        try:
            if sensor_data and len(sensor_data[0]) != N_FEATURES:
                print(f"Warning: expected {N_FEATURES} features, got {len(sensor_data[0])}")
            payload = {
                "sensor_data": sensor_data,
                "labels": [object_name] * len(sensor_data),
            }
            r = requests.post(
                f"{self.base_url}/smell/online_learning",
                json=payload, timeout=30,
            )
            return self._handle_response(r)
        except requests.exceptions.Timeout:
            return None, "Online learning timeout"
        except Exception as e:
            return None, f"Online learning error: {e}"

    def learn_from_csv(
        self,
        csv_file_path: str,
        target_column: str = "Gas name",
        use_augmentation: bool = True,
        n_augmentations: int = 5,
        lowercase_labels: bool = True,
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
            }
            print(f"Sending CSV learning request: aug={use_augmentation}(n={n_augmentations})")
            r = requests.post(
                f"{self.base_url}/smell/learn_from_csv",
                json=payload, timeout=120,
            )
            return self._handle_response(r)
        except FileNotFoundError:
            return None, f"CSV file not found: {csv_file_path}"
        except requests.exceptions.Timeout:
            return None, "CSV learning timeout"
        except Exception as e:
            return None, f"CSV learning error: {e}"

    # ── Analytics ─────────────────────────────────────────────────────────
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
