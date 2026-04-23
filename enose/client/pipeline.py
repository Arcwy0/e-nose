"""Interactive training pipeline: capture → detect → record 22-feature sample → learn."""

from __future__ import annotations

import datetime
import os
import time
from typing import Dict, List

import pandas as pd

from enose.config import (
    DEFAULT_RECORDING_TIME,
    DEFAULT_TARGET_SAMPLES,
    ENVIRONMENTAL_SENSORS,
    N_FEATURES,
    RESISTANCE_SENSORS,
    TEMP_IMAGE_PATH,
)

from .api import ServerAPI
from .sensors import ENoseSensor
from .webcam import WebcamHandler


class TrainingPipeline:
    """Orchestrates webcam + server + e-nose for training / inference / analysis flows."""

    def __init__(
        self,
        server_api: ServerAPI,
        webcam_handler: WebcamHandler,
        enose_sensor: ENoseSensor,
        recording_time: int = DEFAULT_RECORDING_TIME,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
        local_log_path: str = "training_data_22features_log.csv",
    ) -> None:
        self.server_api = server_api
        self.webcam_handler = webcam_handler
        self.enose_sensor = enose_sensor
        self.recording_time = recording_time
        self.target_samples = target_samples
        self.local_data_path = local_log_path

    # ── Local data logging ────────────────────────────────────────────────
    def save_local_data(self, sensor_data: List[Dict[str, float]], object_name: str) -> None:
        if not sensor_data:
            return
        try:
            df = pd.DataFrame(sensor_data)
            df["smell_label"] = object_name
            df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = not os.path.isfile(self.local_data_path)
            df.to_csv(self.local_data_path, mode="a", header=header, index=False)
            mode = "simulated" if self.enose_sensor.offline_mode else "hardware"
            print(f"Logged {len(sensor_data)} {mode} samples → {self.local_data_path}")
        except Exception as e:
            print(f"Local save error: {e}")

    def display_22_feature_sample(self, sample: Dict[str, float]) -> None:
        print("22-Feature Sensor Reading:")
        print("  Resistance (R1–R17):")
        active = {k: v for k, v in sample.items() if k in RESISTANCE_SENSORS and v > 0.01}
        if active:
            items = list(active.items())
            for i in range(0, len(items), 5):
                row = ", ".join(f"{k}={v:.3f}" for k, v in items[i:i + 5])
                print(f"    {row}")
        else:
            print("    (no significant activity)")

        print("  Environmental:")
        env = {k: sample.get(k, 0) for k in ENVIRONMENTAL_SENSORS}
        print("    " + ", ".join(f"{k}={v:.1f}" for k, v in env.items()))

    def display_learning_feedback(self, result) -> None:
        if not result:
            return
        print("\nLearning complete.")
        print(f"  update_type      : {result.get('update_type', '?')}")
        print(f"  samples          : {result.get('samples_processed', '?')}")
        acc = result.get("current_accuracy")
        if isinstance(acc, (int, float)):
            print(f"  accuracy         : {acc:.4f}")

        fb = result.get("feature_breakdown", {})
        if fb:
            print(f"  features         : {fb.get('resistance_sensors', 0)} R + "
                  f"{fb.get('environmental_sensors', 0)} env = {fb.get('total_features', 0)}")

        if result.get("update_type") == "retrain_for_consistency":
            print("  note             : model inconsistency was resolved")
        elif result.get("new_classes_detected"):
            print("  note             : new classes added")
        if result.get("model_reloaded"):
            print("  model reloaded   : yes")

        classes = result.get("classes", [])
        print(f"  classes ({len(classes)})    : {', '.join(classes)}")
        if result.get("visualizations"):
            print(f"  visualizations   : {len(result['visualizations'])} generated")

    # ── Main flows ────────────────────────────────────────────────────────
    def run_training_cycle(self) -> bool:
        """Full pipeline: capture image → detect → confirm → record → learn."""
        mode = "SIMULATED" if self.enose_sensor.offline_mode else "HARDWARE"
        print("\n" + "=" * 70)
        print(f"TRAINING CYCLE ({mode})")
        print("=" * 70)

        try:
            print("Step 1: capture image")
            _, image_path = self.webcam_handler.capture_image()
            if not image_path:
                print("Image capture failed")
                return False

            object_name = input("Enter object name: ").strip()
            if not object_name:
                print("No object name")
                return False

            if self.enose_sensor.offline_mode:
                self.enose_sensor.set_simulation_smell(object_name)

            print(f"Step 2: detecting '{object_name}'")
            detection, err = self.server_api.detect_object(image_path, object_name)
            if err:
                print(f"Detection failed: {err}")
                return False
            self.webcam_handler.display_detection_result(image_path, detection)

            print(f"Step 3: record {self.target_samples} samples in {self.recording_time}s")
            if not self.enose_sensor.offline_mode:
                print(f"  Position sensors near '{object_name}' and let them settle (~2 min)")
                print("  Do NOT move the sensors during recording")

            if not self.enose_sensor.prepare_recording():
                return False

            while True:
                ans = input(f"Start recording for '{object_name}'? (yes/no): ").lower().strip()
                if ans in ("yes", "y"):
                    if not self.enose_sensor.start_recording_after_confirmation(
                        self.recording_time, self.target_samples
                    ):
                        return False
                    break
                if ans in ("no", "n"):
                    print("Cancelled")
                    return False

            print("Recording in progress…")
            time.sleep(self.recording_time + 2)
            sensor_data = self.enose_sensor.stop_recording()
            if not sensor_data:
                print("No data recorded")
                return False

            self.save_local_data(sensor_data, object_name)
            avg = self.enose_sensor.get_average_reading(sensor_data)
            if avg:
                self.display_22_feature_sample(avg)

            print("Step 4: online learning")
            result, err = self.server_api.online_learning(sensor_data, object_name)
            if err:
                print(f"Online learning failed: {err}")
                return False
            self.display_learning_feedback(result)
            return True

        except Exception as e:
            print(f"Training cycle error: {e}")
            return False
        finally:
            _remove_if_exists(TEMP_IMAGE_PATH)

    def run_vision_only(self) -> bool:
        print("\n" + "=" * 50)
        print("VISION-ONLY DETECTION")
        print("=" * 50)
        try:
            _, image_path = self.webcam_handler.capture_image()
            if not image_path:
                return False
            object_name = input("Enter object to detect: ").strip()
            if not object_name:
                return False
            detection, err = self.server_api.detect_object(image_path, object_name)
            if err:
                print(f"Detection failed: {err}")
                return False
            self.webcam_handler.display_detection_result(image_path, detection)
            return True
        except Exception as e:
            print(f"Vision cycle error: {e}")
            return False
        finally:
            _remove_if_exists(TEMP_IMAGE_PATH)

    def identify_current_smell(self) -> bool:
        """Single sensor read → classify."""
        mode = "SIMULATED" if self.enose_sensor.offline_mode else "HARDWARE"
        print("\n" + "=" * 60)
        print(f"SMELL IDENTIFICATION ({mode})")
        print("=" * 60)
        try:
            if self.enose_sensor.offline_mode:
                smell_name = input("Simulation smell (default 'air'): ").strip() or "air"
                self.enose_sensor.set_simulation_smell(smell_name)

            reading = self.enose_sensor.read_single_measurement()
            if not reading:
                print("Failed to read sensor data")
                return False
            if len(reading) != N_FEATURES:
                print(f"Warning: expected {N_FEATURES} features, got {len(reading)}")

            self.display_22_feature_sample(reading)
            result, err = self.server_api.classify_smell(reading)
            if err:
                print(f"Classification failed: {err}")
                return False

            print(f"\nPredicted: {result.get('predicted_smell', '?')}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            probs = result.get("probabilities", {})
            if probs:
                print("\nProbabilities:")
                for smell, prob in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
                    bar = "█" * int(prob * 20)
                    print(f"  {smell:<15} {prob:.3f} {bar}")
            return True
        except Exception as e:
            print(f"Identification error: {e}")
            return False

    def test_manual_22_feature_input(self) -> None:
        """Manual comma-separated entry → classify. 'generate' makes a sample."""
        print("\n" + "=" * 60)
        print("MANUAL SENSOR INPUT")
        print("=" * 60)
        print("Format: R1,...,R17 (17 values) OR R1,...,R17,T,H,CO2,H2S,CH2O (22 values)")
        print("Commands: 'generate' (synth sample), 'back' (exit)")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nValues (or 'generate'): ").strip()
                if user_input.lower() == "back":
                    return
                if user_input.lower() == "generate":
                    smell_name = input("Smell name (default 'coffee'): ").strip() or "coffee"
                    sample = self.enose_sensor.generate_realistic_sensor_data(smell_name)
                    values = [sample[f"R{i}"] for i in range(1, 18)] + \
                             [sample[s] for s in ENVIRONMENTAL_SENSORS]
                    user_input = ",".join(f"{v:.3f}" for v in values)
                    print(f"Generated for '{smell_name}': {user_input}")
                if not user_input:
                    continue

                result, err = self.server_api.test_console_input(user_input)
                if err:
                    print(f"Failed: {err}")
                    continue

                print(f"\nPredicted: {result.get('predicted_smell', '?')}")
                print(f"Confidence: {result.get('confidence', 0):.3f}")
                for smell, prob in result.get("sorted_probabilities", []):
                    bar = "█" * int(prob * 20)
                    print(f"  {smell:<15} {prob:.3f} {bar}")
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(f"Error: {e}")

    def learn_from_csv_file(self) -> bool:
        print("\n" + "=" * 60)
        print("CSV LEARNING")
        print("=" * 60)
        try:
            csv_path = input("CSV file path: ").strip()
            if not csv_path or not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                return False

            target = input("Target column (default 'Gas name'): ").strip() or "Gas name"
            use_aug = input("Use augmentation? (y/n, default y): ").lower().strip() != "n"
            n_aug = 5
            if use_aug:
                try:
                    n_aug_in = input("Augmentations (default 5): ").strip()
                    if n_aug_in:
                        n_aug = int(n_aug_in)
                except ValueError:
                    n_aug = 5
            lowercase = input("Lowercase labels? (y/n, default y): ").lower().strip() != "n"

            result, err = self.server_api.learn_from_csv(csv_path, target, use_aug, n_aug, lowercase)
            if err:
                print(f"CSV learning failed: {err}")
                return False
            self.display_learning_feedback(result)
            return True
        except Exception as e:
            print(f"CSV learning error: {e}")
            return False

    def view_model_info(self) -> None:
        print("\n" + "=" * 60)
        print("MODEL INFO")
        print("=" * 60)
        result, err = self.server_api.get_model_info()
        if err:
            print(f"Failed: {err}")
            return
        if not isinstance(result, dict):
            print(f"Unexpected response: {result}")
            return

        print(f"  loaded     : {result.get('model_loaded', False)}")
        print(f"  type       : {result.get('model_type', '?')}")
        print(f"  fitted     : {result.get('is_fitted', False)}")
        print(f"  online     : {result.get('supports_online_learning', False)}")

        fc = result.get("feature_configuration")
        if fc:
            print(f"  features   : {fc.get('total_features', 0)}")
            print(f"  resistance : {len(fc.get('resistance_sensors', []))}")
            print(f"  env        : {len(fc.get('environmental_sensors', []))}")

        if result.get("is_fitted"):
            classes = result.get("classes", [])
            print(f"  n_classes  : {result.get('n_classes', 0)}")
            print(f"  classes    : {', '.join(classes) if classes else '(none)'}")
            acc = result.get("current_accuracy")
            if isinstance(acc, (int, float)):
                print(f"  accuracy   : {acc:.4f}")
            if "training_samples" in result:
                print(f"  samples    : {result['training_samples']}")
        else:
            print("  (model not trained yet)")

    def visualize_and_analyze(self) -> bool:
        print("\n" + "=" * 60)
        print("DATA VISUALIZATION + ANALYSIS")
        print("=" * 60)
        try:
            print("Generating visualizations…")
            viz, err = self.server_api.visualize_data()
            if err:
                print(f"Visualization failed: {err}")
            else:
                plots = (viz or {}).get("plots", [])
                print(f"  plots: {plots}")

            print("Analyzing data quality…")
            _, err = self.server_api.analyze_data_quality()
            if err:
                print(f"Analysis failed: {err}")
            else:
                print("  ok")

            print("Environmental sensor analysis…")
            env, err = self.server_api.environmental_analysis()
            if err:
                print(f"Env analysis failed: {err}")
            else:
                sensors = (env or {}).get("environmental_sensors", [])
                print(f"  analyzed {len(sensors)} sensors: {', '.join(sensors)}")
            return True
        except Exception as e:
            print(f"Visualization/analysis error: {e}")
            return False


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
