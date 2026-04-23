#!/usr/bin/env python3
"""
Enhanced Vision-Smell Training Client - Updated for 22 Features with Offline Mode
Now supports: R1-R17 (resistance sensors) + T, H, CO2, H2S, CH2O (environmental sensors)
Offline mode allows testing without physical sensors by generating simulated data
"""

import os
import sys
import time
import json
import datetime
import threading
import queue
import numpy as np
import pandas as pd
import cv2
import requests
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import serial
import serial.tools.list_ports
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
import random

# Constants - Updated for 22 features
DEFAULT_RECORDING_TIME = 60  # Fixed to 60 seconds
TARGET_SAMPLES = 100  # Exactly 100 samples
global SENSOR_DATA_COLUMNS
# Updated: R1-R17 + T, H, CO2, H2S, CH2O = 22 total features
SENSOR_DATA_COLUMNS = [f"R{i}" for i in range(1, 18)] + ["T", "H", "CO2", "H2S", "CH2O"]
DEFAULT_BAUD_RATE = 9600
TEMP_IMAGE_PATH = "temp_capture.png"

# Data transformation constants (only for R1-R17)
RLOW = 1
VCC = 4.49
EG = 1.02
VREF = 1.227
COEF = 65536

# Environmental sanity ranges and hard medians (client-side protection)
ENV_SANITY_RANGES = {"T": (5.0, 30.0), "H": (25.0, 65.0), "CO2": (300.0, 2500.0), "H2S": (0.0, float("inf")), "CH2O": (0.0, 200.0)}
ENV_SANITY_MEDIANS = {"T": 20.0, "H": 45.0, "CO2": 200.0, "H2S": 0.0, "CH2O": 10.0}

def sanitize_environmentals_inplace(d: dict) -> None:
    for k in ["T", "H", "CO2", "H2S", "CH2O"]:
        if k not in d: 
            continue
        low, high = ENV_SANITY_RANGES[k]
        med = ENV_SANITY_MEDIANS[k]
        try:
            v = float(d[k])
        except Exception:
            v = med
        if k == "H2S":
            v = max(0.0, v)
        if (v < low) or (v > high) or np.isnan(v):
            v = med
        d[k] = float(v)

def transform_sensor_values(values: List[float]) -> List[float]:
    """Transform raw sensor values to resistance values (only R1-R17)."""
    transformed = []
    for i, value in enumerate(values[:17]):  # Only transform R1-R17
        if value == 0.0:
            transformed.append(value)
        else:
            try:
                new_value = (RLOW * COEF * VCC * EG / (VREF * value)) - RLOW
                transformed.append(max(0.0, new_value))  # Prevent negative values
            except ZeroDivisionError:
                print(f"Warning: Division by zero for sensor R{i+1}, using 0.0")
                transformed.append(0.0)
            except Exception as e:
                print(f"Warning: Transform error for sensor R{i+1}: {e}, using 0.0")
                transformed.append(0.0)
    return transformed

class ENoseSensor:
    """Enhanced e-nose sensor handler with 22-feature support and offline mode."""
    
    def __init__(self, port=None, baud_rate_enose=DEFAULT_BAUD_RATE, baud_rate_UART=DEFAULT_BAUD_RATE, offline_mode=False):
        self.port = port
        self.baud_rate_enose = baud_rate_enose
        self.baud_rate_UART = baud_rate_UART
        self.offline_mode = offline_mode
        self.serial_conn_enose = None
        self.serial_conn_UART = None
        self._is_recording = False
        self._recording_ready = False
        self._lock = threading.Lock()
        self._data_buffer = queue.Queue(maxsize=200)
        self._recording_thread = None
        
        # Offline mode variables
        self._current_smell_profile = None
        self._base_environmental = {"T": 21.0, "H": 49.0, "CO2": 400.0, "H2S": 0.0, "CH2O": 5.0}
        
    def generate_realistic_sensor_data(self, smell_name: str = "air") -> Dict[str, float]:
        """Generate realistic 22-feature sensor data for offline mode."""
        
        # Define smell profiles (resistance sensor patterns)
        smell_profiles = {
            "air": [0.1, 0.05, 0.08, 0.02, 0.12, 0.09, 0.06, 0.04, 0.11, 0.07, 0.08, 0.05, 0.03, 0.09, 0.06, 0.04, 0.08],
            "coffee": [15.2, 8.3, 12.1, 3.4, 18.9, 11.2, 9.8, 6.7, 14.3, 10.5, 13.2, 7.8, 5.6, 12.4, 8.9, 6.3, 11.7],
            "rose": [8.7, 12.4, 6.8, 9.2, 5.3, 14.7, 11.8, 13.2, 7.9, 15.1, 6.4, 12.8, 10.9, 8.5, 13.6, 9.7, 7.2],
            "lemon": [6.3, 4.8, 9.7, 12.1, 7.5, 6.9, 8.2, 5.4, 10.8, 4.6, 11.3, 7.7, 9.1, 6.8, 5.2, 8.9, 10.4],
            "vanilla": [11.8, 9.2, 7.4, 5.8, 13.6, 8.7, 10.3, 12.1, 6.9, 9.8, 8.5, 11.4, 7.2, 10.6, 9.1, 7.8, 8.3],
            "apple": [4.2, 7.8, 5.6, 8.9, 6.3, 9.4, 7.1, 5.8, 8.7, 6.5, 9.2, 5.4, 7.6, 8.1, 6.8, 9.3, 5.9],
            "mint": [9.4, 6.2, 11.8, 7.3, 8.9, 5.7, 12.4, 8.6, 7.1, 10.2, 6.8, 9.5, 11.1, 7.4, 8.2, 6.9, 10.7],
            "chocolate": [13.7, 11.4, 9.8, 8.2, 15.3, 10.6, 12.9, 9.5, 11.7, 8.8, 14.2, 10.1, 9.3, 12.6, 8.4, 11.9, 10.8]
        }
        
        # Get base profile or use air as default
        if smell_name.lower() in smell_profiles:
            base_profile = smell_profiles[smell_name.lower()]
        else:
            # Generate random profile for unknown smells
            base_profile = [random.uniform(0.5, 20.0) for _ in range(17)]
        
        # Add noise to make it realistic
        resistance_values = []
        for base_val in base_profile:
            noise = random.gauss(0, base_val * 0.1)  # 10% noise
            final_val = max(0.0, base_val + noise)
            resistance_values.append(final_val)
        
        # Generate environmental sensor values with small variations
        environmental_values = []
        for sensor in ["T", "H", "CO2", "H2S", "CH2O"]:
            base_val = self._base_environmental[sensor]
            if sensor == "T":
                # Temperature: 18-25°C with small variations
                val = base_val + random.gauss(0, 1.5)
                val = max(15.0, min(30.0, val))
            elif sensor == "H":
                # Humidity: 30-70% with variations
                val = base_val + random.gauss(0, 8.0)
                val = max(20.0, min(80.0, val))
            elif sensor == "CO2":
                # CO2: 300-800 ppm with variations
                val = base_val + random.gauss(0, 50.0)
                val = max(250.0, min(1000.0, val))
            elif sensor == "H2S":
                # H2S: usually very low, 0-2 ppm
                val = base_val + random.gauss(0, 0.3)
                val = max(0.0, min(5.0, val))
            else:  # CH2O
                # CH2O: formaldehyde, usually 0-20 ppm
                val = base_val + random.gauss(0, 2.0)
                val = max(0.0, min(30.0, val))
            
            environmental_values.append(val)
        
        # Combine all values
        all_values = resistance_values + environmental_values
        
        # Create dictionary
        global SENSOR_DATA_COLUMNS
        if len(SENSOR_DATA_COLUMNS) != 22:
            SENSOR_DATA_COLUMNS = [f"R{i}" for i in range(1, 18)] + ["T", "H", "CO2", "H2S", "CH2O"]
        
        result = dict(zip(SENSOR_DATA_COLUMNS, all_values))
        sanitize_environmentals_inplace(result) 
        return result
    
    def list_available_ports(self):
        """List available serial ports."""
        if self.offline_mode:
            print("\nOffline mode: Skipping serial port detection")
            return []
        
        ports = serial.tools.list_ports.comports()
        print("\nAvailable serial ports:")
        if not ports:
            print("  No serial ports found.")
            return []
        for i, port in enumerate(ports):
            print(f"  [{i+1}] {port.device} - {port.description}")
        return ports
    
    def select_port(self):
        """Interactive port selection for both e-nose and UART."""
        if self.offline_mode:
            print("Offline mode: Using simulated sensors")
            return ("OFFLINE_ENOSE", "OFFLINE_UART")
        
        ports = self.list_available_ports()
        if not ports:
            return None, None
        
        port_devices = [p.device for p in ports]
        while True:
            try:
                selection_enose = input(f"Select enose port (1-{len(ports)} or device name): ").strip()
                selection_UART = input(f"Select UART port (1-{len(ports)} or device name): ").strip()
                if not selection_enose or not selection_UART:
                    continue
                
                # Try numeric selection
                try:
                    enose_idx = int(selection_enose) - 1
                    UART_idx = int(selection_UART) - 1
                    if 0 <= enose_idx < len(port_devices) and 0 <= UART_idx < len(port_devices):
                        return port_devices[enose_idx], port_devices[UART_idx]
                except ValueError:
                    # Try device name
                    if selection_enose in port_devices and selection_UART in port_devices:
                        return selection_enose, selection_UART
                
                print("Invalid selection. Please try again.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                return None, None
    
    def connect(self):
        """Connect to both e-nose and UART sensors (or simulate in offline mode)."""
        if self.offline_mode:
            print("Offline mode: Simulating sensor connections")
            self.serial_conn_enose = "SIMULATED_ENOSE"
            self.serial_conn_UART = "SIMULATED_UART"
            print("Connected to simulated e-nose and UART devices")
            print("22-feature offline mode: R1-R17 (simulated) + T,H,CO2,H2S,CH2O (simulated)")
            return True
        
        if not self.port:
            self.port = self.select_port()
            if not self.port or len(self.port) != 2 or not self.port[0] or not self.port[1]:
                return False
        
        try:
            with self._lock:
                if (self.serial_conn_enose and self.serial_conn_UART and 
                    hasattr(self.serial_conn_enose, 'is_open') and self.serial_conn_enose.is_open and
                    hasattr(self.serial_conn_UART, 'is_open') and self.serial_conn_UART.is_open):
                    print(f"Already connected to {self.port[0]} and {self.port[1]}")
                    return True
                
                try:
                    self.serial_conn_enose = serial.Serial(self.port[0], self.baud_rate_enose, timeout=1)
                except Exception as e:
                    print(f"Failed to connect to e-nose on {self.port[0]}: {e}")
                    return False

                try:
                    self.serial_conn_UART = serial.Serial(self.port[1], self.baud_rate_UART, timeout=1)
                except Exception as e:
                    print(f"Failed to connect to UART on {self.port[1]}: {e}")
                    if self.serial_conn_enose:
                        self.serial_conn_enose.close()
                    return False
                
                time.sleep(1)  # Allow devices to settle
                
                # Clear input buffers
                if self.serial_conn_enose.in_waiting:
                    self.serial_conn_enose.read(self.serial_conn_enose.in_waiting)
                
                if self.serial_conn_UART.in_waiting:
                    self.serial_conn_UART.read(self.serial_conn_UART.in_waiting)
                
                print(f"Connected to e-nose on {self.port[0]} at {self.baud_rate_enose} baud")
                print(f"Connected to UART on {self.port[1]} at {self.baud_rate_UART} baud")
                print("22-feature mode: R1-R17 (e-nose) + T,H,CO2,H2S,CH2O (UART)")
                return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from sensors."""
        if self.offline_mode:
            print("Offline mode: Disconnecting from simulated sensors")
            self.serial_conn_enose = None
            self.serial_conn_UART = None
            return
        
        with self._lock:
            if self.serial_conn_enose and hasattr(self.serial_conn_enose, 'is_open') and self.serial_conn_enose.is_open:
                try:
                    self.serial_conn_enose.close()
                    print("Disconnected from e-nose sensor")
                except Exception as e:
                    print(f"E-nose disconnect error: {e}")
            
            if self.serial_conn_UART and hasattr(self.serial_conn_UART, 'is_open') and self.serial_conn_UART.is_open:
                try:
                    self.serial_conn_UART.close()
                    print("Disconnected from UART device")
                except Exception as e:
                    print(f"UART disconnect error: {e}")
            
            self.serial_conn_enose = None
            self.serial_conn_UART = None
    
    def _parse_and_transform_line(self, line: str, line_UART: str) -> Optional[Dict[str, float]]:
        """Parse and transform sensor data for 22 features."""
        try:
            print(f"Parsing e-nose data: {line.strip()}")
            print(f"Parsing UART data: {line_UART.strip()}")
            
            # Parse e-nose data (R1-R17)
            raw_values = [float(x) for x in line.strip().split()]
            if len(raw_values) < 17:
                print(f'Insufficient e-nose values: got {len(raw_values)}, need 17')
                return None
            
            # Parse UART data (T, H, CO2, H2S, CH2O)
            raw_UART = line_UART.strip().split('\t')
            raw_UART = [float(str(i).strip()) for i in raw_UART]
            if len(raw_UART) < 5:
                print(f'Insufficient UART values: got {len(raw_UART)}, need 5')
                return None
            
            # Transform only the resistance sensors (R1-R17)
            transformed_resistance = transform_sensor_values(raw_values)
            
            # Combine: transformed resistance + raw environmental
            all_values = transformed_resistance + raw_UART[:5]  # Take exactly 5 environmental values
            
            global SENSOR_DATA_COLUMNS
            if len(SENSOR_DATA_COLUMNS) != 22:
                SENSOR_DATA_COLUMNS = [f"R{i}" for i in range(1, 18)] + ["T", "H", "CO2", "H2S", "CH2O"]
            
            # Create 22-feature dictionary
            result = dict(zip(SENSOR_DATA_COLUMNS, all_values))
            sanitize_environmentals_inplace(result)

            print(f"22-feature sample created successfully")
            print(f"  Environmental: T={result['T']:.1f}, H={result['H']:.1f}, CO2={result['CO2']:.0f}")
            
            return result
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def read_single_measurement(self) -> Optional[Dict[str, float]]:
        """Read and transform single 22-feature measurement (or generate in offline mode)."""
        if self.offline_mode:
            # Generate simulated data
            current_smell = getattr(self, '_current_smell_name', 'air')
            data = self.generate_realistic_sensor_data(current_smell)
            print(f"Generated simulated 22-feature data for '{current_smell}'")
            return data
        
        with self._lock:
            if (not self.serial_conn_enose or not hasattr(self.serial_conn_enose, 'is_open') or not self.serial_conn_enose.is_open or 
                not self.serial_conn_UART or not hasattr(self.serial_conn_UART, 'is_open') or not self.serial_conn_UART.is_open):
                print("One or both serial connections not available")
                return None
            
            try:
                # Check for data availability
                if self.serial_conn_enose.in_waiting == 0:
                    time.sleep(0.1)
                    if self.serial_conn_enose.in_waiting == 0:
                        return None
                        
                if self.serial_conn_UART.in_waiting == 0:
                    time.sleep(0.1)
                    if self.serial_conn_UART.in_waiting == 0:
                        return None
                
                # Read from both devices
                line_enose = self.serial_conn_enose.readline().decode('utf-8')
                line_UART = self.serial_conn_UART.readline().decode('utf-8')
                
                if not line_enose.strip() or not line_UART.strip():
                    return None
                
                return self._parse_and_transform_line(line_enose, line_UART)
                
            except Exception as e:
                print(f"Read measurement error: {e}")
                return None
    
    def set_simulation_smell(self, smell_name: str):
        """Set the smell for simulation mode."""
        self._current_smell_name = smell_name.lower()
        print(f"Simulation mode: Set current smell to '{smell_name}'")
    
    def _recording_worker_fixed(self, duration_seconds: int = 60, target_samples: int = 100):
        """FIXED recording worker for 22 features with precise timing (supports offline mode)."""
        mode_text = "SIMULATED" if self.offline_mode else "HARDWARE"
        print(f"\n22-FEATURE {mode_text} RECORDING STARTED!")
        print(f"Target: {target_samples} samples in {duration_seconds} seconds")
        print(f"Sample rate: {target_samples/duration_seconds:.2f} samples/second")
        print(f"Features: R1-R17 (transformed) + T,H,CO2,H2S,CH2O (raw)")
        if self.offline_mode:
            current_smell = getattr(self, '_current_smell_name', 'air')
            print(f"Simulating smell: {current_smell}")
        print("-" * 60)
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        sample_count = 0
        
        # Clear buffer
        while not self._data_buffer.empty():
            try:
                self._data_buffer.get_nowait()
            except queue.Empty:
                break
        
        self._is_recording = True
        last_print = time.time()
        
        # Calculate ideal sample interval
        sample_interval = duration_seconds / target_samples
        next_sample_time = start_time
        
        while time.time() < end_time and self._is_recording and sample_count < target_samples:
            current_time = time.time()
            
            # Only collect sample if we're at the right time interval
            if current_time >= next_sample_time:
                data_point = self.read_single_measurement()
                if data_point and len(data_point) == 22:  # Ensure we have all 22 features
                    try:
                        self._data_buffer.put_nowait(data_point)
                        sample_count += 1
                        next_sample_time += sample_interval
                        
                    except queue.Full:
                        # Remove oldest sample and add new one
                        try:
                            self._data_buffer.get_nowait()
                            self._data_buffer.put_nowait(data_point)
                        except queue.Empty:
                            pass
            
            # Progress update every second
            if current_time - last_print >= 1.0:
                elapsed = current_time - start_time
                progress = min(100, round(sample_count / target_samples * 100))
                remaining = max(0, round(duration_seconds - elapsed))
                print(f"Recording: {progress:3d}% | Samples: {sample_count:3d}/{target_samples} | Time: {remaining:2d}s remaining", end='\r')
                last_print = current_time
            
            # In offline mode, add small delay to simulate real timing
            if self.offline_mode:
                time.sleep(0.05)
            else:
                time.sleep(0.01)
        
        self._is_recording = False
        actual_duration = time.time() - start_time
        print(f"\n22-FEATURE {mode_text} RECORDING COMPLETED!")
        print(f"Final: {sample_count} samples in {actual_duration:.1f}s")
        print(f"Rate: {sample_count/actual_duration:.2f} samples/second")
        
        if sample_count < target_samples:
            print(f"Warning: Only collected {sample_count}/{target_samples} samples")
        elif sample_count > target_samples:
            print(f"Info: Collected {sample_count} samples (target: {target_samples})")
    
    def prepare_recording(self) -> bool:
        """Prepare for 22-feature recording."""
        if self.offline_mode:
            self._recording_ready = True
            print("22-feature simulation recording prepared. Waiting for start command...")
            return True
        
        with self._lock:
            if not self.serial_conn_enose or not hasattr(self.serial_conn_enose, 'is_open') or not self.serial_conn_enose.is_open:
                print("Cannot prepare recording: e-nose not connected")
                return False
            if not self.serial_conn_UART or not hasattr(self.serial_conn_UART, 'is_open') or not self.serial_conn_UART.is_open:
                print("Cannot prepare recording: UART device not connected")
                return False
            if self._is_recording:
                print("Recording already in progress")
                return False
        
        self._recording_ready = True
        print("22-feature recording prepared. Waiting for start command...")
        return True
    
    def start_recording_after_confirmation(self, duration_seconds: int = 60, target_samples: int = 100) -> bool:
        """Start 22-feature recording immediately after this call."""
        if not self._recording_ready:
            print("Recording not prepared. Call prepare_recording() first.")
            return False
        
        self._recording_ready = False
        
        self._recording_thread = threading.Thread(
            target=self._recording_worker_fixed,
            args=(duration_seconds, target_samples),
            daemon=True
        )
        self._recording_thread.start()
        return True
    
    def stop_recording(self) -> List[Dict[str, float]]:
        """Stop recording and return 22-feature data."""
        if self._is_recording:
            self._is_recording = False
            if self._recording_thread and self._recording_thread.is_alive():
                self._recording_thread.join(timeout=3.0)
        
        # Collect data from buffer
        collected_data = []
        while not self._data_buffer.empty():
            try:
                sample = self._data_buffer.get_nowait()
                if len(sample) == 22:  # Verify 22 features
                    collected_data.append(sample)
            except queue.Empty:
                break
        
        mode_text = "simulated" if self.offline_mode else "hardware"
        print(f"Collected {len(collected_data)} valid 22-feature {mode_text} samples")
        return collected_data
    
    def get_average_reading(self, data: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Calculate average 22-feature reading from data points."""
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if len(df.columns) != 22:
                print(f"Warning: Expected 22 features, got {len(df.columns)}")
            return df.mean().to_dict()
        except Exception as e:
            print(f"Error calculating average: {e}")
            return None

class WebcamHandler:
    """Simple and reliable webcam handler (unchanged)."""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
    
    def open_camera(self):
        """Open camera connection."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Could not open camera {self.camera_index}")
                return True
            print(f"Camera {self.camera_index} opened successfully")
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def close_camera(self):
        """Close camera and cleanup."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def capture_image(self, save_path=TEMP_IMAGE_PATH):
        """Capture image with preview window."""
        if not self.cap or not self.cap.isOpened():
            if not self.open_camera():
                return None, None
        
        print("Camera preview active. Press SPACE to capture, ESC to cancel.")
        window_name = f'Camera {self.camera_index} Preview'
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 32:  # SPACE
                print("Capturing image...")
                try:
                    cv2.imwrite(save_path, frame)
                    cv2.destroyWindow(window_name)
                    print(f"Image saved to {save_path}")
                    return frame, save_path
                except Exception as e:
                    print(f"Save error: {e}")
                    return frame, None
            elif key == 27:  # ESC
                print("Capture cancelled")
                break
        
        cv2.destroyWindow(window_name)
        return None, None
    
    def display_detection_result(self, image_path, detection_result):
        """Display image with detection results."""
        try:
            # Parse detection result
            if isinstance(detection_result, str):
                try:
                    parsed_result = json.loads(detection_result)
                except json.JSONDecodeError:
                    try:
                        parsed_result = ast.literal_eval(detection_result)
                    except (ValueError, SyntaxError):
                        parsed_result = None
            else:
                parsed_result = detection_result
            
            # Load and display image
            image = Image.open(image_path)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image)
            ax.set_title("Object Detection Results")
            ax.axis('off')
            
            # Process detection data
            if parsed_result and isinstance(parsed_result, dict):
                if '<OPEN_VOCABULARY_DETECTION>' in parsed_result:
                    detection_data = parsed_result['<OPEN_VOCABULARY_DETECTION>']
                    
                    bboxes = detection_data.get('bboxes', [])
                    labels = detection_data.get('bboxes_labels', detection_data.get('labels', []))
                    
                    print(f"Detection results: {len(bboxes)} objects found")
                    
                    # Draw bounding boxes
                    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = map(float, bbox)
                            
                            # Create rectangle
                            rect = patches.Rectangle(
                                (x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='red', facecolor='none'
                            )
                            ax.add_patch(rect)
                            
                            # Add label
                            ax.text(x1, y1 - 5, label, 
                                   color='white', fontsize=10, weight='bold',
                                   bbox=dict(facecolor='red', alpha=0.8, pad=2))
                            
                            print(f"  {i+1}. {label}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    print("No detection data found in result")
                    ax.text(0.5, 0.5, "No detection data found", 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=16, bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
            else:
                print("Could not parse detection result")
                ax.text(0.5, 0.5, "Could not parse detection result", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16, bbox=dict(boxstyle="round", facecolor='orange', alpha=0.7))
            
            plt.tight_layout()
            plt.show(block=True)
            
        except Exception as e:
            print(f"Display error: {e}")
            import traceback
            traceback.print_exc()

class ServerAPI:
    """Server communication handler updated for 22 features."""
    
    def __init__(self, base_url="http://localhost:8016"):
        self.base_url = base_url
    
    def _handle_response(self, response):
        """Handle HTTP response with detailed error information."""
        try:
            response.raise_for_status()
            try:
                return response.json(), None
            except json.JSONDecodeError:
                return response.text, None
        except requests.exceptions.HTTPError:
            # Try to get detailed error information
            try:
                error_detail = response.json()
                if isinstance(error_detail, dict) and 'detail' in error_detail:
                    return None, f"HTTP {response.status_code}: {error_detail['detail']}"
                else:
                    return None, f"HTTP {response.status_code}: {str(error_detail)}"
            except:
                return None, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return None, f"Response error: {e}"
    
    def test_connection(self):
        """Test server connection and verify 22-feature support."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            result, error = self._handle_response(response)
            if error:
                return None, error
            
            # Check for 22-feature support
            if isinstance(result, dict):
                sensor_config = result.get('sensor_configuration', {})
                total_features = sensor_config.get('total_features', 0)
                if total_features == 22:
                    print("Server confirmed 22-feature support")
                else:
                    print(f"Warning: Server reports {total_features} features, expected 22")
            
            return result, None
        except requests.exceptions.Timeout:
            return None, "Connection timeout"
        except Exception as e:
            return None, f"Connection error: {e}"
    
    def detect_object(self, image_path, object_name):
        """Send object detection request."""
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, 'image/png')}
                data = {"text": f"Find {object_name}"}
                response = requests.post(f"{self.base_url}/predict/object", 
                                       files=files, data=data, timeout=30)
            
            return self._handle_response(response)
        except FileNotFoundError:
            return None, f"Image not found: {image_path}"
        except requests.exceptions.Timeout:
            return None, "Detection request timeout"
        except Exception as e:
            return None, f"Detection error: {e}"
    
    def online_learning(self, sensor_data: List[Dict[str, float]], object_name: str):
        """Send 22-feature data for online learning."""
        try:
            # Verify 22-feature data
            if sensor_data:
                sample = sensor_data[0]
                if len(sample) != 22:
                    print(f"Warning: Expected 22 features, got {len(sample)}")
                    print(f"Features: {list(sample.keys())}")
            
            payload = {
                "sensor_data": sensor_data,
                "labels": [object_name] * len(sensor_data)
            }
            
            response = requests.post(f"{self.base_url}/smell/online_learning", 
                                   json=payload, timeout=30)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Online learning timeout"
        except Exception as e:
            return None, f"Online learning error: {e}"
    
    def learn_from_csv(self, csv_file_path: str, target_column: str = "Gas name", 
                      use_augmentation: bool = True, n_augmentations: int = 5,
                      lowercase_labels: bool = True):
        """Learn from 22-feature CSV file."""
        try:
            # Validate CSV before sending
            try:
                df = pd.read_csv(csv_file_path)
                print(f"CSV validation: {len(df)} rows, {len(df.columns)} columns")
                print(f"Columns: {df.columns.tolist()}")
                
                if target_column not in df.columns:
                    print(f"Warning: Target column '{target_column}' not found in CSV")
                    print(f"Available columns: {df.columns.tolist()}")
                    return None, f"Target column '{target_column}' not found"
                
                # Check for 22-feature columns
                expected_sensors = [f"R{i}" for i in range(1, 18)] + ["T", "H", "CO2", "H2S", "CH2O"]
                missing_sensors = [s for s in expected_sensors if s not in df.columns]
                if missing_sensors:
                    print(f"Warning: Missing sensors in CSV: {missing_sensors}")
                    print("Server will add default values for missing sensors")
                
            except Exception as e:
                print(f"CSV validation error: {e}")
                return None, f"CSV validation failed: {e}"
            
            # Read CSV file
            with open(csv_file_path, 'r') as f:
                csv_content = f.read()
            
            payload = {
                "csv_data": csv_content,
                "target_column": target_column,
                "use_augmentation": use_augmentation,
                "n_augmentations": n_augmentations,
                "noise_std": 0.0015,
                "lowercase_labels": lowercase_labels
            }
            
            print(f"Sending CSV learning request...")
            print(f"  File: {csv_file_path}")
            print(f"  Target: {target_column}")
            print(f"  Samples: {len(df)}")
            print(f"  Augmentation: {use_augmentation} (n={n_augmentations})")
            
            response = requests.post(f"{self.base_url}/smell/learn_from_csv", 
                                   json=payload, timeout=120)  # Longer timeout for training
            return self._handle_response(response)
        except FileNotFoundError:
            return None, f"CSV file not found: {csv_file_path}"
        except requests.exceptions.Timeout:
            return None, "CSV learning request timeout"
        except Exception as e:
            return None, f"CSV learning error: {e}"
    
    def test_console_input(self, sensor_values: str):
        """Test with 22-feature console sensor input."""
        try:
            # Validate input format
            try:
                values = [float(x.strip()) for x in sensor_values.split(',')]
                if len(values) != 22:
                    return None, f"Expected 22 values, got {len(values)}. Format: R1,...,R17,T,H,CO2,H2S,CH2O"
            except ValueError as e:
                return None, f"Invalid number format: {e}"
            
            payload = {"values": sensor_values}
            response = requests.post(f"{self.base_url}/smell/test_console", 
                                   json=payload, timeout=10)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Console test request timeout"
        except Exception as e:
            return None, f"Console test error: {e}"
    
    def classify_smell(self, sensor_data: Dict[str, float]):
        """Classify single 22-feature smell sample."""
        try:
            # Verify 22-feature data
            if len(sensor_data) != 22:
                print(f"Warning: Expected 22 features, got {len(sensor_data)}")
                print(f"Features: {list(sensor_data.keys())}")
            
            response = requests.post(f"{self.base_url}/smell/classify", 
                                   json=sensor_data, timeout=5)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Classification timeout"
        except Exception as e:
            return None, f"Classification error: {e}"
    
    def get_model_info(self):
        """Get 22-feature model information."""
        try:
            response = requests.get(f"{self.base_url}/smell/model_info", timeout=5)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Model info timeout"
        except Exception as e:
            return None, f"Model info error: {e}"
    
    def visualize_data(self):
        """Request 22-feature data visualization generation."""
        try:
            response = requests.get(f"{self.base_url}/smell/visualize_data", timeout=30)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Visualization timeout"
        except Exception as e:
            return None, f"Visualization error: {e}"
    
    def analyze_data_quality(self):
        """Request 22-feature data quality analysis."""
        try:
            response = requests.get(f"{self.base_url}/smell/analyze_data", timeout=30)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Analysis timeout"
        except Exception as e:
            return None, f"Analysis error: {e}"
    
    def environmental_analysis(self):
        """Request environmental sensor analysis (22-feature specific)."""
        try:
            response = requests.get(f"{self.base_url}/smell/environmental_analysis", timeout=30)
            return self._handle_response(response)
        except requests.exceptions.Timeout:
            return None, "Environmental analysis timeout"
        except Exception as e:
            return None, f"Environmental analysis error: {e}"

class TrainingPipeline:
    """Training pipeline updated for 22 features with offline support."""
    
    def __init__(self, server_api, webcam_handler, enose_sensor, recording_time=60, target_samples=100):
        self.server_api = server_api
        self.webcam_handler = webcam_handler
        self.enose_sensor = enose_sensor
        self.recording_time = recording_time
        self.target_samples = target_samples
        self.local_data_path = "training_data_22features_log.csv"
    
    def save_local_data(self, sensor_data: List[Dict[str, float]], object_name: str):
        """Save 22-feature training data locally."""
        if not sensor_data:
            return
        
        try:
            df = pd.DataFrame(sensor_data)
            df['smell_label'] = object_name
            df['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Verify 22 features
            if len(df.columns) - 2 != 22:  # -2 for smell_label and timestamp
                print(f"Warning: Expected 22 sensor features, got {len(df.columns) - 2}")
            
            file_exists = os.path.isfile(self.local_data_path)
            df.to_csv(self.local_data_path, mode='a', header=not file_exists, index=False)
            mode_text = "simulated" if self.enose_sensor.offline_mode else "hardware"
            print(f"Logged {len(sensor_data)} 22-feature {mode_text} samples to {self.local_data_path}")
        except Exception as e:
            print(f"Local save error: {e}")
    
    def display_22_feature_sample(self, sample: Dict[str, float]):
        """Display 22-feature sample in organized way."""
        resistance_sensors = [f"R{i}" for i in range(1, 18)]
        environmental_sensors = ["T", "H", "CO2", "H2S", "CH2O"]
        
        print(f"22-Feature Sensor Reading:")
        print(f"  Resistance Sensors (R1-R17):")
        resistance_active = {k: v for k, v in sample.items() if k in resistance_sensors and v > 0.01}
        if resistance_active:
            for i in range(0, len(resistance_active), 5):  # Show 5 per line
                items = list(resistance_active.items())[i:i+5]
                values_str = ", ".join([f"{k}={v:.3f}" for k, v in items])
                print(f"    {values_str}")
        else:
            print(f"    No significant resistance sensor activity")
        
        print(f"  Environmental Sensors:")
        env_values = {k: sample.get(k, 0) for k in environmental_sensors}
        env_str = ", ".join([f"{k}={v:.1f}" for k, v in env_values.items()])
        print(f"    {env_str}")
    
    def display_learning_feedback(self, result):
        """Display enhanced feedback for 22-feature learning operations."""
        if not result:
            return
        
        print(f"\n22-Feature Learning completed successfully!")
        print(f"   Update type: {result.get('update_type', 'Unknown')}")
        print(f"   Samples processed: {result.get('samples_processed', 'Unknown')}")
        print(f"   Current accuracy: {result.get('current_accuracy', 'Unknown'):.4f}")
        
        # Show feature breakdown
        if 'feature_breakdown' in result:
            breakdown = result['feature_breakdown']
            print(f"   Features: {breakdown.get('resistance_sensors', 0)} resistance + {breakdown.get('environmental_sensors', 0)} environmental = {breakdown.get('total_features', 0)} total")
        
        # Show special feedback for retraining and consistency fixes
        if result.get('update_type') == 'retrain_for_consistency':
            print(f"   FIXED: Model inconsistency detected and resolved!")
        elif result.get('new_classes_detected', False):
            print(f"   NEW CLASSES: Model retrained to include new classes!")
        
        if result.get('inconsistency_fixed', False):
            print(f"   Model inconsistency was detected and fixed")
        
        if result.get('model_reloaded', False):
            print(f"   Model was reloaded for consistency")
        
        classes = result.get('classes', [])
        print(f"   Available classes ({len(classes)}): {', '.join(classes)}")
        
        if 'visualizations' in result and result['visualizations']:
            print(f"   Generated {len(result['visualizations'])} visualizations")
    
    def run_training_cycle(self):
        """Execute complete 22-feature training cycle (supports offline mode)."""
        mode_text = "SIMULATED" if self.enose_sensor.offline_mode else "HARDWARE"
        print("\n" + "="*70)
        print(f"22-FEATURE TRAINING CYCLE ({mode_text} MODE)")
        print("="*70)
        
        try:
            # 1. Capture image
            print("Step 1: Capture Image")
            _, image_path = self.webcam_handler.capture_image()
            if not image_path:
                print("Image capture failed")
                return False
            
            # 2. Get object name
            print("Step 2: Specify Object")
            object_name = input("Enter object name: ").strip()
            if not object_name:
                print("No object name provided")
                return False
            
            # Set simulation smell for offline mode
            if self.enose_sensor.offline_mode:
                self.enose_sensor.set_simulation_smell(object_name)
            
            # 3. Object detection
            print(f"Step 3: Detecting '{object_name}'")
            detection_result, error = self.server_api.detect_object(image_path, object_name)
            if error:
                print(f"Detection failed: {error}")
                return False
            
            print("Detection successful!")
            
            # 4. Display results
            print("Step 4: Displaying Results")
            self.webcam_handler.display_detection_result(image_path, detection_result)
            
            # 5. Record 22-feature smell data
            print(f"Step 5: Record 22-Feature Smell Data ({mode_text})")
            print(f"\n22-FEATURE RECORDING INSTRUCTIONS:")
            print(f"   Target: {self.target_samples} samples in {self.recording_time} seconds")
            print(f"   Features: R1-R17 (e-nose) + T,H,CO2,H2S,CH2O (UART)")
            if self.enose_sensor.offline_mode:
                print(f"   Mode: SIMULATION (generating realistic data for '{object_name}')")
            else:
                print(f"   Mode: HARDWARE (reading from physical sensors)")
                print(f"   Please position both sensors near the '{object_name}' object")
                print(f"   Wait approximately 2 minutes for sensors to stabilize")
            print(f"   The recording will start IMMEDIATELY when you confirm")
            if not self.enose_sensor.offline_mode:
                print(f"   Do NOT move the sensors during recording")
            
            # Prepare recording
            if not self.enose_sensor.prepare_recording():
                print(f"Failed to prepare 22-feature {mode_text.lower()} recording")
                return False
            
            # Wait for user confirmation
            print(f"\nWaiting for confirmation...")
            while True:
                user_input = input(f"Start 22-feature {mode_text.lower()} recording for '{object_name}'? (yes/no): ").lower().strip()
                if user_input in ['yes', 'y']:
                    print(f"\nSTARTING 22-FEATURE {mode_text} RECORDING NOW!")
                    if not self.enose_sensor.start_recording_after_confirmation(
                        self.recording_time, self.target_samples
                    ):
                        print("Failed to start recording")
                        return False
                    break
                elif user_input in ['no', 'n']:
                    print("Recording cancelled by user")
                    return False
                else:
                    print("Please enter 'yes' or 'no'")
            
            # Wait for recording to complete
            print(f"22-feature {mode_text.lower()} recording in progress...")
            time.sleep(self.recording_time + 2)
            
            sensor_data = self.enose_sensor.stop_recording()
            
            if not sensor_data:
                print(f"No 22-feature sensor data recorded")
                return False
            
            print(f"22-feature {mode_text.lower()} recording completed: {len(sensor_data)} samples collected")
            
            # 6. Save locally and show stats
            self.save_local_data(sensor_data, object_name)
            avg_reading = self.enose_sensor.get_average_reading(sensor_data)
            if avg_reading:
                self.display_22_feature_sample(avg_reading)
            
            # 7. Send for training
            print("Step 6: Sending for 22-Feature Online Learning")
            result, error = self.server_api.online_learning(sensor_data, object_name)
            if error:
                print(f"Online learning failed: {error}")
                return False
            
            # Display enhanced feedback
            self.display_learning_feedback(result)
            
            return True
            
        except Exception as e:
            print(f"22-feature training cycle error: {e}")
            return False
        finally:
            # Cleanup
            if os.path.exists(TEMP_IMAGE_PATH):
                try:
                    os.remove(TEMP_IMAGE_PATH)
                except:
                    pass
    
    def run_vision_only(self):
        """Vision-only detection cycle."""
        print("\n" + "="*50)
        print("VISION-ONLY DETECTION")
        print("="*50)
        
        try:
            # Capture and detect
            _, image_path = self.webcam_handler.capture_image()
            if not image_path:
                return False
            
            object_name = input("Enter object to detect: ").strip()
            if not object_name:
                return False
            
            print(f"Detecting '{object_name}'...")
            detection_result, error = self.server_api.detect_object(image_path, object_name)
            if error:
                print(f"Detection failed: {error}")
                return False
            
            # Display results
            self.webcam_handler.display_detection_result(image_path, detection_result)
            return True
            
        except Exception as e:
            print(f"Vision cycle error: {e}")
            return False
        finally:
            if os.path.exists(TEMP_IMAGE_PATH):
                try:
                    os.remove(TEMP_IMAGE_PATH)
                except:
                    pass
    
    def identify_current_smell(self):
        """Identify current smell using 22 features (supports offline mode)."""
        mode_text = "SIMULATED" if self.enose_sensor.offline_mode else "HARDWARE"
        print("\n" + "="*60)
        print(f"22-FEATURE SMELL IDENTIFICATION ({mode_text} MODE)")
        print("="*60)
        
        try:
            if self.enose_sensor.offline_mode:
                smell_name = input("Enter smell name for simulation (or press enter for 'air'): ").strip()
                if not smell_name:
                    smell_name = "air"
                self.enose_sensor.set_simulation_smell(smell_name)
            
            print(f"Reading current 22-feature sensor data ({mode_text.lower()})...")
            sensor_reading = self.enose_sensor.read_single_measurement()
            
            if not sensor_reading:
                print("Failed to read 22-feature sensor data")
                return False
            
            if len(sensor_reading) != 22:
                print(f"Warning: Expected 22 features, got {len(sensor_reading)}")
            
            # Display current readings
            self.display_22_feature_sample(sensor_reading)
            
            print("Requesting 22-feature classification...")
            result, error = self.server_api.classify_smell(sensor_reading)
            if error:
                print(f"Classification failed: {error}")
                return False
            
            # Enhanced classification results
            print(f"\n22-Feature Classification Results:")
            print(f"   Predicted smell: {result.get('predicted_smell', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            if 'probabilities' in result:
                print(f"\nAll probabilities:")
                sorted_probs = sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for smell, prob in sorted_probs:
                    confidence_bar = "█" * int(prob * 20)  # Visual bar
                    print(f"     {smell:<15}: {prob:.3f} {confidence_bar}")
            
            # Show feature info
            if 'feature_info' in result:
                feature_info = result['feature_info']
                print(f"\nModel info: {feature_info.get('total_features', 0)} total features")
            
            return True
            
        except Exception as e:
            print(f"22-feature smell identification error: {e}")
            return False
    
    def test_manual_22_feature_input(self):
        """Test with manual 22-feature sensor input."""
        print("\n" + "="*60)
        print("MANUAL 22-FEATURE INPUT TESTING")
        print("="*60)
        print("Enter sensor values as comma-separated values:")
        print("Format: R1,R2,...,R17,T,H,CO2,H2S,CH2O (22 values total)")
        print("Example: 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,21.5,45.2,400,0.1,5.0")
        print("\nTips:")
        print("- R1-R17 should be transformed resistance values")
        print("- T,H,CO2,H2S,CH2O should be raw environmental sensor values")
        print("- Type 'back' to return to main menu")
        print("- Type 'generate' to generate realistic sample values")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nEnter 22 values (or 'generate'): ").strip()
                
                if user_input.lower() == 'back':
                    print("Returning to main menu...")
                    break
                
                if user_input.lower() == 'generate':
                    smell_name = input("Enter smell name for sample generation (default: coffee): ").strip()
                    if not smell_name:
                        smell_name = "coffee"
                    
                    sample_data = self.enose_sensor.generate_realistic_sensor_data(smell_name)
                    values_list = [sample_data[f"R{i}"] for i in range(1, 18)] + [sample_data[s] for s in ["T", "H", "CO2", "H2S", "CH2O"]]
                    user_input = ",".join([f"{v:.3f}" for v in values_list])
                    print(f"Generated 22-feature sample for '{smell_name}':")
                    print(f"  {user_input}")
                
                if not user_input:
                    continue
                
                # Send to server for testing
                result, error = self.server_api.test_console_input(user_input)
                
                if error:
                    print(f"Testing failed: {error}")
                    continue
                
                # Enhanced results display
                print(f"\n22-Feature Prediction Results:")
                print(f"   Predicted smell: {result.get('predicted_smell', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                
                if 'sensor_input' in result:
                    sensor_input = result['sensor_input']
                    if 'resistance_sensors' in sensor_input and 'environmental_sensors' in sensor_input:
                        print(f"\nInput breakdown:")
                        print(f"   Resistance: {len(sensor_input['resistance_sensors'])} values")
                        print(f"   Environmental: {len(sensor_input['environmental_sensors'])} values")
                
                if 'sorted_probabilities' in result:
                    print(f"\nAll probabilities (sorted by confidence):")
                    for smell, prob in result['sorted_probabilities']:
                        confidence_bar = "█" * int(prob * 20)
                        print(f"     {smell:<15}: {prob:.3f} {confidence_bar}")
                
                print(f"\n22-feature classification completed successfully!")
                
            except KeyboardInterrupt:
                print("\nReturning to main menu...")
                break
            except Exception as e:
                print(f"Error during testing: {e}")
    
    def learn_from_csv_file(self):
        """Learn from 22-feature CSV file."""
        print("\n" + "="*60)
        print("22-FEATURE CSV LEARNING")
        print("="*60)
        
        try:
            # Get CSV file path
            csv_path = input("Enter CSV file path: ").strip()
            if not csv_path:
                print("No file path provided")
                return False
            
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                return False
            
            # Get parameters
            target_col = input("Enter target column name (default: Gas name): ").strip()
            if not target_col:
                target_col = "Gas name"
            
            use_aug = input("Use data augmentation? (y/n, default: y): ").lower().strip()
            use_augmentation = use_aug != 'n'
            
            n_aug = 5  # Default optimal value
            if use_augmentation:
                try:
                    n_aug_input = input("Number of augmentations (default: 5 - optimal): ").strip()
                    if n_aug_input:
                        n_aug = int(n_aug_input)
                except ValueError:
                    print("Invalid number, using default (5)")
                    n_aug = 5
            
            lowercase = input("Convert labels to lowercase? (y/n, default: y): ").lower().strip()
            lowercase_labels = lowercase != 'n'
            
            print(f"\nLearning from 22-feature CSV...")
            print(f"   File: {csv_path}")
            print(f"   Target column: {target_col}")
            print(f"   Augmentation: {use_augmentation}")
            if use_augmentation:
                print(f"   Augmentations: {n_aug}")
            print(f"   Lowercase labels: {lowercase_labels}")
            
            # Send learning request
            result, error = self.server_api.learn_from_csv(
                csv_path, target_col, use_augmentation, n_aug, lowercase_labels
            )
            
            if error:
                print(f"22-feature CSV learning failed: {error}")
                return False
            
            # Display enhanced feedback
            self.display_learning_feedback(result)
            
            return True
            
        except Exception as e:
            print(f"22-feature CSV learning error: {e}")
            return False
    
    def view_model_info(self):
        """Display comprehensive 22-feature model information."""
        print("\n" + "="*60)
        print("22-FEATURE MODEL INFORMATION")
        print("="*60)
        
        result, error = self.server_api.get_model_info()
        
        if error:
            print(f"Failed to get model info: {error}")
            return
        
        if not isinstance(result, dict):
            print(f"Unexpected response format: {result}")
            return
        
        # Display model status
        print(f"Model Status:")
        print(f"   Loaded: {result.get('model_loaded', False)}")
        print(f"   Type: {result.get('model_type', 'Unknown')}")
        print(f"   Fitted: {result.get('is_fitted', False)}")
        print(f"   Online Learning: {result.get('supports_online_learning', False)}")
        
        # Display 22-feature configuration
        if 'feature_configuration' in result:
            feature_config = result['feature_configuration']
            print(f"\n22-Feature Configuration:")
            print(f"   Total features: {feature_config.get('total_features', 0)}")
            print(f"   Resistance sensors: {len(feature_config.get('resistance_sensors', []))}")
            print(f"   Environmental sensors: {len(feature_config.get('environmental_sensors', []))}")
            
            preprocessing = feature_config.get('preprocessing', {})
            print(f"   Preprocessing:")
            print(f"     Resistance: {preprocessing.get('resistance', 'Unknown')}")
            print(f"     Environmental: {preprocessing.get('environmental', 'Unknown')}")
        
        # Display feature information
        if 'feature_info' in result:
            feature_info = result['feature_info']
            print(f"\nFeature Information:")
            print(f"   Original features: {feature_info.get('original_features', 0)}")
            print(f"   Active features: {feature_info.get('active_features', 0)}")
            print(f"   Selected features: {feature_info.get('selected_features', 0)}")
        
        # Display class information
        if result.get('is_fitted', False):
            print(f"\nClassification Information:")
            classes = result.get('classes', [])
            print(f"   Number of classes: {result.get('n_classes', 0)}")
            print(f"   Classes: {', '.join(classes) if classes else 'None'}")
            
            if 'current_accuracy' in result:
                print(f"   Current accuracy: {result['current_accuracy']:.4f}")
            if 'training_samples' in result:
                print(f"   Training samples: {result['training_samples']}")
        else:
            print(f"\nModel not trained yet")
        
        print(f"\n22-feature model information retrieved successfully!")
    
    def visualize_and_analyze(self):
        """Generate 22-feature visualizations and analysis."""
        print("\n" + "="*60)
        print("22-FEATURE DATA VISUALIZATION AND ANALYSIS")
        print("="*60)
        
        try:
            # Generate visualizations
            print("Generating 22-feature visualizations...")
            viz_result, viz_error = self.server_api.visualize_data()
            
            if viz_error:
                print(f"Visualization failed: {viz_error}")
            else:
                print("22-feature visualizations generated successfully!")
                if 'plots' in viz_result:
                    print(f"Generated plots: {viz_result['plots']}")
                if 'feature_info' in viz_result:
                    feature_info = viz_result['feature_info']
                    print(f"Features: {feature_info.get('resistance_sensors', 0)} resistance + {feature_info.get('environmental_sensors', 0)} environmental")
            
            # Perform data analysis
            print("\nPerforming 22-feature data quality analysis...")
            analysis_result, analysis_error = self.server_api.analyze_data_quality()
            
            if analysis_error:
                print(f"Analysis failed: {analysis_error}")
            else:
                print("22-feature data quality analysis completed!")
            
            # Environmental sensor analysis (22-feature specific)
            print("\nPerforming environmental sensor analysis...")
            env_result, env_error = self.server_api.environmental_analysis()
            
            if env_error:
                print(f"Environmental analysis failed: {env_error}")
            else:
                print("Environmental sensor analysis completed!")
                if 'environmental_sensors' in env_result:
                    env_sensors = env_result['environmental_sensors']
                    print(f"Analyzed {len(env_sensors)} environmental sensors: {', '.join(env_sensors)}")
            
            return True
            
        except Exception as e:
            print(f"22-feature visualization/analysis error: {e}")
            return False

def main():
    """Main application updated for 22 features with offline mode support."""
    parser = argparse.ArgumentParser(description='Enhanced Vision-Smell Training Client - 22 Features with Offline Mode')
    parser.add_argument('--server', default="http://localhost:8016", help='Server URL')
    parser.add_argument('--port_enose', help='E-nose serial port')
    parser.add_argument('--port_UART', help='UART serial port')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--time', type=int, default=60, help='Recording time (seconds)')
    parser.add_argument('--samples', type=int, default=100, help='Target number of samples')
    parser.add_argument('--baud_enose', type=int, default=DEFAULT_BAUD_RATE, help='Enose baud rate')
    parser.add_argument('--baud_UART', type=int, default=DEFAULT_BAUD_RATE, help='UART baud rate')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode (simulate sensors)')
    args = parser.parse_args()
    
    mode_text = "OFFLINE MODE" if args.offline else "HARDWARE MODE"
    print("=" * 80)
    print(f"ENHANCED VISION-SMELL TRAINING CLIENT - 22 FEATURES ({mode_text})")
    if args.offline:
        print("Features: R1-R17 (simulated resistance) + T,H,CO2,H2S,CH2O (simulated environmental)")
        print("Offline mode: Generates realistic sensor data without physical hardware")
    else:
        print("Features: R1-R17 (resistance sensors) + T,H,CO2,H2S,CH2O (environmental sensors)")
    print("=" * 80)
    
    # Initialize components
    server_api = ServerAPI(args.server)
    webcam_handler = WebcamHandler(args.camera)
    enose_sensor = ENoseSensor([args.port_enose, args.port_UART], args.baud_enose, args.baud_UART, args.offline)
    
    # Test connections
    print("Testing server connection...")
    result, error = server_api.test_connection()
    if error:
        print(f"Server connection failed: {error}")
        print("Make sure the server is running on http://localhost:8016")
        return
    
    server_info = result if isinstance(result, dict) else {}
    print(f"Server connected: {server_info.get('message', 'OK')}")
    print(f"   Version: {server_info.get('version', 'Unknown')}")
    
    # Check 22-feature support
    sensor_config = server_info.get('sensor_configuration', {})
    total_features = sensor_config.get('total_features', 0)
    if total_features == 22:
        print(f"   22-feature support: CONFIRMED")
    else:
        print(f"   WARNING: Server reports {total_features} features, expected 22")
    
    models = server_info.get('models', {})
    print(f"   VLM loaded: {models.get('vlm_loaded', False)}")
    print(f"   Smell classifier: {models.get('smell_classifier_loaded', False)}")
    
    print(f"\nConnecting to sensors ({mode_text.lower()})...")
    if not enose_sensor.connect():
        print("Sensor connection failed")
        if not args.offline:
            print("Try running with --offline flag to use simulated sensors")
        return
    
    print("\nTesting camera...")
    if not webcam_handler.open_camera():
        print("Camera connection failed")
        return
    
    # Initialize pipeline
    pipeline = TrainingPipeline(server_api, webcam_handler, enose_sensor, args.time, args.samples)
    
    try:
        while True:
            print("\n" + "="*80)
            print(f"22-FEATURE MAIN MENU ({mode_text})")
            print("="*80)
            print("1. Full Training Cycle (Vision + 22-Feature Smell)")
            print("2. Vision Only (Object Detection)")
            print("3. 22-Feature Smell Identification")
            print("4. View 22-Feature Model Information")
            print("5. Learn from 22-Feature CSV File")
            print("6. Manual 22-Feature Sensor Testing")
            print("7. Generate 22-Feature Visualizations & Analysis")
            print("8. Exit")
            print("="*80)
            print(f"Recording settings: {args.samples} samples in {args.time}s (22 features)")
            if args.offline:
                print("Offline mode: All sensor data will be simulated")
            
            choice = input("Select option (1-8): ").strip()
            
            if choice == '1':
                pipeline.run_training_cycle()
            elif choice == '2':
                pipeline.run_vision_only()
            elif choice == '3':
                pipeline.identify_current_smell()
            elif choice == '4':
                pipeline.view_model_info()
            elif choice == '5':
                pipeline.learn_from_csv_file()
            elif choice == '6':
                pipeline.test_manual_22_feature_input()
            elif choice == '7':
                pipeline.visualize_and_analyze()
            elif choice == '8':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-8.")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        print("Cleaning up...")
        webcam_handler.close_camera()
        enose_sensor.disconnect()
        print("Goodbye!")

if __name__ == "__main__":
    main()