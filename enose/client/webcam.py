"""Webcam capture + detection overlay display.

Opens an OpenCV preview window; SPACE captures, ESC cancels. After a detection
response comes back from the server, `display_detection_result` renders the image
with bboxes drawn via matplotlib.
"""

from __future__ import annotations

import ast
import json
import traceback
from typing import Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from enose.config import TEMP_IMAGE_PATH


class WebcamHandler:
    """Webcam wrapper: open → preview → capture-on-SPACE."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None

    def open_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Could not open camera {self.camera_index}")
                return False
            print(f"Camera {self.camera_index} opened")
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False

    def close_camera(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def capture_image(self, save_path: str = TEMP_IMAGE_PATH) -> Tuple[Optional[object], Optional[str]]:
        """Show live preview; SPACE saves a frame to `save_path`, ESC aborts."""
        if not self.cap or not self.cap.isOpened():
            if not self.open_camera():
                return None, None

        print("Preview active. SPACE = capture, ESC = cancel.")
        window_name = f"Camera {self.camera_index} Preview"

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(10) & 0xFF
            if key == 32:  # SPACE
                try:
                    cv2.imwrite(save_path, frame)
                    cv2.destroyWindow(window_name)
                    print(f"Image saved → {save_path}")
                    return frame, save_path
                except Exception as e:
                    print(f"Save error: {e}")
                    return frame, None
            if key == 27:  # ESC
                print("Capture cancelled")
                break

        cv2.destroyWindow(window_name)
        return None, None

    def display_detection_result(self, image_path: str, detection_result) -> None:
        """Render bboxes on top of the captured image using matplotlib."""
        try:
            parsed = _parse_detection(detection_result)

            image = Image.open(image_path)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image)
            ax.set_title("Object Detection Results")
            ax.axis("off")

            if parsed and isinstance(parsed, dict) and "<OPEN_VOCABULARY_DETECTION>" in parsed:
                det = parsed["<OPEN_VOCABULARY_DETECTION>"]
                bboxes = det.get("bboxes", [])
                labels = det.get("bboxes_labels", det.get("labels", []))
                polygons = det.get("polygons", [])
                polygon_labels = det.get("polygons_labels", [])
                print(f"Detection: {len(bboxes)} box(es), {len(polygons)} polygon region(s)")

                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = map(float, bbox)
                    ax.add_patch(patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor="red", facecolor="none",
                    ))
                    ax.text(
                        x1, y1 - 5, label,
                        color="white", fontsize=10, weight="bold",
                        bbox=dict(facecolor="red", alpha=0.8, pad=2),
                    )
                    print(f"  {i + 1}. {label}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

                # OPEN_VOCABULARY_DETECTION can return polygons instead of boxes.
                for parts, label in zip(polygons, polygon_labels):
                    for vertices in parts:
                        if len(vertices) < 6 or len(vertices) % 2:
                            continue
                        points = list(zip(map(float, vertices[::2]), map(float, vertices[1::2])))
                        ax.add_patch(patches.Polygon(
                            points, closed=True, fill=False, linewidth=2, edgecolor="orange",
                        ))
                        ax.text(
                            points[0][0], points[0][1] - 5, label,
                            color="white", fontsize=10, weight="bold",
                            bbox=dict(facecolor="darkorange", alpha=0.8, pad=2),
                        )
                    print(f"  polygon: {label}")

                if not bboxes and not polygons:
                    ax.text(
                        0.5, 0.5, "No matching regions returned by the model",
                        transform=ax.transAxes, ha="center", va="center", fontsize=14,
                        bbox=dict(facecolor="orange", alpha=0.7, pad=3),
                    )
            else:
                msg = "No detection data found" if parsed else "Could not parse detection result"
                color = "yellow" if parsed else "orange"
                ax.text(
                    0.5, 0.5, msg, transform=ax.transAxes,
                    ha="center", va="center", fontsize=16,
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                )

            plt.tight_layout()
            plt.show(block=True)
        except Exception as e:
            print(f"Display error: {e}")
            traceback.print_exc()


def _parse_detection(result):
    """Decode legacy payloads and unwrap the server's /predict/object envelope."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(result)
            except (ValueError, SyntaxError):
                return None
    # The HTTP API returns {prompt, task, result: {<TASK>: {...}}}; old
    # callers also pass the inner {<TASK>: {...}} directly. Accept both.
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        return result["result"]
    return result
