"""HTTP-backed vision client — used by both sim (with a real Florence-2 server)
and the real robot mission node. Independent of ROS2.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .interfaces import Detection


class HttpVisionClient:
    """Wraps :class:`enose.client.api.ServerAPI` and adapts its responses to
    the :class:`enose.robot.interfaces.VisionClient` protocol."""

    def __init__(
        self,
        server_api,
        capture_callback: Callable[[], Optional[str]],
    ) -> None:
        """
        ``server_api`` — an ``enose.client.api.ServerAPI`` instance.

        ``capture_callback`` — a callable that returns a fresh image path
        (e.g. ``WebcamHandler.capture_image``-style). Returns ``None`` if the
        capture failed.
        """
        self.api = server_api
        self.capture_cb = capture_callback

    def capture(self) -> Optional[str]:
        try:
            return self.capture_cb()
        except Exception:
            return None

    def detect_scene(self, image_path: str, labels: Sequence[str]) -> List[Detection]:
        payload, err = self.api.detect_scene(image_path, list(labels))
        if err or not isinstance(payload, dict):
            return []
        out: List[Detection] = []
        for row in payload.get("detections", []) or []:
            if not isinstance(row, dict):
                continue
            try:
                out.append(
                    Detection(
                        label=str(row["label"]),
                        bbox=[float(c) for c in row["bbox"]],
                        score=float(row.get("score", 0.0)),
                        area_fraction=float(row["area_fraction"]) if "area_fraction" in row else None,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return out
