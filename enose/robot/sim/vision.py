"""Scripted vision client — returns canned detections per call.

Lets the sim harness drive the policy through SEARCH/APPROACH/COMMIT without
needing Florence-2.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..interfaces import Detection


class ScriptedVision:
    """Returns detections from a callable the harness can hot-swap mid-mission.

    Two construction modes:

    * ``script``: ``(image_path, labels) -> list[Detection]`` — full control
      from the harness.
    * ``fixed``: a static list of detections returned for every call.
    """

    def __init__(
        self,
        script: Optional[Callable[[str, Sequence[str]], List[Detection]]] = None,
        fixed: Optional[List[Detection]] = None,
        image_writer: Optional[Callable[[], str]] = None,
    ) -> None:
        if script is None and fixed is None:
            fixed = []
        self._script = script
        self._fixed: List[Detection] = list(fixed) if fixed is not None else []
        self._image_writer = image_writer

    def capture(self) -> Optional[str]:
        if self._image_writer is not None:
            return self._image_writer()
        # Default: write a 1×1 PNG so the policy has *something* to pass around.
        path = os.path.join(tempfile.gettempdir(), "sim_capture.png")
        if not os.path.exists(path):
            import struct, zlib  # noqa: E401 — tiny deps, keep them local
            sig = b"\x89PNG\r\n\x1a\n"
            def c(t: bytes, d: bytes) -> bytes:
                return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            idat = zlib.compress(b"\x00\x80\x80\x80")
            with open(path, "wb") as f:
                f.write(sig + c(b"IHDR", ihdr) + c(b"IDAT", idat) + c(b"IEND", b""))
        return path

    def detect_scene(self, image_path: str, labels: Sequence[str]) -> List[Detection]:
        if self._script is not None:
            return list(self._script(image_path, labels))
        return list(self._fixed)
