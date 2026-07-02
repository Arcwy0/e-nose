"""TF2-backed localizer (skeleton).

The real impl will use :class:`tf2_ros.Buffer` to look up
``map → base_link`` (the FAST-LIO2 output frame chain) and also publish the
static ``base_link → enose_intake`` transform for the 15 cm-forward sensor
mount.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..interfaces import Pose
from . import require_ros2


class TFLocalizer:
    """Reads pose from a TF2 buffer. ROS2-only."""

    def __init__(
        self,
        node_name: str = "enose_localizer",
        target_frame: str = "map",
        source_frame: str = "base_link",
    ) -> None:
        require_ros2()
        raise NotImplementedError("TFLocalizer is a skeleton — wired up in Step 1.9.")

    def current(self) -> Optional[Pose]: raise NotImplementedError
    def pose_dict(self) -> Optional[Dict[str, float]]: raise NotImplementedError
    def nose_frame(self) -> Optional[Pose]: raise NotImplementedError
