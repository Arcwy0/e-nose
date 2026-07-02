"""Robot integration package — Unitree Go-1 + e-nose mission orchestration.

Hard rule: everything that depends on ROS2 (``rclpy``, ``geometry_msgs``,
``tf2_ros`` …) is import-guarded. The package must be importable on a plain
laptop/Docker container for the simulation harness (Step 1.6) and the smoke
test to work.

Public entry points:

* :class:`Pose` — the (x, y, θ, frame_id) tuple used everywhere here.
* :class:`MotionAdapter`, :class:`Localizer`, :class:`PumpController`,
  :class:`VisionClient`, :class:`PolicyAdapters` — Protocol-style interfaces
  the policy depends on. The sim and ROS2 impls live in
  :mod:`enose.robot.sim` and :mod:`enose.robot.ros2` respectively.
* :class:`Policy` — the Idea-1 state machine (full impl lands in Step 1.7).
"""

from __future__ import annotations

from .interfaces import (
    MotionAdapter,
    Localizer,
    PumpController,
    VisionClient,
    PolicyAdapters,
    Pose,
)
from .manual_pump import ManualPump

__all__ = [
    "Pose",
    "MotionAdapter",
    "Localizer",
    "PumpController",
    "VisionClient",
    "PolicyAdapters",
    "ManualPump",
]
