"""ROS2-backed implementations of the robot interfaces.

Imports are guarded so importing :mod:`enose.robot.ros2` on a host without
``rclpy`` raises a clear error only when you actually call into it. The
modules themselves can still be imported at the package level for inspection
(e.g. ``enose.robot.ros2.unitree_adapter.UnitreeAdapter``).
"""

from __future__ import annotations

try:
    import rclpy  # type: ignore  # noqa: F401
    _ROS2_AVAILABLE = True
except Exception:  # pragma: no cover — host without ROS2
    _ROS2_AVAILABLE = False


def require_ros2() -> None:
    if not _ROS2_AVAILABLE:
        raise RuntimeError(
            "ROS2 (rclpy) is not available. This adapter only works on the Go-1 "
            "Jetson (or any machine with a ROS2 install). Use enose.robot.sim "
            "for laptop / Docker testing."
        )
