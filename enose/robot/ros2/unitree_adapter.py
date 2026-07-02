"""Unitree Go-1 high-level SDK adapter (skeleton).

Real implementation will:

* Publish ``Twist`` on ``/cmd_vel`` for low-rate velocity control (visual
  servo).
* Call the Unitree ROS2 driver's ``sport_client``-style action for
  ``walk_to``, ``face``, ``stand``, ``stop``.

The skeleton here gives the policy a class to depend on; instantiating it
without ROS2 raises a clear error so the sim path is never confused with
the real path.
"""

from __future__ import annotations

from typing import Optional

from . import require_ros2


class UnitreeAdapter:
    """High-level motion adapter for the Go-1 EDU. ROS2-only."""

    def __init__(self, node_name: str = "enose_unitree_adapter") -> None:
        require_ros2()
        # Real impl will create an rclpy.node.Node and the relevant pubs/clients.
        raise NotImplementedError("UnitreeAdapter is a skeleton — wired up in Step 1.9.")

    # Signature mirrors :class:`enose.robot.interfaces.MotionAdapter`.
    def stand(self) -> bool: raise NotImplementedError
    def stop(self) -> bool: raise NotImplementedError
    def face(self, yaw: float) -> bool: raise NotImplementedError
    def walk_to(
        self,
        x: float, y: float,
        yaw: Optional[float] = None,
        frame: str = "map", timeout: float = 30.0,
    ) -> bool: raise NotImplementedError
    def cmd_vel(self, vx: float, vy: float, wz: float) -> None: raise NotImplementedError
