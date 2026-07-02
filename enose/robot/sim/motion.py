"""Instantaneous-teleport motion adapter for tests."""

from __future__ import annotations

import time
from typing import Optional

from ..interfaces import Pose


class SimMotion:
    """Updates a shared :class:`Pose` directly. Stand/stop are no-ops.

    Cooperates with :class:`SimLocalizer` via a shared ``state`` dict so the
    localizer always returns the latest sim pose.
    """

    def __init__(self, state: Optional[dict] = None) -> None:
        # state is a dict shared with SimLocalizer: {"pose": Pose|None, "cmd_vel": (vx, vy, wz)}
        self.state = state if state is not None else {"pose": Pose(0.0, 0.0, 0.0), "cmd_vel": (0.0, 0.0, 0.0)}

    def stand(self) -> bool:
        return True

    def stop(self) -> bool:
        self.state["cmd_vel"] = (0.0, 0.0, 0.0)
        return True

    def face(self, yaw: float) -> bool:
        p = self.state.get("pose") or Pose(0.0, 0.0, 0.0)
        self.state["pose"] = Pose(p.x, p.y, float(yaw), p.frame_id, t=time.time())
        return True

    def walk_to(
        self,
        x: float,
        y: float,
        yaw: Optional[float] = None,
        frame: str = "map",
        timeout: float = 30.0,
    ) -> bool:
        p = self.state.get("pose") or Pose(0.0, 0.0, 0.0)
        self.state["pose"] = Pose(float(x), float(y), float(yaw) if yaw is not None else p.theta, frame, t=time.time())
        return True

    def cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        self.state["cmd_vel"] = (float(vx), float(vy), float(wz))
