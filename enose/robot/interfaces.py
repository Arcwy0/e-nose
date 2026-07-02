"""Pure-Python interfaces the policy depends on.

Using ``typing.Protocol`` (structural) rather than ABCs (nominal) so the sim
and ROS2 impls don't need to inherit anything — easier to mock for tests and
keeps ROS2 imports out of the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass
class Pose:
    """2D robot pose in a named frame.

    Convention: ``theta`` is yaw in radians, right-handed (counter-clockwise
    positive when viewed from above). ``frame_id`` defaults to ``"map"`` to
    match REP-105 / what FAST-LIO2 publishes.
    """

    x: float
    y: float
    theta: float
    frame_id: str = "map"
    t: Optional[float] = None  # capture time (unix s); None = unknown

    def as_dict(self) -> Dict[str, float]:
        """Wire format matching ``LivePose`` on the server side."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "theta": float(self.theta),
            "frame_id": str(self.frame_id),
        }


@dataclass
class Detection:
    """One grounded object from ``/predict/scene``.

    Mirrors the server response; we keep it as a dataclass so the policy
    can reason about it without dict-key typos.
    """

    label: str
    bbox: List[float]               # [x1, y1, x2, y2] in image pixels
    score: float                    # bbox-area-fraction proxy, [0, 1]
    area_fraction: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class Localizer(Protocol):
    """Anything that can answer 'where is the robot right now?'."""

    def current(self) -> Optional[Pose]:
        """Latest pose, or ``None`` if the source has not yet produced one."""
        ...

    def pose_dict(self) -> Optional[Dict[str, float]]:
        """Convenience wrapper for ``LivePublisher.pose_lookup``.

        Returns ``None`` (so the publisher just omits the pose) when the
        source isn't ready yet — never raises.
        """
        ...


class MotionAdapter(Protocol):
    """High-level motion primitives. All blocking until completion or timeout."""

    def stand(self) -> bool: ...
    def stop(self) -> bool: ...
    def face(self, yaw: float) -> bool: ...
    def walk_to(
        self,
        x: float,
        y: float,
        yaw: Optional[float] = None,
        frame: str = "map",
        timeout: float = 30.0,
    ) -> bool:
        """Drive to ``(x, y, yaw)`` in ``frame``. Returns True if reached."""
        ...

    def cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        """Low-rate velocity setpoint — used by the visual servo loop."""
        ...


class PumpController(Protocol):
    """USB-relay-driven air pump controller. Plain on/off; reverse left for v2."""

    def on(self) -> bool: ...
    def off(self) -> bool: ...
    def is_on(self) -> bool: ...


class VisionClient(Protocol):
    """What the policy needs from the vision side — a thin wrapper over the
    server's ``/predict/scene`` (and optionally ``/predict/object``)."""

    def capture(self) -> Optional[str]:
        """Take a fresh image and return a path the server can be POSTed."""
        ...

    def detect_scene(self, image_path: str, labels: Sequence[str]) -> List[Detection]:
        """Ground every candidate label in ``labels``. Empty list if none match."""
        ...


@dataclass
class PolicyAdapters:
    """Bundle of dependencies the policy needs. Lets the sim and the real
    mission node configure them differently."""

    motion: MotionAdapter
    localizer: Localizer
    pump: PumpController
    vision: VisionClient
    sensor: Any                       # an ENoseSensor-like duck (read_single_measurement)
    server_api: Any                   # an enose.client.api.ServerAPI
    settling_endpoint: str = "/smell/settling"
