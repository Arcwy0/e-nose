"""USB-relay-driven pump controller (skeleton).

Real impl: ROS2 service client over ``/pump/set`` (boolean request) backed
by a tiny ``pump_node`` that toggles the relay via ``pyudev`` / serial.
"""

from __future__ import annotations

from . import require_ros2


class Ros2Pump:
    def __init__(self, service_name: str = "/pump/set") -> None:
        require_ros2()
        raise NotImplementedError("Ros2Pump is a skeleton — wired up in Step 1.9.")

    def on(self) -> bool: raise NotImplementedError
    def off(self) -> bool: raise NotImplementedError
    def is_on(self) -> bool: raise NotImplementedError
