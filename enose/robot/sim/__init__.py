"""Simulation impls of the robot interfaces. No hardware, no ROS2.

Used by the simulation harness (``scripts/sim_run.py``, Step 1.6) so the
mission policy can be exercised end-to-end on a laptop / inside Docker.
"""

from .localization import SimLocalizer
from .motion import SimMotion
from .pump import SimPump
from .vision import ScriptedVision

__all__ = ["SimLocalizer", "SimMotion", "SimPump", "ScriptedVision"]
