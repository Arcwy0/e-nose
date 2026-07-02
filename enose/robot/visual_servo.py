"""Pure (bbox + depth) → cmd_vel logic. No ROS2.

This module is intentionally side-effect-free so we can unit-test it without
the robot. The real mission node calls :func:`step` in a control loop:

    cmd = visual_servo.step(bbox, image_size, depth_at_center)
    motion.cmd_vel(cmd.vx, cmd.vy, cmd.wz)
    if cmd.done:
        ...

Control strategy:

* **Yaw rate** ``wz`` proportional to horizontal bbox-center offset from the
  image center. Negative offset (target left of center) → rotate left.
* **Forward speed** ``vx`` proportional to remaining distance ``depth - d_record``.
  Held at 0 if the bbox isn't roughly centered yet — we want the robot to
  *face* the target before walking at it.
* **Done** when the bbox center is within ``bbox_centered_px_tol`` of the
  image x-center AND ``depth_at_center < d_record``.

The gains are conservative; tune empirically against a real depth camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ServoCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    done: bool = False
    reason: Optional[str] = None
    # Diagnostics for the policy / mission log:
    offset_px: Optional[float] = None
    depth_m: Optional[float] = None


@dataclass
class ServoConfig:
    d_record: float = 0.30           # m — stop distance
    bbox_centered_px_tol: int = 30   # half-width tolerance around image x-center
    max_v: float = 0.25              # m/s
    max_w: float = 0.6               # rad/s
    yaw_gain: float = 0.004          # rad/s per pixel offset
    fwd_gain: float = 0.6            # m/s per m of remaining distance
    bbox_centered_for_fwd: int = 60  # only drive forward when offset < this


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def step(
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    depth_at_center: Optional[float],
    config: Optional[ServoConfig] = None,
) -> ServoCommand:
    """One control step. ``bbox`` is ``(x1, y1, x2, y2)`` in pixels;
    ``image_size`` is ``(W, H)``. ``depth_at_center`` is the depth in metres
    at the bbox-center pixel (the camera driver's job to provide). Pass
    ``None`` if depth is unavailable — the function will still center on
    the target but will not declare ``done`` (caller must use an alternate
    proximity check, e.g. bbox-area-fraction)."""
    cfg = config or ServoConfig()

    if not bbox or len(bbox) != 4:
        return ServoCommand(reason="invalid bbox")
    x1, y1, x2, y2 = (float(c) for c in bbox)
    W, _ = image_size
    if W <= 0:
        return ServoCommand(reason="invalid image_size")

    cx = 0.5 * (x1 + x2)
    image_cx = 0.5 * W
    offset_px = cx - image_cx
    centered = abs(offset_px) <= cfg.bbox_centered_px_tol
    centered_enough_for_fwd = abs(offset_px) <= cfg.bbox_centered_for_fwd

    # Yaw: gain × offset, clamped. Sign convention: positive offset → target
    # is to the right → rotate right (negative wz in REP-103 / ROS REP-105).
    wz = _clamp(-cfg.yaw_gain * offset_px, -cfg.max_w, cfg.max_w)

    # Forward only when we're roughly facing the target.
    vx = 0.0
    if centered_enough_for_fwd and depth_at_center is not None and math.isfinite(depth_at_center):
        remaining = depth_at_center - cfg.d_record
        if remaining > 0.0:
            vx = _clamp(cfg.fwd_gain * remaining, 0.0, cfg.max_v)

    done = bool(
        centered
        and depth_at_center is not None
        and math.isfinite(depth_at_center)
        and depth_at_center <= cfg.d_record
    )
    reason = None
    if done:
        reason = "centered and within d_record"

    return ServoCommand(
        vx=vx, vy=0.0, wz=wz, done=done, reason=reason,
        offset_px=offset_px, depth_m=depth_at_center,
    )
