"""Sim localizer — just hands back whatever the sim motion has set."""

from __future__ import annotations

import time
from typing import Dict, Optional

from ..interfaces import Pose


class SimLocalizer:
    """Reads from the same shared ``state`` dict :class:`SimMotion` writes."""

    def __init__(self, state: dict) -> None:
        self.state = state

    def current(self) -> Optional[Pose]:
        p = self.state.get("pose")
        if p is None:
            return None
        # Stamp with capture time on read so LivePublisher gets a fresh ``t``.
        return Pose(p.x, p.y, p.theta, p.frame_id, t=time.time())

    def pose_dict(self) -> Optional[Dict[str, float]]:
        p = self.current()
        return p.as_dict() if p else None
