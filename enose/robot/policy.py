"""Idea-1 mission policy — state machine.

`IDLE → SEARCH → APPROACH → SETTLE → RECORD → COMMIT → PURGE → SEARCH (next)`

Pluggable via :class:`PolicyAdapters` so the sim harness and the real ROS2
mission node share the exact same logic. The state machine itself is
deliberately plain-Python (no behaviour-tree library, no asyncio): the
hardest part of an autonomous-training loop is *what to do*, not *how to
schedule it*, and a procedural impl is the easiest to debug.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from enose.client.live import LivePublisher

from .interfaces import Detection, PolicyAdapters, Pose


class State(str, enum.Enum):
    IDLE = "idle"
    SEARCH = "search"
    APPROACH = "approach"
    SETTLE = "settle"
    RECORD = "record"
    COMMIT = "commit"
    PURGE = "purge"
    DONE = "done"
    ABORT = "abort"


@dataclass
class Target:
    """One row in the mission's target list.

    ``waypoints``: world-frame poses to scan from in order. SEARCH walks to
    each in turn, captures + grounds, and moves on if nothing's detected.

    ``aliases``: extra phrases to feed Florence-2 alongside ``label`` to
    increase recall (e.g. ``label="rose"``, ``aliases=["red rose", "flower"]``).
    """

    label: str
    waypoints: List[Pose] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


@dataclass
class PolicyConfig:
    """Knobs all in one place. Provisional values from the plan §3.5."""

    tau_grounding: float = 0.05
    tau_commit: float = 0.10
    d_record: float = 0.30
    settling_timeout: float = 60.0
    settling_window: float = 8.0
    settling_threshold: float = 0.02
    settling_k_consec: int = 3
    settling_poll_period: float = 1.0
    record_seconds: float = 30.0
    purge_seconds: float = 30.0
    publisher_rate_hz: float = 5.0
    # When True, after detection the policy reads detection.extras["target_pose"]
    # and walks straight there (sim path / known-waypoint path). When False, the
    # policy expects ``visual_servo`` to drive the approach (real depth-camera
    # path; landing in Step 1.8).
    use_target_pose_extras: bool = True


@dataclass
class Transition:
    """One row of the policy log. Used by the sim harness's assertions."""

    t: float
    from_state: State
    to_state: State
    note: str = ""


@dataclass
class CommitRecord:
    """What a single successful COMMIT produced. Useful for the audit tool."""

    label: str
    grounding_score: float
    n_samples: int
    pose_at_record: Optional[Pose]
    server_response: Dict[str, Any]


class Policy:
    """Idea-1 state machine. See module docstring for the high-level flow."""

    def __init__(
        self,
        adapters: PolicyAdapters,
        targets: List[Target],
        config: Optional[PolicyConfig] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.adapters = adapters
        self.targets = list(targets)
        self.config = config or PolicyConfig()
        self._sleep = sleep

        self.state: State = State.IDLE
        self._idx: int = 0
        self._wp_idx: int = 0
        self._active_target: Optional[Target] = None
        self._active_detection: Optional[Detection] = None
        self._active_image_path: Optional[str] = None
        self._t_state_entered: float = time.time()

        # Per-target scratch:
        self._collected_samples: List[Dict[str, float]] = []
        self._pose_at_record: Optional[Pose] = None
        self._collecting: bool = False  # True only while RECORD is active

        # Long-running publisher (one per mission). on_sample appends to the
        # current target's list when ``_collecting`` is True. Outside RECORD
        # the publisher still pushes samples to the server's live buffer so
        # ``/smell/settling`` and ``/smell/drift`` have data to look at.
        self._publisher: Optional[LivePublisher] = None

        self.transitions: List[Transition] = []
        self.commits: List[CommitRecord] = []
        self.skipped: List[str] = []          # target.label values that were skipped
        self.last_error: Optional[str] = None

    # ── orchestration ─────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """Run to completion. Blocks. Returns a mission summary."""
        self._start_publisher()
        try:
            self._transition(State.SEARCH, note="mission start")
            while self.state not in (State.DONE, State.ABORT):
                self.step()
        finally:
            self._stop_publisher()
        return self.summary()

    def _start_publisher(self) -> None:
        if self._publisher is not None:
            return

        def _on_sample(entry: Dict[str, Any]) -> None:
            if not self._collecting:
                return
            s = entry.get("sample")
            if isinstance(s, dict) and s:
                self._collected_samples.append(dict(s))

        self._publisher = LivePublisher(
            sensor=self.adapters.sensor,
            server_url=getattr(self.adapters.server_api, "base_url", "") or "",
            rate_hz=self.config.publisher_rate_hz,
            classify=False,
            session_label=None,
            on_sample=_on_sample,
            pose_lookup=getattr(self.adapters.localizer, "pose_dict", None),
        )
        self._publisher.start()

    def _stop_publisher(self) -> None:
        if self._publisher is None:
            return
        try:
            self._publisher.stop()
        except Exception as e:  # pragma: no cover
            self.last_error = f"publisher.stop: {e}"
        self._publisher = None

    def step(self) -> None:
        # Idempotent — lets callers that drive the state machine manually
        # (sim harness, debugging) work without remembering to call run().
        self._start_publisher()
        handler = {
            State.SEARCH: self._search,
            State.APPROACH: self._approach,
            State.SETTLE: self._settle,
            State.RECORD: self._record,
            State.COMMIT: self._commit,
            State.PURGE: self._purge,
        }.get(self.state)
        if handler is None:
            # Terminal state — nothing to do.
            return
        handler()

    def summary(self) -> Dict[str, Any]:
        return {
            "final_state": self.state.value,
            "n_transitions": len(self.transitions),
            "n_commits": len(self.commits),
            "n_skipped": len(self.skipped),
            "skipped": list(self.skipped),
            "commits": [
                {
                    "label": c.label,
                    "grounding_score": c.grounding_score,
                    "n_samples": c.n_samples,
                    "pose": c.pose_at_record.as_dict() if c.pose_at_record else None,
                }
                for c in self.commits
            ],
            "transitions": [
                {"t": tr.t, "from": tr.from_state.value, "to": tr.to_state.value, "note": tr.note}
                for tr in self.transitions
            ],
            "last_error": self.last_error,
        }

    # ── state handlers ───────────────────────────────────────────────────

    def _search(self) -> None:
        if self._idx >= len(self.targets):
            self._transition(State.DONE, note="all targets processed")
            return

        target = self.targets[self._idx]
        self._active_target = target

        # If there are waypoints, walk to the next one before scanning.
        if target.waypoints and self._wp_idx < len(target.waypoints):
            wp = target.waypoints[self._wp_idx]
            self.adapters.motion.walk_to(wp.x, wp.y, wp.theta, frame=wp.frame_id)

        # Capture + ground.
        image_path = self.adapters.vision.capture()
        if image_path is None:
            self.last_error = "vision.capture returned None"
            self._next_waypoint_or_skip("no image captured")
            return

        labels = [target.label] + list(target.aliases)
        detections = self.adapters.vision.detect_scene(image_path, labels)
        # Pick the highest-scoring detection above tau_grounding whose label
        # matches the target (or any of its aliases) case-insensitively.
        wanted = {l.lower() for l in labels}
        best = max(
            (d for d in detections if d.label.lower() in wanted and d.score >= self.config.tau_grounding),
            key=lambda d: d.score,
            default=None,
        )
        if best is None:
            self._next_waypoint_or_skip("no detection above tau_grounding")
            return

        self._active_detection = best
        self._active_image_path = image_path
        self._transition(State.APPROACH, note=f"detected {best.label} score={best.score:.3f}")

    def _next_waypoint_or_skip(self, why: str) -> None:
        target = self._active_target
        if target is None:
            self._idx += 1
            self._wp_idx = 0
            self._transition(State.SEARCH, note=why)
            return
        self._wp_idx += 1
        if self._wp_idx >= max(1, len(target.waypoints)):
            self.skipped.append(target.label)
            self._idx += 1
            self._wp_idx = 0
            self._active_target = None
            self._transition(State.SEARCH, note=f"skip {target.label}: {why}")
        else:
            # Stay in SEARCH; the next call advances to the next waypoint.
            self._transition(State.SEARCH, note=f"next waypoint: {why}")

    def _approach(self) -> None:
        det = self._active_detection
        if det is None:
            self._transition(State.ABORT, note="approach without active detection")
            return

        # Pump OFF during approach so travel air doesn't pre-saturate sensors.
        try:
            self.adapters.pump.off()
        except Exception as e:
            self.last_error = f"pump.off: {e}"

        if self.config.use_target_pose_extras and "target_pose" in det.extras:
            tp = det.extras["target_pose"]
            try:
                # Accept either a Pose, a dict, or a (x, y, theta) tuple.
                if isinstance(tp, Pose):
                    x, y, yaw, frame = tp.x, tp.y, tp.theta, tp.frame_id
                elif isinstance(tp, dict):
                    x = float(tp["x"]); y = float(tp["y"])
                    yaw = float(tp.get("theta", 0.0))
                    frame = str(tp.get("frame_id", "map"))
                else:
                    x, y, yaw = (float(v) for v in tp[:3])
                    frame = "map"
                self.adapters.motion.walk_to(x, y, yaw, frame=frame)
            except Exception as e:
                self.last_error = f"walk_to(target_pose) failed: {e}"
                self._transition(State.ABORT, note="approach walk_to failed")
                return
            self._transition(State.SETTLE, note=f"arrived at target_pose for {det.label}")
            return

        # Real-robot path: hand off to visual_servo. Not exercised in sim.
        self.last_error = "visual_servo path not yet implemented (Step 1.8)"
        self._transition(State.ABORT, note="no visual_servo path")

    def _settle(self) -> None:
        # Pump ON during settle and record.
        try:
            self.adapters.pump.on()
        except Exception as e:
            self.last_error = f"pump.on: {e}"

        cfg = self.config
        consec = 0
        t_started = time.time()
        while time.time() - t_started < cfg.settling_timeout:
            result, err = self.adapters.server_api.get_settling(
                window=cfg.settling_window,
                threshold=cfg.settling_threshold,
            )
            if err:
                # If the buffer is empty (start of mission) the server returns
                # settled=false with a reason — that's fine, keep polling.
                self.last_error = f"settling: {err}"
            elif isinstance(result, dict) and result.get("settled"):
                consec += 1
                if consec >= cfg.settling_k_consec:
                    self._transition(State.RECORD, note=f"settled (k={consec})")
                    return
            else:
                consec = 0
            self._sleep(cfg.settling_poll_period)
        # Timeout — proceed to RECORD anyway with a note. The PDF agreed: a
        # stuck sensor must not hang the mission.
        self._transition(State.RECORD, note="settle timeout — recording anyway")

    def _record(self) -> None:
        det = self._active_detection
        assert det is not None  # invariant from SEARCH

        # Pose at the start of RECORD — saved with the commit for provenance.
        try:
            self._pose_at_record = self.adapters.localizer.current()
        except Exception:
            self._pose_at_record = None

        # Toggle the long-running publisher's per-target collection on for
        # ``record_seconds``. The publisher itself keeps running so SETTLE
        # for the next target still has live data to look at.
        self._collected_samples = []
        if self._publisher is not None:
            self._publisher.session_label = det.label
        self._collecting = True
        try:
            self._sleep(self.config.record_seconds)
        finally:
            self._collecting = False

        self.last_error = f"recorded {len(self._collected_samples)} samples"
        self._transition(State.COMMIT, note=f"n={len(self._collected_samples)}")

    def _commit(self) -> None:
        det = self._active_detection
        target = self._active_target
        assert det is not None and target is not None

        if not self._collected_samples:
            self.skipped.append(target.label)
            self._transition(State.PURGE, note="no samples to commit")
            return

        if det.score < self.config.tau_commit:
            self.skipped.append(target.label)
            self._transition(
                State.PURGE,
                note=f"score {det.score:.3f} < tau_commit {self.config.tau_commit:.3f}; skipping",
            )
            return

        provenance = self._build_provenance(det)
        result, err = self.adapters.server_api.online_learning(
            self._collected_samples, target.label, provenance=provenance,
        )
        if err:
            self.last_error = f"online_learning: {err}"
            self.skipped.append(target.label)
            self._transition(State.PURGE, note="online_learning failed")
            return

        self.commits.append(
            CommitRecord(
                label=target.label,
                grounding_score=det.score,
                n_samples=len(self._collected_samples),
                pose_at_record=self._pose_at_record,
                server_response=result if isinstance(result, dict) else {"result": result},
            )
        )
        self._transition(State.PURGE, note=f"committed {target.label}")

    def _build_provenance(self, det: Detection) -> Optional[Dict[str, Any]]:
        """Return a ``CommitProvenance``-shaped dict for the audit trail.

        Best-effort: a missing image or an unreadable file is fine — the
        server persists whatever subset we manage to ship.
        """
        prov: Dict[str, Any] = {
            "grounding_score": float(det.score),
            "bbox": list(det.bbox) if det.bbox else None,
            "pose": self._pose_at_record.as_dict() if self._pose_at_record else None,
            "session_id": getattr(self._publisher, "session_id", None) if self._publisher else None,
            "detector_task": det.extras.get("task") if isinstance(det.extras, dict) else None,
            "extras": {
                "label": det.label,
                "area_fraction": det.area_fraction,
            },
        }
        path = self._active_image_path
        if path:
            try:
                import base64, os as _os
                with open(path, "rb") as f:
                    prov["image_b64"] = base64.b64encode(f.read()).decode("ascii")
                prov["image_filename"] = _os.path.basename(path)
            except Exception as e:  # pragma: no cover — image gone, just skip
                self.last_error = f"image read for provenance failed: {e}"
        return prov

    def _purge(self) -> None:
        try:
            self.adapters.pump.off()
        except Exception as e:
            self.last_error = f"pump.off (purge): {e}"
        self._sleep(self.config.purge_seconds)

        # Done with this target — advance.
        self._active_target = None
        self._active_detection = None
        self._active_image_path = None
        self._collected_samples = []
        self._pose_at_record = None
        self._idx += 1
        self._wp_idx = 0
        self._transition(State.SEARCH, note="purge complete")

    # ── helpers ──────────────────────────────────────────────────────────
    def _transition(self, to: State, note: str = "") -> None:
        prev = self.state
        self.transitions.append(Transition(time.time(), prev, to, note))
        self.state = to
        self._t_state_entered = time.time()
