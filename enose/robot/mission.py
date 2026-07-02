"""Shared mission orchestration used by ``sim_run.py`` and ``run_robot_trainer.py``.

The actual policy state machine lives in :mod:`enose.robot.policy`. This
module just wires up the adapters, loads the targets file, and provides the
manual ``step()`` drive loop used by both entry points.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .interfaces import PolicyAdapters, Pose
from .policy import Policy, PolicyConfig, State, Target


def load_targets_file(path: str) -> List[Dict[str, Any]]:
    """JSON or YAML (auto-detected from suffix). YAML requires ``pyyaml`` to
    be installed; if unavailable we fall back to JSON only."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"targets file not found: {path}")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YAML targets require pyyaml — `pip install pyyaml` or use a .json file."
            ) from e
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"targets file must be a list, got {type(data).__name__}")
    return data


def make_targets(rows: List[Dict[str, Any]]) -> List[Target]:
    out: List[Target] = []
    for row in rows:
        wps = [
            Pose(float(w["x"]), float(w["y"]), float(w.get("theta", 0.0)), str(w.get("frame_id", "map")))
            for w in row.get("waypoints", [])
        ]
        out.append(Target(
            label=str(row["label"]),
            waypoints=wps,
            aliases=[str(a) for a in row.get("aliases", [])],
        ))
    return out


@dataclass
class MissionResult:
    summary: Dict[str, Any]
    ok: bool


def run_mission(
    adapters: PolicyAdapters,
    targets: List[Target],
    config: Optional[PolicyConfig] = None,
    sleep: Callable[[float], None] = time.sleep,
    on_transition: Optional[Callable[[State, State, str], None]] = None,
    cue_sensor: Optional[Callable[[Optional[Target], State], None]] = None,
) -> MissionResult:
    """Drive the policy to completion and return the summary.

    ``on_transition(from_state, to_state, note)`` — optional; called every
    time the state changes. Used by the CLIs for human-readable logs.

    ``cue_sensor(target, state)`` — optional hook fired before each ``step``;
    sim harnesses use it to flip the offline e-nose to the right
    ``simulation_smell``. Real robot leaves it ``None``.
    """
    policy = Policy(adapters, targets, config=config, sleep=sleep)

    last_state = policy.state
    policy._transition(State.SEARCH, note="mission start")  # type: ignore[attr-defined]
    try:
        while policy.state not in (State.DONE, State.ABORT):
            if cue_sensor is not None:
                cue_sensor(policy._active_target, policy.state)  # type: ignore[attr-defined]
            policy.step()
            if policy.state != last_state:
                note = policy.transitions[-1].note if policy.transitions else ""
                if on_transition is not None:
                    on_transition(last_state, policy.state, note)
                last_state = policy.state
    finally:
        policy._stop_publisher()  # type: ignore[attr-defined]

    summary = policy.summary()
    ok = summary["final_state"] == "done" and len(policy.commits) >= 1
    return MissionResult(summary=summary, ok=ok)


def print_summary(summary: Dict[str, Any]) -> None:
    print("[mission] complete:")
    print(f"  final_state    = {summary['final_state']}")
    print(f"  transitions    = {summary['n_transitions']}")
    print(f"  commits        = {summary['n_commits']}")
    print(f"  skipped        = {summary['skipped']}")
    print(f"  last_error     = {summary.get('last_error')}")
    for c in summary["commits"]:
        pose = c.get("pose")
        pose_s = (
            f"x={pose['x']:.2f} y={pose['y']:.2f} θ={pose['theta']:.2f}"
            if pose else "—"
        )
        print(
            f"    + commit  {c['label']:<12s}  score={c['grounding_score']:.3f}  "
            f"n={c['n_samples']:<4d}  pose=({pose_s})"
        )
