#!/usr/bin/env python3
"""End-to-end simulation harness for the Idea-1 mission policy.

Runs the full state machine without robot hardware — uses sim adapters from
:mod:`enose.robot.sim` plus :class:`ENoseSensor` in offline mode.

Requires a running server (the Phase-0 server with ``/predict/scene`` and
``/smell/settling``). The VLM is *not* consulted: the sim's
:class:`ScriptedVision` returns canned detections directly.

Targets file (JSON): see ``scripts/sim_targets.json`` for an example.

Usage::

    python scripts/sim_run.py --server http://localhost:8080 \\
        --targets scripts/sim_targets.json --speed 8
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any, Dict, List, Sequence

from enose.client.api import ServerAPI
from enose.client.sensors import ENoseSensor
from enose.robot.interfaces import Detection, PolicyAdapters, Pose
from enose.robot.mission import (
    load_targets_file,
    make_targets,
    print_summary,
    run_mission,
)
from enose.robot.policy import PolicyConfig, State
from enose.robot.sim import ScriptedVision, SimLocalizer, SimMotion, SimPump


def _build_script(
    target_specs: List[Dict[str, Any]],
    shared_state: dict,
    detection_radius: float = 0.5,
):
    by_label: Dict[str, Dict[str, Any]] = {row["label"].lower(): row for row in target_specs}
    for row in target_specs:
        for a in row.get("aliases", []) or []:
            by_label[a.lower()] = row

    def _script(image_path: str, labels: Sequence[str]) -> List[Detection]:
        pose = shared_state.get("pose")
        if pose is None:
            return []
        out: List[Detection] = []
        seen = set()
        for label in labels:
            row = by_label.get(label.lower())
            if not row or row["label"] in seen:
                continue
            wp_world = row.get("world_pose")
            if wp_world is None:
                continue
            dx = float(wp_world["x"]) - pose.x
            dy = float(wp_world["y"]) - pose.y
            dist = math.hypot(dx, dy)
            if dist > detection_radius:
                continue
            seen.add(row["label"])
            out.append(Detection(
                label=row["label"],
                bbox=[100.0, 100.0, 300.0, 300.0],
                score=float(row.get("detection_score", 0.3)),
                area_fraction=0.1,
                extras={"target_pose": dict(wp_world), "_distance": dist},
            ))
        return out

    return _script


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--speed", type=float, default=8.0,
                    help="Time-compression factor (sleep / speed). 1.0 = real time.")
    ap.add_argument("--tau-grounding", type=float, default=0.05)
    ap.add_argument("--tau-commit", type=float, default=0.10)
    ap.add_argument("--record-seconds", type=float, default=6.0)
    ap.add_argument("--purge-seconds", type=float, default=3.0)
    ap.add_argument("--settling-timeout", type=float, default=20.0)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    args = ap.parse_args(argv)

    rows = load_targets_file(args.targets)
    shared = {"pose": Pose(0.0, 0.0, 0.0), "cmd_vel": (0.0, 0.0, 0.0)}
    motion = SimMotion(shared)
    localizer = SimLocalizer(shared)
    pump = SimPump()
    vision = ScriptedVision(script=_build_script(rows, shared))
    sensor = ENoseSensor(offline_mode=True)
    sensor.set_simulation_smell("air")
    server_api = ServerAPI(args.server)
    info, err = server_api.test_connection()
    if err:
        print(f"[sim] server unreachable at {args.server}: {err}", file=sys.stderr)
        return 1

    adapters = PolicyAdapters(
        motion=motion, localizer=localizer, pump=pump, vision=vision,
        sensor=sensor, server_api=server_api,
    )
    config = PolicyConfig(
        tau_grounding=args.tau_grounding,
        tau_commit=args.tau_commit,
        record_seconds=args.record_seconds,
        purge_seconds=args.purge_seconds,
        settling_timeout=args.settling_timeout,
        publisher_rate_hz=args.rate_hz,
        settling_poll_period=max(0.05, 1.0 / args.speed),
    )

    def fast_sleep(s: float) -> None:
        time.sleep(max(0.0, s / max(1.0, args.speed)))

    def on_transition(prev: State, cur: State, note: str) -> None:
        print(f"  [sim] {prev.value:>9s} → {cur.value:<9s}  {note}")

    def cue_sensor(target, state) -> None:
        if state == State.RECORD and target is not None:
            sensor.set_simulation_smell(target.label)

    targets = make_targets(rows)
    print(f"[sim] mission start: {len(targets)} targets, speed={args.speed}")
    result = run_mission(
        adapters, targets, config=config, sleep=fast_sleep,
        on_transition=on_transition, cue_sensor=cue_sensor,
    )
    print_summary(result.summary)
    return 0 if result.summary["final_state"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
