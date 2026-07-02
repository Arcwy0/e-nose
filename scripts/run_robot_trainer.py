#!/usr/bin/env python3
"""Mission entry point for Idea-1 autonomous training on the real Go-1.

The intent is that this script is what you run on the Jetson once the
ROS2 stack (FAST-LIO2, Unitree driver, pump_node) is up. The same script
also has a ``--backend sim`` mode that's exercised in CI / Docker; it
behaves like :mod:`scripts.sim_run`.

Layout::

    python scripts/run_robot_trainer.py --backend ros2 \\
        --server http://lab-gpu:8080 --targets configs/mission.yaml

    # CI / Docker path:
    python scripts/run_robot_trainer.py --backend sim \\
        --server http://localhost:8080 --targets scripts/sim_targets.json

When ``--backend ros2`` is selected the script imports the ROS2 adapters
from :mod:`enose.robot.ros2`; those are currently skeletons that raise
:class:`NotImplementedError` until the Jetson-side wiring is finished
(part of the Week-1 Phase-0 work in the integration plan).
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from enose.client.api import ServerAPI
from enose.robot.interfaces import PolicyAdapters
from enose.robot.mission import (
    load_targets_file,
    make_targets,
    print_summary,
    run_mission,
)
from enose.robot.policy import PolicyConfig, State


def _build_sim_backend(rows, args):
    """Delegate to the existing sim_run.py logic so the CLIs stay in sync."""
    from scripts.sim_run import _build_script  # type: ignore[import-not-found]
    import math, time  # noqa: F401
    from enose.client.sensors import ENoseSensor
    from enose.robot.interfaces import Pose
    from enose.robot.sim import ScriptedVision, SimLocalizer, SimMotion

    shared = {"pose": Pose(0.0, 0.0, 0.0), "cmd_vel": (0.0, 0.0, 0.0)}
    motion = SimMotion(shared)
    localizer = SimLocalizer(shared)
    pump = _build_pump(args.pump)
    vision = ScriptedVision(script=_build_script(rows, shared))
    sensor = ENoseSensor(offline_mode=True)
    sensor.set_simulation_smell("air")
    return motion, localizer, pump, vision, sensor


def _build_pump(mode: str):
    """Pick a pump implementation from the ``--pump`` flag.

    Defaults to ``manual`` on the real robot because the rig has no
    programmable relay yet — the policy still calls ``pump.on()`` /
    ``pump.off()`` at the right state transitions; ManualPump just logs
    them for a human to follow.
    """
    if mode == "manual":
        from enose.robot.manual_pump import ManualPump
        return ManualPump(mode="log")
    if mode == "manual-confirm":
        from enose.robot.manual_pump import ManualPump
        return ManualPump(mode="confirm")
    if mode == "ros2":
        from enose.robot.ros2.pump import Ros2Pump
        return Ros2Pump()
    if mode == "sim":
        from enose.robot.sim.pump import SimPump
        return SimPump()
    raise ValueError(f"unknown --pump mode: {mode!r}")


def _build_ros2_backend(rows, args):
    """Wires up the real Unitree / FAST-LIO2 adapters.

    Motion + localization are still ROS2 skeletons (raise
    ``NotImplementedError`` until the Jetson-side ``rclpy`` wiring lands —
    see ``docs/NEXT_STEPS.md`` §3.2 / §3.3). The pump path is already
    real-robot-ready via ``ManualPump`` (the rig has no programmable relay).
    """
    from enose.robot.ros2.localization import TFLocalizer
    from enose.robot.ros2.unitree_adapter import UnitreeAdapter
    from enose.robot.vision_http import HttpVisionClient

    motion = UnitreeAdapter()
    localizer = TFLocalizer()
    pump = _build_pump(args.pump)

    # Real on-Jetson capture: WebcamHandler from enose.client wraps OpenCV.
    from enose.client.sensors import ENoseSensor
    from enose.client.webcam import WebcamHandler

    webcam = WebcamHandler()

    def _capture():
        _, path = webcam.capture_image()
        return path

    server_api = ServerAPI(args.server)
    vision = HttpVisionClient(server_api, _capture)
    sensor = ENoseSensor(port=args.port_pair, offline_mode=args.offline_sensor)
    return motion, localizer, pump, vision, sensor


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "ros2"], default="sim",
                    help="sim runs offline with ScriptedVision; ros2 talks to the real robot.")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--targets", required=True, help="JSON or YAML target list")
    ap.add_argument("--tau-grounding", type=float, default=0.05)
    ap.add_argument("--tau-commit", type=float, default=0.10)
    ap.add_argument("--record-seconds", type=float, default=30.0)
    ap.add_argument("--purge-seconds", type=float, default=30.0)
    ap.add_argument("--settling-timeout", type=float, default=60.0)
    ap.add_argument("--settling-window", type=float, default=8.0)
    ap.add_argument("--settling-threshold", type=float, default=0.02)
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Sim time-compression. Ignored for ros2 backend.")
    ap.add_argument(
        "--pump",
        choices=["manual", "manual-confirm", "ros2", "sim"],
        default=None,
        help=(
            "Pump backend. Defaults: sim→sim, ros2→manual (rig has no "
            "programmable relay; ManualPump logs requested state for a human). "
            "'manual-confirm' additionally blocks until the user presses Enter."
        ),
    )
    # Pass-through for the real ENoseSensor on the Jetson.
    ap.add_argument("--port-pair", default=None,
                    help="(ros2) e-nose serial ports as 'enose,uart' (default: auto-detect).")
    ap.add_argument("--offline-sensor", action="store_true",
                    help="(ros2) Force ENoseSensor offline_mode even on the robot — debug only.")
    args = ap.parse_args(argv)

    if args.port_pair:
        try:
            a, b = args.port_pair.split(",")
            args.port_pair = (a.strip(), b.strip())
        except ValueError:
            print("--port-pair must be 'enose,uart'", file=sys.stderr)
            return 2

    # Default pump per backend: sim→sim, ros2→manual.
    if args.pump is None:
        args.pump = "sim" if args.backend == "sim" else "manual"

    rows = load_targets_file(args.targets)
    if args.backend == "sim":
        motion, localizer, pump, vision, sensor = _build_sim_backend(rows, args)
    elif args.backend == "ros2":
        try:
            motion, localizer, pump, vision, sensor = _build_ros2_backend(rows, args)
        except (RuntimeError, NotImplementedError) as e:
            print(f"[run_robot_trainer] ros2 backend not ready: {e}", file=sys.stderr)
            print(
                "Wire up enose.robot.ros2.{unitree_adapter,localization,pump} on the Jetson "
                "(Phase-0 step in docs/IDEA1_PLAN.md §3.1).",
                file=sys.stderr,
            )
            return 3
    else:  # pragma: no cover — argparse enforces choices
        raise SystemExit(f"unknown backend: {args.backend}")

    server_api = ServerAPI(args.server)
    info, err = server_api.test_connection()
    if err:
        print(f"[run_robot_trainer] server unreachable at {args.server}: {err}", file=sys.stderr)
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
        settling_window=args.settling_window,
        settling_threshold=args.settling_threshold,
        publisher_rate_hz=args.rate_hz,
        settling_poll_period=max(0.05, 1.0 / max(1.0, args.speed)),
    )

    import time as _t
    def speed_sleep(s: float) -> None:
        _t.sleep(max(0.0, s / max(1.0, args.speed)))

    def on_transition(prev: State, cur: State, note: str) -> None:
        print(f"  [mission] {prev.value:>9s} → {cur.value:<9s}  {note}")

    cue = None
    if args.backend == "sim":
        # Cue the offline e-nose simulator to the target smell at RECORD.
        def cue(target, state):
            if state == State.RECORD and target is not None:
                sensor.set_simulation_smell(target.label)

    targets = make_targets(rows)
    print(f"[mission] backend={args.backend}  targets={len(targets)}  server={args.server}")
    result = run_mission(
        adapters, targets, config=config, sleep=speed_sleep,
        on_transition=on_transition, cue_sensor=cue,
    )
    print_summary(result.summary)
    return 0 if result.summary["final_state"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
