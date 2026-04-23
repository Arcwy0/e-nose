"""Interactive CLI entry point. Wires ServerAPI + WebcamHandler + ENoseSensor + TrainingPipeline."""

from __future__ import annotations

import argparse
import time

from enose.config import (
    DEFAULT_BAUD_RATE,
    DEFAULT_RECORDING_TIME,
    DEFAULT_SERVER_URL,
    DEFAULT_TARGET_SAMPLES,
    N_FEATURES,
)

from .api import ServerAPI
from .pipeline import TrainingPipeline
from .sensors import ENoseSensor
from .webcam import WebcamHandler


MENU = """
================================================================================
MAIN MENU
================================================================================
1. Full training cycle (vision + smell)
2. Vision only (object detection)
3. Smell identification
4. Model info
5. Learn from CSV
6. Manual 22-feature input testing
7. Generate visualizations + analysis
8. Live sensor stream (plot + push to server)
9. Replay a recorded session
0. Exit
================================================================================
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E-Nose vision-smell training client")
    p.add_argument("--server", default=DEFAULT_SERVER_URL, help="Server URL")
    p.add_argument("--port_enose", help="E-nose serial port")
    p.add_argument("--port_UART", help="UART serial port")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--time", type=int, default=DEFAULT_RECORDING_TIME, help="Recording seconds")
    p.add_argument("--samples", type=int, default=DEFAULT_TARGET_SAMPLES, help="Target samples")
    p.add_argument("--baud_enose", type=int, default=DEFAULT_BAUD_RATE)
    p.add_argument("--baud_UART", type=int, default=DEFAULT_BAUD_RATE)
    p.add_argument("--offline", action="store_true", help="Simulate sensors")
    p.add_argument(
        "--live",
        action="store_true",
        help=(
            "Skip the interactive menu and jump straight into the live "
            "sensor-stream dashboard (publishes samples to --server and "
            "draws a matplotlib window)."
        ),
    )
    p.add_argument("--live-rate", type=float, default=5.0,
                   help="Live stream rate in Hz (default 5)")
    p.add_argument("--live-window", type=float, default=60.0,
                   help="Live plot rolling window in seconds (default 60)")
    p.add_argument("--live-baseline", action="store_true",
                   help="Subtract the first few seconds as an air baseline")
    p.add_argument("--live-classify", action="store_true",
                   help="Also call /smell/classify per sample so the plot shows confidence")
    p.add_argument("--live-label",
                   help="Tag every live sample with this label (useful when recording a known odour)")
    p.add_argument("--record-session", metavar="PATH",
                   help=(
                       "Also save a session recording to this .npz file "
                       "(works with --live or menu option 8)"
                   ))
    return p


def _launch_live(args, sensor) -> None:
    """Shared entry for both ``--live`` (non-interactive) and menu option 8."""
    from .live import run_live_dashboard

    run_live_dashboard(
        sensor=sensor,
        server_url=args.server,
        rate_hz=args.live_rate,
        window_seconds=args.live_window,
        subtract_baseline=args.live_baseline,
        classify=args.live_classify,
        session_label=args.live_label,
        record_path=args.record_session,
    )


def _launch_replay(args) -> None:
    """Menu 9 — delegates to scripts/replay.py so there's one replay code path.

    We prompt for the session path here rather than forcing it through CLI
    args so the interactive menu is self-contained.
    """
    path = input("Session .npz path: ").strip().strip('"').strip("'")
    if not path:
        print("Aborted.")
        return
    from scripts.replay import replay_session
    try:
        replay_session(path, server_url=args.server)
    except Exception as e:
        print(f"Replay failed: {e}")


def main() -> None:
    args = build_parser().parse_args()
    mode = "OFFLINE" if args.offline else "HARDWARE"

    print("=" * 80)
    print(f"E-NOSE VISION-SMELL CLIENT — 22 FEATURES ({mode})")
    print("=" * 80)

    server_api = ServerAPI(args.server)
    webcam = WebcamHandler(args.camera)
    port_tuple = (args.port_enose, args.port_UART) if (args.port_enose or args.port_UART) else None
    sensor = ENoseSensor(port_tuple, args.baud_enose, args.baud_UART, args.offline)

    print("Testing server connection…")
    info, err = server_api.test_connection()
    if err:
        print(f"Server connection failed: {err}")
        print(f"Is the server running at {args.server}?")
        # --live against a dead server is still useful (local plot only), so
        # we don't return here when --live is set.
        if not args.live:
            return
    info = info if isinstance(info, dict) else {}
    print(f"Server: {info.get('message', 'OK')} v{info.get('version', '?')}")
    cfg = info.get("sensor_configuration", {})
    total = cfg.get("total_features", 0)
    print(f"  feature support : {total} ({'OK' if total == N_FEATURES else 'MISMATCH'})")
    models = info.get("models", {})
    print(f"  vlm             : {models.get('vlm_loaded', False)}")
    print(f"  smell clf       : {models.get('smell_classifier_loaded', False)}")
    backend = models.get("smell_classifier_backend")
    if backend:
        print(f"  backend         : {backend}")

    print(f"\nConnecting sensors ({mode.lower()})…")
    if not sensor.connect():
        print("Sensor connection failed" + (" — try --offline" if not args.offline else ""))
        return

    # Fast path: --live skips the menu entirely.
    if args.live:
        try:
            _launch_live(args, sensor)
        finally:
            sensor.disconnect()
        return

    # Camera is opened lazily when menu option 1 or 2 is selected.
    # No open_camera() call here so the client works headless / without hardware.

    pipeline = TrainingPipeline(server_api, webcam, sensor, args.time, args.samples)
    try:
        while True:
            print(MENU)
            print(f"Recording: {args.samples} samples in {args.time}s ({mode})")
            choice = input("Select (0-9): ").strip()
            if choice == "1":
                pipeline.run_training_cycle()
            elif choice == "2":
                pipeline.run_vision_only()
            elif choice == "3":
                pipeline.identify_current_smell()
            elif choice == "4":
                pipeline.view_model_info()
            elif choice == "5":
                pipeline.learn_from_csv_file()
            elif choice == "6":
                pipeline.test_manual_22_feature_input()
            elif choice == "7":
                pipeline.visualize_and_analyze()
            elif choice == "8":
                _launch_live(args, sensor)
            elif choice == "9":
                _launch_replay(args)
            elif choice == "0":
                break
            else:
                print("Invalid — select 0-9")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Cleaning up…")
        if webcam.cap is not None:
            webcam.close_camera()
        sensor.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
