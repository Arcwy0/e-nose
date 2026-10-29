"""Graceful signal cleanup. SIGKILL, power loss and USB loss remain external risks."""

from __future__ import annotations

import atexit
import signal
import threading
from contextlib import contextmanager


@contextmanager
def sensor_lifecycle(sensor):
    previous = {}
    closing = False

    def interrupted(signum, frame):
        if not closing:
            raise KeyboardInterrupt(f"signal {signum}")

    def cleanup():
        nonlocal closing
        if closing:
            return
        closing = True
        try:
            sensor.disconnect()
        except Exception as exc:
            print(f"[shutdown] Cleanup failed: {exc}; heater OFF is NOT verified")

    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, interrupted)
    atexit.register(cleanup)
    try:
        yield sensor
    except KeyboardInterrupt:
        print("[shutdown] Interrupted; waiting for controlled heater shutdown")
    finally:
        cleanup()
        atexit.unregister(cleanup)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
