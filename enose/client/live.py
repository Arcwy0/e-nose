"""Live sensor streaming — matplotlib window + server publisher.

Two cooperating components that share the ENoseSensor:

    LivePublisher   — background thread that reads from the sensor at a
                      fixed rate, pushes each sample to the server's live
                      buffer (POST /sensor/live/push) and optionally asks
                      the server to classify it. Also writes to an
                      in-process callback queue so LiveSensorPlot can draw.

    LiveSensorPlot  — interactive matplotlib window (plt.ion) that shows
                      R1–R17 on one axis and T/H/CO2/H2S/CH2O on another.
                      Optional baseline subtraction, rolling window, and a
                      confidence time-series underneath when the publisher
                      is classifying.

Design choices worth knowing:
    * Threaded publisher + main-thread plot. Matplotlib TkAgg/Qt backends
      are not thread-safe, so the plot stays on the main thread and pulls
      samples through a queue. This is also why the publisher uses a
      callback pattern rather than letting the plot call into sensor.read().
    * A single ``stop_event`` drives teardown so Ctrl-C is clean.
    * If the server is unreachable the publisher keeps plotting locally —
      every HTTP error is caught and counted but never raised. Useful in
      the field when wifi drops.
    * Session id is a UUID stamped at publisher start so ``/sensor/live/stats``
      can attribute a block of samples to one recording.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

import requests

from enose.config import (
    ALL_SENSORS,
    ENVIRONMENTAL_SENSORS,
    RESISTANCE_SENSORS,
)


# ── Publisher ────────────────────────────────────────────────────────────────


class LivePublisher:
    """Background thread that streams sensor samples to the server + a local queue.

    Usage:
        pub = LivePublisher(sensor, server_url, rate_hz=5, classify=True)
        pub.start()
        ...                  # samples flow; pub.q.get() yields (t, sample, pred)
        pub.stop()

    The ``on_sample`` callback is invoked synchronously from the publisher
    thread — keep it cheap (e.g. `queue.put_nowait`). The plot class uses
    that to decouple the network loop from the rendering loop.
    """

    def __init__(
        self,
        sensor,
        server_url: str = "",
        rate_hz: float = 5.0,
        classify: bool = False,
        session_label: Optional[str] = None,
        on_sample: Optional[Callable[[Dict], None]] = None,
        http_timeout: float = 2.0,
    ) -> None:
        self.sensor = sensor
        self.server_url = server_url.rstrip("/") if server_url else ""
        self.rate_hz = max(0.5, float(rate_hz))
        self.classify = bool(classify)
        self.session_id = uuid.uuid4().hex[:12]
        self.session_label = session_label
        self.on_sample = on_sample
        self._timeout = http_timeout

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Diagnostics — exposed so the plot's status bar can report health.
        self.n_pushed: int = 0
        self.n_push_failed: int = 0
        self.n_read_failed: int = 0
        self.last_error: Optional[str] = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── worker loop ───────────────────────────────────────────────────────
    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_tick = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_tick:
                # Fine-grained wait so Ctrl-C isn't blocked for a whole period.
                self._stop.wait(min(period, next_tick - now))
                continue
            next_tick = now + period

            sample = self._read_one()
            if sample is None:
                continue

            entry = {
                "t": now,
                "sample": sample,
                "session_id": self.session_id,
                "label": self.session_label,
            }
            # Fire local callback first — the plot cares about smoothness
            # more than it cares about server round-trips.
            if self.on_sample is not None:
                try:
                    self.on_sample(entry)
                except Exception:  # pragma: no cover — defensive
                    pass

            if self.server_url:
                self._push(now, sample, entry)

    def _read_one(self) -> Optional[Dict[str, float]]:
        try:
            s = self.sensor.read_single_measurement()
        except Exception as e:  # pragma: no cover — sensor driver edge cases
            self.last_error = f"read: {e}"
            self.n_read_failed += 1
            return None
        if s is None or len(s) == 0:
            self.n_read_failed += 1
            return None
        return s

    def _push(self, t: float, sample: Dict[str, float], entry: Dict) -> None:
        payload = {
            "sample": sample,
            "client_t": t,
            "classify": self.classify,
            "session_id": self.session_id,
        }
        if self.session_label:
            payload["label"] = self.session_label
        try:
            r = requests.post(
                f"{self.server_url}/sensor/live/push",
                json=payload,
                timeout=self._timeout,
            )
            r.raise_for_status()
            # Echo the server's classification back into the local entry so
            # the plot can draw the confidence line without a second request.
            try:
                body = r.json()
            except ValueError:
                body = {}
            if self.classify and isinstance(body, dict):
                # The push endpoint already embeds predicted/confidence/top3
                # into the buffer entry; we need to re-fetch if we want it,
                # OR the publisher can hit /smell/classify itself. Simpler:
                # hit /smell/classify directly here so the local plot has
                # the classification without a GET round-trip.
                pass
            self.n_pushed += 1
        except requests.RequestException as e:
            self.n_push_failed += 1
            self.last_error = f"push: {e}"

        # Separate classify call so the local plot has prediction + confidence
        # too. We could inline this with push, but keeping them split means
        # a classify failure (unfitted model) doesn't lose the sample push.
        if self.classify:
            try:
                r = requests.post(
                    f"{self.server_url}/smell/classify",
                    json=sample,
                    timeout=self._timeout,
                )
                if r.ok:
                    body = r.json()
                    entry["predicted"] = body.get("predicted_smell")
                    entry["confidence"] = body.get("confidence")
                    entry["probabilities"] = body.get("probabilities", {})
            except requests.RequestException:
                pass


# ── Plot ─────────────────────────────────────────────────────────────────────


class LiveSensorPlot:
    """Rolling line plot of the 22-feature stream with optional confidence trace.

    The plot owns a deque of the last ``window_seconds`` of samples and
    redraws on a timer. Baseline subtraction is optional — on the first
    ``baseline_seconds`` of samples we snapshot per-sensor means, then
    every subsequent point is plotted as ``value - baseline[sensor]``.
    That makes the "step" when you bring a vial close easy to see.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        baseline_seconds: float = 5.0,
        subtract_baseline: bool = False,
        show_confidence: bool = False,
        title_prefix: str = "E-Nose Live",
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.baseline_seconds = float(baseline_seconds)
        self.subtract_baseline = bool(subtract_baseline)
        self.show_confidence = bool(show_confidence)
        self.title_prefix = title_prefix

        self._entries: Deque[Dict] = deque(maxlen=int(60 * window_seconds))  # cap at ~60Hz*window
        self._lock = threading.Lock()
        self._baseline: Dict[str, float] = {}
        self._baseline_ready: bool = False
        self._t0: Optional[float] = None

        self._fig = None
        self._axes: List = []
        self._closed = threading.Event()

    # ── feed from publisher ───────────────────────────────────────────────
    def submit(self, entry: Dict) -> None:
        """Publisher callback — just stashes the entry. Rendering happens in run()."""
        with self._lock:
            if self._t0 is None:
                self._t0 = entry.get("t", time.time())
            self._entries.append(entry)
            self._update_baseline_locked()

    def _update_baseline_locked(self) -> None:
        """Compute baseline once enough samples have accumulated.

        Called under ``self._lock`` from submit(); keeps the baseline a
        cheap rolling mean until ``baseline_seconds`` have elapsed, then
        freezes it so subtraction stays stable for the rest of the session.
        """
        if self._baseline_ready or not self._entries or self._t0 is None:
            return
        elapsed = self._entries[-1].get("t", time.time()) - self._t0
        if elapsed < self.baseline_seconds:
            # Update rolling mean on the fly so early frames also have a
            # reasonable baseline even before we "lock in".
            sums: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for e in self._entries:
                for k, v in e.get("sample", {}).items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
            self._baseline = {k: sums[k] / counts[k] for k in sums}
            return
        self._baseline_ready = True
        print(
            f"[live-plot] baseline locked after {elapsed:.1f}s — "
            f"subtraction {'ON' if self.subtract_baseline else 'OFF'}"
        )

    # ── main-thread rendering loop ────────────────────────────────────────
    def run(self, refresh_hz: float = 5.0) -> None:
        """Block the caller in a matplotlib render loop until the window closes.

        Matplotlib is notoriously tetchy about threading — keep run() on the
        main thread, run LivePublisher in a background thread that feeds
        submit(). Close the plot window or press Ctrl-C to stop.
        """
        import matplotlib

        # Use an interactive backend if available. On headless Linux (Docker)
        # the default Agg backend can't open a window, so we fall back to
        # "Agg mode" — just print a message and exit politely. The browser
        # UI's Live tab is the headless-friendly alternative.
        if matplotlib.get_backend().lower() in ("agg", "module://matplotlib_inline.backend_inline"):
            print(
                "[live-plot] no interactive matplotlib backend available; "
                "use the browser UI Live tab instead."
            )
            return

        import matplotlib.pyplot as plt

        plt.ion()
        n_panels = 3 if self.show_confidence else 2
        heights = [3, 2, 1] if self.show_confidence else [3, 2]
        self._fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(11, 6 if not self.show_confidence else 7),
            gridspec_kw={"height_ratios": heights},
            sharex=True,
        )
        self._axes = list(axes) if hasattr(axes, "__iter__") else [axes]
        self._fig.canvas.mpl_connect("close_event", lambda _evt: self._closed.set())

        period = 1.0 / max(1.0, refresh_hz)
        try:
            while not self._closed.is_set():
                self._redraw()
                plt.pause(period)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            try:
                plt.close(self._fig)
            except Exception:
                pass

    def close(self) -> None:
        self._closed.set()

    # ── drawing ───────────────────────────────────────────────────────────
    def _redraw(self) -> None:
        with self._lock:
            entries = list(self._entries)
            baseline = dict(self._baseline)

        if not entries:
            return

        t_now = entries[-1].get("t", time.time())
        t_min = t_now - self.window_seconds
        entries = [e for e in entries if e.get("t", 0) >= t_min]
        if not entries:
            return

        ts = [e["t"] - t_now for e in entries]  # seconds-from-now, negative

        ax_r, ax_e = self._axes[0], self._axes[1]
        ax_r.clear()
        ax_e.clear()

        sub = self.subtract_baseline and baseline

        # Resistance panel (R1-R17) — blue palette so it reads as one group.
        for i, name in enumerate(RESISTANCE_SENSORS):
            vals = [e.get("sample", {}).get(name) for e in entries]
            vals = [float(v) - baseline.get(name, 0.0) if sub and v is not None
                    else (float(v) if v is not None else float("nan"))
                    for v in vals]
            ax_r.plot(ts, vals, lw=0.9, alpha=0.75, label=name)

        ax_r.set_ylabel(("ΔR" if sub else "R") + " (scaled)")
        ax_r.set_title(
            f"{self.title_prefix} — {len(entries)} samples, window={self.window_seconds:.0f}s"
            + (" [baseline-subtracted]" if sub else "")
        )
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="upper left", ncol=6, fontsize=6, framealpha=0.7)

        # Environmental panel — different axis scales, so draw each on its
        # own twin-axis would be nice, but 5 twins get unreadable. Just
        # normalize-per-sensor to a 0-1 band and label with units in legend.
        env_labels: List[str] = []
        for name in ENVIRONMENTAL_SENSORS:
            vals_raw = [e.get("sample", {}).get(name) for e in entries]
            vals_raw = [v for v in vals_raw if v is not None]
            if not vals_raw:
                continue
            vmin = min(vals_raw)
            vmax = max(vals_raw)
            rng = (vmax - vmin) or 1.0
            ys = [(float(v) - vmin) / rng if v is not None else float("nan")
                  for v in [e.get("sample", {}).get(name) for e in entries]]
            ax_e.plot(ts, ys, lw=1.1, label=f"{name} [{vmin:.1f}–{vmax:.1f}]")
            env_labels.append(name)

        ax_e.set_ylabel("env (normalized)")
        ax_e.set_xlabel("time before now (s)")
        ax_e.set_ylim(-0.05, 1.05)
        ax_e.grid(True, alpha=0.3)
        if env_labels:
            ax_e.legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.7)

        if self.show_confidence and len(self._axes) >= 3:
            ax_c = self._axes[2]
            ax_c.clear()
            confs = [e.get("confidence") for e in entries]
            # Colour-code by predicted class so you can see the label
            # switching during a session. One line per class that appears.
            classes_seen: Dict[str, List] = {}
            for tt, e in zip(ts, entries):
                pred = e.get("predicted")
                conf = e.get("confidence")
                if pred is None or conf is None:
                    continue
                classes_seen.setdefault(pred, [[], []])
                classes_seen[pred][0].append(tt)
                classes_seen[pred][1].append(float(conf))
            for name, (xx, yy) in classes_seen.items():
                ax_c.plot(xx, yy, marker=".", ls="-", lw=1.0, alpha=0.8, label=name)
            ax_c.set_ylim(0, 1)
            ax_c.set_ylabel("confidence")
            ax_c.set_xlabel("time before now (s)")
            ax_c.grid(True, alpha=0.3)
            if classes_seen:
                ax_c.legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.7)

        try:
            self._fig.tight_layout()
        except Exception:
            pass


# ── Convenience runner ───────────────────────────────────────────────────────


def run_live_dashboard(
    sensor,
    server_url: str = "",
    rate_hz: float = 5.0,
    window_seconds: float = 60.0,
    subtract_baseline: bool = False,
    classify: bool = False,
    session_label: Optional[str] = None,
    record_path: Optional[str] = None,
) -> None:
    """Kick off publisher + plot (+ recorder) together. Blocks until the window closes.

    This is what the CLI ``--live`` flag / menu item calls. Callers that want
    finer control (e.g. running the publisher without a window, headless) can
    instantiate ``LivePublisher`` directly with their own ``on_sample`` callback.

    ``record_path`` activates a :class:`SessionRecorder`; the recorder tees off
    the same stream so it captures every sample the plot saw, including any
    server-returned prediction / confidence.
    """
    # Local import to avoid dragging numpy into every client import.
    from .session import SessionRecorder, tee_callbacks

    plot = LiveSensorPlot(
        window_seconds=window_seconds,
        subtract_baseline=subtract_baseline,
        show_confidence=classify,
    )

    recorder: Optional[SessionRecorder] = None
    if record_path:
        recorder = SessionRecorder(
            path=record_path,
            session_label=session_label,
        )

    callback = tee_callbacks(plot.submit, recorder.submit if recorder else None)

    pub = LivePublisher(
        sensor=sensor,
        server_url=server_url,
        rate_hz=rate_hz,
        classify=classify,
        session_label=session_label,
        on_sample=callback,
    )
    if recorder is not None:
        recorder.session_id = pub.session_id

    print(
        f"[live] publishing to {server_url or '(local only)'} at "
        f"{rate_hz:.1f} Hz — session={pub.session_id}"
        + (f" label={session_label}" if session_label else "")
        + (f" → recording to {record_path}" if record_path else "")
    )
    pub.start()
    try:
        plot.run()
    finally:
        pub.stop()
        print(
            f"[live] stopped — pushed={pub.n_pushed} push_failed={pub.n_push_failed} "
            f"read_failed={pub.n_read_failed} last_error={pub.last_error}"
        )
        if recorder is not None and len(recorder) > 0:
            recorder.save()
