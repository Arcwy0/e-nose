"""Session recorder — saves a live sensor run to an .npz file for replay.

Design goals:
    * Zero coupling with the server. The recorder is an ``on_sample`` callback
      for :class:`LivePublisher`; any other callback chain can tee into the
      same data the server / plot sees.
    * Flush-on-close rather than incremental I/O. Sessions are bounded
      (minutes, maybe 1–2 hours) so keeping the rows in memory is fine and
      avoids per-frame fsync latency stealing from the sample rate.
    * One npz per session with stable keys so ``scripts/replay.py`` can load
      sessions from any future client build without a schema migration:

        timestamps   (N,)   float64 — server-clock seconds since epoch
        client_t     (N,)   float64 — client-clock seconds (may be NaN)
        samples      (N, 22) float32 — raw sensor values in ALL_SENSORS order
        sensor_names object       — the column ordering as a 1-D array of strings
        predictions  (N,)   str   — predicted label (empty when classify=False)
        confidences  (N,)   float — max probability (NaN when unknown)
        labels       (N,)   str   — ground-truth label passed by the user
        session_id   scalar str
        session_label scalar str
        created_at   scalar str — ISO-8601

Loader is in ``scripts/replay.py``.
"""

from __future__ import annotations

import datetime
import os
import threading
from typing import Dict, List, Optional

import numpy as np

from enose.config import ALL_SENSORS


class SessionRecorder:
    """Append-only in-memory recorder.

    Plug into ``LivePublisher(on_sample=recorder.submit)`` (possibly via a
    tee — see :func:`tee_callbacks`) and call :meth:`save` when done.
    """

    def __init__(
        self,
        path: str,
        session_id: Optional[str] = None,
        session_label: Optional[str] = None,
    ) -> None:
        self.path = path
        self.session_id = session_id or ""
        self.session_label = session_label or ""
        self._rows: List[Dict] = []
        self._lock = threading.Lock()
        self._created_at = datetime.datetime.now().isoformat(timespec="seconds")

    # ── writer ────────────────────────────────────────────────────────────
    def submit(self, entry: Dict) -> None:
        """Callback plugged into the publisher — stashes one frame."""
        with self._lock:
            self._rows.append(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    # ── flush ─────────────────────────────────────────────────────────────
    def save(self, path: Optional[str] = None) -> str:
        """Serialize to .npz. Returns the resolved path.

        Intentionally tolerant of missing fields — an ``entry`` with no
        ``predicted`` (because the run wasn't classifying) lands as an
        empty-string row; downstream ``replay.py`` treats that as "not
        classified yet, please classify now".
        """
        out = path or self.path
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with self._lock:
            rows = list(self._rows)
        n = len(rows)
        if n == 0:
            print(f"[session] nothing to save — empty recorder for {out}")
            return out

        timestamps = np.array([float(r.get("t", np.nan)) for r in rows], dtype=np.float64)
        client_t = np.array(
            [float(r.get("client_t", r.get("t", np.nan))) for r in rows],
            dtype=np.float64,
        )
        samples = np.zeros((n, len(ALL_SENSORS)), dtype=np.float32)
        for i, r in enumerate(rows):
            s = r.get("sample", {}) or {}
            for j, name in enumerate(ALL_SENSORS):
                v = s.get(name)
                samples[i, j] = float(v) if v is not None else np.nan
        predictions = np.array([str(r.get("predicted") or "") for r in rows])
        confidences = np.array(
            [float(r["confidence"]) if r.get("confidence") is not None else np.nan for r in rows],
            dtype=np.float32,
        )
        labels = np.array([str(r.get("label") or "") for r in rows])

        np.savez(
            out,
            timestamps=timestamps,
            client_t=client_t,
            samples=samples,
            sensor_names=np.array(list(ALL_SENSORS)),
            predictions=predictions,
            confidences=confidences,
            labels=labels,
            session_id=np.array(self.session_id),
            session_label=np.array(self.session_label),
            created_at=np.array(self._created_at),
        )
        print(
            f"[session] wrote {n} samples → {out} "
            f"(session_id={self.session_id or '?'}, label={self.session_label or '?'})"
        )
        return out


def tee_callbacks(*callbacks):
    """Combine several `on_sample` callbacks into one.

    Used when the live dashboard needs to feed both the plot and the session
    recorder from a single :class:`LivePublisher`.
    """
    cbs = [cb for cb in callbacks if cb is not None]

    def _tee(entry):
        for cb in cbs:
            try:
                cb(entry)
            except Exception as e:  # pragma: no cover — defensive
                print(f"[session.tee] callback failed: {e}")

    return _tee
