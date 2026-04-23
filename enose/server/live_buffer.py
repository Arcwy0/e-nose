"""Thread-safe ring buffer for live sensor samples pushed by the client.

The client's `LivePublisher` POSTs one sample per tick to `/sensor/live/push`;
the browser UI polls `/sensor/live/recent` to draw a scrolling plot and the
drift endpoint reads from the same buffer. Keeping the buffer here (not in
``state.py``) avoids bloating the shared-state module with a chunk of logic
that's specific to the live-streaming feature.

Each entry is a plain dict so the payload is cheap to serialize as JSON:

    {
        "t": 1714482093.517,        # unix seconds, server-assigned
        "client_t": 1714482093.410, # client's own clock (optional)
        "sample": {"R1": .., ..., "CH2O": ..},  # 22 sensor dict
        "predicted": "coffee",      # optional — filled if the publisher
                                    # asked the server to classify too
        "confidence": 0.87,
        "probs": {"air": .., "coffee": ..},     # top-k probabilities
    }
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional


# Capacity chosen so ~10 Hz publisher ≈ 100 s of history at once. Tunable via
# ``resize()`` if a longer trail is ever needed; default is the right trade-off
# between RAM and how much scroll-back the UI actually shows.
_DEFAULT_CAPACITY = 1000


class LiveSensorBuffer:
    """FIFO deque with monotonic id per entry so the UI can do ``?since=<id>``.

    The buffer is intentionally simple — one lock for the deque, one for the
    monotonic counter. We don't need a proper queue/event system because the
    UI polls (every ~500 ms) and the only producer is the publisher thread.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_id: int = 0

    # ── writer ────────────────────────────────────────────────────────────
    def push(self, entry: Dict[str, Any]) -> int:
        """Append an entry. Stamps it with ``id`` and server-side ``t`` if absent.

        Returns the assigned id so the caller can echo it back to the client.
        """
        with self._lock:
            entry = dict(entry)
            entry.setdefault("t", time.time())
            entry["id"] = self._next_id
            self._next_id += 1
            self._buf.append(entry)
            return entry["id"]

    def clear(self) -> int:
        """Drop every entry but keep the id counter — so the UI's ``since=``
        cursor doesn't accidentally re-fetch entries that happen to land on
        the same index after a reset."""
        with self._lock:
            n = len(self._buf)
            self._buf.clear()
            return n

    # ── reader ────────────────────────────────────────────────────────────
    def recent(
        self,
        since_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Entries with ``id > since_id`` (or the last ``limit`` if since is None).

        Returns a list of plain dicts; callers get a shallow copy so the
        buffer can keep mutating without corrupting the response.
        """
        with self._lock:
            items = list(self._buf)
        if since_id is not None:
            items = [e for e in items if e.get("id", -1) > since_id]
        if limit is not None and len(items) > limit:
            items = items[-limit:]
        return items

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._buf[-1]) if self._buf else None

    def snapshot(self) -> List[Dict[str, Any]]:
        """Full buffer copy. Used by ``/smell/drift`` to compare the last
        live window against the training distribution."""
        with self._lock:
            return [dict(e) for e in self._buf]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen or 0


# Module-level singleton — importers do `from .live_buffer import buffer`.
buffer = LiveSensorBuffer()
