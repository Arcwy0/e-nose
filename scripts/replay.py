#!/usr/bin/env python3
"""Replay a recorded live session (.npz) against a running server.

Pairs with :mod:`enose.client.session`. The recorder writes one `.npz` per
run; this script loads it, POSTs each sample to ``/smell/classify`` on the
target server, and reports aggregate stats so you can answer questions like:

    * Does the current server model agree with what was recorded?
    * How confident was the model, on average?
    * Did a known-label session actually get classified as that label?

Usage
-----
    python scripts/replay.py path/to/session.npz
    python scripts/replay.py path/to/session.npz --server http://localhost:8080
    python scripts/replay.py path/to/session.npz --rate 0          # as fast as possible
    python scripts/replay.py path/to/session.npz --limit 50        # first 50 samples only

The function :func:`replay_session` is also imported by the interactive menu
(option 9) so there is only one replay code path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from enose.config import ALL_SENSORS, DEFAULT_SERVER_URL


# ── loader ────────────────────────────────────────────────────────────────────
def load_session(path: str) -> Dict[str, Any]:
    """Load a recorder .npz and return a plain-python dict.

    Kept forgiving because older recordings may predate newer keys. Missing
    keys come back as ``None`` / empty arrays rather than raising.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Session file not found: {p}")
    npz = np.load(p, allow_pickle=False)

    def _get(key, default=None):
        return npz[key] if key in npz.files else default

    sensor_names = _get("sensor_names")
    if sensor_names is not None:
        sensor_names = [str(x) for x in sensor_names.tolist()]
    else:
        sensor_names = list(ALL_SENSORS)

    session = {
        "timestamps": _get("timestamps"),
        "client_t": _get("client_t"),
        "samples": _get("samples"),
        "sensor_names": sensor_names,
        "predictions": _get("predictions"),
        "confidences": _get("confidences"),
        "labels": _get("labels"),
        "session_id": str(_get("session_id", np.array("")).item() if _get("session_id") is not None else ""),
        "session_label": str(_get("session_label", np.array("")).item() if _get("session_label") is not None else ""),
        "created_at": str(_get("created_at", np.array("")).item() if _get("created_at") is not None else ""),
    }
    return session


def _row_to_payload(row: np.ndarray, sensor_names: List[str]) -> Dict[str, float]:
    """Convert one (22,) row + column order into the server's SensorData dict."""
    out: Dict[str, float] = {}
    for name, val in zip(sensor_names, row.tolist()):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue  # leave missing values out; server defaults will fill them
        out[name] = float(val)
    return out


# ── stats ─────────────────────────────────────────────────────────────────────
def _summarize(
    replayed_preds: List[str],
    replayed_confs: List[float],
    original_preds: List[str],
    original_confs: List[float],
    session_label: str,
) -> Dict[str, Any]:
    n = len(replayed_preds)
    if n == 0:
        return {"n": 0}

    counts = Counter(replayed_preds)
    top_label, top_count = counts.most_common(1)[0]

    confs_arr = np.array(replayed_confs, dtype=np.float64)
    summary: Dict[str, Any] = {
        "n": n,
        "label_distribution": dict(counts.most_common()),
        "majority_label": top_label,
        "majority_fraction": round(top_count / n, 4),
        "confidence_mean": round(float(np.nanmean(confs_arr)), 4),
        "confidence_median": round(float(np.nanmedian(confs_arr)), 4),
        "confidence_p10": round(float(np.nanpercentile(confs_arr, 10)), 4),
        "confidence_p90": round(float(np.nanpercentile(confs_arr, 90)), 4),
    }

    if session_label:
        correct = sum(1 for p in replayed_preds if p == session_label)
        summary["session_label"] = session_label
        summary["accuracy_vs_session_label"] = round(correct / n, 4)

    # Compare against whatever the recorder wrote down at capture time (may be
    # empty strings when classify=False). Only score rows that had a stored pred.
    if original_preds and any(original_preds):
        agree = 0
        scored = 0
        for old, new in zip(original_preds, replayed_preds):
            if not old:
                continue
            scored += 1
            if old == new:
                agree += 1
        if scored > 0:
            summary["n_with_original_prediction"] = scored
            summary["agreement_with_recorded"] = round(agree / scored, 4)

    return summary


# ── main entry point ──────────────────────────────────────────────────────────
def replay_session(
    path: str,
    server_url: str = DEFAULT_SERVER_URL,
    rate_hz: float = 10.0,
    limit: Optional[int] = None,
    timeout: float = 5.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Replay a recorded session against a server and return summary stats.

    Parameters
    ----------
    path : str
        Path to the .npz produced by ``SessionRecorder.save()``.
    server_url : str
        Base URL of the e-nose server (e.g. ``http://localhost:8080``).
    rate_hz : float
        Target replay rate. ``0`` means "as fast as the server responds".
    limit : int, optional
        Stop after this many samples.
    timeout : float
        Per-request HTTP timeout in seconds.
    verbose : bool
        If True, prints a running tally every 25 samples.

    Returns
    -------
    dict
        Summary stats; also printed to stdout.
    """
    session = load_session(path)
    samples = session["samples"]
    if samples is None or len(samples) == 0:
        print(f"[replay] empty session: {path}")
        return {"n": 0, "path": str(path)}

    n_total = len(samples)
    if limit is not None and limit > 0:
        n_total = min(n_total, limit)

    sensor_names = session["sensor_names"]
    original_preds = [str(x) for x in (session["predictions"] or [])][:n_total]
    original_confs = list(session["confidences"] or [])[:n_total]

    url = server_url.rstrip("/") + "/smell/classify"
    print(f"[replay] {n_total} samples → {url}")
    if session["session_label"]:
        print(f"[replay] session_label = {session['session_label']!r}")
    if session["session_id"]:
        print(f"[replay] session_id    = {session['session_id']}")
    print(f"[replay] created_at    = {session['created_at']}")

    period = (1.0 / rate_hz) if rate_hz and rate_hz > 0 else 0.0

    replayed_preds: List[str] = []
    replayed_confs: List[float] = []
    n_errors = 0
    t_start = time.monotonic()

    for i in range(n_total):
        t_req = time.monotonic()
        payload = _row_to_payload(samples[i], sensor_names)
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            replayed_preds.append(str(body.get("predicted_smell", "")))
            replayed_confs.append(float(body.get("confidence", float("nan"))))
        except Exception as e:
            n_errors += 1
            replayed_preds.append("")
            replayed_confs.append(float("nan"))
            if verbose and n_errors <= 3:
                print(f"[replay] sample {i} failed: {e}")

        if verbose and (i + 1) % 25 == 0:
            last = replayed_preds[-1] or "?"
            conf = replayed_confs[-1]
            conf_str = f"{conf:.2f}" if not np.isnan(conf) else "n/a"
            print(f"  [{i + 1}/{n_total}] {last} ({conf_str})  errors={n_errors}")

        if period > 0:
            elapsed = time.monotonic() - t_req
            wait = period - elapsed
            if wait > 0:
                time.sleep(wait)

    elapsed_total = time.monotonic() - t_start
    summary = _summarize(
        [p for p in replayed_preds if p],
        [c for c in replayed_confs if not np.isnan(c)],
        original_preds,
        original_confs,
        session["session_label"],
    )
    summary.update({
        "path": str(path),
        "server_url": server_url,
        "elapsed_seconds": round(elapsed_total, 2),
        "errors": n_errors,
        "session_id": session["session_id"],
    })

    print("\n" + "─" * 72)
    print("REPLAY SUMMARY")
    print("─" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay a recorded e-nose session against a server.")
    p.add_argument("path", help="Session .npz path")
    p.add_argument("--server", default=DEFAULT_SERVER_URL, help="Server base URL")
    p.add_argument("--rate", type=float, default=10.0,
                   help="Replay rate in Hz (0 = no throttling)")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after this many samples (0 = replay all)")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")
    p.add_argument("--quiet", action="store_true", help="Suppress per-25-sample progress")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        replay_session(
            args.path,
            server_url=args.server,
            rate_hz=args.rate,
            limit=args.limit or None,
            timeout=args.timeout,
            verbose=not args.quiet,
        )
        return 0
    except FileNotFoundError as e:
        print(f"[replay] {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[replay] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
