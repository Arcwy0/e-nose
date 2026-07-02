#!/usr/bin/env python3
"""Inspect the autonomous-training commit history (``data/provenance/``).

The robot's policy attaches an image, grounding score, pose, and session id
to every commit it sends to ``/smell/online_learning``. The server persists
that under ``data/provenance/<id>.{json,png}``. This script lets a human
flip through those entries to find mis-grounded labels before they poison
the classifier weeks later.

Modes:

* ``list`` (default) — table of recent commits.
* ``inspect <id>`` — full metadata for one commit.
* ``low-score`` — list commits whose ``grounding_score`` falls below
  ``--threshold``; first thing to audit after any mission.

Examples::

    python scripts/audit_auto_labels.py --server http://localhost:8080
    python scripts/audit_auto_labels.py inspect 20260522-014203-ab12cd34
    python scripts/audit_auto_labels.py low-score --threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from enose.client.api import ServerAPI


def _fetch(api: ServerAPI, limit: int, label: Optional[str]) -> List[Dict[str, Any]]:
    payload, err = api.list_provenance(limit=limit, label=label)
    if err:
        print(f"[audit] server error: {err}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(payload, dict):
        return []
    return list(payload.get("entries", []))


def _pose_str(pose: Optional[Dict[str, float]]) -> str:
    if not pose:
        return "—"
    return f"x={pose.get('x', 0):.2f} y={pose.get('y', 0):.2f} θ={pose.get('theta', 0):.2f}"


def cmd_list(api: ServerAPI, args: argparse.Namespace) -> int:
    rows = _fetch(api, args.limit, args.label)
    if not rows:
        print("[audit] no provenance entries found")
        return 0
    print(f"{'provenance_id':<30s} {'label':<14s} {'score':>6s} {'n':>4s}  pose")
    print("-" * 80)
    for r in rows:
        print(
            f"{r.get('provenance_id', '?'):<30s} "
            f"{(r.get('label') or '?'):<14s} "
            f"{(r.get('grounding_score') or 0.0):>6.3f} "
            f"{(r.get('n_samples') or 0):>4d}  "
            f"{_pose_str(r.get('pose'))}"
        )
    return 0


def cmd_inspect(api: ServerAPI, args: argparse.Namespace) -> int:
    rows = _fetch(api, args.limit, None)
    match = next((r for r in rows if r.get("provenance_id") == args.id), None)
    if match is None:
        print(f"[audit] no entry with id {args.id} in last {args.limit}", file=sys.stderr)
        return 1
    print(json.dumps(match, indent=2, sort_keys=True))
    return 0


def cmd_low_score(api: ServerAPI, args: argparse.Namespace) -> int:
    rows = _fetch(api, args.limit, args.label)
    flagged = [r for r in rows if (r.get("grounding_score") or 0.0) < args.threshold]
    if not flagged:
        print(f"[audit] no commits below score {args.threshold} in last {args.limit}")
        return 0
    print(f"{'provenance_id':<30s} {'label':<14s} {'score':>6s} {'n':>4s}  pose")
    print("-" * 80)
    for r in flagged:
        print(
            f"{r.get('provenance_id', '?'):<30s} "
            f"{(r.get('label') or '?'):<14s} "
            f"{(r.get('grounding_score') or 0.0):>6.3f} "
            f"{(r.get('n_samples') or 0):>4d}  "
            f"{_pose_str(r.get('pose'))}"
        )
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--label", default=None, help="filter by label")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    s_inspect = sub.add_parser("inspect")
    s_inspect.add_argument("id")
    s_lowscore = sub.add_parser("low-score")
    s_lowscore.add_argument("--threshold", type=float, default=0.10)
    args = ap.parse_args(argv)

    api = ServerAPI(args.server)
    handler = {
        "list": cmd_list,
        "inspect": cmd_inspect,
        "low-score": cmd_low_score,
        None: cmd_list,
    }.get(args.cmd)
    assert handler is not None  # subparsers default to list
    return handler(api, args)


if __name__ == "__main__":
    sys.exit(main())
