# Idea 1 — Implementation Progress

This document records what was implemented in the first integration session, file by file, with the verification evidence for each piece. Companion docs:

* [`IDEA1_PLAN.md`](IDEA1_PLAN.md) — the design plan this work executes against.
* [`NEXT_STEPS.md`](NEXT_STEPS.md) — what's left, ordered by priority.

> **Status at session end:** all 11 plan steps for Idea 1 are landed and verified inside Docker. The robot integration is *complete on the software-only path* — what remains is the Jetson-side ROS2 wiring (skeletons raise `NotImplementedError` until the real-hardware bring-up) and tuning thresholds against real photos.

---

## 1. Architecture delta

The existing repo's split is unchanged:

* `enose/server/` — FastAPI on the lab GPU. *Reusable for non-robotic chemistry work; never imports anything robotics-flavoured.*
* `enose/client/` — Python client on the dog's Jetson. Hardware I/O + HTTP.

What was added:

* **Server (`enose/server/routes/vision.py`)** — vision endpoints decoupled from training.
* **Server (`enose/server/routes/analytics.py`)** — `/smell/settling` endpoint.
* **Server (`enose/server/routes/live.py`, `live_buffer.py`, `schemas.py`)** — optional `pose` on every live sample + `CommitProvenance` schema.
* **Server (`enose/server/routes/training.py`)** — provenance persistence + `GET /smell/provenance` listing.
* **Client (`enose/client/api.py`, `enose/client/live.py`)** — new methods + `LivePublisher.pose_lookup` hook.
* **`enose/robot/` (new package)** — Protocols + sim impls + ROS2 stubs + the mission state machine.
* **`scripts/`** — `sim_run.py`, `run_robot_trainer.py`, `audit_auto_labels.py`, `sim_targets.json`.

The hard rule from the plan holds: **`enose/server/` stays clean**; all robot logic lives in `enose/robot/`.

---

## 2. Bug found and fixed before any new work

The previous Claude session claimed `POST /predict/object` already existed on the server. It did not — `enose/client/api.py:78` POSTed to it for months and got 404. Vision was only reachable through `/training_pipeline`, which couples grounding + training in one call. That coupling is the wrong shape for a robotics policy that needs to apply a `τ_grounding` threshold *before* deciding whether to commit.

Fix: `enose/server/routes/vision.py` restores `POST /predict/object` (open-vocabulary detection, one label per call) and adds `POST /predict/scene` (`<CAPTION_TO_PHRASE_GROUNDING>` for multi-label batches with bbox-area-fraction scores).

---

## 3. Step-by-step implementation log

Each step's "Verified" line is what was actually run inside Docker.

### Step 1.1 — `/predict/object` and `/predict/scene`

* **New:** `enose/server/routes/vision.py` (`POST /predict/object`, `POST /predict/scene`).
* **Edited:** `enose/server/routes/__init__.py`, `enose/server/app.py` (router registration).
* **Verified:** OpenAPI lists both routes; `NO_VLM` mode → 503 ("VLM not loaded"); validation errors → 400 for bad JSON / wrong types / empty labels; live Florence-2 inference → HTTP 200 with the right JSON shape on both endpoints.

### Step 1.2 — `pose` field on the live buffer

* **Edited:** `enose/server/routes/live.py` (`LivePushPayload.pose: LivePose`), `enose/server/live_buffer.py` (docstring only).
* **Verified:** push without pose → still works; push with pose → server stores and `/sensor/live/recent` returns it; malformed pose (missing `y`) → 422 with a per-field error.

### Step 1.3 — `GET /smell/settling`

* **Edited:** `enose/server/routes/analytics.py`.
* **Algorithm:** per-resistance-sensor least-squares slope `dR/dt`, normalized by mean (`rel_slope`). `settled` iff `|rel_slope| < threshold` for *every* `R*` sensor, with ≥ `min_samples` over ≥ `0.5 * window` seconds.
* **Verified:** 120 stable samples (R≈100±0.05) → `settled=True`. R1 ramping from 100→150 over 6s → `settled=False, worst_sensor=R1, worst_rel_slope=0.062`.

### Step 1.4 — `LivePublisher` pose hook + `detect_scene` client method

* **Edited:** `enose/client/live.py` (added `pose_lookup` ctor param + `_lookup_pose` helper + pass-through to the push payload), `enose/client/api.py` (`detect_scene` method).
* **Verified:** publisher with a pose ticker simulating 1 m/s along x → 20 pushes in 2 s, every entry has pose, x monotonically increasing on the server side.

### Step 1.5 — `enose/robot/` package skeleton

* **New files:** `interfaces.py`, `policy.py` (initially stub), `visual_servo.py` (initially stub), `vision_http.py`, `sim/{__init__,motion,localization,pump,vision}.py`, `ros2/{__init__,unitree_adapter,localization,pump}.py`.
* **Design:** `typing.Protocol`-style interfaces (`MotionAdapter`, `Localizer`, `PumpController`, `VisionClient`). Dataclasses `Pose`, `Detection`, `PolicyAdapters`. Sim impls work; ROS2 impls are guarded skeletons that raise `RuntimeError`/`NotImplementedError` on a no-rclpy host so the sim path can never be confused with the real path.
* **Verified:** smoke test passes (65 files, 0 failures); package imports cleanly on host + inside both Docker images; ROS2 stub instantiation raises a clear error message.

### Step 1.6 + 1.7 — Sim harness + Idea-1 policy state machine

* **New:** `scripts/sim_run.py`, `scripts/sim_targets.json`, `enose/robot/mission.py`.
* **Edited:** `enose/robot/policy.py` (full state machine body — `IDLE → SEARCH → APPROACH → SETTLE → RECORD → COMMIT → PURGE → SEARCH/DONE`).
* **Design choices baked in:**
  * Long-running `LivePublisher` for the whole mission so `/smell/settling` has live data; per-target collection toggled by a `_collecting` flag.
  * Pump off during APPROACH (no travel air saturating sensors), on during SETTLE + RECORD.
  * Two-threshold guardrail: `τ_grounding` to *attempt*, `τ_commit > τ_grounding` to *trust*.
  * Hard `settling_timeout` so a stuck sensor never hangs the mission.
* **Verified:** 3-target sim (rose at world (2,0), coffee at (2,2), lemon at unreachable (8,8) — out of detection radius from any waypoint). Run reproducibly transitions 15 times, settles each time at k=3, records ~10–40 samples per target depending on speed, skips lemon for `no detection above tau_grounding`, attempts commits for rose and coffee.

### Step 1.8 — Visual-servo module body

* **Edited:** `enose/robot/visual_servo.py` (was stub).
* **Algorithm:** P-controller on bbox horizontal offset → `wz`; linear on `depth - d_record` → `vx`, gated by "bbox roughly centered" to avoid driving while turning. Outputs `ServoCommand(vx, vy, wz, done, offset_px, depth_m)`.
* **Verified:** 7 unit cases — left/right yaw direction, fwd zero when not centered, fwd positive when centered + depth > d_record, `done=True` when centered + depth ≤ d_record, max-rate clamping, invalid bbox handled.

### Step 1.9 — Mission entry script

* **New:** `scripts/run_robot_trainer.py`.
* **Design:** `--backend sim | ros2`. `sim` reuses the harness; `ros2` builds `UnitreeAdapter` / `TFLocalizer` / `Ros2Pump` / `HttpVisionClient` (the ROS2 adapters currently raise `NotImplementedError`, so the CLI exits with code 3 + a clear instruction string).
* **Verified:** `--backend sim` runs end-to-end with 1 successful commit; `--backend ros2` (no rclpy in Docker) prints "wire up `enose.robot.ros2.{...}` on the Jetson" and exits.

### Step 1.10 — Provenance + audit tool

* **New:** `enose/server/schemas.py:CommitProvenance`, `scripts/audit_auto_labels.py`, `data/provenance/` (created on first commit).
* **Edited:** `enose/server/routes/training.py` (`_persist_provenance` + new `GET /smell/provenance`), `enose/client/api.py` (provenance kwarg on `online_learning` + `list_provenance`), `enose/robot/policy.py` (`_build_provenance` — base64-encodes the source image).
* **Key safety property:** the server persists provenance to `data/provenance/<id>.{json,png}` *before* attempting the classifier retrain, so the audit trail survives commits that fail server-side. The plan §3.3 called this out as mandatory; it's now enforced.
* **Audit tool subcommands:** `list`, `inspect <id>`, `low-score --threshold X`.
* **Verified:** sim mission produces 2 provenance entries (rose at score 0.350, coffee at 0.220); `audit list` table renders correctly; `audit inspect` returns the full JSON; provenance is preserved even when the underlying retrain 500s.

### Step 1.11 — Docker image deps

* **No Dockerfile changes required.** Every import in the new code is either stdlib, already in `Dockerfile-cu124` (numpy, pandas, fastapi, pydantic, PIL), or already in `Dockerfile-client` (numpy, pandas, requests, pyserial, opencv, matplotlib, pillow).
* **Verified:** `docker build -t enose-client -f docker/Dockerfile-client .` succeeds; `enose.robot` + `scripts/run_robot_trainer.py` import inside the client image; `visual_servo.step` runs.

---

## 4. Pre-existing server quirks discovered along the way

Two real things, neither caused by this work, both relevant for what comes next:

1. **`/predict/object` was missing.** Fixed in Step 1.1 (`routes/vision.py`).
2. **`BalancedRFClassifier` retrain on small new-class batches is fragile.** `CalibratedClassifierCV` (default `cv=5`) can land a CV fold with only one class, which breaks `imblearn`'s `BalancedRandomForest.fit` with `ValueError: The target 'y' needs to have more than 1 class. Got 1 class instead`. Symptom on the sim missions: one of two new-class commits typically fails with HTTP 500. The policy already handles it gracefully (logs `online_learning failed`, marks the target as skipped, moves on). Real fix lives in [`NEXT_STEPS.md`](NEXT_STEPS.md) §2. Also written into memory.

One hardware constraint discovered after the initial plan:

3. **Pump is mechanically switched only.** The plan's §3.1 assumed a USB-relay-controlled pump; the rig has no such hardware and one is not planned in the near future. Resolved by adding `enose.robot.manual_pump.ManualPump`, which logs the requested state for a human operator to act on. The policy's state transitions are unchanged; only the binding is different. CLI flag: `--pump manual` (default for `--backend ros2`) or `--pump manual-confirm` (blocks until Enter at each transition).

---

## 5. Files touched (canonical list)

```
enose/server/
  app.py                    — register vision router
  schemas.py                — CommitProvenance + OnlineLearningData.provenance field
  live_buffer.py            — docstring (schema docs only; storage is dict-opaque)
  routes/__init__.py        — register vision router
  routes/vision.py          — NEW. /predict/object, /predict/scene
  routes/live.py            — LivePushPayload.pose
  routes/analytics.py       — GET /smell/settling
  routes/training.py        — provenance persistence + GET /smell/provenance

enose/client/
  api.py                    — detect_scene, get_settling, online_learning(provenance=…), list_provenance
  live.py                   — LivePublisher.pose_lookup + payload pass-through

enose/robot/                — NEW PACKAGE
  __init__.py
  interfaces.py             — Pose, Detection, Localizer, MotionAdapter, PumpController,
                              VisionClient, PolicyAdapters
  manual_pump.py            — ManualPump (logs "switch pump ON/OFF" prompts; rig has no
                              programmable relay yet — default for --backend ros2)
  mission.py                — load_targets_file, make_targets, run_mission, print_summary
  policy.py                 — Policy state machine + Target + PolicyConfig + provenance assembly
  visual_servo.py           — step(bbox, image_size, depth) → ServoCommand
  vision_http.py            — HttpVisionClient (ServerAPI → VisionClient adapter)
  sim/
    __init__.py
    motion.py               — SimMotion (instantaneous teleport)
    localization.py         — SimLocalizer (reads sim state)
    pump.py                  — SimPump (bool flag)
    vision.py               — ScriptedVision (canned detections)
  ros2/
    __init__.py             — guarded rclpy import
    unitree_adapter.py      — UnitreeAdapter (skeleton, raises NotImplementedError)
    localization.py         — TFLocalizer (skeleton)
    pump.py                  — Ros2Pump (skeleton)

scripts/
  sim_run.py                — NEW. Sim harness CLI
  run_robot_trainer.py      — NEW. Real-robot CLI (--backend sim|ros2)
  sim_targets.json          — NEW. Example mission spec for the harness
  audit_auto_labels.py      — NEW. list/inspect/low-score subcommands

docs/
  IDEA1_PLAN.md             — NEW. The plan this implementation executes against
  IDEA1_PROGRESS.md         — NEW. This file
  NEXT_STEPS.md             — NEW. What to do next
```

---

## 6. How to reproduce the verification

Inside the repo root (`/home/vlm-workspace/e-nose`):

```bash
# 1. Smoke test — AST-parse all 66 .py files + check every package's __all__.
python3 scripts/smoke_test.py

# 2. Start the server (NO_VLM mode is enough for everything that doesn't
#    touch Florence-2 — i.e. all of Steps 1.2–1.11).
docker run --rm -d --name enose-srv --network enose-net \
    -p 18080:8080 -v /home/vlm-workspace/e-nose:/app \
    -e PYTHONPATH=/app -e ENOSE_NO_VLM=1 \
    enose-server:latest python scripts/run_server.py
until curl -fsS http://localhost:18080/health 2>/dev/null >/dev/null; do sleep 1; done

# 3. Run a full sim mission.
docker run --rm --network enose-net \
    -v /home/vlm-workspace/e-nose:/app -e PYTHONPATH=/app \
    enose-server:latest python scripts/run_robot_trainer.py \
    --backend sim --server http://enose-srv:8080 \
    --targets /app/scripts/sim_targets.json \
    --speed 10 --record-seconds 10 --purge-seconds 2 --settling-timeout 8

# 4. Inspect what was committed.
docker run --rm --network enose-net \
    -v /home/vlm-workspace/e-nose:/app -e PYTHONPATH=/app \
    enose-server:latest python scripts/audit_auto_labels.py \
    --server http://enose-srv:8080 list

# 5. With the VLM — verify /predict/scene against a real photo.
docker run --rm -d --name enose-srv --gpus '"device=0"' --network enose-net \
    -p 18080:8080 -v /home/vlm-workspace/e-nose:/app \
    -e PYTHONPATH=/app enose-server:latest python scripts/run_server.py
# ... wait ~60s for Florence-2 to load (poll /health for "vlm":true) ...
curl -X POST http://localhost:18080/predict/scene \
    -F image=@your_photo.jpg -F 'labels=["rose","cup","book"]'
```

---

## 7. Quality gates that hold at session end

* `scripts/smoke_test.py` — 66 files parsed, 0 failures.
* `enose.robot` imports cleanly on host *and* inside both Docker images.
* `enose.robot.ros2.UnitreeAdapter()` raises `RuntimeError` on a no-rclpy host with the right message.
* All seven `visual_servo.step` unit cases pass.
* End-to-end sim mission: 15 transitions, ≥ 1 successful commit, audit trail persisted with image + score + pose.
