# Vision + Smell + Robot Dog — Integration Plan

## Context

This document is the integration plan for adding robotics (Unitree Go-1 EDU) to the existing e-nose stack (FastAPI server + Python client). Two research outputs:

- **Idea 1 — Autonomous training in the wild:** dog walks up to objects (rose, coffee, …) using VLM grounding, records e-nose data, and online-trains the classifier with the VLM-inherited label. *Near-term deadline.*
- **Idea 2 — Smell-aware navigation:** dog uses smell **and** vision jointly to *find* a target (`"find strawberry"`) — source localization via olfactory cues, with the VLM scoping candidates. *Long-term research, publication target.*

The previous Claude session produced an analysis (`docs/Robot dog integration analysis.pdf`). This plan keeps what was right, fixes mistakes, and locks in concrete decisions (ROS2 already up on the Jetson, pump acquirable, no SLAM yet, both ideas long-term).

---

## 1. Audit of the previous Claude session

### What was correct and survives

- **Compute split** (mission control on Jetson, FastAPI server stays on lab GPU, HTTP over Wi-Fi) — correct for both ideas; matches your hardware.
- **Pump first.** The pump is the single biggest force-multiplier (settle/recover drops from ~30–60 s to ~5–15 s). Without it, the dog spends 80% of every mission standing still. Keep this as the very first hardware task.
- **High-level Unitree SDK only.** Low-level adds weeks of stability work for a "nose-pointing" gesture that isn't required.
- **FAST-LIO2 for SLAM.** Right call for Go-1 EDU + L1 LIDAR + IMU; de-facto standard in the Unitree ROS2 community.
- **Auto-labelling guardrails:** two grounding thresholds (`τ_grounding` to *attempt*, `τ_commit > τ_grounding` to *trust*), plus source-image provenance saved with every committed batch. Without this you poison the classifier weeks later and can't tell why.
- **Cross-contamination purge between targets** is a real, non-negotiable budget item.
- **State machine for Idea 1:** `SEARCH → APPROACH → SETTLE → RECORD → COMMIT → PURGE`. Clean and ships fast.
- **Pose annotation on every live sample,** even for Idea 1. The plumbing is cheap and pre-pays for Idea 2.
- **Florence-2 latency.** ~1–3 s/inference; needs a fast onboard detector in the inner loop for Idea 2.

### What was wrong / needs correction

1. **`/predict/object` endpoint does not exist on the server.** The previous Claude listed it as an existing endpoint, but the actual server only has `/training_pipeline` (which couples vision + online-training into a single multipart call). `enose/client/api.py:78` POSTs to `/predict/object`, which currently 404s. This is a pre-existing bug and must be fixed in Phase 0, *before* any robotics integration that depends on standalone grounding.
2. **`/smell/drift` is not a settling detector.** It computes live-vs-training distribution distance, not per-sensor first-derivative magnitude. A new `/smell/settling` endpoint is genuinely needed.
3. **Confidence-as-intensity is a weak proxy.** Classifier confidence saturates near 1.0 across a wide concentration range and conflates "confident this is rose" with "lots of rose." Better default: baseline-deviation magnitude on the *target-relevant* raw sensors (top-K by feature-importance, pulled from `/smell/model_info`). Cheap, monotone, grounded in physics.
4. **Surge-cast in your environment is the wrong v0.** The PDF said "start with surge-cast" then later admitted it "won't shine in weak indoor airflow." Skip it. See §4.
5. **Florence-2 grounding scores need a different task.** `<OPEN_VOCABULARY_DETECTION>` (what the current code uses) does not return calibrated per-box scores. Use `<CAPTION_TO_PHRASE_GROUNDING>` for the multi-label `/predict/scene` endpoint.

### What I'd improve beyond the PDF

- **Define a clear comms boundary.** ROS2 stays on the dog (pose, control, fast sensors, mission orchestration). HTTP/FastAPI is only for VLM + classifier inference and live-buffer storage. Don't put ROS2 on the lab GPU server.
- **Build a hardware-free simulation harness early** (mock sensor + mock pose publisher) so the full mission policy can be debugged end-to-end without touching the robot. Elevate it to a Week-1 deliverable.
- **Calibrate the 15 cm nose offset explicitly** as a static TF (`base_link → enose_intake`), not in policy logic.

---

## 2. Architecture (both ideas)

```
┌───────────────────────────────────────────────────────────────┐
│  Lab GPU box — existing FastAPI server (mostly unchanged)     │
│   existing:                                                   │
│     /smell/classify, /smell/online_learning, /smell/drift,    │
│     /smell/model_info, /smell/learn_from_csv,                 │
│     /sensor/live/{push,recent,stats,clear},                   │
│     /training_pipeline (existing vision+train one-shot)       │
│   add in Phase 0/1:                                           │
│     /predict/object        (FIX: currently called but missing)│
│     /predict/scene         (multi-label grounding, one call)  │
│     /smell/settling        (per-sensor derivative magnitudes) │
│     pose field in /sensor/live/push payload + buffer          │
│   add in Phase 2 (Idea 2):                                    │
│     /smell/intensity?target=<class>                           │
│     /smell/spatial_map?target=<class>                         │
└─────────────────────────┬─────────────────────────────────────┘
                          │ Wi-Fi / HTTP
┌─────────────────────────┴─────────────────────────────────────┐
│ Go-1 EDU Jetson — ROS2 (already up)                           │
│                                                               │
│  ROS2 nodes (new):                                            │
│    unitree_driver       (existing/community)                  │
│    fast_lio2            (community, Phase 0)                  │
│    mission_node         (owns policy, calls VLM via HTTP)     │
│    live_publisher_node  (wraps existing LivePublisher + pose) │
│    pump_node            (USB-relay service)                   │
│                                                               │
│  Python package (new): enose/robot/                           │
│    unitree_adapter.py   walk_to / face / stand / stop         │
│    localization.py      TF lookup → (x, y, θ, t)              │
│    pump.py              ROS2 service client                   │
│    visual_servo.py      bbox + depth → cmd_vel                │
│    fast_detector.py     onboard YOLO for inner-loop (Idea 2)  │
│    intensity.py         baseline-deviation scalar (Idea 2)    │
│    plume_tracer.py      gradient → infotaxis (Idea 2)         │
│    policy.py            Idea-1 state machine                  │
│    fusion_policy.py     Idea-2 state machine                  │
│    nodes/mission_node.py, nodes/live_publisher_node.py        │
│                                                               │
│  Reused unchanged:                                            │
│    enose/client/{sensors,live,api,session,webcam}.py          │
└───────────────────────────────────────────────────────────────┘
```

Hard rule: **`enose/server/` stays reusable for non-robotic chemistry work.** All robot-specific code lives in `enose/robot/`. The server only learns new endpoints; it never imports anything robot-flavoured.

---

## 3. Idea 1 — Autonomous Training in the Wild

### 3.1 Files to add / modify

**Server (`enose/server/`)**

- `routes/vision.py` *(new)*
  - `POST /predict/object` — restores the missing endpoint the client already calls. Reuses `process_image_with_vlm` from `enose/vision/florence.py`. One label per call.
  - `POST /predict/scene` — image + list of candidate labels → list of `(label, bbox, score)`. Switch to `<CAPTION_TO_PHRASE_GROUNDING>` because it returns per-phrase scores usable for thresholding.
- `routes/analytics.py` *(extend)*
  - `GET /smell/settling?window=10` — per-sensor first-derivative magnitudes (RMS of `dR/dt` over the last `window` seconds), with a boolean `settled`.
- `live_buffer.py` *(extend)*
  - Optional `pose: {x, y, theta, frame_id}` on entries.
- `routes/live.py` *(extend)*
  - Accept `pose` in `POST /sensor/live/push`; pass through unchanged.

**Robot package (`enose/robot/`, new)**

- `unitree_adapter.py` — only `stand()`, `walk_to(x, y, yaw, frame="map")`, `face(yaw)`, `stop()`. Implementation calls the Unitree ROS2 driver via topics/services.
- `localization.py` — `current() -> (x, y, θ, t)` via `tf2_ros.Buffer.lookup_transform("map", "base_link")`. Also `nose_frame()` returning the 15 cm forward mount pose (static TF).
- `pump.py` — ROS2 service client `/pump/set` for the USB relay. `on()`, `off()`.
- `visual_servo.py` — bbox + depth → `cmd_vel` toward target. Stop when `bbox_centered AND depth_at_bbox_center < d_record (≈30 cm)`.
- `policy.py` — state machine.
- `nodes/mission_node.py` — owns the policy.
- `nodes/live_publisher_node.py` — wraps `LivePublisher` with `pose_lookup=localization.current`.

**Existing files to touch**

- `enose/client/live.py` — `LivePublisher` gains optional `pose_lookup: Callable[[], dict | None] = None`; attach `pose=pose_lookup()` to every push.
- `enose/client/api.py` — once `/predict/object` is restored, `detect_object` works. Add `detect_scene(image_path, labels)`.

**Mission entry**

- `scripts/run_robot_trainer.py` — boots ROS2, loads target list from YAML, starts `mission_node`.

### 3.2 Idea-1 state machine

```
IDLE → SEARCH → APPROACH → SETTLE → RECORD → COMMIT → PURGE → SEARCH (next)
                                       │
                                       └→ ABORT (low grounding score / OOD blocks)
```

- **SEARCH:** drive to next waypoint or scan current pose. `/predict/scene` with the target list. If any `score > τ_grounding`, → APPROACH carrying the bbox + label.
- **APPROACH:** visual servo until `bbox_centered AND depth < d_record`. Pump **off** during approach.
- **SETTLE:** stand still, pump **on**. Poll `/smell/settling` until settled for `K = 3` consecutive seconds, cap at 60 s.
- **RECORD:** stream ~30 s of samples via the live publisher (pose attached) → server live buffer.
- **COMMIT:** if Florence-2 grounding score > `τ_commit` *and* mean OOD score reasonable → `/smell/online_learning` with the batch and label. Save provenance.
- **PURGE:** pump off, walk to clean zone; wait until `/smell/settling` reports baseline drift returned.

### 3.3 Auto-labelling guardrails

- Two thresholds `τ_grounding < τ_commit`, tuned empirically.
- Provenance fields on `/smell/online_learning`: `image_path, grounding_score, bbox, pose`.
- `scripts/audit_auto_labels.py` — walks provenance, lets human flag/rollback bad commits.

### 3.4 Idea-1 milestones (≈4–6 weeks)

| Week | Deliverable | Validation |
|---|---|---|
| 1 | Pump + USB relay + `pump_node`. FAST-LIO2 up. `unitree_adapter` (walk_to / face / stand). `localization.py`. **Sim harness** (mock sensor + mock pose). | Manual teleop produces pose-tagged live samples on the server. |
| 2 | Server: `/predict/object` (fix), `/predict/scene`, `/smell/settling`, pose field in live buffer. Client: `LivePublisher` pose hook. `policy.py` skeleton. | Policy dry-runs on sim harness end-to-end. |
| 3 | `visual_servo.py` stable. Full `IDLE → … → COMMIT` on one target. | Dog autonomously trains on rose, accuracy bumps. |
| 4 | Multi-target (N = 5) with PURGE. Threshold tuning. `audit_auto_labels.py`. | Paper-quality run. |
| 5–6 | Buffer, hardening, write-up. | Reproducible demo. |

### 3.5 Cuttable scope

| Priority | Item |
|---|---|
| Last to cut | Pump, robot adapter, any pose source (AprilTag if no SLAM), state machine, `τ_commit` |
| Mid | `/predict/scene`, `/smell/settling` (replaced by hardcoded 60 s), audit tool |
| First | FAST-LIO2 (use AprilTags), `/smell/auto_record` shortcut |

Absolute minimum demo (≈2 weeks): pump + AprilTag-marked targets + walk-to-tag + record + commit, no SLAM.

---

## 4. Idea 2 — Smell-Aware Navigation

### 4.1 Algorithm choice (recommended, with reasoning)

**Skip surge-cast. Build gradient-ascent baseline (v0), then infotaxis-with-VLM-prior (v1–v2, publication target).**

Reasoning:
- **Surge-cast** needs a reliable upwind estimate. Your indoor lab with HVAC + 15 cm passive intake gives a well-mixed, weakly directional flow. Surge-cast thrashes. Not v0.
- **Gradient ascent (stop-and-sample)** is the right baseline: implementable, debuggable, makes the rest of the pipeline (pose-tagged samples, intensity endpoint, decision logic) testable. Also the paper's "classical baseline" for comparison.
- **Infotaxis** (Vergassola et al., *Nature* 2007) maintains a Bayesian posterior over source position, picks actions that maximize expected entropy reduction. It does **not** require a wind direction — plug in a diffusion-only forward sensor model and the framework still works.
- **Infotaxis-with-VLM-prior** is the publishable novelty: Florence-2 grounds candidate source objects in the visible scene → each candidate becomes a broad Gaussian peak in the prior over source positions (back-project bbox center via depth + pose to world-frame). Infotaxis updates the posterior using smell observations; actions disambiguate among visible candidates.

Sequence:
- **v0:** stop-and-sample gradient ascent (baseline).
- **v1:** infotaxis with diffusion-only forward model, uniform prior (validates the inference machinery).
- **v2 (publication contribution):** infotaxis with Florence-2-derived prior (multimodal fusion).
- **v3 (extension):** multi-source; VLM-grounded *class* prior ("find strawberry, not orange" — the user's "cool experiment").

### 4.2 Files to add (on top of Idea 1's `enose/robot/`)

- `enose/robot/fast_detector.py` — small onboard YOLO (or colour blobs) at ~30 Hz on the Jetson. Hands off to Florence-2 (~1 Hz) only when something interesting is in frame.
- `enose/robot/intensity.py` — `intensity(target_class) = baseline_deviation` on top-K importance sensors for the target (from `/smell/model_info`).
- `enose/robot/plume_tracer.py` — pluggable: `GradientAscentTracer` (v0), `InfotaxisTracer` (v1), `MultimodalInfotaxisTracer` (v2).
- `enose/robot/fusion_policy.py` — Idea-2 state machine implementing the smell/vision handoff (EXPLORE / DETECTED / LOCALIZING / CONFIRM / LOST).

**Server-side additions**

- `GET /smell/intensity?target=<class>&window=<sec>` — current baseline-deviation scalar from the live buffer.
- `GET /smell/spatial_map?target=<class>` — grid of `(x, y, mean_intensity, n_samples)` from the pose-tagged buffer.

### 4.3 Idea-2 milestones (≈2–4 months after Idea 1 ships)

| Month | Deliverable | Validation |
|---|---|---|
| 1 | `/smell/intensity`, `/smell/spatial_map`, `GradientAscentTracer` in sim. | Sim walks up a synthetic gradient. |
| 2 | Controlled plume rig (perfume diffuser). Single-odour gradient ascent in lab. | Time-to-source vs random walk over N trials. |
| 3 | `InfotaxisTracer` on real plume rig. Compare to gradient. | Paper figure: two-line comparison. |
| 4 | `MultimodalInfotaxisTracer` with Florence-2 prior. Multi-candidate scene. | Headline publication experiment. |
| 5+ | Multi-source, outdoors, infotaxis under turbulence, fast PID sensor for inner loop. | Extension experiments. |

### 4.4 Idea-2 risks beyond the PDF

- **Diffusion model mismatch.** Indoor plumes deviate from pure diffusion (HVAC eddies). Validate the forward model on a known plume rig; fall back to a non-parametric model (KDE over observed `intensity(x, y)`) if needed.
- **Sensor-history confound in infotaxis.** MOS sensors have ~minute memory; consecutive samples are correlated. Subsample the buffer to 1 sample/30 s for the posterior update; dense buffer for visualization only.
- **Grounding back-projection error.** `bbox_center + depth → world (x, y)` accumulates depth, extrinsic, and SLAM drift errors. Make the VLM prior a *broad* Gaussian (σ ≈ 30 cm), not a delta.

---

## 5. Sequencing across both ideas

```
Phase 0 (weeks 1–2)  — robot plumbing (shared)
    pump, FAST-LIO2, unitree_adapter, localization, sim harness,
    live-buffer pose field, /predict/object fix, /predict/scene,
    /smell/settling
                ↓
Phase 1 (weeks 3–6)  — Idea 1
    visual_servo, policy state machine, mission_node,
    auto-labelling guardrails, multi-target missions, audit tool
                ↓
Phase 2 (months 2–5) — Idea 2
    intensity + spatial_map endpoints, fast_detector,
    GradientAscentTracer → InfotaxisTracer → MultimodalInfotaxisTracer,
    fusion_policy, controlled plume rig, publication experiments
```

Phase-0 work carries directly into Phase 2. Nothing in Phase 1 is redone for Idea 2 except the policy state machine.

---

## 6. Verification plan

**Phase 0**
- Manual teleop with live publisher running → server UI Live tab shows pose-tagged samples.
- `POST /predict/object` with a known image → returns a bbox (regression check).
- `POST /predict/scene` with `["rose", "cup", "book"]` → three rows, scores in `[0, 1]`.
- Pump ROS2 service toggles → sensor readings respond.

**Phase 1**
- `scripts/run_robot_trainer.py --targets rose.yaml` → single COMMIT, audit row, accuracy bump.
- N = 5 targets → audit output flags any mis-grounded commits.
- Sim harness completes full mission end-to-end without robot.

**Phase 2**
- Plume rig at known `(x₀, y₀)`. 10 trials each of gradient and infotaxis tracers. Time-to-within-30 cm < random walk.
- 3-candidate-objects scene with plume from only one → `MultimodalInfotaxisTracer` selects the correct object faster than baselines.

---

## 7. Open questions / things to lock in before coding

- Hard threshold values: `τ_grounding`, `τ_commit`, `d_record`, settling-derivative threshold. Provisional now, tune in Week 4.
- Provenance storage location — recommend server-side under `data/provenance/`.
- Pump relay interface — USB serial or GPIO; pick whichever your hardware already supports.
- Keep `/training_pipeline`? Yes — chemist-friendly one-shot stays for the existing UI; the robot uses the decoupled endpoints.

---

## 8. Idea 1 — Step-by-step implementation plan

For execution, Idea 1 is broken into the following meaningful, verifiable steps. Each step has a Docker-runnable verification (or an explicit "ask user to verify" hand-off where hardware is required).

| Step | Title | Deliverable | Verification |
|---|---|---|---|
| 1.1 | Restore `/predict/object` and add `/predict/scene` | New `enose/server/routes/vision.py` with both routes; registered in `enose/server/app.py` | `pytest`-style script hitting both endpoints inside the server Docker container |
| 1.2 | Add `pose` field to live buffer | `LiveEntry`/`LiveSample` carries optional `pose`; `/sensor/live/push` accepts and stores it; `/sensor/live/recent` returns it | Unit test: push with pose → read back with pose preserved |
| 1.3 | Add `/smell/settling` | `GET /smell/settling?window=10` returns per-sensor RMS derivative + boolean `settled` | Test: push synthetic stable vs drifting samples → endpoint reports correct `settled` |
| 1.4 | `LivePublisher` pose hook | Optional `pose_lookup` callback in `enose/client/live.py`; attaches `pose` to each push | Unit test: mocked publisher with mock pose_lookup → observe `pose` in push payload |
| 1.5 | `enose/robot/` package skeleton | `unitree_adapter.py`, `localization.py`, `pump.py`, `visual_servo.py`, `policy.py` (no-op/stub bodies but with correct signatures) | `python -c "import enose.robot"` succeeds inside client container |
| 1.6 | Simulation harness | Mock e-nose + mock pose publisher driving the full policy in offline mode | `scripts/sim_run.py --targets sim_targets.yaml` completes a mock mission end-to-end |
| 1.7 | Policy state machine (Idea 1) | Full `IDLE → … → PURGE` transitions wired against `unitree_adapter`/HTTP API | Sim harness completes mission; transition log matches expected sequence |
| 1.8 | Visual-servo module | Bbox + depth → `cmd_vel`; stop on `bbox_centered AND depth < d_record` | Unit test with synthetic depth image + bbox |
| 1.9 | Mission entry point | `scripts/run_robot_trainer.py` (CLI, YAML target list) | User-verified: run on the real dog (hardware gate) |
| 1.10 | Auto-labelling guardrails + audit tool | `/smell/online_learning` accepts provenance; `scripts/audit_auto_labels.py` | Unit test + user-verified UI walkthrough |
| 1.11 | Docker images updated | `docker/Dockerfile-cu124` (server) + `docker/Dockerfile-client` (client) install any new deps | `docker build` succeeds for both |

Execution starts at Step 1.1 with a Docker-only workflow.
