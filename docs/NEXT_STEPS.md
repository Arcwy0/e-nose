# What to do next

Companion to [`IDEA1_PROGRESS.md`](IDEA1_PROGRESS.md). The software-only path of Idea 1 is complete; what's left is everything that needs your hardware in the room and the longer Idea-2 research arc.

Sections are ordered roughly by *what unblocks what*. The Idea-1 deadline-critical work is §§1–4. Idea-2 starts at §6.

---

## 1. Tune `τ_grounding` / `τ_commit` against real photos *(half a day, no robot needed)*

The current defaults (`τ_grounding=0.05`, `τ_commit=0.10`, both as bbox-area-fraction) are placeholders. They have to be calibrated against your scene.

What to do:

1. Bring up the server with the VLM (no robot):
   ```bash
   docker run --rm -d --name enose-srv --gpus '"device=0"' --network enose-net \
       -p 18080:8080 -v "$(pwd)":/app -e PYTHONPATH=/app \
       enose-server:latest python scripts/run_server.py
   ```
2. Collect 20–30 photos that include the objects you're going to train on (rose, coffee cup, lemon, …), plus 10 that *don't*.
3. For each photo, hit `POST /predict/scene` with the full candidate list and log `(label, ground_truth, score)`.
4. Plot scores; pick `τ_grounding` at the 90th percentile of true-positive scores' lower edge and `τ_commit` at the 95th percentile (so false-positives don't sneak in).
5. Update `PolicyConfig.tau_grounding` / `tau_commit` defaults *or* pass them as CLI args.

**Why this matters:** Florence-2's `<CAPTION_TO_PHRASE_GROUNDING>` does not emit a calibrated confidence — we proxy with bbox area fraction. The proxy is monotone in "how close / how prominent" the object is in frame, which is what we actually care about, but the absolute numbers depend on your scene's typical framing. Get the calibration right *once* and the policy is robust.

---

## 2. Server-side fix for the small-batch retrain quirk *(1 day)*

The `BalancedRFClassifier`'s `CalibratedClassifierCV` (default `cv=5`) chokes when a tiny new-class batch lands a CV fold with one class only, raising `ValueError: The target 'y' needs to have more than 1 class`. The policy already catches this and moves on, but it costs you a commit per mission.

Two fixes, pick one:

**Option A — lower `cv` adaptively (recommended).** In `enose/classifier/balanced_rf.py`, in the retrain path, when *any* class has fewer than 5 samples in the combined training frame, fall back to `cv=2`. ~10 lines.

**Option B — skip calibration on incremental updates.** Keep the uncalibrated `BalancedRandomForest` for online updates; only run the full calibrated wrapper during the periodic full-retrain (e.g. once a day or after N batches). Cleaner but more code.

Verification: rerun the sim mission with 3 targets; both rose and coffee should commit successfully.

---

## 3. Real-robot bring-up on the Jetson *(Phase 0 of the plan, ≈ week 1)*

The ROS2 adapters in `enose/robot/ros2/` are currently skeletons that raise `NotImplementedError`. Bring them up in this order — each step is independently verifiable, so don't skip ahead.

### 3.1 Pump — already handled by `ManualPump` *(no work needed)*

The rig's pump is mechanically switched only — there is no programmable relay and one is not planned in the near future. The policy already supports this: `enose.robot.manual_pump.ManualPump` (default for `--backend ros2`) logs the requested state as `[pump] please switch pump ON / OFF` so the human watching the terminal can flip the physical switch. Use `--pump manual-confirm` if you want the policy to block until you press Enter at each transition.

The `enose.robot.ros2.pump.Ros2Pump` skeleton stays in place so wiring is trivial if/when programmable hardware shows up. Until then, **don't implement it.**

Operational implication: in multi-target missions where odour purge between targets matters, the human has to flip the pump at the right moments OR you run a single odour per mission. For Idea-1 demos with one to two odours this is fine.

### 3.2 FAST-LIO2 + TF tree *(2 days, mostly extrinsic calibration)*

Bring up FAST-LIO2 with the Go-1 EDU L1 LIDAR + IMU. Publish `map → odom → base_link`. Then publish a *static* transform `base_link → enose_intake` for the 15 cm forward sensor mount (configure in a one-line launch file).

Implement `enose.robot.ros2.localization.TFLocalizer` (the skeleton's signatures are locked):
```python
def current(self) -> Optional[Pose]:
    msg = self._tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
    ...
```

Verify: drive the dog around manually; the live tab on the server UI (with the new pose plumbing from Step 1.4) shows samples with plausible `(x, y, θ)`.

### 3.3 `UnitreeAdapter` *(1 day)*

The Go-1 EDU's `unitree_ros2` driver exposes high-level sport commands. Wire `walk_to(x, y, yaw, frame, timeout)` to whatever action / topic the driver supports (typically a NavigateToPose-style action or a Twist publisher with a closed-loop wrapper). `stand`, `stop`, `face` are smaller helpers.

`cmd_vel(vx, vy, wz)` is what `visual_servo.step` outputs — publish on `/cmd_vel` at ≥ 10 Hz.

Verify: a tiny script that calls `UnitreeAdapter.walk_to(1.0, 0.0, 0.0)` and watches the dog move 1 m forward.

### 3.4 `HttpVisionClient` capture wiring *(half a day)*

Replace `WebcamHandler.capture_image` with whatever camera you actually use on the dog. The depth camera on the Go-1 EDU exposes an RGB stream over ROS2 (`sensor_msgs/Image`); subscribe, convert with `cv_bridge`, save the latest frame to a tempfile, return the path.

While there, also implement a `depth_at_bbox_center(bbox)` helper that reads the corresponding depth frame at the bbox center pixel. The visual-servo control loop needs it.

### 3.5 Integrate visual servo into `_approach`

Currently the policy uses the "walk to detection.extras['target_pose']" path which only the sim has. Add the real path: when `extras["target_pose"]` is absent, run the visual-servo loop:

```python
while not done:
    bbox, depth_im = vision_client.latest_frame()
    cmd = visual_servo.step(bbox, image_size, depth_at_bbox_center(bbox, depth_im))
    motion.cmd_vel(cmd.vx, cmd.vy, cmd.wz)
    done = cmd.done
    if elapsed > timeout: break
```

Verify: AprilTag on a stand; dog drives to it autonomously, stops at the right distance.

---

## 4. First real autonomous-training run *(1 day, after §1–3)*

Mission profile:

* 1 known target (rose) at a known waypoint.
* Pump on for SETTLE + RECORD, off for APPROACH and PURGE.
* `record_seconds = 30`, `purge_seconds = 30`.
* `--backend ros2` (the real path).

Success criterion: classifier accuracy on the "rose" class measurably improves vs the pre-mission baseline.

Once that works, extend to N = 5 targets in one mission and rerun. The plan's Week 4 milestone is "paper-quality experiment: N missions × N targets, accuracy gains over baseline, false-label rate from Florence-2 grounding."

---

## 5. Audit + cleanup hooks

Two small things worth doing right after the first real mission:

* **Audit script polish.** `scripts/audit_auto_labels.py` currently prints to the terminal. Two additions to consider:
  * `--html out.html` writes a static HTML report with embedded images for offline review.
  * `--rollback <id>` removes a bad commit's samples from `training_data.csv` and triggers a full retrain (think `git revert` for a labelled batch).
* **Cross-contamination diagnostics.** After each PURGE, log whether `/smell/drift` came back to baseline before SEARCH starts. Right now we just wait `purge_seconds`; the dog should *check* that the sensors actually recovered before moving on. This is a 30-line addition to `policy._purge`.

---

## 6. Idea-2 prerequisites *(starts after Idea 1 ships, ≈ month 2+)*

Everything below carries directly over from Phase 0 — pump, SLAM, robot adapter, pose-tagged buffer, sim harness. Don't rebuild them.

### 6.1 Server endpoints for spatial smell *(2 days)*

Implement in `enose/server/routes/analytics.py`:

* `GET /smell/intensity?target=<class>&window=<sec>` — baseline-deviation magnitude over the top-K sensors by feature-importance for the target class (`get_model_info` already returns the importances). Defined in §4.1 of the plan; the cheap-tier proxy from the previous Claude (`confidence × (1 − OOD/threshold)`) is **not** what we want — see [`IDEA1_PLAN.md`](IDEA1_PLAN.md) §1.3 for why.
* `GET /smell/spatial_map?target=<class>` — given the pose-tagged live buffer, return a grid of `(x, y, mean_intensity, n_samples)`. High-leverage for both the controller and paper figures.

### 6.2 Idea-2 algorithm sequence

Per the plan §4.1, the recommendation is to skip surge-cast entirely:

1. **v0 (month 1) — stop-and-sample gradient ascent.** New file `enose/robot/plume_tracer.py:GradientAscentTracer`. Sample at 3+ poses, fit a local gradient, walk along it. Use this as both the baseline implementation *and* the paper's classical comparison.
2. **v1 (month 2) — infotaxis with diffusion-only forward model.** Add `InfotaxisTracer`. Maintain a discretized grid posterior `p(source | observations)`; each observation updates it via a Gaussian-diffusion likelihood; the action is the cell that maximizes expected entropy reduction. *No wind estimate needed.*
3. **v2 (month 3) — infotaxis with VLM prior.** Add `MultimodalInfotaxisTracer`. Initialize the prior from `/predict/scene`'s back-projected bbox centers (broad Gaussian per candidate, σ ≈ 30 cm to absorb depth + extrinsic + SLAM noise). **This is the publishable contribution.**
4. **v3 (extension) — multi-source + class-grounded prior.** "Find strawberry, not orange" — VLM picks out *all* visible plausible sources, smell disambiguates by class.

### 6.3 Idea-2 policy

New `enose/robot/fusion_policy.py` replaces `policy.py` for Idea 2. Implements the smell/vision handoff table in plan §4.2: `EXPLORE → DETECTED → LOCALIZING → CONFIRM`, with `LOST` as a back-track recovery state.

### 6.4 Fast onboard detector

`enose/robot/fast_detector.py` — a small YOLO (or even a colour-blob heuristic for the first lab demo) running at ~30 Hz on the Jetson. Florence-2 (~1 Hz over Wi-Fi) is too slow for the inner-loop control of Idea 2; the fast detector handles "is anything interesting in front of me?", Florence-2 is reserved for periodic re-grounding at decision points.

### 6.5 Controlled plume rig

Hardware: a perfume diffuser at a known position, or a vial with known volatility. You will be asked "how do we know it found the source and didn't just walk in circles?" — the answer has to be "known source location, repeatable plume, time-to-source over N trials vs random walk."

---

## 7. Risks worth tracking now *(from the plan's §7 + new ones from this session)*

* **FAST-LIO2 extrinsic calibration is the most common time sink.** Budget a week.
* **Wi-Fi reliability for the HTTP loop.** Verify `LivePublisher`'s reconnection tolerance; add a `NETWORK_LOSS` state in the policy.
* **Cross-contamination between strong odours.** Even with the pump, sequential strong odours need real purge time. Build the budget into mission planning.
* **Florence-2 latency** stays an issue for Idea 2. Already addressed by §6.4.
* **Power.** Pump + Jetson + LIDAR + dog is a lot. Confirm battery duration covers a full multi-target mission, or build between-mission charging into the protocol.
* **Lab safety / IRB** for some odorants. Plan around it for the lab phase.
* **Server's small-batch retrain quirk** (§2 above) — fix before first real mission.

---

## 8. Suggested order for the next session

If you want to keep moving without me asking for direction:

1. §1 — calibrate `τ_grounding` / `τ_commit` with a handful of real photos (no robot needed).
2. §2 — server-side fix for the small-batch retrain quirk (so commits succeed reliably).
3. §3.2 — FAST-LIO2 (the slow one; do this early because §3.3 + §3.5 depend on it). Pump (§3.1) needs no software work.
4. §3.3 + §3.4 + §3.5 — fill in the ROS2 adapters and integrate visual servo.
5. §4 — first real autonomous-training run with one target.
6. §5 — audit hardening + cross-contamination diagnostics.
7. §6.1 + §6.2-v0 — start Idea 2 with the simplest tracer.

Each item is small enough to land in a session with verification.
