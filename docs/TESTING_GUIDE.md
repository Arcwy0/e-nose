# Testing Guide — from Docker to the Robot Dog

This guide walks through every check that's actually possible *with what the codebase currently implements*. Stages skipped: anything that depends on Jetson-side ROS2 adapters (`enose.robot.ros2.*` is still a skeleton; see [`NEXT_STEPS.md`](NEXT_STEPS.md) §3) and Idea-2 plume tracing.

Companion docs:
* [`IDEA1_PLAN.md`](IDEA1_PLAN.md) — design plan.
* [`IDEA1_PROGRESS.md`](IDEA1_PROGRESS.md) — what was built and how it was verified.
* [`NEXT_STEPS.md`](NEXT_STEPS.md) — what's left.

> **Hardware note on the pump.** The rig's e-nose pump is mechanically switched only — there is no programmable relay yet. The policy still calls `pump.on()` / `pump.off()` at the right transitions; the default real-robot binding is `ManualPump`, which **logs** the requested state as `[pump] please switch pump ON / OFF` so the human watching the terminal can flip the physical switch. Use `--pump manual-confirm` if you want the policy to *block* until you press Enter.

---

## Stage 0 — Prerequisites

| Item | Why | Where |
|---|---|---|
| Linux host with Docker + NVIDIA Container Toolkit | server image needs GPU | lab GPU box |
| Florence-2-Large weights | VLM grounding | `model/Florence-2-Large/` (already on disk) |
| `enose-net` Docker network | server ↔ client comms | `docker network create enose-net` if missing |
| Repo clone at `/home/vlm-workspace/e-nose` (or wherever) | source code | git or rsync |
| Go-1 EDU with Jetson, e-nose on UART, front depth + RGB camera | for stages 6–7 only | the robot |

Optional, only when you reach Stage 5: real photos of your target objects in their lab setting.

---

## Stage 1 — Docker image bring-up

Two images: one for the server (CUDA, Florence-2, classifier) and one for the client (small, deployable to the Jetson). Build only what you need.

### 1.1 Network

```bash
docker network ls | grep -q enose-net || docker network create enose-net
```

### 1.2 Server image — `enose-server` *(~15 min first build, mostly flash-attn)*

```bash
cd /home/vlm-workspace/e-nose
docker build -t enose-server -f docker/Dockerfile-cu124 .
```

**Verify:** `docker images | grep enose-server` shows a row.

If the build fails on `flash-attn==2.7.3`, free up RAM (the build caps at `MAX_JOBS=4` nvcc workers but still needs ~16 GB).

### 1.3 Client image — `enose-client` *(~2 min)*

```bash
docker build -t enose-client -f docker/Dockerfile-client .
```

**Verify:** `docker run --rm enose-client python --version` prints a 3.11.x line.

---

## Stage 2 — Server smoke (no VLM, no robot)

The "smoke mode" skips Florence-2 entirely so it boots in seconds. Everything that doesn't *touch* the VLM works.

### 2.1 Start the server in NO_VLM mode

```bash
docker run --rm -d --name enose-srv --network enose-net \
    -p 18080:8080 -v "$(pwd)":/app \
    -e PYTHONPATH=/app -e ENOSE_NO_VLM=1 \
    enose-server python scripts/run_server.py
```

Wait until ready:

```bash
until curl -fsS http://localhost:18080/health 2>/dev/null; do sleep 1; done; echo READY
```

Expected JSON: `"status":"healthy", … "models":{"vlm":false,"smell_classifier":true}`.

### 2.2 Codebase smoke test

```bash
python3 scripts/smoke_test.py
```

Expected: `[smoke] OK` — 67 files parsed, 0 failures.

### 2.3 OpenAPI sanity — the new endpoints should be there

```bash
curl -fsS http://localhost:18080/openapi.json \
  | python3 -c "import json,sys; print('\n'.join(sorted(json.load(sys.stdin)['paths'])))"
```

Look for the additions:
* `/predict/object`, `/predict/scene`
* `/smell/settling`
* `/smell/provenance`
* and the pre-existing `/smell/classify`, `/sensor/live/{push,recent,stats,clear}`, `/training_pipeline`, etc.

### 2.4 Vision endpoints return 503 without VLM *(contract check)*

```bash
# Make a tiny PNG (no PIL on host needed):
python3 -c "
import struct, zlib
def png(w,h):
    sig=b'\x89PNG\r\n\x1a\n'
    def c(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    ihdr=struct.pack('>IIBBBBB',w,h,8,2,0,0,0)
    idat=zlib.compress((b'\x00'+bytes((180,80,80))*w)*h)
    open('/tmp/t.png','wb').write(sig+c(b'IHDR',ihdr)+c(b'IDAT',idat)+c(b'IEND',b''))
png(64,48)"

curl -s -w '\nHTTP %{http_code}\n' -X POST http://localhost:18080/predict/object \
    -F image=@/tmp/t.png -F 'text=Find rose'
# expect: HTTP 503, body {"detail":"VLM not loaded"}

curl -s -w '\nHTTP %{http_code}\n' -X POST http://localhost:18080/predict/scene \
    -F image=@/tmp/t.png -F 'labels=[]'
# expect: HTTP 400, "labels list cannot be empty" (validation before VLM check)
```

### 2.5 Live buffer round-trip with pose

```bash
curl -s -X POST http://localhost:18080/sensor/live/push \
    -H 'Content-Type: application/json' \
    -d '{"sample":{"R1":1.0,"R2":2.0},"pose":{"x":1.5,"y":2.3,"theta":0.78}}' | python3 -m json.tool
# expect: {"ok": true, "id": …}

curl -s 'http://localhost:18080/sensor/live/recent?limit=1' | python3 -m json.tool
# expect: items[0].pose == {"x":1.5,"y":2.3,"theta":0.78,"frame_id":"map"}
```

### 2.6 `/smell/settling` distinguishes stable from drifting traces

```bash
curl -s -X DELETE http://localhost:18080/sensor/live/clear >/dev/null

# Stable trace
python3 -c "
import requests, time, random
for i in range(120):
    s = {f'R{k}': 100.0 + random.uniform(-0.05,0.05) for k in range(1,18)}
    s.update(T=22, H=50, CO2=400, H2S=0, CH2O=5)
    requests.post('http://localhost:18080/sensor/live/push', json={'sample': s})
    time.sleep(0.05)
print(requests.get('http://localhost:18080/smell/settling?window=8&threshold=0.02').json()['settled'])
"
# expect: True

# Ramped R1
curl -s -X DELETE http://localhost:18080/sensor/live/clear >/dev/null
python3 -c "
import requests, time
for i in range(120):
    s = {f'R{k}': 100.0 + (50*i/120 if k==1 else 0) for k in range(1,18)}
    s.update(T=22, H=50, CO2=400, H2S=0, CH2O=5)
    requests.post('http://localhost:18080/sensor/live/push', json={'sample': s})
    time.sleep(0.05)
print(requests.get('http://localhost:18080/smell/settling?window=8&threshold=0.02').json())
"
# expect: settled=False, worst_sensor='R1', worst_rel_slope ≈ 0.06
```

---

## Stage 3 — End-to-end sim mission (no robot)

This is the most important software-only test. It exercises everything: vision (mocked), classifier (real), pose-tagged live buffer (real), settling endpoint (real), policy state machine (real), commit + provenance (real), audit tool (real).

### 3.1 Run

```bash
# Clear last mission's provenance so the audit list is clean
rm -rf data/provenance

# With the NO_VLM server from Stage 2.1 still running:
docker run --rm --network enose-net \
    -v "$(pwd)":/app -e PYTHONPATH=/app \
    enose-server python scripts/run_robot_trainer.py \
    --backend sim --pump sim \
    --server http://enose-srv:8080 \
    --targets /app/scripts/sim_targets.json \
    --speed 10 --record-seconds 10 --purge-seconds 2 \
    --settling-timeout 8 --rate-hz 20
```

### 3.2 What you should see

A clean transition log, e.g.:

```
[mission] backend=sim  targets=3  server=http://enose-srv:8080
  [mission]      idle → approach   detected rose score=0.350
  [mission]  approach → settle     arrived at target_pose for rose
  [mission]    settle → record     settled (k=3)
  [mission]    record → commit     n=20
  [mission]    commit → purge      committed rose         ← (or "online_learning failed" — see note)
  [mission]     purge → search     purge complete
  [mission]    search → approach   detected coffee score=0.220
  ...
  [mission]    search → done       all targets processed
[mission] complete:
  final_state = done
  commits     = ≥ 1
  skipped     = ['lemon', …]
```

* `lemon` is *expected* to be skipped — its world_pose is 8 m away and beyond the `detection_radius`. That tests the `no detection above tau_grounding` branch.
* You may see "online_learning failed" on one of the two new-class commits — that's a pre-existing classifier-side quirk (`BalancedRFClassifier`'s calibration CV chokes on tiny batches). The policy correctly catches it, marks the target as skipped, and moves on. Fix lives in [`NEXT_STEPS.md`](NEXT_STEPS.md) §2.

### 3.3 Check the provenance side-effects

```bash
ls data/provenance/        # one .json + one .png per commit attempt
docker run --rm --network enose-net -v "$(pwd)":/app -e PYTHONPATH=/app \
    enose-server python scripts/audit_auto_labels.py \
    --server http://enose-srv:8080 list
```

Expected table:

```
provenance_id                  label          score    n  pose
--------------------------------------------------------------------
20260522-…-…                   coffee         0.220   20  x=2.00 y=2.00 θ=0.00
20260522-…-…                   rose           0.350   20  x=2.00 y=0.00 θ=0.00
```

### 3.4 Inspect one entry + low-score filter

```bash
PID=$(ls data/provenance/*.json | head -1 | xargs -I {} basename {} .json)

docker run --rm --network enose-net -v "$(pwd)":/app -e PYTHONPATH=/app \
    enose-server python scripts/audit_auto_labels.py \
    --server http://enose-srv:8080 inspect "$PID"

docker run --rm --network enose-net -v "$(pwd)":/app -e PYTHONPATH=/app \
    enose-server python scripts/audit_auto_labels.py \
    --server http://enose-srv:8080 low-score --threshold 0.30
# expect: the rose entry NOT shown, the coffee entry shown (its score 0.22 < 0.30).
```

### 3.5 Dry-run with `ManualPump` (preview what the human will see on the real robot)

```bash
docker run --rm --network enose-net \
    -v "$(pwd)":/app -e PYTHONPATH=/app \
    enose-server python scripts/run_robot_trainer.py \
    --backend sim --pump manual \
    --server http://enose-srv:8080 \
    --targets /app/scripts/sim_targets.json \
    --speed 15 --record-seconds 6 --purge-seconds 2 --settling-timeout 6
```

You should see prompts interleaved with the state-machine log:

```
  [mission]  approach → settle     ...
[pump] please switch pump ON
  [mission]    settle → record     settled (k=3)
  ...
[pump] please switch pump OFF
  [mission]     purge → search     purge complete
```

`--pump manual-confirm` additionally blocks until you press Enter at each prompt — useful when you actually have the physical pump in your hand.

### 3.6 Stop the server

```bash
docker stop enose-srv && docker rm enose-srv
```

---

## Stage 4 — Server with Florence-2 (real VLM)

Now bring the GPU online and verify Florence-2 actually grounds objects.

### 4.1 Start the server with VLM

```bash
docker run --rm -d --name enose-srv --gpus '"device=0"' --network enose-net \
    -p 18080:8080 -v "$(pwd)":/app -e PYTHONPATH=/app \
    enose-server python scripts/run_server.py
```

Florence-2-Large takes ~60–120 s to load on an A100. Watch for it:

```bash
until curl -fsS http://localhost:18080/health 2>/dev/null | grep -q '"vlm":true'; do
    docker ps -q -f name=enose-srv | grep -q . || { docker logs enose-srv | tail; break; }
    sleep 3
done
echo VLM READY
```

If it never reaches `vlm:true`, check `docker logs enose-srv` for `Florence-2 load failed:` — usually a transformers version mismatch (the image pins 4.44.2 for a reason) or a corrupted local snapshot in `model/Florence-2-Large/`.

### 4.2 `/predict/object` on a real photo

Put any photo of a clearly identifiable object at `./test_photo.jpg`, then:

```bash
curl -s -X POST http://localhost:18080/predict/object \
    -F image=@test_photo.jpg -F 'text=Find cup' | python3 -m json.tool
```

Expected: `{"prompt":"Find cup","task":"<OPEN_VOCABULARY_DETECTION>","result":{...with bboxes if found...}}`.

### 4.3 `/predict/scene` with multiple labels

```bash
curl -s -X POST http://localhost:18080/predict/scene \
    -F image=@test_photo.jpg \
    -F 'labels=["cup","table","plant","laptop"]' | python3 -m json.tool
```

Expected: `{"labels":[…],"detections":[{"label":"cup","bbox":[…],"score":0.XX},…],"image_size":[W,H],…}`. Empty `detections` on a featureless image is fine — that's the contract.

### 4.4 Threshold calibration *(half a day; do this before any real mission — see `NEXT_STEPS.md` §1)*

Collect ~20–30 photos of your target objects (rose, coffee cup, etc.) in their lab setting; record `(true_label, predicted_score)` for each. Pick `τ_grounding` around the 90th percentile of true-positive scores' lower edge and `τ_commit` at the 95th percentile. Update `enose.robot.policy.PolicyConfig` defaults, or pass `--tau-grounding 0.X --tau-commit 0.Y` on the CLI.

### 4.5 Stop the VLM server

```bash
docker stop enose-srv && docker rm enose-srv
```

---

## Stage 5 — Client image sanity (still no robot)

Verify the deployable client image can import the robot package and exercise `visual_servo`.

```bash
docker run --rm -v "$(pwd)":/app -e PYTHONPATH=/app enose-client python -c "
import enose.robot
from enose.robot import Pose, ManualPump
from enose.robot.policy import Policy, PolicyConfig
from enose.robot.visual_servo import step, ServoConfig
from enose.robot.sim import SimMotion, SimLocalizer, ScriptedVision
# end-to-end import
cmd = step((100,100,300,300), (640,480), 2.0, ServoConfig())
print('visual_servo:', cmd)
mp = ManualPump('log'); mp.on(); mp.off()
print('OK — client image is robot-ready')
"
```

Expected: the `visual_servo` line plus two `[pump]` prompts, then `OK …`.

---

## Stage 6 — Robot dog: connection and code deployment

> Everything from here forward requires the actual Go-1 EDU. None of these steps depend on ROS2 adapters being finished — they only need (a) network connectivity, (b) the e-nose serial ports, (c) the camera, and (d) the Python client running on the Jetson.

### 6.1 Network setup

The dog's Jetson needs to reach the lab-GPU server over Wi-Fi.

1. Power on the Go-1; SSH into the Jetson (default credentials per Unitree docs — adjust IP):
   ```bash
   ssh unitree@192.168.123.13
   ```
2. From the Jetson, ping the lab-GPU box and curl the server health endpoint:
   ```bash
   curl -fsS http://<lab-gpu-ip>:18080/health
   ```
   If this fails, debug Wi-Fi / firewall before going further. The whole stack relies on this.

### 6.2 Code deployment

You have three options; pick whichever fits your workflow.

**Option A — rsync from your laptop:**

```bash
rsync -av --delete --exclude '.git' --exclude '__pycache__' --exclude 'data' --exclude 'plots' \
    /home/vlm-workspace/e-nose/ unitree@192.168.123.13:~/e-nose/
```

**Option B — git pull on the Jetson** (recommended for traceability):

```bash
ssh unitree@192.168.123.13
git clone <your-repo-url> ~/e-nose
# or, if already cloned:
cd ~/e-nose && git pull
```

**Option C — Docker image deploy** (if the Jetson has Docker installed and you want hermetic deps):

```bash
# On the lab box, save the client image:
docker save enose-client | gzip > /tmp/enose-client.tar.gz
scp /tmp/enose-client.tar.gz unitree@192.168.123.13:~/
# On the Jetson:
gunzip -c ~/enose-client.tar.gz | docker load
```

### 6.3 Python deps on the Jetson

If you went with Option A or B (not Docker), install client deps:

```bash
ssh unitree@192.168.123.13
cd ~/e-nose
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# Equivalent to the Dockerfile-client's pip line:
pip install numpy pandas requests pyserial opencv-python-headless matplotlib pillow
```

### 6.4 Serial ports for the e-nose

The e-nose ships data on two UART devices. Identify them:

```bash
ls /dev/ttyUSB*    # or /dev/ttyACM*
# Usually /dev/ttyUSB0 (e-nose) and /dev/ttyUSB1 (env sensors), but verify.
```

Grant access:

```bash
sudo usermod -aG dialout $USER  # then log out / back in
```

Quick sanity read:

```bash
python3 - <<'EOF'
from enose.client.sensors import ENoseSensor
s = ENoseSensor(port=('/dev/ttyUSB0', '/dev/ttyUSB1'))
print(s.read_single_measurement())   # expect a 22-key dict
EOF
```

If you get junk or zeros, double-check baud rates and that the device isn't held by another process.

### 6.5 Webcam check

```bash
python3 - <<'EOF'
from enose.client.webcam import WebcamHandler
w = WebcamHandler()
ok, path = w.capture_image()
print('captured:', ok, '→', path)
EOF
```

Expected: a PNG at the printed path.

---

## Stage 7 — Robot integration tests (without ROS2 adapters yet)

The full autonomous mission needs `enose.robot.ros2.{UnitreeAdapter,TFLocalizer}` to be implemented — those are skeletons today, so `run_robot_trainer.py --backend ros2` will exit with the `ros2 backend not ready` message. That's expected and unblocks no testing you can do right now. What you *can* and *should* test on the dog before then:

### 7.1 Manual teleop + live publisher (sensor pipeline end-to-end)

This is the single most important real-robot test you can run today. It verifies:

* e-nose UART reads on the dog
* HTTP push to the lab-GPU server
* Live buffer fills, drift / settling endpoints have data to look at

```bash
# On the Jetson:
cd ~/e-nose
python3 scripts/run_client.py \
    --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1 \
    --server http://<lab-gpu-ip>:18080 \
    --live --live-classify \
    --record-session runs/teleop_$(date +%Y%m%d_%H%M%S).npz
```

Teleop the dog with the standard Unitree controller while this runs. On the lab GPU side:

* Open `http://<lab-gpu-ip>:18080/ui` → **Live** tab in a browser.
* You should see the rolling R1–R17 trace, drift panel populated.
* `curl http://<lab-gpu-ip>:18080/smell/settling?window=10` returns a real per-sensor report (not "live buffer empty").

If the live tab is empty, debug push errors on the Jetson:
* `pub.last_error` is logged by `run_client.py`.
* Most common cause: the server IP isn't reachable from the dog's Wi-Fi (Stage 6.1).

### 7.2 Pose plumbing dry-run *(only after a Localizer exists)*

When `enose.robot.ros2.TFLocalizer` is implemented (see `NEXT_STEPS.md` §3.2), run the same teleop with the localizer injected:

```bash
# Sketch — exact integration arrives in NEXT_STEPS §3.4.
python3 - <<'EOF'
from enose.client.sensors import ENoseSensor
from enose.client.live import LivePublisher
from enose.robot.ros2.localization import TFLocalizer

sensor = ENoseSensor(port=('/dev/ttyUSB0', '/dev/ttyUSB1'))
loc = TFLocalizer()
pub = LivePublisher(
    sensor=sensor,
    server_url='http://<lab-gpu-ip>:18080',
    rate_hz=5.0,
    pose_lookup=loc.pose_dict,
)
pub.start()
input('press Enter to stop')
pub.stop()
EOF
```

Then on the server side:

```bash
curl http://<lab-gpu-ip>:18080/sensor/live/recent?limit=5
```

Every entry should have a `pose` field with non-zero `(x,y,θ)` once the dog moves.

### 7.3 Vision capture from the dog's camera

Make sure your `WebcamHandler` (or its replacement that subscribes to the depth camera's RGB topic) actually returns a path the server can grind through Florence-2.

```bash
python3 - <<'EOF'
from enose.client.api import ServerAPI
from enose.client.webcam import WebcamHandler
api = ServerAPI('http://<lab-gpu-ip>:18080')
w = WebcamHandler()
ok, path = w.capture_image()
print('captured:', path)
result, err = api.detect_scene(path, ['table', 'cup', 'plant'])
print('err:', err)
print('result:', result)
EOF
```

Expected: `result['detections']` is a list (possibly empty) and `err` is `None`.

### 7.4 Provenance from a hand-fed mission *(no autonomous nav required)*

You can exercise the full *commit* pipeline today, with the dog stationary, by manually walking the e-nose past a target and submitting the batch directly. This lets you test the audit trail end-to-end before the autonomous path lands.

```bash
python3 - <<'EOF'
import base64, requests, time
from enose.client.sensors import ENoseSensor
from enose.client.api import ServerAPI
from enose.client.webcam import WebcamHandler

api = ServerAPI('http://<lab-gpu-ip>:18080')
w = WebcamHandler()
sensor = ENoseSensor(port=('/dev/ttyUSB0', '/dev/ttyUSB1'))

# 1) Capture the source image.
_, img_path = w.capture_image()

# 2) Detect the target (single-label OVD).
det, err = api.detect_object(img_path, 'rose')
print('detect:', det, err)

# 3) Position e-nose by hand near the rose, wait ~30s for MOS settle (PUMP ON).
print('[pump] switch pump ON; let sensors settle for 30s...')
time.sleep(30)

# 4) Record ~30s of samples.
samples = []
for _ in range(150):
    s = sensor.read_single_measurement()
    if s: samples.append(s)
    time.sleep(0.2)

# 5) Commit with provenance.
with open(img_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')
provenance = {
    'grounding_score': 0.5,   # placeholder; real policy would pull this from /predict/scene
    'bbox': None,
    'pose': None,
    'image_b64': img_b64,
    'image_filename': img_path.rsplit('/', 1)[-1],
    'extras': {'note': 'hand-fed test commit'},
}
res, err = api.online_learning(samples, 'rose', provenance=provenance)
print('commit:', res, err)
print('[pump] switch pump OFF; walk to clean zone for purge.')
EOF
```

Then on the server:

```bash
python3 scripts/audit_auto_labels.py --server http://<lab-gpu-ip>:18080 list
ls -la data/provenance/
```

You should see your hand-fed entry plus the source image.

### 7.5 Things you CANNOT test yet (and why)

* **`run_robot_trainer.py --backend ros2`** — exits with `ros2 backend not ready` until the adapters in `enose.robot.ros2.{unitree_adapter,localization}` are implemented (`NEXT_STEPS.md` §3.2 / §3.3).
* **Autonomous SEARCH→APPROACH→COMMIT loop on the real dog** — same blocker. The whole *policy* is tested (Stage 3.x), what's missing is the bridge between the policy and the robot.
* **Visual-servo control loop on real depth data** — implementation exists (Stage 5 verified pure-function output) but it isn't wired into the policy's `_approach` until `NEXT_STEPS.md` §3.5.
* **Idea-2 plume tracing** — not yet implemented.

---

## Stage 8 — Triage cheatsheet

| Symptom | Probable cause | Where to look |
|---|---|---|
| `enose-server` not found | Image was pruned | Rebuild: `docker build -t enose-server -f docker/Dockerfile-cu124 .` |
| `models.vlm: false` after starting with GPU | Florence-2 load failed | `docker logs enose-srv \| grep Florence-2` — usually a corrupted `model/Florence-2-Large/` |
| `/predict/scene` returns empty `detections` | Florence-2 doesn't see the labels in this image | Try `/predict/object` with single label; tune `τ_grounding` ([`NEXT_STEPS.md`](NEXT_STEPS.md) §1) |
| `[mission] online_learning failed` for one of two new-class commits | `BalancedRFClassifier` calibration CV quirk | Known issue, [`NEXT_STEPS.md`](NEXT_STEPS.md) §2 |
| `[mission] settle timeout — recording anyway` every time | Sensor never stabilizes; thresholds too tight, or pump isn't on | Increase `--settling-threshold` to 0.05, or flip the physical pump on |
| Live tab empty in the browser UI | Push failures from the client | Check `pub.last_error`, Wi-Fi, server URL |
| `ros2 backend not ready` from `run_robot_trainer.py` | ROS2 adapters not implemented yet | Expected. See [`NEXT_STEPS.md`](NEXT_STEPS.md) §3 |
| Hand-fed commit's image isn't in `data/provenance/` | `image_b64` is empty or path was unreadable | Confirm `WebcamHandler.capture_image()` returned a valid path before encoding |

---

## Stage 9 — Where this guide ends

You've now exercised:

* Both Docker images build and import the package cleanly.
* The new server endpoints (`/predict/object`, `/predict/scene`, `/smell/settling`, `/smell/provenance`) work and validate input correctly.
* The pose-tagged live buffer round-trips, with `/smell/drift` and `/smell/settling` reading real data.
* The full Idea-1 policy state machine runs to completion in sim with provenance + audit.
* On the real dog: sensor I/O, Wi-Fi to the lab-GPU server, camera capture, hand-fed commits.

Next concrete unblock: implement the ROS2 adapters per [`NEXT_STEPS.md`](NEXT_STEPS.md) §3. Once those land, the same `run_robot_trainer.py --backend ros2` command exercises the whole stack end-to-end on the physical robot.
