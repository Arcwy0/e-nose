# E-Nose — Client Quickstart

Everything a colleague needs to use the lab server.

> **Server URL:** the lab admin will share the base URL with you privately
> (something like `http://<host>:<port>`). In the examples below, replace
> `<SERVER_URL>` with whatever they give you — e.g.
> `http://192.168.1.42:8080` on the lab LAN or a private tunnel URL.

Pick the section that matches your setup. Sections are independent — you don't need to read the ones above.

- [A. Just the browser (no install)](#a--just-the-browser-no-install)
- [B. Python client — native install](#b--python-client--native-install)
- [C. Python client — Docker (no pip, no venv)](#c--python-client--docker-no-pip-no-venv)
- [D. Scenario cheatsheet](#d--scenario-cheatsheet)
  - [D1. Smell-only, no hardware](#d1-smell-only-no-hardware)
  - [D2. Smell-only with e-nose hardware](#d2-smell-only-with-e-nose-hardware)
  - [D3. Vision-only (camera, no e-nose)](#d3-vision-only-camera-no-e-nose)
  - [D4. Full training cycle (camera + e-nose)](#d4-full-training-cycle-camera--e-nose)
  - [D5. Live online graphs (streaming)](#d5-live-online-graphs-streaming)
  - [D6. Replay a saved session](#d6-replay-a-saved-session)
  - [D7. Train the server from a CSV](#d7-train-the-server-from-a-csv)
- [E. Sanity check](#e--sanity-check)
- [F. Troubleshooting](#f--troubleshooting)

---

## A — Just the browser (no install)

Open in any browser:

```
<SERVER_URL>/ui
```

| Tab | What to do there |
|---|---|
| **Classify** | Paste 22 comma-separated sensor values → predicted smell + confidence + OOD traffic light |
| **Train** | Upload / paste a CSV (22 sensor columns + a label column) → retrain the model |
| **Live** | Click **Start polling** — shows live sensor charts + drift panel while someone is `--live` streaming |
| **Model Info** | Known classes, accuracy, per-class P/R/F1, confusion matrix |

No terminal, no Python. Good enough for most work.

---

## B — Python client — native install

Use this if you want to plug in hardware (e-nose, camera), stream live graphs, or script anything.

### Prerequisites
- Python 3.9+
- `git`
- On Linux: serial-port access (`sudo usermod -aG dialout $USER`, then log out + back in)

### Install once

```bash
git clone https://github.com/YOUR_USERNAME/e-nose.git
cd e-nose
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[classifier-extras]"
```

That's the full client. No CUDA, no Florence-2 weights — the server handles all the heavy lifting.

Now jump to the [scenario cheatsheet](#d--scenario-cheatsheet).

---

## C — Python client — Docker (no pip, no venv)

Use this if you don't want to touch Python on your machine, or you're on a locked-down OS.

### Build once

```bash
git clone https://github.com/YOUR_USERNAME/e-nose.git
cd e-nose
docker build -t enose-client -f docker/Dockerfile-client .
```

### Run

**Offline (no hardware, no camera):**
```bash
docker run --rm -it --network host \
    -v "$(pwd)":/app \
    enose-client \
    python scripts/run_client.py --offline --server <SERVER_URL>
```

**With e-nose hardware (Linux):**
```bash
docker run --rm -it --network host \
    -v "$(pwd)":/app \
    --device=/dev/ttyUSB0 --device=/dev/ttyUSB1 \
    enose-client \
    python scripts/run_client.py \
        --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1 \
        --server <SERVER_URL>
```

**With camera (Linux):**
```bash
docker run --rm -it --network host \
    -v "$(pwd)":/app \
    --device=/dev/video0 \
    enose-client \
    python scripts/run_client.py --server <SERVER_URL>
```

Notes:
- `--network host` is needed so the container can reach the server address.
- Live streaming opens a matplotlib window. Inside Docker that needs X-forwarding (`-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`) or you run live natively instead. For most people, run live natively via section B.
- Windows/macOS have no `/dev/ttyUSB*` passthrough — use native install (B) for hardware.

---

## D — Scenario cheatsheet

All commands assume you did either section B (native) or section C (Docker). Native commands are shown; for Docker, prepend the `docker run … enose-client` wrapper from section C.

The interactive menu looks like this once you're running:

```
1. Full training cycle (vision + smell)
2. Vision only (object detection)
3. Smell identification
4. Model info
5. Learn from CSV
6. Manual 22-feature input testing
7. Generate visualizations + analysis
8. Live sensor stream (plot + push to server)
9. Replay a recorded session
0. Exit
```

### D1. Smell-only, no hardware

Simulated sensors. Smoke-tests the server connection and menu without needing any kit.

```bash
python scripts/run_client.py --offline --server <SERVER_URL>
```

Useful menu options: **3** (classify simulated sample), **4** (model info), **5** (upload CSV), **6** (type 22 values by hand), **7** (regenerate all server-side plots).

### D2. Smell-only with e-nose hardware

**Find your serial ports:**

```bash
# Linux
ls /dev/ttyUSB*                      # usually /dev/ttyUSB0 + /dev/ttyUSB1

# Windows: Device Manager → Ports (COM & LPT) → usually COM3 + COM4
# macOS:   ls /dev/tty.usbserial-*
```

**Run:**

```bash
# Linux
python scripts/run_client.py \
    --port_enose /dev/ttyUSB0 \
    --port_UART  /dev/ttyUSB1 \
    --server     <SERVER_URL>

# Windows
python scripts/run_client.py \
    --port_enose COM3 --port_UART COM4 \
    --server     <SERVER_URL>
```

Pick menu **3** for a one-shot classification of what the e-nose currently smells.

### D3. Vision-only (camera, no e-nose)

Uses Florence-2 on the server to describe what the webcam sees. No sensors needed — the e-nose connection is skipped if you don't pass `--port_enose`.

```bash
python scripts/run_client.py \
    --offline \
    --camera 0 \
    --server <SERVER_URL>
```

`--offline` disables e-nose; `--camera 0` is the default webcam (try `1` / `2` if you have multiple). Pick menu **2** for object detection.

> **Note:** this only works if the server was started with Florence-2 loaded (full mode). If the server is in smell-only mode (`ENOSE_NO_VLM=1`), option 2 returns 503.

### D4. Full training cycle (camera + e-nose)

Menu option 1. Opens the camera, waits for you to trigger recording, captures 22 sensor features over `--time` seconds while snapping a photo, then sends both to the server.

```bash
python scripts/run_client.py \
    --port_enose /dev/ttyUSB0 \
    --port_UART  /dev/ttyUSB1 \
    --camera 0 \
    --time 10 \
    --samples 100 \
    --server <SERVER_URL>
```

Then pick menu **1** and follow the prompts (you'll label the sample, e.g. `coffee`). The server stores the (image, 22 features, label) triplet and can retrain from the accumulated dataset.

Flags:
- `--time` — recording duration in seconds (default 10)
- `--samples` — target number of sensor reads in that window (default 100)
- `--camera` — camera index (default 0; again needs server with VLM loaded)

### D5. Live online graphs (streaming)

Pushes samples to the server at 5 Hz so anyone watching the browser's **Live** tab sees real-time charts. Optionally classifies each sample and records everything to a `.npz` for later replay.

```bash
python scripts/run_client.py \
    --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1 \
    --server <SERVER_URL> \
    --live --live-classify --live-label coffee \
    --record-session runs/coffee.npz
```

Flags:
- `--live` — go straight to the live dashboard, skip the menu
- `--live-rate 5` — Hz (default 5)
- `--live-window 60` — plot rolling-window seconds (default 60)
- `--live-baseline` — subtract the first few seconds as an air baseline
- `--live-classify` — also call `/smell/classify` per sample so the plot shows confidence
- `--live-label coffee` — tag every sample with this label (recording known odours)
- `--record-session runs/coffee.npz` — also dump everything to disk

A matplotlib window opens on your machine. `Ctrl-C` to stop — the `.npz` flushes on exit.

Menu option **8** launches the same dashboard interactively.

### D6. Replay a saved session

```bash
python scripts/replay.py runs/coffee.npz \
    --server <SERVER_URL> \
    --verbose
```

Reports label distribution, confidence stats, and how today's server predictions compare to the ones recorded at capture time (useful for spotting drift). Menu option **9** does the same thing from inside the client.

### D7. Train the server from a CSV

Two equivalent ways:

**Browser (easiest):** the **Train** tab at `<SERVER_URL>/ui` — drag a CSV in, click Train.

**Python client:** menu **5** ("Learn from CSV"). It prompts for a path and uploads the file.

CSV format: 22 sensor columns + 1 label column. Column names must match the server's expected features (see `data/enose_points_1906.csv` in the repo for a reference file). Menu **4** on the client or the **Model Info** tab shows which classes / features the server currently knows.

---

## E — Sanity check

```bash
curl <SERVER_URL>/health
# {"status":"healthy", ...}
```

If `/health` works but the browser UI looks wrong, it's a browser cache issue — hard-reload with `Ctrl+Shift+R`.

---

## F — Troubleshooting

| Symptom | Fix |
|---|---|
| `curl .../health` hangs or fails | Server is down or the URL is wrong — ask the admin. |
| Classify returns 503 "classifier not fitted" | Nobody has trained the model yet. Go to the **Train** tab and upload a CSV. |
| Menu option 1 or 2 returns 503 | Server is in smell-only mode (no Florence-2). Use menu **3** for smell-only work or ask the admin to restart the server without `ENOSE_NO_VLM=1`. |
| "permission denied" opening `/dev/ttyUSB0` (Linux) | `sudo usermod -aG dialout $USER`, log out + back in. |
| Camera index 0 "not found" | Try `--camera 1` / `--camera 2`. Laptops with IR cameras sometimes enumerate differently. |
| Browser **Live** tab is empty | No client is currently `--live` streaming. Start one with [D5](#d5-live-online-graphs-streaming). |
| Live dashboard opens but no plot window (Docker) | Matplotlib inside Docker needs X-forwarding — run live natively (section B) instead. |
| `scripts/run_client.py: No such file` | You didn't `cd` into the cloned repo root. |
| Windows: `.venv\Scripts\Activate.ps1` blocked | Run PowerShell as admin once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |

---

For deeper docs — OOD, drift, per-class metrics, custom CSV formats — see the main [`README.md`](../README.md) and [`CHEMIST_GUIDE.md`](CHEMIST_GUIDE.md).
