# Setup Guide — From Clone to First Classification

This guide covers release **3.3.0** and the current `main` branch. Install the
server and hardware client in **separate environments**. The server lock uses
headless OpenCV; the client uses desktop OpenCV.

For wheel-only Windows client installation see
[CLIENT_WINDOWS_INSTALL.md](CLIENT_WINDOWS_INSTALL.md). For an existing
installation see [MIGRATION.md](MIGRATION.md). Hardware controls and the complete
client option reference are in the [client README](../enose/client/README.md).

## 1. Prerequisites and clone

Use 64-bit CPython 3.10 for the closest match to Docker. The server lock supports
Python 3.10–3.12 on Windows/Linux x86_64; the separate client can use Python 3.11.
Windows examples use **uv and PowerShell**; Linux examples use **Bash**.
Install Git and, on Windows, uv before continuing.

Run in either shell:

```text
git clone https://github.com/Arcwy0/e-nose.git
cd e-nose
git switch main
```

A CUDA GPU is optional. CPU Florence inference needs sufficient RAM and can take
considerably longer than GPU inference. USB drivers are needed only for physical
sensors; an environmental UART is optional. Native Windows execution and physical
hardware still require validation on the colleague's machine.

## 2. Server environment: the same runtime lock as Docker

`requirements-server.txt` includes the complete
[`requirements/server-lock.txt`](../requirements/server-lock.txt).
pip, uv and [the Dockerfile](../docker/Dockerfile-cu124) consume that same lock,
including direct and transitive versions: Torch **2.4.1+cu124**, torchvision
**0.19.1+cu124**, Transformers **4.44.2**, tokenizers **0.19.1**, NumPy
**1.26.4**, and scikit-learn **1.5.2**. Do not individually upgrade these packages.

### Windows PowerShell — uv

```powershell
uv venv --python 3.10 .venv-server
.\.venv-server\Scripts\Activate.ps1
uv pip sync requirements-server.txt
uv pip install --no-deps -e .
uv pip check
uv run --active --no-project python scripts/verify_server_environment.py
uv run --active --no-project python scripts/smoke_test.py
```

Use `uv pip sync` with the shared lock, not unconstrained `uv sync` or a broad
server-extra install. `--no-deps` preserves the locked runtime. The repository's
uv index policy matches pip's CUDA-wheel selection.

### Linux Bash — standard venv/pip

On the remote server, prefer the Docker workflow below. For a native Linux
installation where needed:

```bash
python3.10 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade "pip==24.3.1"
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/verify_server_environment.py
python scripts/smoke_test.py
```

### FlashAttention and platform differences

Docker retains **FlashAttention 2.7.3**, installed after the locked Torch runtime.
To match that extension in a native **Linux CUDA** environment, first install
the CUDA 12.4 toolkit and C++ build toolchain, then:

```bash
export MAX_JOBS=4
python -m pip install -r requirements-flash-build.txt
python -m pip install --no-build-isolation -r requirements-flash-attn.txt
python -m pip check
```

CPU and native Windows use eager attention without FlashAttention. The loader
handles Florence's false mandatory-extension check. Shared package versions are
the same; OS-specific CUDA libraries, Linux-only Triton/uvloop, and the optional
CUDA extension cannot be identical across platforms. The CUDA Torch wheel also
runs on CPU without a CUDA GPU.

If Windows reports a missing Torch/OpenCV DLL, install the
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
Linux headless OpenCV may require the OS package `libglib2.0-0`.

## 3. Florence-2-Large model files

Skip this section for smell-only use. Weights are private/local runtime artifacts,
not part of Git or the wheel. Download the tested official revision, not an
unversioned mix of model/tokenizer files.

Windows PowerShell, with the server environment active:

```powershell
uv run --active --no-project huggingface-cli download microsoft/Florence-2-large --revision 21a599d414c4d928c9032694c424fb94458e3594 --include "*.json" "*.py" "model.safetensors" "LICENSE" --local-dir model/Florence-2-Large
uv run --active --no-project python -c "from enose.utils.tokenizer_io import ensure_slow_tokenizer_files; print(ensure_slow_tokenizer_files('model/Florence-2-Large'))"
uv run --active --no-project python scripts/verify_server_environment.py --vlm --device cpu
```

Linux Bash, with the server environment active:

```bash
huggingface-cli download microsoft/Florence-2-large --revision 21a599d414c4d928c9032694c424fb94458e3594 --include "*.json" "*.py" "model.safetensors" "LICENSE" --local-dir model/Florence-2-Large
python -c "from enose.utils.tokenizer_io import ensure_slow_tokenizer_files; print(ensure_slow_tokenizer_files('model/Florence-2-Large'))"
python scripts/verify_server_environment.py --vlm
```

The helper reconstructs missing `merges.txt`/`vocab.json` from the same
`tokenizer.json`; it does not overwrite existing files or weights. Prepare the
folder before mounting it read-only. Keep the snapshot's Python model/processor
code. See [FLORENCE_MODEL_FILES.md](FLORENCE_MODEL_FILES.md) for file layouts,
repair details, and localization checks.

`--device cpu` is a **verification-script** option, not a server option. The
server selects CPU automatically when CUDA is unavailable. Caption verification
does not alone prove object-localization parsing: also test `bottle` or `vial`
on a visible container. Vision cannot identify acetone chemically from an
unlabelled bottle.

## 4. Server configuration and startup

The default paths are relative to the working directory: Florence in
`model/Florence-2-Large/`, active classifier in
`trained_models/smell_classifier_sgd_latest.joblib`, and learning history in
root-level `database_robodog.csv`.

`.env` is **not automatically loaded** by the native server. Use explicit
process environment variables below, or pass a private `--env-file` to Docker.
Do not commit credentials.

| Setting | Purpose |
|---|---|
| `ENOSE_NO_VLM=1` | Smell-only startup, no model weights needed |
| `ENOSE_NO_VLM=0` | Load Florence-2 |
| `ENOSE_CLASSIFIER=balanced_rf` | Default classifier; `xgb` is also available |
| `ENOSE_BASELINE_MODE`, `ENOSE_SNV` | Defaults for new classifier config; saved model and explicit CSV settings take precedence |
| `SERVER_HOST`, `SERVER_PORT` | Defaults overridden by `--host` / `--port` |
| `--workers 1` | Required: live state and deployment locks are process-local |
| `--reload` | Development reloader; avoid for normal acquisition |

Windows PowerShell, server environment:

```powershell
$env:ENOSE_NO_VLM = "1"
uv run --active --no-project enose-server --host 0.0.0.0 --port 8016
```

For full mode, set `$env:ENOSE_NO_VLM = "0"` before the same command.

Linux Bash, server environment:

```bash
ENOSE_NO_VLM=1 enose-server --host 0.0.0.0 --port 8016
# Full mode:
# ENOSE_NO_VLM=0 enose-server --host 0.0.0.0 --port 8016
```

Open [the UI](http://localhost:8016/ui), [health](http://localhost:8016/health),
and [API docs](http://localhost:8016/docs). `0.0.0.0` is a bind address;
colleagues connect using the host's actual IP. Use a trusted network or an
authenticated reverse proxy: training/promotion routes are not access-controlled.
CPU startup/inference latency depends on hardware; a successful load is not a
latency guarantee.

A wheel-only installation provides `enose-server`. Repository helpers such as
`scripts/verify_server_environment.py` require a checkout. To install a wheel
on the server, first install the **matching release's complete server lock**,
then the wheel with `--no-deps`; the wheel does not replace that lock.

## 5. Separate client environment and safe first run

Do not install `[client]` into the locked server environment.

Windows PowerShell, new terminal in the repository:

```powershell
uv venv --python 3.11 .venv-client
.\.venv-client\Scripts\Activate.ps1
uv pip install -e ".[client]"
uv run --active --no-project enose-client --offline --server http://localhost:8016 --preflight
uv run --active --no-project enose-client --offline --server http://localhost:8016
```

Linux Bash, separate client terminal:

```bash
python3.11 -m venv .venv-client
source .venv-client/bin/activate
python -m pip install -e ".[client]"
enose-client --offline --server http://localhost:8016 --preflight
enose-client --offline --server http://localhost:8016
```

The [client README](../enose/client/README.md) lists **every CLI option**, menu
entry, defaults, and environment-variable overrides. The installed command is
`enose-client`; a checkout can alternatively run `scripts/run_client.py`.

Current acquisition choices:

- **PCB (default):** binary USB CDC, resistance output in ohms by default.
- **Legacy:** explicitly select `--sensor-backend legacy`; retain its RLOW
  calibration with `--rlow` / `ENOSE_RLOW`. It has no INFO/STATUS protocol.
- **Replay:** `--sensor-backend replay --sensor-csv PATH` reads supplier
  resistance-mode CSV, not raw ADC or automatic smell labels.
- **Simulation:** `--offline`; no physical hardware.
- Environmental UART: omit `--port_UART` for R1–R17-only use.

Use [PCB_ACQUISITION.md](PCB_ACQUISITION.md) before hardware operation. Heating
defaults to `--heater-control external`: preserve supplier-configured PI
setpoint, observe, and request off on exit. Supplier GUI and our client cannot
own the same port concurrently. Explicit `client` heating defaults to a
300°C setpoint, 350°C host trip guard, and 1800-second session maximum.
These are **not certified hardware limits**. Disable ramps duty down; USB loss,
SIGKILL, host crash, or power loss cannot guarantee heater shutdown without a
firmware watchdog/interlock. Test offline before connecting physical hardware.

CPU VLM requests may require increased `--vision-timeout` / `--scene-timeout`.
For live display without a local window use `--headless --live`. Read-only
heater diagnostics appear in the browser; heater actuation remains local.

## 6. First training and classification

In the browser's Train tab, upload labelled R1–R17 data with `smell_label`
(and optional environmental columns). Declare the correct resistance unit.
The new board's supplier CSV contains resistance values but no chemical labels;
preview it first, then supply verified row annotations or labels.

Preview retained/rejected plateau windows, recording date/session and class
counts. Historical merging remains the default learning policy; select
dataset-only replacement explicitly when wanted.

The browser defaults to **staging a candidate**, leaving the active model and
history unchanged. Evaluate active and candidate models on the same independent
labelled holdout, then promote the candidate in Model Info. Initial deployment
without comparable evidence needs an explicit unevaluated override. Rollback
restores the paired model/history snapshot. See the
[server README](../enose/server/README.md) for endpoints and deployment limits.

If independent class-complete exposure groups do not exist, validation reports
generalization **not estimable**, not a misleading row-split score.
Capture a fresh clean-air baseline for live use after a new session/board,
critical acquisition-setting change, or server restart. Units and provenance
must match the model; placeholders are not environmental measurements.

## 7. Docker server — Linux Bash

Use the approved rootless daemon by default. The image contains dependencies
but expects source/weights to be mounted. Mount under `/workspace`:

```bash
docker build -t enose-server -f docker/Dockerfile-cu124 .
docker run --rm -it --name enose-server --gpus all -p 8016:8080 \
  --mount type=bind,source="$(pwd)",target=/workspace/e-nose \
  --workdir /workspace/e-nose -e PYTHONPATH=/workspace/e-nose -e ENOSE_NO_VLM=0 \
  enose-server python scripts/run_server.py --host 0.0.0.0 --port 8080
```

For a genuinely GPU-hidden smell-only test, omit `--gpus all`, use
`--runtime runc`, set `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`,
and `ENOSE_NO_VLM=1`. The NVIDIA runtime may expose GPUs by default otherwise.
This run mounts no USB/robot devices. Deliberate hardware passthrough belongs
to a separately approved client deployment, not server setup.

For Docker/network details see [SERVER_CLIENT_GUIDE.md](SERVER_CLIENT_GUIDE.md).
Keep durable dependency changes in the Dockerfile/lock and rebuild the image;
do not manually patch a running container.

## 8. Troubleshooting and large files

| Symptom | Check |
|---|---|
| `No module named enose` | Activate the intended environment and install the checkout/wheel into it. |
| No active classifier / HTTP 503 | Stage labelled data, then explicitly promote a candidate. |
| Missing `merges.txt` / BART `NoneType` path error | Run the matching-snapshot tokenizer repair; not a CPU/CUDA issue. |
| Florence dependency/config errors | Recreate the server environment from the complete lock, not individual upgrades. |
| CPU detection timeout | Increase client request deadlines; distinguish timeout from empty parsed regions. |
| OpenCV GUI unavailable | Keep desktop client and headless server environments separate; use `--headless` when appropriate. |
| Legacy preflight lacks INFO/STATUS | Expected: the old board does not implement those commands. |
| Baseline or unit mismatch | Correct declared units and capture clean air for the current board/session. |
| Port already in use | Choose another server `--port`; close the supplier GUI before opening its serial port. |
| `OFF UNVERIFIED` | Follow physical safety procedures; do not treat disconnected hardware as safely off. |

Keep source, locks, and docs in Git. Keep `.env`, raw weights, research CSVs,
recorded sessions, classifier artifacts, and candidate/snapshot directories in
private storage. The wheel does **not** contain these runtime assets.
Do not force-add private data/models or rewrite Git history as an installation
fix. Git LFS is only for deliberately approved tracked artifacts; it will not
recreate ignored files. Copy private state using [MIGRATION.md](MIGRATION.md).
