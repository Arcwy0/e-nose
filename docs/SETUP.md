# Setup Guide — From Clone to First Classification

This is the end-to-end setup for a fresh machine. If you only want to run
the **smell classifier** (no vision, no GPU), skip Step 3; everything else
works unchanged. If you want the **full multimodal system**, do all steps.

A companion guide for non-programmers using the browser UI lives in
[`CHEMIST_GUIDE.md`](CHEMIST_GUIDE.md). The server ↔ client deployment
reference is in [`SERVER_CLIENT_GUIDE.md`](SERVER_CLIENT_GUIDE.md).
If you are moving an existing installation—including local models and data—to
another machine, follow [`MIGRATION.md`](MIGRATION.md) as well.

---

## 0. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10 – 3.12 (64-bit)** | Server reference is Python 3.10, matching Docker. Python 3.13+ is not supported by the Torch 2.4.1 lock. The separate client retains its own requirements. |
| **Git** | Any recent version. |
| **Git LFS** *(optional)* | Only needed if you plan to commit model weights, `.npz` sessions, or other large artifacts — see `.gitattributes`. |
| **CUDA-capable GPU** *(optional)* | Recommended for Florence-2 latency. CPU inference is also supported. |
| **USB-serial drivers** *(client only)* | For the physical e-nose + UART env board. Not needed in `--offline` mode. |

Quick check that you have what you need:

```bash
python --version      # server: 3.10, 3.11, or 3.12
git --version
git lfs version       # optional
nvidia-smi            # optional — shows "No devices" on CPU-only hosts
```

---

## 1. Clone the repository

```bash
# Install Git LFS once per machine (skip if you never plan to push large files)
git lfs install

git clone https://github.com/<YOUR_USERNAME>/e-nose.git
cd e-nose
```

LFS is lazy: it will only download LFS-tracked blobs that actually exist in
the repo. On a fresh clone with no such blobs yet, nothing extra happens.

---

## 2. Create the Python environment

The full server uses **one exact runtime lock**: `requirements-server.txt`
includes `requirements/server-lock.txt`. Docker, standard `venv`/pip, and uv
all install that file. It pins direct **and transitive** versions, including
Torch **2.4.1+cu124**, torchvision **0.19.1+cu124**, Transformers **4.44.2**,
tokenizers **0.19.1**, NumPy **1.26.4**, and scikit-learn **1.5.2**. Do not
upgrade those packages individually or mix server and client environments.

### Option A — `venv` (standard library, Windows/Linux server)

```bash
python -m venv .venv  # use Python 3.10-3.12; Windows can use `py -3.10 -m venv .venv`
# Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd.exe)
.\.venv\Scripts\activate.bat

python -m pip install --upgrade "pip==24.3.1"
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/verify_server_environment.py
```

The editable install registers `enose`; `--no-deps` preserves the exact locked
versions. The CUDA 12.4 Torch wheel also runs on CPU when no CUDA GPU is
available. Model weights are still downloaded separately.

### Option B — uv (same runtime versions as pip and Docker)

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv requirements-server.txt
uv pip install --python .venv --no-deps -e .
uv pip check --python .venv
# Activate as above, then:
python scripts/verify_server_environment.py
```

Use `uv pip sync` with this lock, rather than an unconstrained `uv sync` or
`uv pip install` from broad extras. Both installers consume the same file;
the project config makes uv's two-index selection match pip's. For fully
matching Python versions, use 3.10 in both native environments and Docker.

### FlashAttention — retained for Docker / Linux CUDA

Docker still installs **flash-attn 2.7.3**. A native Linux CUDA environment
can install the same extension **after** the runtime lock:

```bash
# Needs CUDA 12.4 toolkit/nvcc and a C++ compiler; set CUDA_HOME if necessary.
export MAX_JOBS=4
python -m pip install -r requirements-flash-build.txt
python -m pip install --no-build-isolation -r requirements-flash-attn.txt
# uv equivalent (after uv pip sync above):
# uv pip install --python .venv -r requirements-flash-build.txt
# uv pip install --python .venv --no-build-isolation -r requirements-flash-attn.txt
```

Native **Windows** and hosts without the CUDA build toolchain use Florence's
eager attention path. The loader bypasses Transformers 4.44.2's false mandatory
FlashAttention check only for `modeling_florence2.py` when the extension is absent;
it leaves installed FlashAttention intact. This is the deliberate platform
exception: Windows/Linux have the same shared library versions, but CUDA DLLs,
Linux `nvidia-*`/Triton/uvloop wheels, and this extension cannot be identical.

On Windows install the Microsoft Visual C++ Redistributable if Torch reports
a missing DLL. The Linux headless OpenCV wheel may need `libglib2.0-0` from the OS.

> **Just running the client (no server on this machine)?** Use the lightweight
> client extra instead — it pulls none of the fastapi / scikit-learn stack:
> `pip install -e ".[client]"`. See [CLIENT_WINDOWS_INSTALL.md](CLIENT_WINDOWS_INSTALL.md)
> for the prebuilt-wheel path (no `git clone` needed).

### Option C — `conda`

```bash
conda create -n enose python=3.11
conda activate enose
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
```

### What each extra gives you

| Extra | Packages | When you need it |
|---|---|---|
| *(base)* | numpy, pandas, pillow, requests | Always (light, always-wheel-available) |
| `client` | pyserial, matplotlib, opencv-python | Robot-dog / colleague client (sensors, camera, live plot) |
| `server` | Pinned FastAPI, classifier, Torch, and vision dependencies | Full server; use the complete lock above to also pin transitive packages and the CUDA wheel build |
| `classifier-extras` | imbalanced-learn, xgboost | To use `BalancedRFClassifier` or `XGBTabularClassifier` |
| `vision` | Pinned torch/torchvision, transformers/tokenizers, accelerate, einops/timm, HF hub, safetensors, sentencepiece | Florence-2 dependencies (also included in `server`) |
| `dev` | pytest, ruff | Development / CI |

> The `numpy<2` / `scikit-learn<1.6` pins live only in the `server` extra, so a
> `[client]` install never drags them in — that's what previously broke the
> Windows wheel build for colleagues.

Smoke-test the install:

```bash
python scripts/smoke_test.py
# → AST-parsed N files — 0 failures, OK
```

---

## 3. Download Florence-2-Large

> **Skip this step** if you plan to run smell-only (`ENOSE_NO_VLM=1`).

The model is ~1.6 GB and is **not** committed to this repo. Download it
into `model/Florence-2-Large/` — the path `enose/config.VLM_MODEL_PATH`
points at. Pick one of three methods.

### Method A — `huggingface-cli` (recommended)

```bash
huggingface-cli download microsoft/Florence-2-large \
    --local-dir model/Florence-2-Large \
    --local-dir-use-symlinks False
```

If your network needs auth, set `HF_TOKEN` first:

```bash
# Linux / macOS
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Windows PowerShell
$env:HF_TOKEN = "hf_xxx..."
```

### Method B — `git lfs clone` from the HF mirror

```bash
git lfs install
git clone https://huggingface.co/microsoft/Florence-2-large model/Florence-2-Large
```

### Method C — Python snippet (useful inside Docker)

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="microsoft/Florence-2-large",
    local_dir="model/Florence-2-Large",
    local_dir_use_symlinks=False,
)
```

Verify the download:

```bash
ls model/Florence-2-Large/
# Should contain: config.json, model.safetensors (or pytorch_model.bin),
#                 processor_config.json, tokenizer*, vocab.json, etc.
```

### Changing the model path

Edit `VLM_MODEL_PATH` in [`enose/config.py`](../enose/config.py). Everything
downstream reads it from there.

---

## 4. Configure environment variables

```bash
cp .env.example .env
# edit .env as needed, then:

# Linux / macOS
set -a; source .env; set +a

# Windows PowerShell (quick approach)
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
```

The runtime reads env vars directly via `os.environ`; `.env` is **not**
auto-loaded by the server. Docker users can pass `--env-file .env` to
`docker run`.

Key variables (see [`.env.example`](../.env.example) for the full list):

| Variable | Purpose |
|---|---|
| `ENOSE_NO_VLM=1` | Skip Florence-2 — smell classifier only |
| `ENOSE_CLASSIFIER` | `balanced_rf` (default) or `xgb` |
| `SERVER_HOST` / `SERVER_PORT` | Bind address / port |
| `HF_TOKEN` | Only if you fetch Florence-2 from a gated mirror |

---

## 5. First run — server

### Smell-only (fastest, no GPU)

```bash
ENOSE_NO_VLM=1 python scripts/run_server.py
```

Check it's alive from another terminal:

```bash
curl http://localhost:8080/health
# {"status": "healthy", ...}
```

Browser UI: <http://localhost:8080/ui>
Swagger:    <http://localhost:8080/docs>

### Full mode (Florence-2 + classifier)

```bash
# Ensure ENOSE_NO_VLM is unset or 0 (PowerShell: $env:ENOSE_NO_VLM = "0")
python scripts/verify_server_environment.py --vlm
python scripts/run_server.py --host 0.0.0.0 --port 8016
```

`--vlm` checks the installed versions, runs the real server startup lifecycle,
loads the local weights/processor, and generates a caption. It uses temporary
working files and does not overwrite trained classifiers or datasets. Use
`--device cpu` to validate the CPU path explicitly, or `--device cuda` for GPU.
With the example port above the UI is at `http://localhost:8016/ui`.

First start loads Florence-2 into VRAM (~5–20 s depending on disk speed).
Any failure during the VLM load aborts startup with a clear error
message.

---

## 6. First run — client (no hardware needed)

In a second terminal, with the same venv active:

```bash
python scripts/run_client.py --offline --server http://localhost:8080
```

Pick menu option **6** to type 22 values by hand, or **3** to stream
synthetic samples through the classifier. The
[`client README`](../enose/client/README.md) has the full menu and every
CLI flag.

---

## 7. (Optional) Docker setup

### Server image — GPU, with Florence-2

```bash
docker build -t enose-server -f docker/Dockerfile-cu124 .

docker network create enose-net         # once per host

docker run -it --name florence2 \
    --gpus '"device=0"' \
    --network enose-net \
    -p 8016:8080 \
    -v "$(pwd)":/app \
    --env-file .env \
    enose-server \
    python scripts/run_server.py
```

The `-v "$(pwd)":/app` bind-mount means `model/Florence-2-Large/` you
downloaded in Step 3 is visible inside the container — no need to bake
it into the image.

### Client image — headless, UART pass-through

```bash
docker build -t enose-client -f docker/Dockerfile-client .

# Offline (no hardware):
docker run --rm -it \
    -v "$(pwd)":/app \
    -e PYTHONPATH=/app \
    enose-client \
    python scripts/run_client.py --offline --server http://host.docker.internal:8080
```

See [`SERVER_CLIENT_GUIDE.md`](SERVER_CLIENT_GUIDE.md) for the full
Docker + network walkthrough.

---

## 8. Publish your fork to GitHub

If you cloned from someone else and want to push your own copy:

```bash
# 1. Create an empty repo on github.com (no README, no .gitignore, no license)
#    e.g. https://github.com/YOUR_USERNAME/e-nose

# 2. In your local clone:
git remote remove origin                     # drop the original remote
git remote add origin git@github.com:YOUR_USERNAME/e-nose.git

# 3. First-time push
git push -u origin main
```

If you are turning **this** working directory into a new repo:

```bash
git init -b main
git lfs install --local
git add .gitignore .gitattributes .env.example
git commit -m "Initial: git infrastructure"

git add .
git commit -m "Import e-nose codebase"

git remote add origin git@github.com:YOUR_USERNAME/e-nose.git
git push -u origin main
```

Before the first push, double-check that no secret slipped through:

```bash
git ls-files | grep -Ei 'env|token|secret|key|credential' || echo "clean"
```

And that no bulky file made it into the commit:

```bash
git ls-files | xargs -I{} wc -c "{}" 2>/dev/null | sort -n | tail -20
```

Anything over a few MB should either be LFS-tracked (via `.gitattributes`)
or gitignored.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: enose` | You forgot `pip install -e .` after activating the venv. |
| Server starts but `/smell/classify` returns 503 | The classifier has no artifact yet. Go to the UI's Train tab and upload a CSV (a minimal one is at `data/enose_points_1906.csv`). |
| Florence-2 load fails with `Can't find config.json` | Download incomplete or path wrong. Re-run Step 3; verify `ls model/Florence-2-Large/config.json` succeeds. |
| Florence-2 reports missing `forced_bos_token_id`, an unexpected `dtype` argument, or tokenizer `ModelWrapper` errors | Recreate the server environment using the complete shared lock above. It fixes Torch/torchvision, Transformers/tokenizers, and the loader's slow-tokenizer setting together. Avoid upgrading only one library. |
| Florence-2 reports missing `flash_attn` on Windows | Use the current loader and shared lock. Windows runs eager attention without that extension; Docker/Linux retain the separately pinned extension. |
| `CUDA out of memory` on Florence-2 load | Set `ENOSE_NO_VLM=1` or use a smaller GPU variant. |
| `permission denied: /dev/ttyUSB0` on Linux | `sudo usermod -aG dialout $USER` then log out + back in. |
| Push rejected: "file over 100 MB" | LFS wasn't installed before the commit. Install LFS, run `git lfs migrate import --include="*.safetensors,*.bin,*.joblib,*.npz"`, then re-push. |
| `git lfs: command not found` after clone, and model files are tiny stubs | Install LFS (`git lfs install`) then `git lfs pull`. |

---

## 10. What to commit vs. keep local

| Commit | Keep local (gitignored) |
|---|---|
| Source under `enose/`, `scripts/`, `docs/` | `model/Florence-2-Large/` weights |
| `pyproject.toml`, `.gitignore`, `.gitattributes`, `.env.example` | `.env` |
| `docker/Dockerfile-*` | `trained_models/*.joblib` |
| `data/enose_points_1906.csv` (small example) | `data/database_robodog*.csv` (large logs) |
| `README.md`, sub-package READMEs | `data/plots/`, `runs/`, `temp_capture.png` |
|  | `.venv/`, `__pycache__/`, `.claude/` |

The rules are encoded in `.gitignore`; this table is just a human summary.
