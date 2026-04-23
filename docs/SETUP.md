# Setup Guide — From Clone to First Classification

This is the end-to-end setup for a fresh machine. If you only want to run
the **smell classifier** (no vision, no GPU), skip Step 3; everything else
works unchanged. If you want the **full multimodal system**, do all steps.

A companion guide for non-programmers using the browser UI lives in
[`CHEMIST_GUIDE.md`](CHEMIST_GUIDE.md). The server ↔ client deployment
reference is in [`SERVER_CLIENT_GUIDE.md`](SERVER_CLIENT_GUIDE.md).

---

## 0. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.9 – 3.12** | Pin via `pyenv` if your system Python is older. |
| **Git** | Any recent version. |
| **Git LFS** *(optional)* | Only needed if you plan to commit model weights, `.npz` sessions, or other large artifacts — see `.gitattributes`. |
| **CUDA-capable GPU** *(optional)* | Required only for Florence-2 in full mode. 8 GB VRAM is enough. |
| **USB-serial drivers** *(client only)* | For the physical e-nose + UART env board. Not needed in `--offline` mode. |

Quick check that you have what you need:

```bash
python --version      # ≥ 3.9
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

### Option A — `venv` (standard library, works everywhere)

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd.exe)
.\.venv\Scripts\activate.bat

pip install --upgrade pip
pip install -e ".[classifier-extras]"          # server + classifier, CPU-only
# add vision extras if you want Florence-2 too:
pip install -e ".[classifier-extras,vision]"
```

### Option B — `conda`

```bash
conda create -n enose python=3.11
conda activate enose
pip install -e ".[classifier-extras,vision]"
```

### What each extra gives you

| Extra | Packages | When you need it |
|---|---|---|
| *(base)* | fastapi, uvicorn, numpy, pandas, scikit-learn, matplotlib, requests, pyserial, opencv-python | Always |
| `classifier-extras` | imbalanced-learn, xgboost | To use `BalancedRFClassifier` or `XGBTabularClassifier` |
| `vision` | torch, transformers, einops, timm | Full Florence-2 mode |
| `dev` | pytest, ruff | Development / CI |

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
pip install -U "huggingface_hub[cli]"

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
python scripts/run_server.py
```

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
| Florence-2 load fails with `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'` | `transformers` is too new. Pin to `<4.50`: `pip install --force-reinstall "transformers==4.44.2"` and `rm -rf /root/.cache/huggingface/modules/transformers_modules/microsoft`. The Dockerfile and pyproject now pin this upper bound, so this only hits pre-existing environments. |
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
