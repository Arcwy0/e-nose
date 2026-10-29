# Migrating the project to another PC

Release **3.3.0** separates public source/package installation from private
experimental state. GitHub and the wheel do not include model weights, recorded
sessions, private datasets, credentials, or the learned classifier.

## 1. Freeze and inventory the old installation

Stop acquisition and server writes before copying. Do not copy a classifier
while history/candidate promotion is being updated. Keep the original until the
new installation is verified.

| Path | What to preserve |
|---|---|
| `model/Florence-2-Large/` | Complete pinned weights, tokenizer, processor and model code; otherwise download again |
| `trained_models/smell_classifier_sgd_latest.joblib` | Active classifier |
| **Root `database_robodog.csv`** | Active model's historical learning dataset; not just files under `data/` |
| All `trained_models/` | Candidate models, their history/metadata and rollback snapshots |
| `data/`, including `data/provenance/` | Private measurements, autonomous-label images and metadata |
| `runs/` or custom recorder paths | NPZ sessions and acquisition provenance |
| Private environment/configuration | Transfer securely; do not publish tokens or copy environment directories |

Migrate the **model/history pair and the whole candidate/snapshot tree
together**. Copying only a `.joblib` can lose training history and rollback
consistency. Candidate comparison evidence is tied to model/history hashes.
Keep relative directory structure intact.

Rollback manifests also bind absolute active-artifact paths. If the new PC
uses different paths, preserve old snapshots as archival evidence; create new
deployment snapshots there rather than editing manifests to bypass checks.

Linux Bash inventory/checksums, run from the old project root:

```bash
du -sh model trained_models data runs database_robodog.csv 2>/dev/null
find trained_models data runs -type f -print0 | sort -z | xargs -0 sha256sum > migration-checksums.sha256
if test -f database_robodog.csv; then sha256sum database_robodog.csv >> migration-checksums.sha256; fi
```

Omit unavailable directories from `find`; add custom recording directories.
The manifest above covers experimental state, not weights/secrets. Verify
weights separately if copying rather than downloading.

For a Windows PowerShell inventory:

```powershell
Get-Item .\database_robodog.csv, .\trained_models, .\data, .\runs -ErrorAction SilentlyContinue
Get-ChildItem .\trained_models, .\data, .\runs -File -Recurse -ErrorAction SilentlyContinue |
    Get-FileHash -Algorithm SHA256
Get-FileHash -Algorithm SHA256 .\database_robodog.csv -ErrorAction SilentlyContinue
```

Record hashes privately and compare after transfer. Use encrypted external
storage or institutional storage/SSH. No public force-add of model/data files.

## 2. Clone main or obtain the matching release

Run in either shell:

```text
git clone https://github.com/Arcwy0/e-nose.git
cd e-nose
git switch main
```

For an exact release checkout instead of current main, use `git switch --detach
v3.3.0`. The old `idea_1_refinement` branch is not required.

Client-only machines can install
[`enose-3.3.0-py3-none-any.whl`](https://github.com/Arcwy0/e-nose/releases/tag/v3.3.0)
without a checkout. Follow [CLIENT_WINDOWS_INSTALL.md](CLIENT_WINDOWS_INSTALL.md)
for download/checksum checks, uv installation and installed commands.
Repository `scripts/` helpers are not included in the wheel.

Git LFS only retrieves deliberately tracked artifacts. Ignored Florence weights
and experimental files do not appear after `git lfs pull`.

## 3. Recreate environments; do not copy .venv

Use [SETUP.md](SETUP.md) for full installation, Docker and pinned versions.
Server and hardware client use different environments.

### Windows PowerShell — server

```powershell
uv venv --python 3.10 .venv-server
.\.venv-server\Scripts\Activate.ps1
uv pip sync requirements-server.txt
uv pip install --no-deps -e .
uv pip check
uv run --active --no-project python scripts/verify_server_environment.py
```

For the separate client terminal:

```powershell
uv venv --python 3.11 .venv-client
.\.venv-client\Scripts\Activate.ps1
uv pip install -e ".[client]"
uv pip check
```

### Linux Bash — server

Prefer rebuilding [the Docker image](../docker/Dockerfile-cu124) on the remote
server. A native environment uses the same complete runtime lock:

```bash
python3.10 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade "pip==24.3.1"
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
python -m pip check
python scripts/verify_server_environment.py
```

The lock covers shared direct/transitive versions across Docker, pip and uv.
Linux CUDA additionally installs pinned FlashAttention 2.7.3 after Torch as
described in setup. CPU/Windows use eager attention; OS-specific libraries
necessarily differ. A client-only Linux/Jetson installation uses a separate
environment with `python -m pip install -e ".[client]"`; native dependency-wheel
availability on the target architecture needs validation.

## 4. Restore Florence and configuration

Use [FLORENCE_MODEL_FILES.md](FLORENCE_MODEL_FILES.md) for revision
`21a599d414c4d928c9032694c424fb94458e3594`, the filtered download and missing
slow-tokenizer preparation. Keep all model/processor Python files and one
complete weight format. Never combine tokenizer files from different models.

For offline migration, copy the full model folder and verify the copied files.
Prepare `merges.txt`/`vocab.json` before using read-only mounts. Smell-only
startup needs no Florence download.

Native runtime does **not autoload `.env`**. Reapply private environment
variables explicitly in the new process; Docker can accept `--env-file`.
Never commit the private file.

RLOW belongs to the **legacy client-side conversion** via `ENOSE_RLOW`,
`--rlow`, or menu 10. New PCB resistance output is ohms by default, with no
legacy RLOW multiplier. Preserve original units and acquisition identity in
datasets; do not relabel legacy values as ohms without verifying their scale.

The new PCB backend is default. Old-board clients must explicitly select
`--sensor-backend legacy`, which never probes unsupported INFO/STATUS.
Close supplier software before our client opens its COM port.

Heating defaults to supplier-configured `external` mode. Explicit `client`
mode defaults to 300°C target/350°C host trip guard, both configurable.
Check [PCB_ACQUISITION.md](PCB_ACQUISITION.md) before physical testing.
Graceful disable/cooldown is monitored; no software guarantee exists after
USB loss, SIGKILL or host failure without a device watchdog/interlock.

## 5. Restore and verify experimental state

Copy the active model, root history and candidate/snapshot tree while the new
server is stopped. Retain the backup and source checksums. On Linux, from the
new repository root:

```bash
sha256sum -c migration-checksums.sha256
```

Never load classifier/joblib artifacts from an untrusted source: deserialization
can execute code. Use the matching runtime lock for copied classifiers.
If an old model lacks acquisition/unit provenance, verify compatibility rather
than guessing that it matches the new PCB.

A **live clean-air baseline does not migrate**: persisted models reset live
binding on load. Capture fresh clean air for the new board/session, and after
restart or relevant acquisition-setting changes.

If no model is copied, use the browser Train preview and verified labels. The
default workflow stages a candidate without changing active state; independently
evaluate and explicitly promote it. Historical merging remains the default data
policy unless dataset-only replacement is deliberately chosen. With insufficient
independent exposure groups, validation accuracy is not estimable.

## 6. Verify before hardware connection

Windows PowerShell, server terminal:

```powershell
$env:ENOSE_NO_VLM = "1"
uv run --active --no-project enose-server --host 0.0.0.0 --port 8016
```

Windows PowerShell, separate client terminal:

```powershell
uv run --active --no-project enose-client --offline --server http://localhost:8016 --preflight
```

Linux Bash, server terminal:

```bash
ENOSE_NO_VLM=1 enose-server --host 0.0.0.0 --port 8016
```

Linux Bash, separate client environment:

```bash
enose-client --offline --server http://localhost:8016 --preflight
```

Check `/health`, `/ui`, known classes and the expected active-model history.
For VLM verification use the repository helper in the server environment:
Windows `uv run --active --no-project python scripts/verify_server_environment.py
--vlm --device cpu`; Linux `python scripts/verify_server_environment.py --vlm`.
`--device` belongs to this helper, not `enose-server`. The server automatically
uses CPU if no CUDA GPU is available; allow sufficient RAM and potentially
longer inference/request deadlines.

Next test labelled replay and candidate/rollback behaviour on a private copy,
then acquisition preflight, and only then an approved physical-sensor test.
Do not enable heating or robot actuation as part of an offline migration check.
All client options are documented in the [client README](../enose/client/README.md).

## 7. Large-file policy

Keep weights, raw CSVs, recordings, generated models, candidate/snapshot state,
credentials and private reports out of public Git. Use reproducible model
downloads and private research storage. The wheel contains code, not these
runtime assets. No Git-history rewriting or force-adding ignored files is
needed to migrate.
