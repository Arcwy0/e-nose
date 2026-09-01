# Migrating the project to another PC

GitHub carries the source code and documentation, but intentionally does **not**
carry local secrets, Florence-2 weights, trained classifier artifacts, recorded
sessions, or most experimental datasets. Follow this checklist so the new PC
reproduces both the software and the current experimental state.

## 1. On the old PC: identify what must be copied

The following paths are normally local-only because `.gitignore` excludes them:

| Path | Contents | Recommended migration method |
|---|---|---|
| `.env` | Local configuration and possible tokens | Encrypted/private transfer; never commit |
| `model/` | Florence-2 weights and Hugging Face files | Prefer downloading again; copy only for an offline migration |
| `trained_models/` | Trained smell classifier `.joblib` files | Copy privately if the learned state must be preserved |
| `data/*.csv` | Experimental/training datasets | Copy privately or to approved research storage |
| `data/provenance/` | Autonomous-label images and metadata | Copy privately if audit history matters |
| `runs/` | Recorded `.npz` sessions | Copy privately if replay is required |

Check their sizes before copying:

```bash
du -sh model trained_models data runs 2>/dev/null
```

Create checksums so the copy can be verified on the new PC:

```bash
find trained_models data runs -type f -print0 2>/dev/null \
  | sort -z | xargs -0 sha256sum > migration-checksums.sha256
```

Copy the required directories using an encrypted external drive, `rsync` over
SSH, or your institution's approved storage. For example:

```bash
rsync -a --info=progress2 trained_models/ user@NEW_PC:/path/to/e-nose/trained_models/
rsync -a --info=progress2 data/ user@NEW_PC:/path/to/e-nose/data/
rsync -a --info=progress2 runs/ user@NEW_PC:/path/to/e-nose/runs/
```

Do not upload research data or credentials to a public GitHub repository merely
to simplify migration.

## 2. Clone and select the project branch

On the new PC:

```bash
git lfs install
git clone https://github.com/Arcwy0/e-nose.git
cd e-nose
git switch idea_1_refinement
```

Git LFS is a safety mechanism for deliberately tracked large artifacts. The
normal Florence weights and runtime classifier files remain ignored, so
`git lfs pull` will not recreate them unless a future commit explicitly tracks
such a file.

## 3. Install the software

For a complete native installation (server, classifier, and Florence vision):

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[server,classifier-extras,vision,client]"
python scripts/smoke_test.py
```

For a client/Jetson machine that only reads sensors and talks to a remote server:

```bash
pip install -e ".[client]"
```

CUDA, Docker, Windows, and USB-serial details are covered in
[`SETUP.md`](SETUP.md), [`SERVER_CLIENT_GUIDE.md`](SERVER_CLIENT_GUIDE.md), and
[`CLIENT_WINDOWS_INSTALL.md`](CLIENT_WINDOWS_INSTALL.md).

## 4. Restore or download Florence-2

Downloading the official weights again is usually safer than copying them:

```bash
huggingface-cli download microsoft/Florence-2-large \
  --local-dir model/Florence-2-Large \
  --local-dir-use-symlinks False
```

If the new PC will initially run smell-only, weights are unnecessary:

```bash
ENOSE_NO_VLM=1 python scripts/run_server.py
```

For an offline move, copy the complete `model/Florence-2-Large/` directory and
verify that at least `config.json`, tokenizer files, processor configuration,
and a `.safetensors` or `.bin` weight file arrived.

## 5. Restore configuration and experimental state

Start from the public template rather than committing the old secret file:

```bash
cp .env.example .env
```

Then restore the required private values. In particular, set `ENOSE_RLOW` in
the **client/robot process** because ADC-to-resistance conversion occurs there.
It can also be overridden with client flag `--rlow` or menu option 10.

Copy back any required `trained_models/`, `data/`, `data/provenance/`, and
`runs/` content. Verify copied files from the repository root:

```bash
sha256sum -c migration-checksums.sha256
```

If a trained model is not copied, the server still starts but the smell model
must be trained again through the browser UI or `/smell/learn_from_csv`.

## 6. Verify the new PC

Run the checks in increasing order of hardware dependence:

```bash
python scripts/smoke_test.py
ENOSE_NO_VLM=1 python scripts/run_server.py
python scripts/run_client.py --offline --server http://localhost:8080
```

Then verify the browser UI at `http://localhost:8080/ui`, load the trained model,
run one known-smell classification, and only then connect the UART devices and
camera. The full staged procedure is in [`TESTING_GUIDE.md`](TESTING_GUIDE.md).

## 7. Large-file policy

- Do not commit `.env`, credentials, raw model weights, generated classifiers,
  or unrestricted experimental data.
- GitHub rejects ordinary Git blobs larger than 100 MB. The repository's
  `.gitattributes` routes common ML formats through Git LFS only when they are
  deliberately force-added.
- Prefer reproducible model downloads and private research storage over Git LFS
  for multi-gigabyte weights and datasets.
- Before every push, inspect `git status` and the staged file sizes:

  ```bash
  git diff --cached --name-only
  git diff --cached --stat
  ```
