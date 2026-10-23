# Florence-2 model files: download, repair, and migration

Use this guide if a new machine's `model/Florence-2-Large/` folder looks
different from an existing working copy, or startup fails inside BART with
`TypeError: expected str, bytes or os.PathLike object, not NoneType`.
Run the commands from the repository root with the **server** environment
activated; see [SETUP.md](SETUP.md) for the shared pip/uv/Docker dependency lock.

## Why the folders differ

The tested [official Microsoft snapshot](https://huggingface.co/microsoft/Florence-2-large/tree/21a599d414c4d928c9032694c424fb94458e3594)
contains `tokenizer.json` and `vocab.json`, but **does not publish**
`merges.txt`, `added_tokens.json`, or `special_tokens_map.json`. Repeating the
same download will not produce files that are absent upstream. This is a
tokenizer serialization/layout difference, not evidence of a failed download
or a missing CUDA GPU.

The combined `tokenizer.json` already contains the BPE vocabulary and merge
rules. The Florence processor adds its task/location tokens when it is loaded.
Saving a loaded tokenizer with Transformers' `save_pretrained()` exports
separate files, including `merges.txt`, `added_tokens.json`, and
`special_tokens_map.json`. The exact layout also depends on tokenizer/library
version and whether the fast or slow tokenizer was saved.

Our existing working folder has the same base vocabulary and BPE merge order
as the official snapshot, but a newer tokenizer JSON representation and an
expanded `tokenizer_config.json`. Exporting the official processor's tokenizer
reproduces its 1,024-entry `added_tokens.json`. This is evidence of a locally
saved/exported tokenizer layout; it does **not** establish which historical
command created that folder.

| File | What to do |
|---|---|
| `config.json`, `preprocessor_config.json` | Download from the same model revision. |
| `configuration_florence2.py`, `modeling_florence2.py`, `processing_florence2.py` | Keep the official model/processor code. Exported copies without it can depend on the Hugging Face code cache/network. |
| `model.safetensors` | Download the weights. A separate `pytorch_model.bin` is unnecessary when this file is available. |
| `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, `vocab.json` | Keep the files from the same snapshot. |
| `merges.txt` | Required by our **slow BART** tokenizer; reconstruct from the same snapshot's `tokenizer.json` if absent. |
| `added_tokens.json`, `special_tokens_map.json` | Optional separate exports for the official processor. Do not invent them or copy them from another model. |

Our loader uses slow tokenization because newer fast-tokenizer JSON exports
can be incompatible with the pinned `tokenizers` library. The compatibility
gap was that the official snapshot omitted a file this slow path needs.

## 1. Download a reproducible snapshot

This single-line command works in Bash and Windows PowerShell. The revision
is pinned to the snapshot tested with our server environment; the include
filter avoids downloading a second copy of the weights in `.bin` format.

```bash
huggingface-cli download microsoft/Florence-2-large --revision 21a599d414c4d928c9032694c424fb94458e3594 --include "*.json" "*.py" "model.safetensors" "LICENSE" --local-dir model/Florence-2-Large
```

If a matching snapshot is already downloaded, no weight redownload is needed
for the tokenizer repair below. Do not mix files from Florence-2-base,
Florence-2-large-ft, or an unrelated BART tokenizer. For a different revision,
validate startup and inference before treating it as a supported replacement.

## 2. Reconstruct missing slow-tokenizer files

With the updated source installed (`python -m pip install --no-deps -e .`),
the server and VLM verification command do this automatically. To prepare
the folder explicitly, without loading the model weights:

```bash
python -c "from enose.utils.tokenizer_io import ensure_slow_tokenizer_files; print(ensure_slow_tokenizer_files('model/Florence-2-Large'))"
```

The helper creates only missing `merges.txt`/`vocab.json`, preserves vocabulary
IDs and BPE merge order, and never overwrites existing files or weights.
The directory must be writable once. If Docker mounts it read-only, prepare
it on the host before starting the container. Missing/corrupt `tokenizer.json`
must be downloaded again from the same model revision.

### Existing v3.2.1 installations without the recovery helper

For an immediate one-time repair, the following PowerShell command uses only
Python's standard library and does not change installed packages or weights:

```powershell
@'
import json
from pathlib import Path

p = Path("model/Florence-2-Large")
target = p / "merges.txt"
if not target.exists():
    data = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
    merges = data["model"]["merges"]
    lines = [m if isinstance(m, str) else " ".join(m) for m in merges]
    target.write_text("#version: 0.2\n" + "\n".join(lines) + "\n", encoding="utf-8")
print("merges.txt is ready")
'@ | python -
```

In Bash, run the same Python body between `python - <<'PY'` and a closing
`PY` line. This fallback assumes the official snapshot's valid `vocab.json`
and `tokenizer.json` are already present.

## 3. Optional: reproduce the extra exported token files

Neither file below is needed to start the official processor. If you want a
tokenizer backup or to reproduce the additional files seen in a working copy,
first complete Step 2, then export to a **separate, new directory**:

```bash
python -c "from transformers import AutoProcessor; p = AutoProcessor.from_pretrained('model/Florence-2-Large', trust_remote_code=True, use_fast=False); print(p.tokenizer.save_pretrained('model/Florence-2-tokenizer-export'))"
```

On the pinned environment this writes `added_tokens.json`, `merges.txt`,
`special_tokens_map.json`, `tokenizer_config.json`, and `vocab.json`. It does
not copy model weights or make that export directory a complete VLM. The
server continues using `model/Florence-2-Large/`; there is no need to copy the
optional files back or overwrite the original snapshot's configuration.

## 4. Verify and start on CPU

```bash
python scripts/verify_server_environment.py --vlm --device cpu
```

This checks locked dependencies, runs real server startup, and generates a
caption from an image. It may create the missing tokenizer text files but
does not overwrite weights, datasets, or smell-classifier artifacts.

Windows PowerShell:

```powershell
$env:ENOSE_NO_VLM = "0"
python scripts/run_server.py --host 0.0.0.0 --port 8016
```

Bash:

```bash
ENOSE_NO_VLM=0 python scripts/run_server.py --host 0.0.0.0 --port 8016
```

Open `http://localhost:8016/ui`. On a PC without a CUDA GPU the server selects
CPU automatically. FlashAttention is not needed on CPU; do not try installing
it to repair a tokenizer error. Real CPU startup and caption generation were
validated on Linux with the official snapshot and FlashAttention absent;
native Windows execution has not been validated here.

## Empty detections after successful startup

Version 3.2.2 fixes a separate decoding/display problem. Slow BART's default
decoding inserts spaces between location tokens, while Florence's postprocessor
expects adjacent tokens such as `<loc_100><loc_200><loc_300><loc_400>`. The model
can generate valid coordinates but return empty parsed boxes. The loader now
decodes with `spaces_between_special_tokens=False`; changing the weights,
prompts, or CPU/CUDA device is not the remedy for that parser failure.

The camera client also unwraps the `/predict/object` response's `result` field
and draws polygon regions as well as boxes. Update both server and client;
rebuild the Docker image when running the server in Docker. Dependency pins
and model weights have not changed.

A successful caption-only startup check does not test localization-token
parsing. Check object detection on a visible bottle with `bottle` or `vial`.
The critical decoding and API-envelope contracts are covered by
`tests/test_florence_detection.py`.

An `acetone` region is **not chemical identification**. An unlabeled bottle's
contents cannot be verified from its appearance. Use visual container labels
for localization, and confirmed sample identity or olfactory measurements for
chemical labels. CPU latency depends on the PC; a timeout is distinct from
an HTTP 200 response containing empty parsed regions.

## GitHub and offline copies

Keep the recovery code and these instructions in Git, rather than a second
copy of regeneratable model/tokenizer artifacts. The model folder remains
ignored. For an offline migration, copy the complete official snapshot plus
the recovered text files. A saved tokenizer alone is not a complete model;
also retain the weights, processor configuration, and Python code. Follow
[MIGRATION.md](MIGRATION.md) for the remaining project artifacts.

Implementation reference: Transformers 4.44.2
[`save_pretrained()`](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/tokenization_utils_base.py)
exports tokenizer configuration, special-token metadata, and vocabulary files.
