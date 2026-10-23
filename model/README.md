# model/ — Vision model files (not stored in Git)

Florence-2-Large is several gigabytes and is intentionally excluded from this
repository. After cloning, download it into `model/Florence-2-Large/`:

```bash
huggingface-cli download microsoft/Florence-2-large \
  --local-dir model/Florence-2-Large \
  --local-dir-use-symlinks False
```

The resulting directory should contain `config.json`, tokenizer and processor
files, and a `.safetensors` or `.bin` weight file. See
[`docs/SETUP.md`](../docs/SETUP.md) for installation details and
[`docs/MIGRATION.md`](../docs/MIGRATION.md) for moving an existing offline copy.

Official snapshots can omit `merges.txt`, `added_tokens.json`, and
`special_tokens_map.json`. The loader recovers missing `merges.txt`/`vocab.json`
from the same snapshot's `tokenizer.json`; it needs a writable folder once and
never overwrites existing tokenizer files or weights. The two separate token
metadata files are optional for the official processor. Keep the snapshot's
Python model/processor code for portable offline loading.

For the tested download revision, exact repair commands, and optional export
of the extra token metadata files, see
[Florence model files](../docs/FLORENCE_MODEL_FILES.md).

Do not force-add the weights to ordinary Git. If a controlled project genuinely
needs to version weights, use dedicated artifact storage or Git LFS and confirm
the repository's storage quota first.
