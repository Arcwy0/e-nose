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

Do not force-add the weights to ordinary Git. If a controlled project genuinely
needs to version weights, use dedicated artifact storage or Git LFS and confirm
the repository's storage quota first.
