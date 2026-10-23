"""Recover Florence's slow BART tokenizer files from its own tokenizer JSON.

Some official Florence-2 snapshots include tokenizer.json and vocab.json but
omit merges.txt. Transformers' slow BART tokenizer still requires both text
files, even though all BPE data is already present in tokenizer.json.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock


def _merge_lines(merges: list) -> list[str]:
    lines = []
    for index, merge in enumerate(merges):
        # Old tokenizers exports store "left right" strings. Newer exports
        # store [left, right] pairs; parse with Python, not the older Rust ABI.
        pair = merge.split(" ") if isinstance(merge, str) else merge
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(token, str) and token and not any(c.isspace() for c in token) for token in pair)
        ):
            raise ValueError(f"Invalid BPE merge at index {index}; download a complete Florence tokenizer")
        lines.append(" ".join(pair))
    return lines


def ensure_slow_tokenizer_files(model_dir: str | Path) -> list[str]:
    """Create only missing vocab.json/merges.txt, retaining original token IDs.

    Existing files and model weights are never overwritten. Recovery uses the
    model's own BPE vocabulary/merge order, not a tokenizer from another model.
    A file lock and atomic replacement keep concurrent server workers from
    observing partially written files. Returns names of newly created files.
    """
    root = Path(model_dir)
    required = ("vocab.json", "merges.txt")
    if not root.is_dir() or all((root / name).is_file() for name in required):
        return []

    source = root / "tokenizer.json"
    try:
        tokenizer = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Incomplete Florence tokenizer in {root}: slow tokenization needs vocab.json and merges.txt. "
            "Download tokenizer.json from the same Florence-2 snapshot to recover missing files."
        ) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not valid tokenizer JSON; download it again") from exc

    model = tokenizer.get("model") if isinstance(tokenizer, dict) else None
    if not isinstance(model, dict) or model.get("type") not in (None, "BPE"):
        raise ValueError(f"{source} does not contain a BPE tokenizer model")
    vocabulary, merges = model.get("vocab"), model.get("merges")
    if (
        not isinstance(vocabulary, dict)
        or not vocabulary
        or not all(isinstance(token, str) and type(token_id) is int and token_id >= 0 for token, token_id in vocabulary.items())
        or not isinstance(merges, list)
        or not merges
    ):
        raise ValueError(f"{source} has no usable BPE vocabulary/merges; download a complete Florence tokenizer")

    # Validate all input before creating either missing file.
    merge_lines = _merge_lines(merges)
    contents = {
        "vocab.json": json.dumps(vocabulary, ensure_ascii=False) + "\n",
        "merges.txt": "#version: 0.2\n" + "\n".join(merge_lines) + "\n",
    }
    created = []
    with FileLock(str(root / ".enose-tokenizer.lock")):
        for name in required:
            destination = root / name
            if destination.exists():
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=root,
                    prefix=".enose-tokenizer-", delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(contents[name])
                os.replace(temporary, destination)
                created.append(name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return created
