"""Fresh Florence snapshots must work without a pre-existing merges.txt."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from enose.utils.tokenizer_io import ensure_slow_tokenizer_files


VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "ab": 6}


def _write_snapshot(path, modern=False):
    model = {"vocab": VOCAB, "merges": [["a", "b"]] if modern else ["a b"]}
    if modern:
        model["type"] = "BPE"
    (path / "tokenizer.json").write_text(json.dumps({"model": model, "added_tokens": []}), encoding="utf-8")
    (path / "tokenizer_config.json").write_text('{"tokenizer_class":"BartTokenizer"}')


@pytest.mark.parametrize("modern", [False, True])
def test_recovers_legacy_and_modern_exports_without_changing_ids(tmp_path, modern):
    _write_snapshot(tmp_path, modern=modern)
    assert ensure_slow_tokenizer_files(tmp_path) == ["vocab.json", "merges.txt"]
    assert json.loads((tmp_path / "vocab.json").read_text()) == VOCAB
    assert (tmp_path / "merges.txt").read_text() == "#version: 0.2\na b\n"
    stat = (tmp_path / "merges.txt").stat().st_mtime_ns
    assert ensure_slow_tokenizer_files(tmp_path) == []
    assert (tmp_path / "merges.txt").stat().st_mtime_ns == stat


def test_existing_vocab_and_other_model_files_are_preserved(tmp_path):
    _write_snapshot(tmp_path)
    original = b'{"existing-token":123}\n'
    (tmp_path / "vocab.json").write_bytes(original)
    (tmp_path / "model.safetensors").write_bytes(b"weights are untouched")
    assert ensure_slow_tokenizer_files(tmp_path) == ["merges.txt"]
    assert (tmp_path / "vocab.json").read_bytes() == original
    assert (tmp_path / "model.safetensors").read_bytes() == b"weights are untouched"


def test_complete_tokenizer_does_not_require_tokenizer_json(tmp_path):
    (tmp_path / "vocab.json").write_text("{}")
    (tmp_path / "merges.txt").write_text("#version: 0.2\na b\n")
    assert ensure_slow_tokenizer_files(tmp_path) == []


@pytest.mark.parametrize("broken", [None, "not json", '{"model":{"type":"WordPiece"}}'])
def test_incomplete_invalid_snapshot_fails_before_writing_files(tmp_path, broken):
    if broken is not None:
        (tmp_path / "tokenizer.json").write_text(broken)
    with pytest.raises((FileNotFoundError, ValueError), match="tokenizer|BPE"):
        ensure_slow_tokenizer_files(tmp_path)
    assert not (tmp_path / "vocab.json").exists()
    assert not (tmp_path / "merges.txt").exists()


@pytest.mark.parametrize("merge", ["a", ["a"], ["a", "b", "c"], ["a\nb", "c"]])
def test_invalid_merges_do_not_write_a_partial_tokenizer(tmp_path, merge):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": VOCAB, "merges": [merge]}}))
    with pytest.raises(ValueError, match="Invalid BPE merge"):
        ensure_slow_tokenizer_files(tmp_path)
    assert not (tmp_path / "vocab.json").exists()
    assert not (tmp_path / "merges.txt").exists()


def test_concurrent_workers_only_create_one_complete_pair(tmp_path):
    _write_snapshot(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        created = list(pool.map(lambda _: ensure_slow_tokenizer_files(tmp_path), range(12)))
    assert sum(len(names) for names in created) == 2
    assert json.loads((tmp_path / "vocab.json").read_text()) == VOCAB
    assert (tmp_path / "merges.txt").read_text() == "#version: 0.2\na b\n"


def test_recovered_files_load_a_real_slow_bart_tokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("4."):
        pytest.skip("Slow BART is tested on the pinned Transformers 4 runtime; v5 changes tokenizer backends")
    _write_snapshot(tmp_path)
    ensure_slow_tokenizer_files(tmp_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tmp_path, use_fast=False, local_files_only=True)
    assert not tokenizer.is_fast
    assert tokenizer.encode("ab", add_special_tokens=False) == [VOCAB["ab"]]
    assert not (tmp_path / "added_tokens.json").exists()
    assert not (tmp_path / "special_tokens_map.json").exists()


def test_recovery_preserves_unicode_byte_level_vocabulary(tmp_path):
    vocabulary = {**VOCAB, "Ġ": 7, "Ġab": 8}
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": vocabulary, "merges": ["a b", "Ġ ab"]}}),
        encoding="utf-8",
    )
    ensure_slow_tokenizer_files(tmp_path)
    assert json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8")) == vocabulary
    assert (tmp_path / "merges.txt").read_text(encoding="utf-8") == "#version: 0.2\na b\nĠ ab\n"
