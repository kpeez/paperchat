from pathlib import Path
from types import SimpleNamespace

from paperchat_backend.benchmarks.docling_validation import docling_adapter


def test_find_cached_sentence_transformer_snapshot_from_hf_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = (
        tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "snapshot-1"
    )
    snapshot_path.mkdir(parents=True)
    for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        (snapshot_path / filename).write_text("x", encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    assert docling_adapter._find_cached_sentence_transformer_snapshot() == snapshot_path


def test_find_cached_sentence_transformer_snapshot_uses_default_home_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / "snapshot-1"
    )
    snapshot_path.mkdir(parents=True)
    for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        (snapshot_path / filename).write_text("x", encoding="utf-8")

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(docling_adapter.Path, "home", lambda: tmp_path)

    assert docling_adapter._find_cached_sentence_transformer_snapshot() == snapshot_path


def test_build_hybrid_chunker_uses_cached_tokenizer(monkeypatch) -> None:
    cached_tokenizer = object()

    class FakeHybridChunker:
        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

    monkeypatch.setattr(
        docling_adapter,
        "_load_cached_hybrid_tokenizer",
        lambda: cached_tokenizer,
    )

    chunker = docling_adapter._build_hybrid_chunker(
        SimpleNamespace(HybridChunker=FakeHybridChunker)
    )

    assert chunker.tokenizer is cached_tokenizer


def test_build_hybrid_chunker_falls_back_without_cached_tokenizer(monkeypatch) -> None:
    class FakeHybridChunker:
        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

    monkeypatch.setattr(
        docling_adapter,
        "_load_cached_hybrid_tokenizer",
        lambda: None,
    )

    chunker = docling_adapter._build_hybrid_chunker(
        SimpleNamespace(HybridChunker=FakeHybridChunker)
    )

    assert chunker.tokenizer is None
