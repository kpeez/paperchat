from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import paperchat.services.embeddings as embedding_module
from paperchat import config as paperchat_config
from paperchat.services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingDependencyError,
    EmbeddingGemmaEmbedder,
    EmbeddingRuntimeError,
    _embed_with_node_runtime,
    _ensure_node_runtime_dependencies,
)


def test_embedding_gemma_embedder_raises_clear_error_when_node_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("paperchat.services.embeddings.shutil.which", lambda _: None)

    embedder = EmbeddingGemmaEmbedder()

    with pytest.raises(EmbeddingDependencyError, match=r"Node\.js"):
        embedder.embed_documents(("doc text",))


def test_embedding_gemma_embedder_uses_qmd_prompt_format_and_model_cache() -> None:
    calls: dict[str, object] = {}

    def fake_runner(texts, model_name: str, model_cache_dir: Path):
        calls["runner"] = {
            "texts": texts,
            "model_name": model_name,
            "model_cache_dir": str(model_cache_dir),
        }
        return ((0.1, 0.2),)

    embedder = EmbeddingGemmaEmbedder(embed_runner=fake_runner)

    assert embedder.embed_documents(("doc text",)) == ((0.1, 0.2),)
    assert calls["runner"] == {
        "texts": ("title: none | text: doc text",),
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "model_cache_dir": str(paperchat_config.get_model_cache_dir()),
    }
    assert embedder.embed_query("query text") == (0.1, 0.2)
    assert calls["runner"] == {
        "texts": ("task: search result | query: query text",),
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "model_cache_dir": str(paperchat_config.get_model_cache_dir()),
    }


def test_embedding_gemma_embedder_uses_paperchat_model_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_runner(texts, model_name: str, model_cache_dir: Path):
        calls["runner"] = {
            "texts": texts,
            "model_name": model_name,
            "model_cache_dir": str(model_cache_dir),
        }
        return ((0.1, 0.2),)

    monkeypatch.setenv("PAPERCHAT_CACHE_DIR", str(tmp_path / "paperchat-cache"))
    paperchat_config.get_cache_dir.cache_clear()
    paperchat_config.get_model_cache_dir.cache_clear()

    embedder = EmbeddingGemmaEmbedder(embed_runner=fake_runner)

    assert embedder.embed_documents(("doc text",)) == ((0.1, 0.2),)
    expected_model_cache = tmp_path / "paperchat-cache" / "models"
    assert expected_model_cache.is_dir()
    assert calls["runner"] == {
        "texts": ("title: none | text: doc text",),
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "model_cache_dir": str(expected_model_cache),
    }

    paperchat_config.get_cache_dir.cache_clear()
    paperchat_config.get_model_cache_dir.cache_clear()


def test_embed_with_node_runtime_runs_helper_and_parses_vectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    deps_called = {"value": False}

    monkeypatch.setattr("paperchat.services.embeddings.shutil.which", lambda _: "/usr/bin/node")
    monkeypatch.setattr(
        "paperchat.services.embeddings._ensure_node_runtime_dependencies",
        lambda: deps_called.__setitem__("value", True),
    )

    def fake_run(*args, **kwargs):
        calls["run"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace(returncode=0, stdout='{"vectors":[[0.1,0.2],[0.3,0.4]]}', stderr="")

    monkeypatch.setattr("paperchat.services.embeddings.subprocess.run", fake_run)

    vectors = _embed_with_node_runtime(
        ("title: none | text: a", "title: none | text: b"),
        DEFAULT_EMBEDDING_MODEL,
        tmp_path / "models",
    )

    assert vectors == ((0.1, 0.2), (0.3, 0.4))
    assert deps_called["value"] is True
    run_call = calls["run"]
    assert run_call["args"] == (["/usr/bin/node", str(embedding_module.EMBEDDING_RUNTIME_SCRIPT)],)
    assert run_call["kwargs"]["cwd"] == embedding_module.EMBEDDING_RUNTIME_DIR
    assert run_call["kwargs"]["capture_output"] is True
    payload = run_call["kwargs"]["input"]
    assert DEFAULT_EMBEDDING_MODEL in payload
    assert str(tmp_path / "models") in payload


def test_embed_with_node_runtime_raises_for_invalid_helper_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("paperchat.services.embeddings.shutil.which", lambda _: "/usr/bin/node")
    monkeypatch.setattr(
        "paperchat.services.embeddings._ensure_node_runtime_dependencies", lambda: None
    )
    monkeypatch.setattr(
        "paperchat.services.embeddings.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )

    with pytest.raises(EmbeddingRuntimeError, match="invalid response"):
        _embed_with_node_runtime(
            ("title: none | text: a",),
            DEFAULT_EMBEDDING_MODEL,
            tmp_path / "models",
        )


def test_embed_with_node_runtime_parses_json_after_download_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("paperchat.services.embeddings.shutil.which", lambda _: "/usr/bin/node")
    monkeypatch.setattr(
        "paperchat.services.embeddings._ensure_node_runtime_dependencies", lambda: None
    )
    monkeypatch.setattr(
        "paperchat.services.embeddings.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='Downloading to ~/.cache/paperchat/models\n{"vectors":[[0.1,0.2]]}',
            stderr="",
        ),
    )

    assert _embed_with_node_runtime(
        ("title: none | text: a",),
        DEFAULT_EMBEDDING_MODEL,
        tmp_path / "models",
    ) == ((0.1, 0.2),)


def test_ensure_node_runtime_dependencies_installs_helper_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper_dir = tmp_path / "embedding_runtime"
    helper_dir.mkdir()
    package_json = helper_dir / "package.json"
    package_json.write_text("{}", encoding="utf-8")
    sentinel = helper_dir / "node_modules" / "node-llama-cpp"

    monkeypatch.setattr(embedding_module, "EMBEDDING_RUNTIME_DIR", helper_dir)
    monkeypatch.setattr(embedding_module, "EMBEDDING_RUNTIME_PACKAGE", package_json)
    monkeypatch.setattr(embedding_module, "NODE_RUNTIME_SENTINEL", sentinel)
    monkeypatch.setattr("paperchat.services.embeddings.shutil.which", lambda _: "/usr/bin/npm")

    calls: dict[str, dict[str, object]] = {}

    def fake_run(*args, **kwargs):
        calls["run"] = {"args": args, "kwargs": kwargs}
        sentinel.mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("paperchat.services.embeddings.subprocess.run", fake_run)

    _ensure_node_runtime_dependencies()

    run_call = calls["run"]
    assert isinstance(run_call, dict)
    assert run_call["args"] == (["/usr/bin/npm", "install", "--no-fund", "--no-audit"],)
    run_kwargs = cast(dict[str, object], run_call["kwargs"])
    assert run_kwargs["cwd"] == helper_dir
    assert sentinel.is_dir()


def test_ensure_node_runtime_dependencies_raises_clear_error_when_install_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper_dir = tmp_path / "embedding_runtime"
    helper_dir.mkdir()
    package_json = helper_dir / "package.json"
    package_json.write_text("{}", encoding="utf-8")
    sentinel = helper_dir / "node_modules" / "node-llama-cpp"

    monkeypatch.setattr(embedding_module, "EMBEDDING_RUNTIME_DIR", helper_dir)
    monkeypatch.setattr(embedding_module, "EMBEDDING_RUNTIME_PACKAGE", package_json)
    monkeypatch.setattr(embedding_module, "NODE_RUNTIME_SENTINEL", sentinel)
    monkeypatch.setattr("paperchat.services.embeddings.shutil.which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        "paperchat.services.embeddings.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="install failed"),
    )

    with pytest.raises(EmbeddingDependencyError, match="install failed"):
        _ensure_node_runtime_dependencies()
