from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from paperchat.config import get_embedding_model_name, get_model_cache_dir

DEFAULT_EMBEDDING_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
BACKEND_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_RUNTIME_DIR = BACKEND_ROOT / "embedding_runtime"
EMBEDDING_RUNTIME_SCRIPT = EMBEDDING_RUNTIME_DIR / "embedder.mjs"
EMBEDDING_RUNTIME_PACKAGE = EMBEDDING_RUNTIME_DIR / "package.json"
NODE_RUNTIME_SENTINEL = EMBEDDING_RUNTIME_DIR / "node_modules" / "node-llama-cpp"


class EmbeddingService(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class EmbeddingRuntimeError(RuntimeError):
    """Raised when the local embedding runtime cannot serve embeddings."""


class EmbeddingDependencyError(EmbeddingRuntimeError):
    """Raised when optional local embedding dependencies are unavailable."""


class EmbeddingGemmaEmbedder:
    """Local EmbeddingGemma adapter backed by qmd-style GGUF inference."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        embed_runner: Any | None = None,
    ) -> None:
        self._model_name = model_name or get_embedding_model_name() or DEFAULT_EMBEDDING_MODEL
        self._embed_runner = embed_runner or _embed_with_node_runtime
        self._model_cache_dir = _ensure_model_cache_dir()

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()

        formatted_texts = tuple(_format_document_for_embedding(text) for text in texts)
        return self._embed_runner(formatted_texts, self._model_name, self._model_cache_dir)

    def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = self._embed_runner(
            (_format_query_for_embedding(text),),
            self._model_name,
            self._model_cache_dir,
        )
        return vectors[0]

    @property
    def model_name(self) -> str:
        return self._model_name


EmbeddingGemmaEmbeddingService = EmbeddingGemmaEmbedder


def _embed_with_node_runtime(
    texts: Sequence[str],
    model_name: str,
    model_cache_dir: Path,
) -> tuple[tuple[float, ...], ...]:
    node_binary = shutil.which("node")
    if node_binary is None:
        msg = (
            "Local EmbeddingGemma GGUF inference requires Node.js because PaperChat uses the "
            "`node-llama-cpp` runtime for this model."
        )
        raise EmbeddingDependencyError(msg)

    _ensure_node_runtime_dependencies()
    request = json.dumps(
        {
            "model": model_name,
            "model_cache_dir": str(model_cache_dir),
            "texts": list(texts),
        }
    )
    result = subprocess.run(
        [node_binary, str(EMBEDDING_RUNTIME_SCRIPT)],
        input=request,
        capture_output=True,
        cwd=EMBEDDING_RUNTIME_DIR,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise EmbeddingRuntimeError(_model_load_error_message(model_name, RuntimeError(detail)))

    try:
        payload = json.loads(_extract_runtime_json(result.stdout))
        vectors = payload["vectors"]
    except (KeyError, json.JSONDecodeError, TypeError) as error:
        msg = "EmbeddingGemma runtime returned an invalid response."
        raise EmbeddingRuntimeError(msg) from error

    return tuple(_vector_to_tuple(vector) for vector in vectors)


def _ensure_node_runtime_dependencies() -> None:
    if NODE_RUNTIME_SENTINEL.exists():
        return

    if not EMBEDDING_RUNTIME_PACKAGE.is_file():
        msg = f"Missing embedding runtime package metadata at {EMBEDDING_RUNTIME_PACKAGE}."
        raise EmbeddingDependencyError(msg)

    npm_binary = shutil.which("npm")
    if npm_binary is None:
        msg = (
            "Local EmbeddingGemma GGUF inference requires npm to install the bundled "
            "`node-llama-cpp` runtime."
        )
        raise EmbeddingDependencyError(msg)

    result = subprocess.run(
        [npm_binary, "install", "--no-fund", "--no-audit"],
        capture_output=True,
        cwd=EMBEDDING_RUNTIME_DIR,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        msg = f"Installing the local EmbeddingGemma runtime failed: {detail}"
        raise EmbeddingDependencyError(msg)


def _ensure_model_cache_dir() -> Path:
    cache_dir = get_model_cache_dir().expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _format_query_for_embedding(query: str) -> str:
    return f"task: search result | query: {query}"


def _format_document_for_embedding(text: str) -> str:
    return f"title: none | text: {text}"


def _model_load_error_message(model_name: str, error: Exception) -> str:
    lowered = str(error).lower()
    if "gated repo" in lowered or "access to model" in lowered:
        return (
            f"EmbeddingGemma GGUF download failed for `{model_name}`. "
            "PaperChat expects a public GGUF model artifact and will download it into the local "
            "PaperChat cache on first use."
        )

    return (
        "EmbeddingGemma could not be loaded. Ensure Node.js is available, the bundled "
        "`node-llama-cpp` runtime can be installed, and the configured GGUF model can be "
        "downloaded into the PaperChat cache."
    )


def _vector_to_tuple(vector: Any) -> tuple[float, ...]:
    raw_values = vector.tolist() if hasattr(vector, "tolist") else vector
    return tuple(float(value) for value in raw_values)


def _extract_runtime_json(stdout: str) -> str:
    json_start = stdout.rfind('{"vectors"')
    if json_start == -1:
        return stdout
    return stdout[json_start:]
