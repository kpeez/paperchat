import importlib
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from paperchat_backend.benchmarks.docling_validation.models import (
    ChunkerRun,
    FixtureDocument,
    FixtureRun,
    NormalizedChunk,
)

T = TypeVar("T")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def run_docling_fixture(fixture: FixtureDocument) -> FixtureRun:
    converter_class, chunker_factories = _load_docling_components()
    converter = converter_class()

    parse_started = perf_counter()
    try:
        conversion_result, parser_warning_codes = _capture_operation(
            lambda: converter.convert(str(fixture.pdf_path))
        )
    except Exception as error:  # pragma: no cover - exercised with Docling installed.
        return FixtureRun(
            doc_id=fixture.doc_id,
            title=fixture.title,
            parser_ok=False,
            parse_seconds=perf_counter() - parse_started,
            parser_warning_codes=(),
            chunker_runs=(),
            error=str(error),
        )

    document = getattr(conversion_result, "document", conversion_result)
    chunker_runs = tuple(
        _run_chunker(
            document=document, doc_id=fixture.doc_id, chunker_id=chunker_id, factory=factory
        )
        for chunker_id, factory in chunker_factories.items()
    )
    return FixtureRun(
        doc_id=fixture.doc_id,
        title=fixture.title,
        parser_ok=True,
        parse_seconds=perf_counter() - parse_started,
        parser_warning_codes=parser_warning_codes,
        chunker_runs=chunker_runs,
    )


def _run_chunker(
    *,
    document: Any,
    doc_id: str,
    chunker_id: str,
    factory: Callable[[], Any],
) -> ChunkerRun:
    started = perf_counter()
    try:
        chunker = factory()
        raw_chunks, warning_codes = _capture_operation(lambda: tuple(chunker.chunk(document)))
        chunks = tuple(
            _normalize_chunk(
                chunk=raw_chunk,
                chunker=chunker,
                doc_id=doc_id,
                chunker_id=chunker_id,
                chunk_index=chunk_index,
            )
            for chunk_index, raw_chunk in enumerate(raw_chunks)
        )
    except Exception as error:  # pragma: no cover - exercised with Docling installed.
        return ChunkerRun(
            chunker_id=chunker_id,
            chunk_seconds=perf_counter() - started,
            chunks=(),
            warning_codes=(),
            error=str(error),
        )

    return ChunkerRun(
        chunker_id=chunker_id,
        chunk_seconds=perf_counter() - started,
        chunks=chunks,
        warning_codes=warning_codes,
    )


def _normalize_chunk(
    *,
    chunk: Any,
    chunker: Any,
    doc_id: str,
    chunker_id: str,
    chunk_index: int,
) -> NormalizedChunk:
    text = str(getattr(chunk, "text", "")).strip()
    retrieval_text, contextualize_warning_codes = _contextualize_chunk(
        chunker, chunk, fallback=text
    )
    page_numbers = _extract_page_numbers(chunk)
    headings = _extract_headings(chunk)

    warning_codes = list(contextualize_warning_codes)
    if not text:
        warning_codes.append("empty_text")
    if not retrieval_text:
        warning_codes.append("empty_retrieval_text")
    if not page_numbers:
        warning_codes.append("missing_page_numbers")
    if not headings:
        warning_codes.append("missing_headings")

    return NormalizedChunk(
        doc_id=doc_id,
        chunker_id=chunker_id,
        chunk_index=chunk_index,
        text=text,
        retrieval_text=retrieval_text,
        page_numbers=page_numbers,
        headings=headings,
        warning_codes=tuple(sorted(set(warning_codes))),
    )


def _contextualize_chunk(chunker: Any, chunk: Any, *, fallback: str) -> tuple[str, tuple[str, ...]]:
    contextualize = getattr(chunker, "contextualize", None)
    if not callable(contextualize):
        return fallback, ()

    try:
        contextualized_text, warning_codes = _capture_operation(lambda: str(contextualize(chunk)))
    except Exception:
        return fallback, ("contextualize_failed",)

    return contextualized_text.strip() or fallback, warning_codes


def _extract_page_numbers(chunk: Any) -> tuple[int, ...]:
    meta = getattr(chunk, "meta", None)
    doc_items = getattr(meta, "doc_items", ()) or ()
    page_numbers: set[int] = set()
    for item in doc_items:
        for provenance in getattr(item, "prov", ()) or ():
            page_number = getattr(provenance, "page_no", None)
            if isinstance(page_number, int):
                page_numbers.add(page_number)
    return tuple(sorted(page_numbers))


def _extract_headings(chunk: Any) -> tuple[str, ...]:
    meta = getattr(chunk, "meta", None)
    raw_headings = getattr(meta, "headings", ()) or ()
    headings = tuple(str(heading).strip() for heading in raw_headings if str(heading).strip())
    return headings


def _capture_operation(operation: Callable[[], T]) -> tuple[T, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = operation()
    warning_codes = tuple(sorted({_warning_code(record) for record in caught_warnings}))
    return result, warning_codes


def _warning_code(record: warnings.WarningMessage) -> str:
    category_name = getattr(record.category, "__name__", "warning")
    slug = SLUG_PATTERN.sub("_", category_name.lower()).strip("_")
    return slug or "warning"


def _load_docling_components() -> tuple[type[Any], dict[str, Callable[[], Any]]]:
    try:
        chunking_module = importlib.import_module("docling.chunking")
        document_converter_module = importlib.import_module("docling.document_converter")
    except ImportError as error:  # pragma: no cover - depends on benchmark dependency install.
        msg = (
            "Docling is not installed. Run `uv sync --locked --group dev` in `backend/` "
            "before executing the Docling validation harness."
        )
        raise RuntimeError(msg) from error

    return document_converter_module.DocumentConverter, {
        "hybrid": lambda: _build_hybrid_chunker(chunking_module),
        "hierarchical": chunking_module.HierarchicalChunker,
    }


def _build_hybrid_chunker(chunking_module: Any) -> Any:
    tokenizer = _load_cached_hybrid_tokenizer()
    if tokenizer is None:
        return chunking_module.HybridChunker()

    return chunking_module.HybridChunker(tokenizer=tokenizer)


def _load_cached_hybrid_tokenizer() -> Any | None:
    try:
        huggingface_module = importlib.import_module(
            "docling_core.transforms.chunker.tokenizer.huggingface"
        )
    except ImportError:  # pragma: no cover - depends on installed Docling extras.
        return None

    snapshot_path = _find_cached_sentence_transformer_snapshot()
    if snapshot_path is None:
        return None

    tokenizer = huggingface_module.HuggingFaceTokenizer.from_pretrained(
        model_name=snapshot_path,
        max_tokens=256,
    )
    inner_tokenizer = tokenizer.get_tokenizer()
    if hasattr(inner_tokenizer, "model_max_length"):
        inner_tokenizer.model_max_length = 10**9

    return tokenizer


def _find_cached_sentence_transformer_snapshot() -> Path | None:
    roots = _hugging_face_roots()
    for root in roots:
        snapshots_root = root / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
        if not snapshots_root.is_dir():
            continue

        for snapshot_path in sorted(snapshots_root.iterdir()):
            if snapshot_path.is_dir() and _has_hybrid_tokenizer_files(snapshot_path):
                return snapshot_path

    return None


def _hugging_face_roots() -> tuple[Path, ...]:
    raw_roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path.home() / ".cache" / "huggingface"),
    ]

    roots: list[Path] = []
    for raw_root in raw_roots:
        if not raw_root:
            continue

        root = Path(raw_root).expanduser()
        roots.append(root)
        roots.append(root / "hub")

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(resolved)

    return tuple(unique_roots)


def _has_hybrid_tokenizer_files(snapshot_path: Path) -> bool:
    required_files = ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
    return all((snapshot_path / filename).is_file() for filename in required_files)
