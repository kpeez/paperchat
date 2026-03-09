from __future__ import annotations

import importlib
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TypeVar
from uuid import UUID

T = TypeVar("T")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
DOCLING_PARSER_ID = "docling"
DOCLING_CHUNKER_ID = "hierarchical"


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    document_id: UUID
    chunk_index: int
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    parser_id: str
    chunker_id: str
    parse_seconds: float
    parser_warning_codes: tuple[str, ...]
    chunks: tuple[ParsedChunk, ...]


@dataclass(frozen=True, slots=True)
class DoclingChunk:
    chunk_index: int
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DoclingParseResult:
    parser_ok: bool
    parser_id: str
    chunker_id: str
    parse_seconds: float
    parser_warning_codes: tuple[str, ...]
    chunks: tuple[DoclingChunk, ...]
    error: str | None = None


class DoclingParseError(RuntimeError):
    """Raised when the Docling production parser cannot produce chunks."""


class DocumentParser(Protocol):
    def parse_document(self, *, document_id: UUID, pdf_path: Path) -> ParsedDocument: ...


class DoclingDocumentParser:
    """Production Docling parser using DocumentConverter and HierarchicalChunker."""

    def parse_document(self, *, document_id: UUID, pdf_path: Path) -> ParsedDocument:
        result = run_docling_parse(pdf_path=pdf_path)
        if not result.parser_ok:
            raise DoclingParseError(result.error or "Docling parsing failed.")

        chunks = tuple(
            ParsedChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                retrieval_text=chunk.retrieval_text,
                page_numbers=chunk.page_numbers,
                headings=chunk.headings,
                warning_codes=chunk.warning_codes,
            )
            for chunk in result.chunks
        )
        return ParsedDocument(
            parser_id=result.parser_id,
            chunker_id=result.chunker_id,
            parse_seconds=result.parse_seconds,
            parser_warning_codes=result.parser_warning_codes,
            chunks=chunks,
        )


def run_docling_parse(*, pdf_path: str | Path) -> DoclingParseResult:
    converter_class, chunker_factory = _load_docling_components()
    converter = converter_class()

    started = perf_counter()
    try:
        conversion_result, parser_warning_codes = _capture_operation(
            lambda: converter.convert(str(pdf_path))
        )
        document = getattr(conversion_result, "document", conversion_result)
        chunker = chunker_factory()
        raw_chunks, chunker_warning_codes = _capture_operation(
            lambda: tuple(chunker.chunk(document))
        )
        chunks = tuple(
            _normalize_chunk(
                chunk=raw_chunk,
                chunker=chunker,
                chunk_index=chunk_index,
                parser_warning_codes=parser_warning_codes,
                chunker_warning_codes=chunker_warning_codes,
            )
            for chunk_index, raw_chunk in enumerate(raw_chunks)
        )
    except Exception as error:
        return DoclingParseResult(
            parser_ok=False,
            parser_id=DOCLING_PARSER_ID,
            chunker_id=DOCLING_CHUNKER_ID,
            parse_seconds=perf_counter() - started,
            parser_warning_codes=(),
            chunks=(),
            error=str(error),
        )

    return DoclingParseResult(
        parser_ok=True,
        parser_id=DOCLING_PARSER_ID,
        chunker_id=DOCLING_CHUNKER_ID,
        parse_seconds=perf_counter() - started,
        parser_warning_codes=parser_warning_codes,
        chunks=chunks,
    )


def _normalize_chunk(
    *,
    chunk: Any,
    chunker: Any,
    chunk_index: int,
    parser_warning_codes: tuple[str, ...],
    chunker_warning_codes: tuple[str, ...],
) -> DoclingChunk:
    text = str(getattr(chunk, "text", "")).strip()
    retrieval_text, contextualize_warning_codes = _contextualize_chunk(
        chunker,
        chunk,
        fallback=text,
    )
    page_numbers = _extract_page_numbers(chunk)
    headings = _extract_headings(chunk)

    warning_codes = list(parser_warning_codes)
    warning_codes.extend(chunker_warning_codes)
    warning_codes.extend(contextualize_warning_codes)
    if not text:
        warning_codes.append("empty_text")
    if not retrieval_text:
        warning_codes.append("empty_retrieval_text")
    if not page_numbers:
        warning_codes.append("missing_page_numbers")
    if not headings:
        warning_codes.append("missing_headings")

    return DoclingChunk(
        chunk_index=chunk_index,
        text=text,
        retrieval_text=retrieval_text or text,
        page_numbers=page_numbers,
        headings=headings,
        warning_codes=tuple(sorted(set(warning_codes))),
    )


def _contextualize_chunk(
    chunker: Any,
    chunk: Any,
    *,
    fallback: str,
) -> tuple[str, tuple[str, ...]]:
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
    return tuple(str(heading).strip() for heading in raw_headings if str(heading).strip())


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


def _load_docling_components() -> tuple[type[Any], Callable[[], Any]]:
    try:
        chunking_module = importlib.import_module("docling.chunking")
        document_converter_module = importlib.import_module("docling.document_converter")
    except ImportError as error:
        msg = (
            "Docling is not installed. Run `uv sync --locked --group dev` in `backend/` before "
            "using the production ingestion path."
        )
        raise RuntimeError(msg) from error

    return document_converter_module.DocumentConverter, chunking_module.HierarchicalChunker
