from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from paperchat.services.docling_ingestion import run_docling_parse

PARSER_ID = "docling"
CHUNKER_ID = "hierarchical"


@dataclass(frozen=True, slots=True)
class IngestionChunk:
    chunk_index: int
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParseResult:
    parser_id: str
    chunker_id: str
    parse_seconds: float
    parser_warning_codes: tuple[str, ...]
    chunks: tuple[IngestionChunk, ...]
    error: str | None = None


class DoclingParser:
    """Compatibility wrapper around the production Docling parsing path."""

    def parse_document(
        self,
        *,
        pdf_path: Path,
        document_id: UUID | None = None,
    ) -> ParseResult:
        result = run_docling_parse(pdf_path=pdf_path)
        chunks = tuple(
            IngestionChunk(
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                retrieval_text=chunk.retrieval_text,
                page_numbers=chunk.page_numbers,
                headings=chunk.headings,
                warning_codes=chunk.warning_codes,
            )
            for chunk in result.chunks
        )
        return ParseResult(
            parser_id=result.parser_id,
            chunker_id=result.chunker_id,
            parse_seconds=result.parse_seconds,
            parser_warning_codes=result.parser_warning_codes,
            chunks=chunks,
            error=result.error,
        )
