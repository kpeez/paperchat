from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FixtureDocument:
    doc_id: str
    title: str
    pdf_path: Path
    traits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldQuery:
    query_id: str
    doc_id: str
    query: str
    expected_pages: tuple[int, ...]
    expected_headings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NormalizedChunk:
    doc_id: str
    chunker_id: str
    chunk_index: int
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkerRun:
    chunker_id: str
    chunk_seconds: float
    chunks: tuple[NormalizedChunk, ...]
    warning_codes: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FixtureRun:
    doc_id: str
    title: str
    parser_ok: bool
    parse_seconds: float
    parser_warning_codes: tuple[str, ...]
    chunker_runs: tuple[ChunkerRun, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class QuerySummary:
    query_id: str
    chunker_id: str
    doc_id: str
    first_doc_hit_rank: int | None
    first_page_hit_rank: int | None
    first_heading_hit_rank: int | None
    first_support_hit_rank: int | None


@dataclass(frozen=True, slots=True)
class ChunkerSummary:
    chunker_id: str
    chunk_count: int
    average_chunk_words: float
    page_coverage: float
    heading_coverage: float
    doc_hit_rate_at_3: float
    page_hit_rate_at_3: float
    heading_hit_rate_at_3: float
    support_hit_rate_at_3: float
    mrr: float
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class Recommendation:
    status: str
    recommended_chunker_id: str | None
    rationale: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    fixture_runs: tuple[FixtureRun, ...]
    query_summaries: tuple[QuerySummary, ...]
    chunker_summaries: tuple[ChunkerSummary, ...]
    recommendation: Recommendation
