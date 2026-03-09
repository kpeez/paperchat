import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from paperchat.benchmarks.docling_validation.docling_adapter import run_docling_fixture
from paperchat.benchmarks.docling_validation.manifests import (
    load_fixture_documents,
    load_gold_queries,
)
from paperchat.benchmarks.docling_validation.models import (
    BenchmarkRun,
    ChunkerSummary,
    FixtureRun,
    GoldQuery,
    QuerySummary,
    Recommendation,
)
from paperchat.benchmarks.docling_validation.retrieval import rank_chunks
from paperchat.config import BACKEND_ROOT

DEFAULT_FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "docling_validation"
DEFAULT_FIXTURES_PATH = DEFAULT_FIXTURE_ROOT / "fixtures.json"
DEFAULT_QUERIES_PATH = DEFAULT_FIXTURE_ROOT / "gold_queries.json"
DEFAULT_OUTPUT_ROOT = Path("/tmp/paperchat-docling-validation")
DEFAULT_TOP_K = 3


def run_validation(
    *,
    fixtures_path: Path = DEFAULT_FIXTURES_PATH,
    queries_path: Path = DEFAULT_QUERIES_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[BenchmarkRun, Path]:
    fixtures = load_fixture_documents(fixtures_path)
    queries = load_gold_queries(queries_path)
    fixture_runs = tuple(run_docling_fixture(fixture) for fixture in fixtures)
    query_summaries = _build_query_summaries(queries=queries, fixture_runs=fixture_runs)
    chunker_summaries = _build_chunker_summaries(
        fixture_runs=fixture_runs,
        query_summaries=query_summaries,
        top_k=top_k,
    )
    recommendation = _build_recommendation(
        fixture_runs=fixture_runs,
        chunker_summaries=chunker_summaries,
    )
    benchmark_run = BenchmarkRun(
        fixture_runs=fixture_runs,
        query_summaries=query_summaries,
        chunker_summaries=chunker_summaries,
        recommendation=recommendation,
    )
    output_directory = _write_reports(
        benchmark_run=benchmark_run,
        fixtures_path=fixtures_path,
        queries_path=queries_path,
        output_root=output_root,
    )
    return benchmark_run, output_directory


def _build_query_summaries(
    *,
    queries: tuple[GoldQuery, ...],
    fixture_runs: tuple[FixtureRun, ...],
) -> tuple[QuerySummary, ...]:
    summaries: list[QuerySummary] = []
    chunker_ids = sorted(
        {
            chunker_run.chunker_id
            for fixture_run in fixture_runs
            for chunker_run in fixture_run.chunker_runs
        }
    )
    chunks_by_chunker = {
        chunker_id: tuple(
            chunk
            for fixture_run in fixture_runs
            for chunker_run in fixture_run.chunker_runs
            if chunker_run.chunker_id == chunker_id
            for chunk in chunker_run.chunks
        )
        for chunker_id in chunker_ids
    }

    for chunker_id, chunks in chunks_by_chunker.items():
        for query in queries:
            ranked_chunks = rank_chunks(query.query, chunks)
            expected_pages = set(query.expected_pages)
            expected_headings = query.expected_headings
            doc_id = query.doc_id
            first_doc_hit_rank = _first_rank(
                ranked_chunks,
                lambda item, doc_id=doc_id: item.chunk.doc_id == doc_id,
            )
            first_page_hit_rank = _first_rank(
                ranked_chunks,
                lambda item, doc_id=doc_id, expected_pages=expected_pages: (
                    item.chunk.doc_id == doc_id
                    and bool(set(item.chunk.page_numbers) & expected_pages)
                ),
            )
            first_heading_hit_rank = _first_rank(
                ranked_chunks,
                lambda item, doc_id=doc_id, expected_headings=expected_headings: (
                    item.chunk.doc_id == doc_id
                    and _heading_overlap(item.chunk.headings, expected_headings)
                ),
            )
            first_support_hit_rank = _support_rank(
                first_doc_hit_rank=first_doc_hit_rank,
                first_page_hit_rank=first_page_hit_rank,
                first_heading_hit_rank=first_heading_hit_rank,
            )
            summaries.append(
                QuerySummary(
                    query_id=query.query_id,
                    chunker_id=chunker_id,
                    doc_id=query.doc_id,
                    first_doc_hit_rank=first_doc_hit_rank,
                    first_page_hit_rank=first_page_hit_rank,
                    first_heading_hit_rank=first_heading_hit_rank,
                    first_support_hit_rank=first_support_hit_rank,
                )
            )

    return tuple(summaries)


def _build_chunker_summaries(
    *,
    fixture_runs: tuple[FixtureRun, ...],
    query_summaries: tuple[QuerySummary, ...],
    top_k: int,
) -> tuple[ChunkerSummary, ...]:
    summaries: list[ChunkerSummary] = []
    chunker_ids = sorted(
        {
            chunker_run.chunker_id
            for fixture_run in fixture_runs
            for chunker_run in fixture_run.chunker_runs
        }
    )

    for chunker_id in chunker_ids:
        chunks = tuple(
            chunk
            for fixture_run in fixture_runs
            for chunker_run in fixture_run.chunker_runs
            if chunker_run.chunker_id == chunker_id
            for chunk in chunker_run.chunks
        )
        chunker_queries = tuple(
            query_summary
            for query_summary in query_summaries
            if query_summary.chunker_id == chunker_id
        )
        runtime_seconds = sum(
            chunker_run.chunk_seconds
            for fixture_run in fixture_runs
            for chunker_run in fixture_run.chunker_runs
            if chunker_run.chunker_id == chunker_id
        )

        chunk_count = len(chunks)
        average_chunk_words = mean(len(chunk.text.split()) for chunk in chunks) if chunks else 0.0
        page_coverage = _coverage(chunks, lambda chunk: bool(chunk.page_numbers))
        heading_coverage = _coverage(chunks, lambda chunk: bool(chunk.headings))

        summaries.append(
            ChunkerSummary(
                chunker_id=chunker_id,
                chunk_count=chunk_count,
                average_chunk_words=average_chunk_words,
                page_coverage=page_coverage,
                heading_coverage=heading_coverage,
                doc_hit_rate_at_3=_hit_rate(chunker_queries, "first_doc_hit_rank", top_k),
                page_hit_rate_at_3=_hit_rate(chunker_queries, "first_page_hit_rank", top_k),
                heading_hit_rate_at_3=_hit_rate(chunker_queries, "first_heading_hit_rank", top_k),
                support_hit_rate_at_3=_hit_rate(chunker_queries, "first_support_hit_rank", top_k),
                mrr=_mrr(chunker_queries),
                runtime_seconds=runtime_seconds,
            )
        )

    return tuple(summaries)


def _build_recommendation(
    *,
    fixture_runs: tuple[FixtureRun, ...],
    chunker_summaries: tuple[ChunkerSummary, ...],
) -> Recommendation:
    failed_documents = tuple(
        fixture_run.doc_id for fixture_run in fixture_runs if not fixture_run.parser_ok
    )
    if failed_documents:
        return Recommendation(
            status="docling_blocked",
            recommended_chunker_id=None,
            rationale=(
                "Docling failed to parse every required fixture successfully.",
                f"Hard parser failures: {', '.join(failed_documents)}.",
                "Reopen parser comparison before PR 4 hardcodes the ingestion path.",
            ),
        )

    viable_summaries = tuple(summary for summary in chunker_summaries if summary.chunk_count > 0)
    if not viable_summaries:
        return Recommendation(
            status="docling_blocked",
            recommended_chunker_id=None,
            rationale=(
                "Docling parsed the fixtures but produced no usable chunks.",
                "Reopen parser comparison before PR 4 continues.",
            ),
        )

    recommended = max(
        viable_summaries,
        key=lambda summary: (
            summary.page_hit_rate_at_3,
            summary.heading_hit_rate_at_3,
            summary.support_hit_rate_at_3,
            summary.doc_hit_rate_at_3,
            summary.mrr,
            -summary.runtime_seconds,
            1 if summary.chunker_id == "hybrid" else 0,
        ),
    )

    status = (
        "docling_approved_hybrid"
        if recommended.chunker_id == "hybrid"
        else "docling_approved_alternate_chunker"
    )
    rationale = (
        f"Docling parsed all fixtures successfully; PR 4 can continue with `{recommended.chunker_id}`.",
        "Chunker choice was ranked by page fidelity first, heading fidelity second, retrieval third, and runtime as the tie-breaker.",
        f"`{recommended.chunker_id}` achieved page_hit_rate@3={recommended.page_hit_rate_at_3:.3f}, heading_hit_rate@3={recommended.heading_hit_rate_at_3:.3f}, and mrr={recommended.mrr:.3f}.",
    )
    return Recommendation(
        status=status,
        recommended_chunker_id=recommended.chunker_id,
        rationale=rationale,
    )


def _write_reports(
    *,
    benchmark_run: BenchmarkRun,
    fixtures_path: Path,
    queries_path: Path,
    output_root: Path,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_directory = output_root / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)

    _write_json(
        output_directory / "manifest.json",
        {
            "fixtures_path": str(fixtures_path),
            "queries_path": str(queries_path),
            "generated_at_utc": timestamp,
        },
    )
    _write_json(
        output_directory / "fixture_runs.json",
        [asdict(fixture_run) for fixture_run in benchmark_run.fixture_runs],
    )
    _write_json(
        output_directory / "query_summaries.json",
        [asdict(query_summary) for query_summary in benchmark_run.query_summaries],
    )
    _write_json(
        output_directory / "chunker_summaries.json",
        [asdict(summary) for summary in benchmark_run.chunker_summaries],
    )
    _write_json(
        output_directory / "recommendation.json",
        asdict(benchmark_run.recommendation),
    )
    with (output_directory / "normalized_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for fixture_run in benchmark_run.fixture_runs:
            for chunker_run in fixture_run.chunker_runs:
                for chunk in chunker_run.chunks:
                    handle.write(json.dumps(asdict(chunk), sort_keys=True))
                    handle.write("\n")
    (output_directory / "recommendation.md").write_text(
        _recommendation_markdown(benchmark_run.recommendation),
        encoding="utf-8",
    )
    return output_directory


def _recommendation_markdown(recommendation: Recommendation) -> str:
    lines = [f"# {recommendation.status.replace('_', ' ').title()}"]
    if recommendation.recommended_chunker_id is not None:
        lines.append("")
        lines.append(f"Recommended chunker: `{recommendation.recommended_chunker_id}`")

    for reason in recommendation.rationale:
        lines.append("")
        lines.append(f"- {reason}")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _first_rank(scored_chunks: tuple[object, ...], predicate) -> int | None:
    for index, scored_chunk in enumerate(scored_chunks, start=1):
        if predicate(scored_chunk):
            return index
    return None


def _support_rank(
    *,
    first_doc_hit_rank: int | None,
    first_page_hit_rank: int | None,
    first_heading_hit_rank: int | None,
) -> int | None:
    candidates = [
        rank for rank in (first_page_hit_rank, first_heading_hit_rank) if rank is not None
    ]
    if candidates:
        return min(candidates)
    return first_doc_hit_rank


def _heading_overlap(chunk_headings: tuple[str, ...], expected_headings: tuple[str, ...]) -> bool:
    normalized_chunk_headings = tuple(heading.lower() for heading in chunk_headings)
    normalized_expected = tuple(heading.lower() for heading in expected_headings)
    return any(
        expected in chunk_heading or chunk_heading in expected
        for expected in normalized_expected
        for chunk_heading in normalized_chunk_headings
    )


def _coverage(chunks, predicate) -> float:
    if not chunks:
        return 0.0
    return sum(1 for chunk in chunks if predicate(chunk)) / len(chunks)


def _hit_rate(query_summaries: tuple[QuerySummary, ...], field_name: str, top_k: int) -> float:
    if not query_summaries:
        return 0.0
    hits = sum(
        1
        for summary in query_summaries
        if (rank := getattr(summary, field_name)) is not None and rank <= top_k
    )
    return hits / len(query_summaries)


def _mrr(query_summaries: tuple[QuerySummary, ...]) -> float:
    if not query_summaries:
        return 0.0
    return sum(
        0.0 if summary.first_support_hit_rank is None else 1 / summary.first_support_hit_rank
        for summary in query_summaries
    ) / len(query_summaries)
