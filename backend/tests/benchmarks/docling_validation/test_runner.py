import json
from pathlib import Path

import pytest

from paperchat.benchmarks.docling_validation.models import (
    ChunkerRun,
    FixtureRun,
    NormalizedChunk,
)
from paperchat.benchmarks.docling_validation.runner import run_validation


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_chunk(
    *,
    chunker_id: str,
    chunk_index: int,
    retrieval_text: str,
    page_numbers: tuple[int, ...],
    headings: tuple[str, ...],
) -> NormalizedChunk:
    return NormalizedChunk(
        doc_id="doc-1",
        chunker_id=chunker_id,
        chunk_index=chunk_index,
        text=retrieval_text,
        retrieval_text=retrieval_text,
        page_numbers=page_numbers,
        headings=headings,
        warning_codes=(),
    )


def test_run_validation_prefers_chunker_with_better_support_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixtures_path = tmp_path / "fixtures.json"
    queries_path = tmp_path / "gold_queries.json"
    output_root = tmp_path / "reports"
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    write_json(
        fixtures_path,
        [
            {
                "doc_id": "doc-1",
                "title": "Sample Paper",
                "pdf": "paper.pdf",
                "traits": ["two-column"],
            }
        ],
    )
    write_json(
        queries_path,
        [
            {
                "query_id": "q-1",
                "doc_id": "doc-1",
                "query": "transformer attention",
                "expected_pages": [1],
                "expected_headings": ["Introduction"],
            }
        ],
    )

    def fake_run_docling_fixture(_fixture) -> FixtureRun:
        return FixtureRun(
            doc_id="doc-1",
            title="Sample Paper",
            parser_ok=True,
            parse_seconds=0.1,
            parser_warning_codes=(),
            chunker_runs=(
                ChunkerRun(
                    chunker_id="hybrid",
                    chunk_seconds=0.2,
                    chunks=(
                        make_chunk(
                            chunker_id="hybrid",
                            chunk_index=0,
                            retrieval_text="transformer attention introduction",
                            page_numbers=(1,),
                            headings=("Introduction",),
                        ),
                    ),
                    warning_codes=(),
                ),
                ChunkerRun(
                    chunker_id="hierarchical",
                    chunk_seconds=0.1,
                    chunks=(
                        make_chunk(
                            chunker_id="hierarchical",
                            chunk_index=0,
                            retrieval_text="unrelated appendix text",
                            page_numbers=(3,),
                            headings=("Appendix",),
                        ),
                    ),
                    warning_codes=(),
                ),
            ),
        )

    monkeypatch.setattr(
        "paperchat.benchmarks.docling_validation.runner.run_docling_fixture",
        fake_run_docling_fixture,
    )

    benchmark_run, output_directory = run_validation(
        fixtures_path=fixtures_path,
        queries_path=queries_path,
        output_root=output_root,
        top_k=3,
    )

    assert benchmark_run.recommendation.status == "docling_approved_hybrid"
    assert benchmark_run.recommendation.recommended_chunker_id == "hybrid"
    assert (output_directory / "recommendation.json").is_file()
    assert (output_directory / "normalized_chunks.jsonl").is_file()


def test_run_validation_blocks_when_docling_parse_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixtures_path = tmp_path / "fixtures.json"
    queries_path = tmp_path / "gold_queries.json"
    output_root = tmp_path / "reports"
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    write_json(
        fixtures_path,
        [
            {
                "doc_id": "doc-1",
                "title": "Sample Paper",
                "pdf": "paper.pdf",
                "traits": ["two-column"],
            }
        ],
    )
    write_json(
        queries_path,
        [
            {
                "query_id": "q-1",
                "doc_id": "doc-1",
                "query": "transformer attention",
                "expected_pages": [1],
                "expected_headings": ["Introduction"],
            }
        ],
    )

    monkeypatch.setattr(
        "paperchat.benchmarks.docling_validation.runner.run_docling_fixture",
        lambda _fixture: FixtureRun(
            doc_id="doc-1",
            title="Sample Paper",
            parser_ok=False,
            parse_seconds=0.1,
            parser_warning_codes=(),
            chunker_runs=(),
            error="parse failed",
        ),
    )

    benchmark_run, _output_directory = run_validation(
        fixtures_path=fixtures_path,
        queries_path=queries_path,
        output_root=output_root,
        top_k=3,
    )

    assert benchmark_run.recommendation.status == "docling_blocked"
    assert benchmark_run.recommendation.recommended_chunker_id is None
