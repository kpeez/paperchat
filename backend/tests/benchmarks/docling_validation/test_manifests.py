import json
from pathlib import Path

import pytest

from paperchat_backend.benchmarks.docling_validation.manifests import (
    load_fixture_documents,
    load_gold_queries,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_fixture_documents_resolves_pdf_paths(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    manifest_path = tmp_path / "fixtures.json"
    write_json(
        manifest_path,
        [
            {
                "doc_id": "doc-1",
                "title": "Sample Paper",
                "pdf": "paper.pdf",
                "traits": ["two-column", "equations"],
            }
        ],
    )

    documents = load_fixture_documents(manifest_path)

    assert len(documents) == 1
    assert documents[0].pdf_path == pdf_path.resolve()
    assert documents[0].traits == ("two-column", "equations")


def test_load_gold_queries_validates_duplicate_ids(tmp_path: Path):
    manifest_path = tmp_path / "queries.json"
    write_json(
        manifest_path,
        [
            {
                "query_id": "q-1",
                "doc_id": "doc-1",
                "query": "attention is all you need",
                "expected_pages": [1],
                "expected_headings": ["Abstract"],
            },
            {
                "query_id": "q-1",
                "doc_id": "doc-1",
                "query": "duplicate id",
                "expected_pages": [2],
                "expected_headings": ["Introduction"],
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate query_id"):
        load_gold_queries(manifest_path)
