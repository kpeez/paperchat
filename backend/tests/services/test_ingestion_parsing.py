from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from paperchat.services.ingestion_parsing import DoclingParser


def test_docling_parser_maps_low_level_parse_result(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_run_docling_parse(*, pdf_path: Path):
        assert pdf_path.name == "paper.pdf"
        return SimpleNamespace(
            parser_ok=True,
            parser_id="docling",
            chunker_id="hierarchical",
            parse_seconds=0.25,
            parser_warning_codes=("userwarning",),
            error=None,
            chunks=(
                SimpleNamespace(
                    chunk_index=0,
                    text="Body",
                    retrieval_text="Contextualized body",
                    page_numbers=(1,),
                    headings=("Intro",),
                    warning_codes=("missing_page_numbers",),
                ),
            ),
        )

    monkeypatch.setattr(
        "paperchat.services.ingestion_parsing.run_docling_parse",
        fake_run_docling_parse,
    )

    result = DoclingParser().parse_document(pdf_path=pdf_path)

    assert result.parser_id == "docling"
    assert result.chunker_id == "hierarchical"
    assert result.error is None
    assert result.parse_seconds == 0.25
    assert result.parser_warning_codes == ("userwarning",)
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Body"
    assert result.chunks[0].retrieval_text == "Contextualized body"
    assert result.chunks[0].page_numbers == (1,)
    assert result.chunks[0].headings == ("Intro",)
    assert result.chunks[0].warning_codes == ("missing_page_numbers",)


def test_docling_parser_returns_failure_result_when_parse_fails(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        "paperchat.services.ingestion_parsing.run_docling_parse",
        lambda *, pdf_path: SimpleNamespace(
            parser_ok=False,
            parser_id="docling",
            chunker_id="hierarchical",
            parse_seconds=0.1,
            parser_warning_codes=(),
            chunks=(),
            error="parse failed",
        ),
    )

    result = DoclingParser().parse_document(pdf_path=pdf_path)

    assert result.error == "parse failed"
    assert result.chunks == ()
