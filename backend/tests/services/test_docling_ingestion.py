import warnings
from types import SimpleNamespace

from paperchat.services.docling_ingestion import run_docling_parse


class ExampleWarning(Warning):
    pass


def make_chunk(
    *, text: str, contextualized: str, page_numbers: tuple[int, ...], headings: tuple[str, ...]
):
    doc_items = (
        SimpleNamespace(
            prov=tuple(SimpleNamespace(page_no=page_number) for page_number in page_numbers)
        ),
    )
    meta = SimpleNamespace(doc_items=doc_items, headings=headings)
    return SimpleNamespace(
        text=text,
        meta=meta,
        contextualize_result=contextualized,
    )


def test_run_docling_parse_uses_hierarchical_chunker_and_normalizes_output(monkeypatch) -> None:
    chunk = make_chunk(
        text="Chunk body",
        contextualized="Document context: Chunk body",
        page_numbers=(3, 1),
        headings=("Methods",),
    )

    class FakeChunker:
        def chunk(self, _document):
            return (chunk,)

        def contextualize(self, raw_chunk):
            assert raw_chunk is chunk
            return raw_chunk.contextualize_result

    class FakeConverter:
        def convert(self, _path):
            return SimpleNamespace(document="parsed-document")

    chunking_module = SimpleNamespace(HierarchicalChunker=FakeChunker)
    converter_module = SimpleNamespace(DocumentConverter=FakeConverter)

    def fake_import_module(name: str):
        if name == "docling.chunking":
            return chunking_module
        if name == "docling.document_converter":
            return converter_module
        msg = f"unexpected import {name}"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "paperchat.services.docling_ingestion.importlib.import_module", fake_import_module
    )

    result = run_docling_parse(pdf_path="/tmp/paper.pdf")

    assert result.parser_ok is True
    assert result.chunker_id == "hierarchical"
    assert result.error is None
    assert result.parser_warning_codes == ()
    assert len(result.chunks) == 1

    parsed_chunk = result.chunks[0]
    assert parsed_chunk.chunk_index == 0
    assert parsed_chunk.text == "Chunk body"
    assert parsed_chunk.retrieval_text == "Document context: Chunk body"
    assert parsed_chunk.page_numbers == (1, 3)
    assert parsed_chunk.headings == ("Methods",)
    assert parsed_chunk.warning_codes == ()


def test_run_docling_parse_returns_failure_result_when_converter_raises(monkeypatch) -> None:
    class FakeConverter:
        def convert(self, _path):
            raise RuntimeError("parse failed")

    chunking_module = SimpleNamespace(HierarchicalChunker=object)
    converter_module = SimpleNamespace(DocumentConverter=FakeConverter)

    def fake_import_module(name: str):
        if name == "docling.chunking":
            return chunking_module
        if name == "docling.document_converter":
            return converter_module
        msg = f"unexpected import {name}"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "paperchat.services.docling_ingestion.importlib.import_module", fake_import_module
    )

    result = run_docling_parse(pdf_path="/tmp/paper.pdf")

    assert result.parser_ok is False
    assert result.chunks == ()
    assert result.error == "parse failed"


def test_run_docling_parse_collects_parse_and_chunk_warnings(monkeypatch) -> None:
    chunk = make_chunk(
        text="Chunk body",
        contextualized="Chunk body",
        page_numbers=(),
        headings=(),
    )

    class FakeChunker:
        def chunk(self, _document):
            warnings.warn("chunk warning", ExampleWarning, stacklevel=1)
            return (chunk,)

        def contextualize(self, raw_chunk):
            warnings.warn("context warning", ExampleWarning, stacklevel=1)
            return raw_chunk.contextualize_result

    class FakeConverter:
        def convert(self, _path):
            warnings.warn("parse warning", ExampleWarning, stacklevel=1)
            return SimpleNamespace(document="parsed-document")

    chunking_module = SimpleNamespace(HierarchicalChunker=FakeChunker)
    converter_module = SimpleNamespace(DocumentConverter=FakeConverter)

    def fake_import_module(name: str):
        if name == "docling.chunking":
            return chunking_module
        if name == "docling.document_converter":
            return converter_module
        msg = f"unexpected import {name}"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "paperchat.services.docling_ingestion.importlib.import_module", fake_import_module
    )

    result = run_docling_parse(pdf_path="/tmp/paper.pdf")

    assert result.parser_ok is True
    assert result.parser_warning_codes == ("examplewarning",)
    assert result.chunks[0].warning_codes == (
        "examplewarning",
        "missing_headings",
        "missing_page_numbers",
    )
