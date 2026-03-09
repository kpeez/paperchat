import json
from pathlib import Path

from paperchat.benchmarks.docling_validation.models import FixtureDocument, GoldQuery


def load_fixture_documents(path: Path) -> tuple[FixtureDocument, ...]:
    raw_documents = _load_json_list(path)
    seen_doc_ids: set[str] = set()
    documents: list[FixtureDocument] = []

    for raw_document in raw_documents:
        doc_id = _require_str(raw_document, "doc_id", path)
        if doc_id in seen_doc_ids:
            msg = f"Duplicate doc_id {doc_id!r} in {path}."
            raise ValueError(msg)
        seen_doc_ids.add(doc_id)

        pdf_name = _require_str(raw_document, "pdf", path)
        documents.append(
            FixtureDocument(
                doc_id=doc_id,
                title=_require_str(raw_document, "title", path),
                pdf_path=(path.parent / pdf_name).resolve(),
                traits=tuple(_require_str_list(raw_document, "traits", path)),
            )
        )

    return tuple(documents)


def load_gold_queries(path: Path) -> tuple[GoldQuery, ...]:
    raw_queries = _load_json_list(path)
    seen_query_ids: set[str] = set()
    queries: list[GoldQuery] = []

    for raw_query in raw_queries:
        query_id = _require_str(raw_query, "query_id", path)
        if query_id in seen_query_ids:
            msg = f"Duplicate query_id {query_id!r} in {path}."
            raise ValueError(msg)
        seen_query_ids.add(query_id)

        queries.append(
            GoldQuery(
                query_id=query_id,
                doc_id=_require_str(raw_query, "doc_id", path),
                query=_require_str(raw_query, "query", path),
                expected_pages=tuple(_require_int_list(raw_query, "expected_pages", path)),
                expected_headings=tuple(_require_str_list(raw_query, "expected_headings", path)),
            )
        )

    return tuple(queries)


def _load_json_list(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        msg = f"Manifest file not found at {path}."
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Manifest file {path} must contain a top-level JSON list."
        raise ValueError(msg)

    records: list[dict[str, object]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            msg = f"Manifest item {index} in {path} must be an object."
            raise ValueError(msg)
        records.append(item)

    return records


def _require_str(record: dict[str, object], key: str, path: Path) -> str:
    value = record.get(key)
    if isinstance(value, str) and value:
        return value

    msg = f"Field {key!r} in {path} must be a non-empty string."
    raise ValueError(msg)


def _require_str_list(record: dict[str, object], key: str, path: Path) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"Field {key!r} in {path} must be a list of strings."
        raise ValueError(msg)

    return [item for item in value if isinstance(item, str)]


def _require_int_list(record: dict[str, object], key: str, path: Path) -> list[int]:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        msg = f"Field {key!r} in {path} must be a list of integers."
        raise ValueError(msg)

    return [item for item in value if isinstance(item, int)]
