"""Tests for the vector store module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from paperchat.core.vector_store import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    VectorStore,
    sanitize_model_identifier,
)

MAIN_FIELD_NAMES = {
    "pk",
    "chunk_id",
    "embedding",
    "text",
    "label",
    "page_numbers",
    "headings",
    "source",
}
METADATA_FIELD_NAMES = {"pk", "source", "original_filename", "date_added"}

DEFAULT_MODEL_ID = "pymilvus/default"
DEFAULT_MODEL_NAME = "GPTCache/paraphrase-albert-onnx"
DEFAULT_MODEL_DIM = 768
EXPLICIT_MODEL_ID = "openai/text-embedding-3-small"
EXPLICIT_MODEL_DIM = 1536


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "paperchat.db")


@pytest.fixture
def mock_milvus_client():
    """Create a mock MilvusClient."""
    with mock.patch("paperchat.core.vector_store.MilvusClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client.query.return_value = []
        mock_client.insert.return_value = {"insert_count": 1}
        mock_delete_result = mock.MagicMock()
        mock_delete_result.delete_count = 1
        mock_client.delete.return_value = mock_delete_result
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def vector_store_init_mocks():
    """Mocks dependencies for VectorStore initialization testing."""
    with (
        mock.patch("paperchat.core.vector_store.MilvusClient") as mock_client_class,
        mock.patch(
            "paperchat.core.vector_store.VectorStore._get_or_create_collection"
        ) as mock_get_create,
        mock.patch("paperchat.core.vector_store.VectorStore._build_indices") as mock_build_indices,
        mock.patch("paperchat.core.vector_store.get_component_logger") as mock_get_logger,
        mock.patch("paperchat.core.vector_store.get_embedding_model") as mock_get_model,
    ):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        mock_get_logger.return_value = mock.MagicMock()

        def mock_model_loader(model_id):
            mock_embed_fn = mock.MagicMock()
            if model_id == DEFAULT_MODEL_ID:
                mock_embed_fn.dim = DEFAULT_MODEL_DIM
                mock_embed_fn.model_name = DEFAULT_MODEL_NAME
            elif model_id == EXPLICIT_MODEL_ID:
                mock_embed_fn.dim = EXPLICIT_MODEL_DIM
            else:
                raise ValueError(f"Unexpected model_id in mock: {model_id}")
            return mock_embed_fn

        mock_get_model.side_effect = mock_model_loader

        yield {
            "mock_client_class": mock_client_class,
            "mock_get_create": mock_get_create,
            "mock_build_indices": mock_build_indices,
            "mock_get_model": mock_get_model,
        }


@pytest.fixture
def vector_store(mock_milvus_client, temp_db_path):
    """Create a VectorStore instance using the default model."""
    with (
        mock.patch("paperchat.core.vector_store.VectorStore._get_or_create_collection"),
        mock.patch("paperchat.core.vector_store.VectorStore._build_indices"),
        mock.patch("paperchat.core.vector_store.get_component_logger") as mock_get_logger,
        mock.patch("paperchat.core.vector_store.get_embedding_model") as mock_get_model,
    ):
        mock_get_logger.return_value = mock.MagicMock()
        mock_embed_fn = mock.MagicMock()
        mock_embed_fn.dim = DEFAULT_MODEL_DIM
        mock_embed_fn.model_name = DEFAULT_MODEL_NAME
        mock_embed_fn.encode_queries.return_value = [np.array([0.1] * DEFAULT_MODEL_DIM)]
        mock_get_model.return_value = mock_embed_fn

        store = VectorStore(db_path=temp_db_path, model_identifier=DEFAULT_MODEL_ID)
        store.client = mock_milvus_client
        store._embed_fn = mock_embed_fn
        store.main_schema = store._define_main_schema()
        store.metadata_schema = store._define_metadata_schema()
        yield store


def test_init_default_model(vector_store_init_mocks, temp_db_path):
    """Test initialization with the default model identifier."""
    mocks = vector_store_init_mocks
    mock_get_model = mocks["mock_get_model"]
    mock_get_create = mocks["mock_get_create"]
    mock_build_indices = mocks["mock_build_indices"]

    store = VectorStore(db_path=temp_db_path, model_identifier=DEFAULT_MODEL_ID)

    assert store.model_identifier == DEFAULT_MODEL_ID
    mock_get_model.assert_called_once_with(DEFAULT_MODEL_ID)
    assert store.embedding_dim == DEFAULT_MODEL_DIM

    expected_default_suffix = sanitize_model_identifier(DEFAULT_MODEL_NAME)
    expected_collection_name = f"paperchat_docs_{expected_default_suffix}"
    assert store.collection_name == expected_collection_name
    assert store.metadata_collection_name == VectorStore.METADATA_COLLECTION_NAME

    mock_get_create.assert_has_calls(
        [
            mock.call(expected_collection_name, mock.ANY),
            mock.call(store.metadata_collection_name, mock.ANY),
        ],
        any_order=True,
    )
    mock_build_indices.assert_called_once()


def test_init_explicit_model(vector_store_init_mocks, temp_db_path):
    """Test initialization with an explicit model identifier."""
    mocks = vector_store_init_mocks
    mock_get_model = mocks["mock_get_model"]
    mock_get_create = mocks["mock_get_create"]
    mock_build_indices = mocks["mock_build_indices"]

    store = VectorStore(db_path=temp_db_path, model_identifier=EXPLICIT_MODEL_ID)

    assert store.model_identifier == EXPLICIT_MODEL_ID
    mock_get_model.assert_called_once_with(EXPLICIT_MODEL_ID)
    assert store.embedding_dim == EXPLICIT_MODEL_DIM

    expected_explicit_suffix = sanitize_model_identifier(EXPLICIT_MODEL_ID)
    expected_collection_name = f"paperchat_docs_{expected_explicit_suffix}"
    assert store.collection_name == expected_collection_name
    assert store.metadata_collection_name == VectorStore.METADATA_COLLECTION_NAME

    mock_get_create.assert_has_calls(
        [
            mock.call(expected_collection_name, mock.ANY),
            mock.call(store.metadata_collection_name, mock.ANY),
        ],
        any_order=True,
    )
    mock_build_indices.assert_called_once()


@mock.patch("paperchat.core.vector_store.datetime")
def test_add_document_success(mock_dt, vector_store):
    """Test successfully adding a new document inserts chunks and metadata."""
    pdf_path = Path("/fake/path/to/document.pdf")
    source_id = pdf_path.stem
    mock_now = datetime(2023, 1, 1, 12, 0, 0)
    mock_dt.now.return_value = mock_now
    iso_timestamp = mock_now.isoformat()

    mock_chunk_data = [{"chunk_id": "chunk1", "embedding": [0.1], "text": "text1"}]
    expected_insert_data = [
        {"chunk_id": "chunk1", "embedding": [0.1], "text": "text1", "source": source_id}
    ]
    expected_metadata = [
        {
            "source": source_id,
            "original_filename": pdf_path.name,
            "date_added": iso_timestamp,
        }
    ]

    with mock.patch(
        "paperchat.core.vector_store.process_pdf_document", return_value=mock_chunk_data
    ) as mock_process:
        vector_store.client.reset_mock()
        vector_store.client.query.return_value = []
        vector_store.client.insert.side_effect = [
            {"insert_count": len(expected_insert_data)},
            {"insert_count": 1},
        ]

        result = vector_store.add_document(str(pdf_path), source_id=source_id)

        assert result is True
        vector_store.client.query.assert_called_once_with(
            collection_name=vector_store.metadata_collection_name,
            filter=f'source == "{source_id}"',
            output_fields=["source"],
            limit=1,
        )
        mock_process.assert_called_once()
        assert vector_store.client.insert.call_count == 2
        vector_store.client.insert.assert_has_calls(
            [
                mock.call(collection_name=vector_store.collection_name, data=expected_insert_data),
                mock.call(
                    collection_name=vector_store.metadata_collection_name, data=expected_metadata
                ),
            ]
        )


def test_add_document_already_exists(vector_store):
    """Test adding a document that already exists skips processing and insertion."""
    pdf_path = Path("/path/to/existing.pdf")
    source_id = pdf_path.stem
    vector_store.client.reset_mock()
    vector_store.client.query.return_value = [{"source": source_id}]

    with mock.patch("paperchat.core.vector_store.process_pdf_document") as mock_process:
        result = vector_store.add_document(str(pdf_path), source_id=source_id)

        assert result is True
        vector_store.client.query.assert_called_once_with(
            collection_name=vector_store.metadata_collection_name,
            filter=f'source == "{source_id}"',
            output_fields=["source"],
            limit=1,
        )
        mock_process.assert_not_called()
        vector_store.client.insert.assert_not_called()
        vector_store.logger.info.assert_any_call(
            f"Document ID '{source_id}' already exists in metadata. Skipping."
        )


@mock.patch("paperchat.core.vector_store.process_pdf_document", return_value=[{"text": "chunk"}])
def test_add_document_chunk_insert_failure(mock_process, vector_store):
    """Test failure during chunk insertion logs error and returns False."""
    pdf_path = Path("/path/to/fail_chunk.pdf")
    source_id = pdf_path.stem
    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()
    vector_store.client.query.return_value = []

    vector_store.client.insert.side_effect = [
        {"insert_count": 0, "primary_keys": []},
        {"insert_count": 1},
    ]

    result = vector_store.add_document(str(pdf_path), source_id=source_id)

    assert result is False
    vector_store.client.query.assert_called_once()
    mock_process.assert_called_once()
    assert vector_store.client.insert.call_count == 1
    vector_store.client.insert.assert_any_call(
        collection_name=vector_store.collection_name, data=mock.ANY
    )
    vector_store.logger.error.assert_called_once()
    assert (
        f"Mismatch in chunk insertion count for {source_id}"
        in vector_store.logger.error.call_args[0][0]
    )


@mock.patch("paperchat.core.vector_store.datetime")
@mock.patch("paperchat.core.vector_store.process_pdf_document", return_value=[{"text": "chunk"}])
def test_add_document_metadata_insert_failure(mock_process, mock_dt, vector_store):
    """Test failure during metadata insertion logs error and returns False."""
    pdf_path = Path("/path/to/fail_meta.pdf")
    source_id = pdf_path.stem
    mock_now = datetime(2023, 1, 1, 12, 0, 0)
    mock_dt.now.return_value = mock_now

    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()
    vector_store.client.query.return_value = []

    vector_store.client.insert.side_effect = [
        {"insert_count": 1},
        {"insert_count": 0},
    ]

    result = vector_store.add_document(str(pdf_path), source_id=source_id)

    assert result is False
    vector_store.client.query.assert_called_once()
    mock_process.assert_called_once()
    assert vector_store.client.insert.call_count == 2
    vector_store.client.insert.assert_has_calls(
        [
            mock.call(collection_name=vector_store.collection_name, data=mock.ANY),
            mock.call(collection_name=vector_store.metadata_collection_name, data=mock.ANY),
        ]
    )
    vector_store.logger.error.assert_called_once()
    assert f"Failed to insert metadata for {source_id}" in vector_store.logger.error.call_args[0][0]


def test_retrieve_success(vector_store):
    """Test retrieving documents successfully."""
    query_text = "test query"
    expected_results = [{"entity": {"text": "result", "pk": 1, "source": "s1"}, "distance": 0.9}]
    vector_store.client.reset_mock()
    mock_client_output = [expected_results]
    vector_store.client.search.return_value = mock_client_output

    output_fields = [f.name for f in vector_store.main_schema.fields if f.name != "embedding"]

    results = vector_store.search(query_text)

    assert results == expected_results
    vector_store.client.search.assert_called_once()
    call_args, call_kwargs = vector_store.client.search.call_args
    assert call_kwargs["collection_name"] == vector_store.collection_name
    assert len(call_kwargs["data"]) == 1
    assert call_kwargs["limit"] == DEFAULT_SIMILARITY_TOP_K
    assert call_kwargs["output_fields"] == output_fields
    assert call_kwargs["filter"] is None
    assert call_kwargs["search_params"]["radius"] == DEFAULT_SIMILARITY_THRESHOLD


def test_retrieve_with_filter_and_custom_params(vector_store):
    """Test retrieve with a filter expression and custom top_k/threshold."""
    query_text = "filtered query"
    filter_expr = "source == 'doc1'"
    custom_k = 3
    custom_threshold = 0.5
    custom_output_fields = ["text", "source"]
    expected_results = [{"entity": {"text": "filtered result", "source": "doc1"}, "distance": 0.6}]
    vector_store.client.reset_mock()
    mock_client_output = [expected_results]
    vector_store.client.search.return_value = mock_client_output

    results = vector_store.search(
        query_text,
        top_k=custom_k,
        threshold=custom_threshold,
        output_fields=custom_output_fields,
        filter_expression=filter_expr,
    )

    assert results == expected_results
    vector_store.client.search.assert_called_once()
    call_args, call_kwargs = vector_store.client.search.call_args
    assert call_kwargs["collection_name"] == vector_store.collection_name
    assert len(call_kwargs["data"]) == 1
    assert call_kwargs["limit"] == custom_k
    assert call_kwargs["output_fields"] == custom_output_fields
    assert call_kwargs["filter"] == filter_expr
    assert call_kwargs["search_params"]["radius"] == custom_threshold


def test_retrieve_embedding_failure(vector_store):
    """Test retrieve returns empty list if embedding fails."""
    vector_store._embed_fn.encode_queries.side_effect = Exception("Embedding error")
    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()

    results = vector_store.search("query")

    assert results == []
    vector_store.client.search.assert_not_called()
    vector_store.logger.exception.assert_called_once()


def test_retrieve_search_failure(vector_store):
    """Test retrieve returns empty list if search fails."""
    vector_store.client.search.side_effect = Exception("Search error")
    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()

    results = vector_store.search("query")

    assert results == []
    vector_store.client.search.assert_called_once()
    vector_store.logger.exception.assert_called_once()


def test_list_documents_success(vector_store):
    """Test listing documents successfully."""
    expected_metadata = [
        {"source": "doc1", "original_filename": "doc1.pdf", "date_added": "t1"},
        {"source": "doc2", "original_filename": "doc2.pdf", "date_added": "t2"},
    ]
    vector_store.client.reset_mock()
    vector_store.client.query.return_value = expected_metadata

    results = vector_store.list_documents()

    assert results == expected_metadata
    vector_store.client.query.assert_called_once_with(
        collection_name=vector_store.metadata_collection_name,
        output_fields=["source", "original_filename", "date_added"],
        limit=100,
    )


def test_list_documents_failure(vector_store):
    """Test listing documents returns empty list on failure."""
    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()
    vector_store.client.query.side_effect = Exception("Query failed")

    results = vector_store.list_documents()

    assert results == []
    vector_store.client.query.assert_called_once()
    vector_store.logger.exception.assert_called_once()


def test_remove_document_success(vector_store):
    """Test removing a document successfully deletes from both collections."""
    source_id = "doc_to_remove"
    vector_store.client.reset_mock()

    mock_meta_delete = mock.MagicMock()
    mock_meta_delete.delete_count = 1
    mock_chunk_delete = mock.MagicMock()
    mock_chunk_delete.delete_count = 5
    vector_store.client.delete.side_effect = [mock_meta_delete, mock_chunk_delete]

    meta_deleted, chunks_deleted = vector_store.remove_document(source_id)

    assert meta_deleted == 1
    assert chunks_deleted == 5
    assert vector_store.client.delete.call_count == 2
    expected_filter = f'source == "{source_id}"'
    vector_store.client.delete.assert_has_calls(
        [
            mock.call(
                collection_name=vector_store.metadata_collection_name, filter=expected_filter
            ),
            mock.call(collection_name=vector_store.collection_name, filter=expected_filter),
        ]
    )


def test_remove_document_not_found(vector_store):
    """Test removing a non-existent document returns (0, 0)."""
    source_id = "doc_not_found"
    vector_store.client.reset_mock()

    mock_zero_delete = mock.MagicMock()
    mock_zero_delete.delete_count = 0
    vector_store.client.delete.return_value = mock_zero_delete

    meta_deleted, chunks_deleted = vector_store.remove_document(source_id)

    assert meta_deleted == 0
    assert chunks_deleted == 0
    assert vector_store.client.delete.call_count == 2


def test_remove_document_metadata_delete_failure(vector_store):
    """Test failure during metadata deletion logs error, attempts chunk deletion, returns 0 meta count."""
    source_id = "fail_meta_delete"
    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()

    mock_chunk_delete = mock.MagicMock()
    mock_chunk_delete.delete_count = 5
    vector_store.client.delete.side_effect = [
        Exception("Metadata delete failed"),
        mock_chunk_delete,
    ]

    meta_deleted, chunks_deleted = vector_store.remove_document(source_id)

    assert meta_deleted == 0
    assert chunks_deleted == 5
    assert vector_store.client.delete.call_count == 2
    vector_store.logger.exception.assert_called_once()
    assert (
        f"Failed to remove from metadata collection for '{source_id}'"
        in vector_store.logger.exception.call_args[0][0]
    )


def test_remove_document_chunk_delete_failure(vector_store):
    """Test failure during chunk deletion logs error and returns 0 chunk count."""
    source_id = "fail_chunk_delete"
    vector_store.client.reset_mock()
    vector_store.logger.reset_mock()

    mock_meta_delete = mock.MagicMock()
    mock_meta_delete.delete_count = 1
    vector_store.client.delete.side_effect = [mock_meta_delete, Exception("Chunk delete failed")]

    meta_deleted, chunks_deleted = vector_store.remove_document(source_id)

    assert meta_deleted == 1
    assert chunks_deleted == 0
    assert vector_store.client.delete.call_count == 2
    vector_store.logger.exception.assert_called_once()
    assert (
        f"Failed to remove chunks from main collection for '{source_id}'"
        in vector_store.logger.exception.call_args[0][0]
    )


@pytest.fixture(autouse=True)
def reset_delete_mock_attributes(mock_milvus_client):
    yield
    if hasattr(mock_milvus_client.delete.return_value, "delete_count"):
        pass
    if hasattr(mock_milvus_client.delete.return_value, "primary_keys"):
        pass
