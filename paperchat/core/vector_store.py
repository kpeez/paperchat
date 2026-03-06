"""Vector storage implementation using Milvus."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from paperchat.core.embeddings import get_embedding_model

from ..utils.logging import get_component_logger
from .ingestion import (
    process_pdf_document,
)

DEFAULT_SIMILARITY_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.2


class VectorStore:
    """
    Handles interactions with a Milvus vector store collection for document chunks.

    Attributes:
        db_path (str): Path to the Milvus Lite database file.
        collection_name (str): Name of the collection managed by this instance (dynamic based on model).
        model_identifier (str): Identifier of the embedding model being used.
        metadata_collection_name (str): Name of the shared metadata collection.
        client (MilvusClient): The initialized Milvus client.
        _embed_fn (Any): The loaded embedding function instance.
        embedding_dim (int): Dimension of the vectors, derived from the model.
        main_schema (CollectionSchema): The schema used for the main collection.
        metadata_schema (CollectionSchema): The schema used for the metadata collection.
    """

    METADATA_COLLECTION_NAME = "paperchat_metadata"
    VECTOR_DB_DIR = str(Path.home() / ".paperchat" / "paperchat.db")

    def __init__(
        self,
        db_path: str = VECTOR_DB_DIR,
        model_identifier: str = "pymilvus/default",
    ):
        """
        Initializes the VectorStore, connects to Milvus, determines the correct
        data collection based on the model_identifier, and ensures both the
        model-specific data collection and the shared metadata collection exist.
        """
        self.db_path = db_path
        self.model_identifier = model_identifier
        self.logger = get_component_logger("VectorStore")

        try:
            self.logger.info(f"Loading embedding model: {model_identifier}")
            self._embed_fn = get_embedding_model(model_identifier)
            self.embedding_dim = self._embed_fn.dim
            self.logger.info(f"Using embedding dimension from _embed_fn.dim: {self.embedding_dim}")

        except (
            ValueError,
            RuntimeError,
            AttributeError,
        ) as e:
            self.logger.exception(
                f"Failed to load embedding model '{model_identifier}' or get its dimension: {e}"
            )
            raise

        if self.model_identifier == "pymilvus/default":
            sanitized_suffix = sanitize_model_identifier(self._embed_fn.model_name)
            self.logger.info(
                f"Default model resolved to: '{self._embed_fn.model_name}'. Using sanitized name for collection suffix."
            )
        else:
            sanitized_suffix = sanitize_model_identifier(self.model_identifier)
        self.collection_name = f"paperchat_docs_{sanitized_suffix}"
        self.logger.info(f"Using data collection: {self.collection_name}")

        self.metadata_collection_name = self.METADATA_COLLECTION_NAME
        self.logger.info(f"Using metadata collection: {self.metadata_collection_name}")

        try:
            self.client = MilvusClient(uri=self.db_path)
            self.logger.info(f"Connected to Milvus: {self.db_path}")
        except Exception as e:
            self.logger.exception(f"Failed to initialize MilvusClient: {e}")
            raise

        self.main_schema = self._define_main_schema()
        self.metadata_schema = self._define_metadata_schema()
        self._get_or_create_collection(self.collection_name, self.main_schema)
        self._get_or_create_collection(self.metadata_collection_name, self.metadata_schema)
        self._build_indices()

    def add_document(self, pdf_path: str | Path, source_id: str, max_tokens: int = 1024) -> bool:
        """
        Processes a PDF document, adds chunks to the main collection,
        and adds metadata to the metadata collection. Skips if the document
        (based on source) already exists in the metadata collection.

        Args:
            pdf_path: Path to the PDF file.
            source_id: The unique identifier (e.g., original filename stem) for this document.
            max_tokens: Maximum number of tokens per chunk.

        Returns:
            True if the document was successfully processed and added (or already existed),
            False if an error occurred during processing or insertion.
        """
        pdf_file = Path(pdf_path)
        self.logger.info(f"Processing request for document: {pdf_file.name} (ID: {source_id})")

        try:
            # 1. check metadata existence
            existing = self.client.query(
                collection_name=self.metadata_collection_name,
                filter=f'source == "{source_id}"',
                output_fields=["source"],
                limit=1,
            )

            if existing:
                self.logger.info(f"Document ID '{source_id}' already exists in metadata. Skipping.")
                return True

            # 2. process pdf into chunks (if new)
            self.logger.info(f"Document ID '{source_id}' not found. Processing PDF...")
            pdf_data = process_pdf_document(
                pdf_path, embed_fn=self._embed_fn, max_tokens_per_chunk=max_tokens
            )

            for chunk in pdf_data:
                chunk["source"] = source_id

            # 3. insert chunks into main collection
            self.logger.info(f"Inserting {len(pdf_data)} chunks into '{self.collection_name}'")
            chunk_result = self.client.insert(collection_name=self.collection_name, data=pdf_data)
            insert_count = chunk_result.get("insert_count", 0)
            if insert_count != len(pdf_data):
                self.logger.error(
                    f"Mismatch in chunk insertion count for {source_id}. Expected {len(pdf_data)}, got {insert_count}. IDs: {chunk_result.get('primary_keys')}"
                )
                # Decide on rollback or error state? For now, log error and continue to metadata attempt, but return False later.
                # Consider deleting inserted chunks if possible? client.delete might be complex with specific IDs.
                return False

            self.logger.info(f"Successfully inserted {insert_count} chunks for {source_id}")

            # 4. insert metadata entry
            iso_timestamp = datetime.now().isoformat()
            metadata_entry = [
                {
                    "source": source_id,
                    "original_filename": pdf_file.name,
                    "date_added": iso_timestamp,
                }
            ]
            self.logger.info(
                f"Inserting metadata for '{source_id}' into '{self.metadata_collection_name}'"
            )
            meta_result = self.client.insert(
                collection_name=self.metadata_collection_name, data=metadata_entry
            )
            meta_insert_count = meta_result.get("insert_count", 0)

            if meta_insert_count != 1:
                self.logger.error(
                    f"Failed to insert metadata for {source_id}. Result: {meta_result}"
                )
                return False

            self.logger.info(f"Successfully added document '{source_id}'")
            return True

        except Exception as e:
            self.logger.exception(f"Failed processing/adding document {source_id}: {e}")
            return False

    def search(
        self,
        query_text: str,
        top_k: int = DEFAULT_SIMILARITY_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        output_fields: list[str] | None = None,
        filter_expression: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Searches the main collection for text chunks similar to the query text.

        Args:
            query_text: The text to search for.
            top_k: The maximum number of results to return.
            threshold: The minimum similarity score to return.
            output_fields: List of field names to include in the results.
                           Defaults to main schema fields except embedding.
            filter_expression: Milvus filter expression string (e.g., "source == 'doc1.pdf'").

        Returns:
            A list of dictionaries representing the retrieved chunks, ordered by similarity.
            Returns an empty list if no results are found or an error occurs.
        """
        if output_fields is None:
            output_fields = [f.name for f in self.main_schema.fields if f.name != "embedding"]

        try:
            self.logger.debug(f"Loading collection '{self.collection_name}' for search.")
            query_embedding = self._embed_fn.encode_queries([query_text])[0]
        except Exception as e:
            self.logger.exception(f"Failed to generate query embedding or load collection: {e}")
            return []

        search_params = {
            "nprobe": 10,
            "metric_type": "COSINE",
            "radius": threshold,
            "range_filter": 1.0,
        }

        try:
            raw_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields,
                filter=filter_expression,
            )
            results = raw_results[0]

        except Exception as e:
            self.logger.exception(f"Search failed: {e}")
            results = []

        return results

    def list_documents(self) -> list[dict[str, Any]]:
        """
        Lists all documents by querying the metadata collection.

        Returns:
            A list of dictionaries, each containing metadata for one document
            (source, original_filename, date_added).
            Returns an empty list if there are no documents or an error occurs.
        """
        self.logger.debug(f"Listing documents from '{self.metadata_collection_name}'")
        try:
            results = self.client.query(
                collection_name=self.metadata_collection_name,
                output_fields=["source", "original_filename", "date_added"],
                limit=100,
            )
            self.logger.info(f"Found {len(results)} documents in metadata.")
            return results

        except Exception as e:
            self.logger.exception(f"Failed to list documents from metadata: {e}")
            return []

    def remove_document(self, source_id: str) -> tuple[int, int]:
        """
        Removes a document and its associated data from both the metadata
        and main document collections based on the source identifier.

        Args:
            source_id: The unique identifier (e.g., file stem) of the document to delete.

        Returns:
            A tuple containing:
            (number of metadata entries deleted, number of chunks deleted).
            Typically (1, N) on success, (0, 0) if not found, or potentially
            (0, N) or (1, 0) if deletion from one collection fails.
        """
        self.logger.info(f"Attempting to remove document with ID: '{source_id}'")
        metadata_deleted_count = 0
        chunks_deleted_count = 0

        # step 1: remove from metadata
        try:
            self.logger.debug(
                f"Removing metadata entry for '{source_id}' from '{self.metadata_collection_name}'"
            )
            meta_del_result = self.client.delete(
                collection_name=self.metadata_collection_name,
                filter=f'source == "{source_id}"',
            )
            if hasattr(meta_del_result, "delete_count"):
                metadata_deleted_count = meta_del_result.delete_count
            elif hasattr(meta_del_result, "primary_keys"):
                metadata_deleted_count = len(meta_del_result.primary_keys)
            else:
                metadata_deleted_count = 1 if meta_del_result else 0

            self.logger.info(
                f"Removed {metadata_deleted_count} entries from metadata for '{source_id}'"
            )
        except Exception as e:
            self.logger.exception(
                f"Failed to remove from metadata collection for '{source_id}': {e}"
            )

        # step 2: remove from main collection
        try:
            self.logger.debug(f"Removing chunks for '{source_id}' from '{self.collection_name}'")
            chunk_del_result = self.client.delete(
                collection_name=self.collection_name,
                filter=f'source == "{source_id}"',
            )
            if hasattr(chunk_del_result, "delete_count"):
                chunks_deleted_count = chunk_del_result.delete_count
            elif hasattr(chunk_del_result, "primary_keys"):
                chunks_deleted_count = len(chunk_del_result.primary_keys)
            else:
                chunks_deleted_count = 0

            self.logger.info(
                f"Removed {chunks_deleted_count} chunks from main collection for '{source_id}'"
            )
        except Exception as e:
            self.logger.exception(
                f"Failed to remove chunks from main collection for '{source_id}': {e}"
            )

        self.logger.info(
            f"Removal complete for '{source_id}'. Metadata removed: {metadata_deleted_count}, Chunks removed: {chunks_deleted_count}"
        )
        return metadata_deleted_count, chunks_deleted_count

    def _define_main_schema(self) -> CollectionSchema:
        """Defines the Milvus collection schema for document chunks."""
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Original chunk identifier",
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(
                name="label", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=64
            ),
            FieldSchema(name="page_numbers", dtype=DataType.ARRAY, element_type=DataType.INT16),
            FieldSchema(name="headings", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Source document filename stem (e.g., pdf_file.stem)",
            ),
        ]
        return CollectionSchema(
            fields, description="Document Chunks Collection", enable_dynamic_field=True
        )

    def _define_metadata_schema(self) -> CollectionSchema:
        """Defines the Milvus collection schema for document metadata."""
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Unique source identifier/filename stem (e.g., pdf_file.stem)",
            ),
            FieldSchema(
                name="original_filename",
                dtype=DataType.VARCHAR,
                max_length=1024,
                description="Original filename of the source document",
            ),
            FieldSchema(
                name="date_added",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="ISO 8601 timestamp when the document was added",
            ),
        ]
        return CollectionSchema(
            fields, description="Document Metadata Collection", enable_dynamic_field=False
        )

    def _get_or_create_collection(self, collection_name: str, schema: CollectionSchema) -> None:
        """Creates the specified collection if it doesn't exist."""
        try:
            collection_exists = self.client.has_collection(collection_name)
            if not collection_exists:
                self.logger.info(f"Creating collection '{collection_name}'")
                self.client.create_collection(collection_name=collection_name, schema=schema)
            else:
                self.logger.info(f"Collection '{collection_name}' already exists.")
            if collection_name == self.collection_name:
                self._check_collection_dimension(collection_name)

        except Exception as e:
            self.logger.exception(f"Failed to check/create collection '{collection_name}': {e}")
            raise

    def _check_collection_dimension(self, collection_name: str) -> None:
        """Verify the dimension of the existing collection's embedding field."""
        try:
            collection_info = self.client.describe_collection(collection_name)
            for field in collection_info["fields"]:
                if field["name"] == "embedding":
                    existing_dim = field["params"]["dim"]
                    if existing_dim != self.embedding_dim:
                        error_msg = (
                            f"Dimension mismatch for collection '{collection_name}'! "
                            f"Expected (from model '{self.model_identifier}'): {self.embedding_dim}. "
                            f"Found in existing collection: {existing_dim}. "
                            f"Must drop and recreate collection or use matching model."
                        )
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                    return  # Found the embedding field
            self.logger.warning(
                f"Could not find 'embedding' field in existing collection '{collection_name}' description."
            )
        except Exception as e:
            self.logger.exception(
                f"Failed to check dimension for existing collection '{collection_name}': {e}"
            )
            # Decide if this should be a critical error? For now, just log.

    def _build_indices(self) -> None:
        """Creates indices for both main and metadata collections if they don't exist."""
        try:
            self.logger.info(
                f"Ensuring vector index exists on 'embedding' for '{self.collection_name}'"
            )
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
            self.client.create_index(
                collection_name=self.collection_name, index_params=index_params
            )
            self.logger.info(f"Vector index ensured for '{self.collection_name}'")
        except Exception as e:
            self.logger.warning(f"Could not ensure vector index for '{self.collection_name}': {e}")

        # metadata db index
        metadata_index_field = "source"
        try:
            metadata_index_params = self.client.prepare_index_params()
            metadata_index_params.add_index(field_name=metadata_index_field)
            self.logger.info(
                f"Ensuring scalar index exists on '{metadata_index_field}' for '{self.metadata_collection_name}'"
            )
            self.client.create_index(
                collection_name=self.metadata_collection_name, index_params=metadata_index_params
            )
            self.logger.info(f"Scalar index ensured for '{self.metadata_collection_name}'")
        except Exception as e:
            self.logger.warning(
                f"Could not ensure scalar index for '{self.metadata_collection_name}': {e}"
            )


def sanitize_model_identifier(model_identifier: str) -> str:
    """
    Sanitizes a model identifier string to be suitable for a Milvus collection name suffix.

    Replaces potentially problematic characters like '/', '.', '-' with underscores '_'
    and ensures the name starts and ends with alphanumeric characters.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", model_identifier)
    sanitized = sanitized.strip("_")
    if not sanitized:
        return "default_model"
    if not re.match(r"^[a-zA-Z_]", sanitized):
        sanitized = "_" + sanitized
    max_len = 200  # Milvus char limit for collection names is 255
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]

    return sanitized
