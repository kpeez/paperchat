"""Runs retrieval on a BEIR dataset and saves results using Typer."""

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm

from paperchat.core import VectorStore, get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEFAULT_EMBEDDING_BATCH_SIZE = 32
BASE_EVAL_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = BASE_EVAL_DIR / "datasets"
DEFAULT_RESULTS_DIR = BASE_EVAL_DIR / "results"
DEFAULT_DB_DIR = BASE_EVAL_DIR / "data"

app = typer.Typer(help="Run BEIR retrieval evaluation and save results.")


def load_beir_dataset(
    data_path: Path, split: str = "test"
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """
    Loads the corpus and queries from a downloaded BEIR dataset folder.

    Args:
        data_path: Path to the BEIR dataset directory (e.g., evaluation/datasets/scidocs).
        split: The dataset split to load (e.g., "test", "train", "dev").

    Returns:
        A tuple containing the corpus and queries dictionaries.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"BEIR dataset not found at {data_path}. Run download_data.py first."
        )
    logging.info(f"Loading BEIR dataset from: {data_path}")
    corpus, queries, _ = GenericDataLoader(data_folder=str(data_path)).load(split=split)
    logging.info(f"Loaded corpus with {len(corpus)} documents and {len(queries)} queries.")
    return corpus, queries


def prepare_and_embed_corpus(
    corpus: dict[str, dict[str, str]], embed_fn: Any, batch_size: int
) -> list[dict[str, Any]]:
    """
    Formats BEIR corpus and generates embeddings for direct Milvus insertion.

    Args:
        corpus: The BEIR corpus dictionary {doc_id: {'title': ..., 'text': ...}}.
        embed_fn: The embedding function instance.
        batch_size: Batch size for embedding generation.

    Returns:
        A list of dictionaries ready for client.insert, including embeddings.
    """
    doc_ids = list(corpus.keys())
    texts_to_embed = [
        (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
        for doc_id in doc_ids
    ]

    all_embeddings = []
    logging.info(
        f"Generating embeddings for {len(texts_to_embed)} documents in batches of {batch_size}..."
    )
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding Corpus"):
        batch_texts = texts_to_embed[i : i + batch_size]
        embeddings = embed_fn.encode_documents(batch_texts)
        if hasattr(embeddings, "tolist"):
            all_embeddings.extend(embeddings.tolist())
        else:
            all_embeddings.extend(embeddings)

    if len(all_embeddings) != len(doc_ids):
        raise RuntimeError("Mismatch between number of documents and generated embeddings.")

    data_for_milvus = []
    for i, doc_id in enumerate(doc_ids):
        data_for_milvus.append(
            {
                "text": texts_to_embed[i],
                "embedding": all_embeddings[i],
                "source": doc_id,
                "chunk_id": f"{doc_id}_chunk0",
                "label": [],
                "page_numbers": [],
                "headings": corpus[doc_id].get("title", ""),
            }
        )

    return data_for_milvus


def run_retrieval(
    vector_store: VectorStore, queries: dict[str, str], k: int = 100
) -> dict[str, dict[str, float]]:
    """
    Runs retrieval for all queries against the vector store.

    Args:
        vector_store: The initialized VectorStore.
        queries: The dictionary of queries {query_id: query_text}.
        k: The number of documents to retrieve for each query.

    Returns:
        A dictionary containing the retrieval results in BEIR format:
        {query_id: {beir_doc_id1: score1, beir_doc_id2: score2, ...}}
    """
    results: dict[str, dict[str, float]] = {}
    logging.info(f"Running retrieval for {len(queries)} queries...")

    output_fields = ["source"]

    for query_id, query_text in tqdm(queries.items(), desc="Retrieving"):
        try:
            retrieved_hits = vector_store.search(query_text, top_k=k, output_fields=output_fields)
            query_results = {}
            for hit in retrieved_hits:
                original_doc_id = hit.get("entity", {}).get("source")
                if original_doc_id:
                    score = hit.get("distance", 0.0)
                    query_results[original_doc_id] = float(score)
                else:
                    logging.warning(
                        f"Retrieved hit for query {query_id} is missing 'source' field. Hit: {hit}"
                    )
            results[query_id] = query_results
        except Exception as e:
            logging.error(f"Error retrieving results for query_id {query_id}: {e}", exc_info=True)
            results[query_id] = {}
    logging.info("Retrieval finished.")
    return results


def save_results(results: dict[str, dict[str, float]], output_path: Path) -> None:
    """Saves the retrieval results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Retrieval results saved to: {output_path}")


def _ingest_corpus_if_needed(
    vector_store: VectorStore,
    corpus: dict[str, dict[str, str]],
    embedding_model: Any,
    batch_size: int,
) -> None:
    """Handles the logic for checking, dropping, creating, and ingesting into a collection automatically."""
    expected_doc_count = len(corpus)
    collection_name = vector_store.collection_name
    needs_ingestion = True

    if vector_store.client.has_collection(collection_name):
        vector_store.client.load_collection(collection_name)
        current_doc_count = vector_store.client.num_entities(collection_name)
        logging.info(f"Collection '{collection_name}' exists with {current_doc_count} entities.")

        if current_doc_count == expected_doc_count:
            logging.info(
                f"Collection appears complete ({current_doc_count}/{expected_doc_count}). Skipping ingestion."
            )
            needs_ingestion = False
        else:
            logging.warning(
                f"Collection '{collection_name}' exists but is incomplete ({current_doc_count}/{expected_doc_count}). "
                f"Dropping and re-creating for full ingestion."
            )
            vector_store.client.release_collection(collection_name)
            vector_store.client.drop_collection(collection_name)
            vector_store._get_or_create_collection(collection_name, vector_store.main_schema)
            vector_store._build_indices()
    else:
        logging.info(f"Collection '{collection_name}' does not exist. Creating and ingesting.")
        vector_store._get_or_create_collection(collection_name, vector_store.main_schema)
        vector_store._build_indices()

    if needs_ingestion:
        logging.info(f"Preparing and embedding corpus for '{collection_name}'...")
        data_to_insert = prepare_and_embed_corpus(corpus, embedding_model, batch_size)
        logging.info(f"Ingesting {len(data_to_insert)} documents into '{collection_name}'...")
        insert_result = vector_store.client.insert(
            collection_name=collection_name, data=data_to_insert
        )
        insert_count = insert_result.get("insert_count", 0)
        if insert_count != len(data_to_insert):
            raise RuntimeError(
                f"Mismatch in document insertion count. Expected {len(data_to_insert)}, got {insert_count}."
            )
        logging.info(f"Ingestion complete and flushed for '{collection_name}'.")


def run_evaluation_pipeline(
    dataset: str,
    data_dir: Path,
    db_dir: Path,
    split: str,
    embedding_model_id: str,
    results_dir: Path,
    top_k: int,
    batch_size: int,
) -> Path | None:
    """Runs the full pipeline: load, init, ingest, retrieve, save. Returns results file path or None."""
    dataset_path = data_dir / dataset
    sanitized_model_name = embedding_model_id.replace("/", "_")
    results_file = results_dir / f"{dataset}_{sanitized_model_name}_results.json"

    if results_file.exists():
        logging.info(f"Results file already exists: {results_file}. Skipping retrieval.")
        typer.echo(
            typer.style(
                f"Skipping retrieval, results found at {results_file}", fg=typer.colors.YELLOW
            )
        )
        return results_file

    db_dir.mkdir(parents=True, exist_ok=True)
    eval_db_path = str(db_dir / f"evaluation_{dataset}_{sanitized_model_name}.db")
    logging.info(f"Using evaluation database: {eval_db_path}")

    vector_store = None
    try:
        corpus, queries = load_beir_dataset(dataset_path, split)

        logging.info(f"Initializing VectorStore for model: {embedding_model_id}")
        embedding_model = get_embedding_model(embedding_model_id)
        vector_store = VectorStore(
            db_path=eval_db_path,
            model_identifier=embedding_model_id,
        )

        _ingest_corpus_if_needed(
            vector_store=vector_store,
            corpus=corpus,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )

        results = run_retrieval(vector_store, queries, k=top_k)
        save_results(results, results_file)
        typer.echo(typer.style(f"Retrieval results saved to {results_file}", fg=typer.colors.GREEN))
        return results_file

    except FileNotFoundError as e:
        logging.error(f"Data loading error: {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logging.error(f"An unexpected error occurred in pipeline: {e}", exc_info=True)
        raise typer.Exit(code=1) from e
    finally:
        if vector_store and hasattr(vector_store, "client") and vector_store.client:
            try:
                logging.info("Milvus client cleanup check.")
            except Exception as disconnect_e:
                logging.warning(f"Error during Milvus cleanup attempt: {disconnect_e}")


@app.command()
def main(
    embedding_model_id: Annotated[
        str,
        typer.Option("--model-id", "-m", help="Embedding model identifier (required)."),
    ],
    dataset: Annotated[
        str,
        typer.Option("--dataset", "-d", help="Name of the BEIR dataset."),
    ] = "scidocs",
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Directory containing BEIR datasets.", resolve_path=True),
    ] = DEFAULT_DATA_DIR,
    split: Annotated[str, typer.Option("--split", help="Dataset split.")] = "test",
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", help="Directory to save results.", resolve_path=True),
    ] = DEFAULT_RESULTS_DIR,
    db_dir: Annotated[
        Path,
        typer.Option(
            "--db-dir", help="Base directory for evaluation databases.", resolve_path=True
        ),
    ] = DEFAULT_DB_DIR,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of documents to retrieve.")] = 100,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Batch size for embedding generation.")
    ] = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> None:
    """
    Runs BEIR retrieval evaluation using dataset-specific DBs.
    """
    try:
        results_path = run_evaluation_pipeline(
            dataset=dataset,
            data_dir=data_dir,
            db_dir=db_dir,
            split=split,
            embedding_model_id=embedding_model_id,
            results_dir=results_dir,
            top_k=top_k,
            batch_size=batch_size,
        )
        if results_path:
            logging.info("Retrieval pipeline completed successfully.")
        else:
            logging.info("Retrieval step skipped as results already exist.")
    except typer.Exit:
        logging.error("Retrieval pipeline failed.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}", exc_info=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
