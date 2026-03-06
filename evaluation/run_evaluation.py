"""Coordinates the BEIR evaluation workflow: download, retrieve, calculate, summarize."""

import csv
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_EVAL_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = BASE_EVAL_DIR / "datasets"
DEFAULT_RESULTS_DIR = BASE_EVAL_DIR / "results"
DEFAULT_DB_DIR = BASE_EVAL_DIR / "data"
DEFAULT_SUMMARY_FILE = DEFAULT_RESULTS_DIR / "evaluation_summary.csv"

app = typer.Typer(help="Run the full BEIR evaluation workflow.")


def _run_command(command: list[str], step_name: str) -> None:
    """Helper function to run a command as a subprocess and handle errors."""
    logging.info(f"Starting step: {step_name}...")
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=BASE_EVAL_DIR.parent,
        )
        logging.info(f"{step_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{step_name} failed with exit code {e.returncode}")
        logging.error(f"Stdout:\n{e.stdout}")
        logging.error(f"Stderr:\n{e.stderr}")
        raise typer.Exit(code=e.returncode) from e
    except Exception as e:
        logging.error(f"An unexpected error occurred during {step_name}: {e}", exc_info=True)
        raise typer.Exit(code=1) from e


def _append_to_summary(
    summary_file: Path, run_info: dict[str, Any], metrics: dict[str, dict[str, float]]
) -> None:
    """Appends run information and key metrics to the summary CSV file."""
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "dataset", "split", "model_id", "metric", "value"]
    key_metrics_to_log = {"NDCG@10", "Recall@100", "Recall@1000", "MAP@1000", "P@10"}

    rows_to_write = []
    file_exists = summary_file.is_file()

    for _metric_group_name, metric_values in metrics.items():
        for metric_name_at_k, value in metric_values.items():
            if metric_name_at_k in key_metrics_to_log:
                row = run_info.copy()
                row["metric"] = metric_name_at_k
                row["value"] = f"{value:.4f}"
                rows_to_write.append(row)

    if not rows_to_write:
        logging.warning("No key metrics found to log in the summary file.")
        return

    try:
        with open(summary_file, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_write)
        logging.info(f"Successfully appended results to {summary_file}")
    except Exception as e:
        logging.error(
            f"Failed to append results to summary file {summary_file}: {e}", exc_info=True
        )


@app.command()
def run(
    model_id: Annotated[
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
    split: Annotated[str, typer.Option("--split", help="Dataset split.")] = "test",
    metrics_ks_arg: Annotated[
        list[int] | None,
        typer.Option("--k", help="Values of k for metrics. Defaults to [1, 3, 5, 10, 100, 1000]."),
    ] = None,
    skip_ingestion: Annotated[
        bool,
        typer.Option("--skip-ingestion", help="Skip ingestion if collection exists."),
    ] = False,
    force_download: Annotated[
        bool,
        typer.Option("--force-download", help="Force download dataset even if it exists."),
    ] = False,
    summary_file: Annotated[
        Path,
        typer.Option("--summary-file", help="Path to the CSV summary file.", resolve_path=True),
    ] = DEFAULT_SUMMARY_FILE,
) -> None:
    """Runs the full BEIR evaluation workflow and logs results to a summary file."""

    timestamp = datetime.datetime.now().isoformat()
    dataset_path = data_dir / dataset
    sanitized_model_name = model_id.replace("/", "_")
    results_json_path = results_dir / f"{dataset}_{sanitized_model_name}_results.json"
    metrics_ks = metrics_ks_arg if metrics_ks_arg is not None else [1, 3, 5, 10, 100, 1000]
    metrics_ks_str = [str(k) for k in metrics_ks]

    if not dataset_path.exists() or force_download:
        logging.info(f"Dataset '{dataset}' not found at {data_dir} or download forced.")
        download_cmd = [
            sys.executable,
            str(BASE_EVAL_DIR / "download_beir.py"),
            "--dataset",
            dataset,
            "--output-dir",
            str(data_dir),
        ]
        _run_command(download_cmd, "Data Download")
    else:
        logging.info(f"Dataset '{dataset}' found at {data_dir}. Skipping download.")

    retrieval_cmd = [
        sys.executable,
        str(BASE_EVAL_DIR / "run_retrieval.py"),
        "--model-id",
        model_id,
        "--dataset",
        dataset,
        "--data-dir",
        str(data_dir),
        "--results-dir",
        str(results_dir),
        "--db-dir",
        str(db_dir),
        "--split",
        split,
    ]
    if skip_ingestion:
        retrieval_cmd.append("--skip-ingestion")

    _run_command(retrieval_cmd, "Retrieval Execution")

    calc_cmd = [
        sys.executable,
        str(BASE_EVAL_DIR / "calculate_metrics.py"),
        str(results_json_path),
        "--dataset",
        dataset,
        "--data-dir",
        str(data_dir),
        "--split",
        split,
    ]
    for k in metrics_ks_str:
        calc_cmd.extend(["--k", k])
    metrics_json_output = results_dir / f"{dataset}_{sanitized_model_name}_metrics.json"
    calc_cmd.extend(["--output", str(metrics_json_output)])

    _run_command(calc_cmd, "Metrics Calculation")

    try:
        with open(metrics_json_output) as f:
            calculated_metrics = json.load(f)

        run_info = {
            "timestamp": timestamp,
            "dataset": dataset,
            "split": split,
            "model_id": model_id,
        }
        _append_to_summary(summary_file, run_info, calculated_metrics)

    except FileNotFoundError as e:
        logging.error(
            f"Metrics JSON file not found: {metrics_json_output}. Cannot append to summary."
        )
        raise typer.Exit(code=1) from e
    except Exception as e:
        logging.error(f"Failed to read metrics JSON or append to summary: {e}", exc_info=True)
        raise typer.Exit(code=1) from e

    logging.info("Evaluation workflow complete.")
    typer.echo(
        typer.style("Evaluation workflow finished successfully!", fg=typer.colors.BRIGHT_GREEN)
    )


if __name__ == "__main__":
    app()
