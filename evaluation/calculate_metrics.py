"""Calculates BEIR evaluation metrics from saved retrieval results."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="Calculate BEIR evaluation metrics from retrieval results.")
console = Console()


def load_qrels(data_path: Path, split: str = "test") -> dict[str, dict[str, int]]:
    """Loads the qrels (relevance judgments) for a given dataset split."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"BEIR dataset not found at {data_path}. Run download_data.py first."
        )
    logging.info(f"Loading qrels from: {data_path}")
    try:
        _, _, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)
        logging.info(f"Loaded {len(qrels)} queries with relevance judgements.")
        return qrels
    except Exception as e:
        logging.error(f"Failed to load qrels from {data_path}: {e}", exc_info=True)
        raise


def load_results(results_path: Path) -> dict[str, dict[str, float]]:
    """Loads the retrieval results from a JSON file."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}")
    logging.info(f"Loading retrieval results from: {results_path}")
    with open(results_path) as f:
        results = json.load(f)
    logging.info(f"Loaded results for {len(results)} queries.")
    return results


def display_metrics(metrics: dict[str, dict[str, float]], title: str = "Retrieval Metrics") -> None:
    """Displays the calculated metrics in a formatted table."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    if metrics:
        first_metric_dict = next(iter(metrics.values()), None)
        if first_metric_dict:
            first_metric_keys = list(first_metric_dict.keys())
            for k_header in first_metric_keys:
                table.add_column(k_header, style="magenta")

            for metric_name, k_values in metrics.items():
                row = [metric_name]
                for k_header in first_metric_keys:
                    value = k_values.get(k_header, float("nan"))
                    row.append(f"{value:.4f}")
                table.add_row(*row)

    console.print(table)


@app.command()
def main(
    results_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the JSON file containing retrieval results.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Name of the BEIR dataset evaluated (e.g., 'scidocs'). Used to find qrels.",
        ),
    ] = "scidocs",
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Directory containing BEIR datasets.", resolve_path=True),
    ] = Path(__file__).parent / "datasets",
    split: Annotated[str, typer.Option("--split", help="Dataset split used for qrels.")] = "test",
    metrics_ks_arg: Annotated[
        list[int] | None,
        typer.Option(
            "--k",
            help="Values of k for recall/precision metrics. Defaults to [1, 3, 5, 10, 100, 1000].",
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path to save calculated metrics as JSON.",
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """
    Calculates and displays BEIR retrieval metrics (NDCG, MAP, Recall, P) for given results.
    """
    metrics_ks = metrics_ks_arg if metrics_ks_arg is not None else [1, 3, 5, 10, 100, 1000]

    dataset_path = data_dir / dataset
    model_name = results_path.stem.replace(f"{dataset}_", "").replace("_results", "")
    title = f"Retrieval Metrics: [{dataset}/{split}] - Model: [{model_name}]"

    try:
        logging.info(f"Calculating metrics for results: {results_path}")
        logging.info(f"Using qrels for dataset: {dataset}, split: {split}")

        qrels = load_qrels(dataset_path, split)
        results = load_results(results_path)

        if not qrels:
            typer.echo(typer.style("Error: Qrels are empty.", fg=typer.colors.RED))
            raise typer.Exit(code=1)
        if not results:
            typer.echo(typer.style("Error: Results are empty.", fg=typer.colors.RED))
            raise typer.Exit(code=1)

        common_queries = set(results.keys()) & set(qrels.keys())
        if len(common_queries) < len(results):
            logging.warning(
                f"Results contain {len(results) - len(common_queries)} queries not found in qrels. Evaluating only on {len(common_queries)} common queries."
            )
            results = {qid: results[qid] for qid in common_queries}
        if not common_queries:
            typer.echo(
                typer.style(
                    "Error: No common queries between results and qrels.", fg=typer.colors.RED
                )
            )
            raise typer.Exit(code=1)

        evaluator = EvaluateRetrieval()
        ndcg, map_score, recall, precision = evaluator.evaluate(qrels, results, metrics_ks)

        all_metrics = {
            "NDCG": ndcg,
            "MAP": map_score,
            "Recall": recall,
            "Precision": precision,
        }

        display_metrics(all_metrics, title=title)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(all_metrics, f, indent=4)
            typer.echo(typer.style(f"Metrics saved to {output_file}", fg=typer.colors.GREEN))

    except FileNotFoundError as e:
        logging.error(f"File loading error: {e}")
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(code=1) from e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        typer.echo(typer.style(f"An unexpected error occurred: {e}", fg=typer.colors.RED))
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
