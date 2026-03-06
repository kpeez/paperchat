"""Download a BEIR dataset (e.g., scidocs) for evaluation using Typer."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from beir import util
from beir.datasets.data_loader import GenericDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="BEIR Dataset Download Utility")


def download_beir_dataset(
    dataset_name: str, output_dir: Path, data_path: Path | None = None
) -> Path:
    """
    Downloads the specified BEIR dataset using beir.util and returns the path.

    Args:
        dataset_name: The name of the BEIR dataset (e.g., "scidocs").
        output_dir: The directory to save the downloaded dataset.
        data_path: The specific path where the dataset was extracted (optional, for logging).

    Returns:
        The path to the downloaded dataset directory.

    Raises:
        Exception: If the download fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    extracted_path_str = util.download_and_unzip(url, str(output_dir))
    extracted_path = Path(extracted_path_str)
    logging.info(f"Dataset '{dataset_name}' downloaded and extracted to: {extracted_path}")
    return extracted_path


@app.command()
def main(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Name of the BEIR dataset to download (e.g., 'scidocs', 'nfcorpus').",
        ),
    ] = "scidocs",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save the downloaded dataset.",
            resolve_path=True,
        ),
    ] = Path(__file__).parent / "datasets",
) -> None:
    """
    Downloads and verifies a specified BEIR dataset.
    """
    logging.info(f"Starting download for dataset: {dataset}")
    try:
        data_path = download_beir_dataset(dataset, output_dir)
        logging.info(f"Verifying dataset integrity by loading '{dataset}/test' split...")
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")
        logging.info(
            f"Successfully downloaded and verified dataset '{dataset}'. "
            f"Corpus size: {len(corpus)}, Queries size: {len(queries)}, Qrels size: {len(qrels)}"
        )
        typer.echo(
            typer.style(
                f"Dataset '{dataset}' downloaded successfully to {data_path}",
                fg=typer.colors.GREEN,
            )
        )
    except Exception as e:
        logging.error(f"Failed to download or verify dataset '{dataset}': {e}", exc_info=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
