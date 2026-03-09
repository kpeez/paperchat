import argparse
from pathlib import Path

from paperchat.benchmarks.docling_validation.runner import (
    DEFAULT_FIXTURES_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QUERIES_PATH,
    DEFAULT_TOP_K,
    run_validation,
)


def main() -> None:
    args = _build_parser().parse_args()
    benchmark_run, output_directory = run_validation(
        fixtures_path=args.fixtures,
        queries_path=args.queries,
        output_root=args.output_root,
        top_k=args.top_k,
    )
    recommendation = benchmark_run.recommendation
    print(f"status={recommendation.status}")
    if recommendation.recommended_chunker_id is not None:
        print(f"recommended_chunker={recommendation.recommended_chunker_id}")
    print(f"output_dir={output_directory}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PR 3 Docling validation harness.")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES_PATH,
        help="Path to the fixture manifest JSON.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to the gold-query manifest JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where per-run reports should be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k cutoff for hit-rate metrics.",
    )
    return parser
