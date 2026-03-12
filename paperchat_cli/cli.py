from __future__ import annotations

import argparse
from collections.abc import Sequence

from paperchat_cli.launcher import LaunchError, launch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperchat")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch", help="Start the local PaperChat app stack.")
    launch_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the PaperChat app in a browser.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "launch":
        try:
            launch(no_open=args.no_open)
        except LaunchError as error:
            print(f"paperchat: {error}")
            return 1
        return 0

    msg = f"Unsupported command: {args.command}"
    raise RuntimeError(msg)
