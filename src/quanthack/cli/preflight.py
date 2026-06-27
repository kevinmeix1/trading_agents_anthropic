from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._time import parse_datetime
from quanthack.trading.preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local dry-run readiness checks.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--now", type=parse_datetime, default=None)
    parser.add_argument(
        "--quote-as-of",
        type=parse_datetime,
        default=None,
        help="Override quote quality evaluation time.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    report = run_preflight(
        config_path=args.config,
        now=args.now,
        quote_as_of=args.quote_as_of,
    )
    for line in report.summary_lines():
        print(line)

    if report.overall == "ATTENTION_REQUIRED":
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
