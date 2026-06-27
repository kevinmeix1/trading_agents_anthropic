from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.rubric import (
    build_technology_prize_rubric,
    write_technology_prize_rubric,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the project against the technology-prize judging rubric."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve source and artifact paths.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_rubric.md",
        help="Markdown rubric output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_rubric.json",
        help="JSON rubric output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    rubric = build_technology_prize_rubric(project_root=args.project_root)
    json_output = args.json_output or None
    write_technology_prize_rubric(
        rubric,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in rubric.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
