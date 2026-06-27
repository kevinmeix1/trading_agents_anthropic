from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.demo_rehearsal import (
    build_technology_prize_demo_rehearsal,
    write_technology_prize_demo_rehearsal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rehearse the safe offline technology-prize judge demo flow."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve source and artifact paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory where judge-facing reports are regenerated.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_demo_rehearsal.md",
        help="Markdown rehearsal output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_demo_rehearsal.json",
        help="JSON rehearsal output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    rehearsal = build_technology_prize_demo_rehearsal(
        project_root=args.project_root,
        output_dir=args.output_dir,
    )
    json_output = args.json_output or None
    write_technology_prize_demo_rehearsal(
        rehearsal,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in rehearsal.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
