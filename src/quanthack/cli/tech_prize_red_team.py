from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.red_team import (
    build_technology_prize_red_team_report,
    write_technology_prize_red_team_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the technology-prize skeptical judge red-team checks."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve source and artifact paths.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_red_team.md",
        help="Markdown red-team output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_red_team.json",
        help="JSON red-team output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    report = build_technology_prize_red_team_report(project_root=args.project_root)
    json_output = args.json_output or None
    write_technology_prize_red_team_report(
        report,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in report.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
