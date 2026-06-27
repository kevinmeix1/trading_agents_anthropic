from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.demo_director import build_judge_demo_runbook, write_judge_demo_runbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a timed technology-prize judge demo runbook."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve evidence artifacts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_demo_runbook.md",
        help="Markdown runbook output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_demo_runbook.json",
        help="JSON runbook output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    runbook = build_judge_demo_runbook(project_root=args.project_root)
    json_output = args.json_output or None
    write_judge_demo_runbook(
        runbook,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in runbook.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
