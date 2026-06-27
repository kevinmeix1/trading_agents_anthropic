from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.guardrails import build_agent_guardrail_suite, write_agent_guardrail_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the technology-prize AI/broker guardrail suite."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve source and evidence artifacts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_guardrails.md",
        help="Markdown guardrail-suite output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_guardrails.json",
        help="JSON guardrail-suite output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    suite = build_agent_guardrail_suite(project_root=args.project_root)
    json_output = args.json_output or None
    write_agent_guardrail_suite(
        suite,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in suite.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
