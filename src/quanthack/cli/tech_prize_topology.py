from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.topology import build_agent_topology_report, write_agent_topology_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the technology-prize AgentSDK topology."
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_topology.md",
        help="Markdown topology output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_topology.json",
        help="JSON topology output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    report = build_agent_topology_report()
    json_output = args.json_output or None
    write_agent_topology_report(
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
