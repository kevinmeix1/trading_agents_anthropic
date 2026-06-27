from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.workflow import run_local_agent_workflow, write_local_agent_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic local technology-prize agent workflow."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve evidence artifacts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_workflow.md",
        help="Markdown workflow output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_workflow.json",
        help="JSON workflow output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    workflow = run_local_agent_workflow(project_root=args.project_root)
    json_output = args.json_output or None
    write_local_agent_workflow(
        workflow,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in workflow.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
