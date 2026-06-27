from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.trace_replay import build_agent_trace_replay, write_agent_trace_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline AgentSDK-style trace replay."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve evidence artifacts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_trace_replay.md",
        help="Markdown trace-replay output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_trace_replay.json",
        help="JSON trace-replay output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    replay = build_agent_trace_replay(project_root=args.project_root)
    json_output = args.json_output or None
    write_agent_trace_replay(
        replay,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in replay.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
