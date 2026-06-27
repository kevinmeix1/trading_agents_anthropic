from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.demo_pack import (
    build_technology_prize_demo_pack,
    write_technology_prize_demo_pack,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a judge-ready technology-prize demo pack."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve source and artifact paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory where component reports are regenerated.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_demo_pack.md",
        help="Markdown demo-pack output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_demo_pack.json",
        help="JSON demo-pack output path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--allow-online-sdk",
        action="store_true",
        help="Allow the AgentSDK Runner to make online model calls when OPENAI_API_KEY is set.",
    )
    parser.add_argument(
        "--allow-online-anthropic",
        action="store_true",
        help="Allow the Anthropic critic to make online model calls when ANTHROPIC_API_KEY is set.",
    )
    parser.add_argument("--sdk-model", default="gpt-5.5")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-6")
    return parser


def run(args: argparse.Namespace) -> None:
    pack = build_technology_prize_demo_pack(
        project_root=args.project_root,
        output_dir=args.output_dir,
        allow_online_sdk=args.allow_online_sdk,
        allow_online_anthropic=args.allow_online_anthropic,
        sdk_model=args.sdk_model,
        anthropic_model=args.anthropic_model,
    )
    json_output = args.json_output or None
    write_technology_prize_demo_pack(
        pack,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in pack.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
