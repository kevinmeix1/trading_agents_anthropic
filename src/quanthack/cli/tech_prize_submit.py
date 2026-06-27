from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.submission_bundle import (
    build_technology_prize_submission_bundle,
    write_technology_prize_submission_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the final technology-prize submission bundle."
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
        default="outputs/reports/technology_prize_submission.md",
        help="Markdown submission-bundle output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_submission.json",
        help="JSON submission-bundle output path. Use an empty string to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    bundle = build_technology_prize_submission_bundle(
        project_root=args.project_root,
        output_dir=args.output_dir,
    )
    json_output = args.json_output or None
    write_technology_prize_submission_bundle(
        bundle,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in bundle.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
