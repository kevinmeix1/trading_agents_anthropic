from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.agents.demo_pack import (
    build_technology_prize_demo_pack,
    write_technology_prize_demo_pack,
)
from quanthack.agents.judge_packet import (
    build_technology_prize_judge_packet,
    write_technology_prize_judge_packet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a requirement-by-requirement technology-prize judge packet."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve source and artifact paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory where supporting demo-pack reports are refreshed.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_judge_packet.md",
        help="Markdown judge-packet output path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_judge_packet.json",
        help="JSON judge-packet output path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Do not refresh the safe offline demo pack before verifying report artifacts.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.skip_refresh:
        pack = build_technology_prize_demo_pack(
            project_root=args.project_root,
            output_dir=args.output_dir,
        )
        output_dir = Path(args.project_root) / args.output_dir
        write_technology_prize_demo_pack(
            pack,
            markdown_path=output_dir / "technology_prize_demo_pack.md",
            json_path=output_dir / "technology_prize_demo_pack.json",
        )

    packet = build_technology_prize_judge_packet(project_root=args.project_root)
    json_output = args.json_output or None
    write_technology_prize_judge_packet(
        packet,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in packet.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
