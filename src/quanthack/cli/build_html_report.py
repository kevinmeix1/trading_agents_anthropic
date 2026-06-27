from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.trading.execution import read_journal
from quanthack.reporting.html_report import build_journal_html_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standalone HTML dry-run report.")
    parser.add_argument("--journal", default="outputs/dry_run_journal.jsonl")
    parser.add_argument("--output", default="outputs/reports/journal_report.html")
    parser.add_argument("--recent", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> None:
    records = read_journal(args.journal)
    report = build_journal_html_report(records=records, recent_limit=args.recent)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.html, encoding="utf-8")

    print(f"Report: {output_path}")
    print(f"Records: {len(records)}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
