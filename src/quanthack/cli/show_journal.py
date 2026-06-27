from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.trading.execution import read_journal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show dry-run journal records.")
    parser.add_argument("--journal", default="outputs/dry_run_journal.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> None:
    records = read_journal(args.journal)
    if not records:
        print("No journal records found.")
        return

    for record in records[-args.limit :]:
        request = record["request"]
        decision = record["decision"]
        print(
            f"{record['created_at_utc']} | {record['status']} | "
            f"{request['side']} {request['symbol']} requested "
            f"${request['target_notional_usd']:,.0f} adjusted "
            f"${decision['adjusted_notional_usd']:,.0f} | {decision['reason']}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
