from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.trading.execution import read_journal
from quanthack.reporting.journal_report import summarize_journal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the dry-run decision journal.")
    parser.add_argument("--journal", default="outputs/dry_run_journal.jsonl")
    parser.add_argument("--recent", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> None:
    records = read_journal(args.journal)
    summary = summarize_journal(records)

    print("Journal Summary")
    print(f"  Journal: {args.journal}")
    print(f"  Records: {summary.total_records}")
    print(f"  Accepted: {summary.accepted}")
    print(f"  Blocked: {summary.blocked}")
    print(f"  Accepted rate: {summary.accepted_rate:.1%}")
    print(f"  Requested notional: {money(summary.requested_notional_usd, cents=False)}")
    print(f"  Adjusted notional: {money(summary.adjusted_notional_usd, cents=False)}")
    print(f"  Trimmed by risk: {money(summary.trimmed_notional_usd, cents=False)}")

    print("By status")
    if summary.by_status:
        for status, count in summary.by_status.items():
            print(f"  {status}: {count}")
    else:
        print("  none")

    print("By mode")
    if summary.by_mode:
        for mode, count in summary.by_mode.items():
            print(f"  {mode}: {count}")
    else:
        print("  none")

    print("By symbol")
    if summary.by_symbol:
        for row in summary.by_symbol:
            print(
                f"  {row.symbol}: records={row.count}, accepted={row.accepted}, "
                f"blocked={row.blocked}, requested={money(row.requested_notional_usd, cents=False)}, "
                f"adjusted={money(row.adjusted_notional_usd, cents=False)}, "
                f"trimmed={money(row.trimmed_notional_usd, cents=False)}"
            )
    else:
        print("  none")

    if args.recent <= 0:
        return

    print(f"Recent records ({min(args.recent, len(records))})")
    for record in records[-args.recent :]:
        request = record.get("request", {})
        decision = record.get("decision", {})
        print(
            f"  {record.get('created_at_utc', 'unknown')} | "
            f"{record.get('status', 'UNKNOWN')} | "
            f"{request.get('side', 'UNKNOWN')} {request.get('symbol', 'UNKNOWN')} | "
            f"requested={money(float(request.get('target_notional_usd', 0.0)), cents=False)} | "
            f"adjusted={money(float(decision.get('adjusted_notional_usd', 0.0)), cents=False)} | "
            f"{decision.get('reason', 'no reason')}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
