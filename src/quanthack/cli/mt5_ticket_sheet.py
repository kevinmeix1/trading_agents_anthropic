from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.market.adapters import parse_symbol_map
from quanthack.trading.mt5_ticket_sheet import (
    build_mt5_ticket_sheet_from_snapshot_csv,
    load_contract_specs,
    write_contract_spec_template_from_snapshot_csv,
    write_mt5_ticket_sheet_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a profile signal snapshot CSV into manual MT5 tickets."
    )
    parser.add_argument(
        "--snapshot-csv",
        default="outputs/research/deployment_profile_signal_snapshot.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/research/mt5_ticket_sheet.csv",
    )
    parser.add_argument("--contract-spec-csv", default=None)
    parser.add_argument(
        "--broker-symbol-map",
        action="append",
        default=None,
        help="Map canonical to broker symbol, for example EURUSD=EURUSD.pro",
    )
    parser.add_argument("--include-holds", action="store_true")
    parser.add_argument(
        "--write-contract-spec-template",
        default=None,
        help="Optional CSV path for an MT5 Symbol Specification template.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if args.write_contract_spec_template:
        write_contract_spec_template_from_snapshot_csv(
            args.snapshot_csv,
            args.write_contract_spec_template,
        )
    contract_specs = (
        load_contract_specs(args.contract_spec_csv)
        if args.contract_spec_csv is not None
        else {}
    )
    broker_map = parse_symbol_map(tuple(args.broker_symbol_map or ()))
    tickets = build_mt5_ticket_sheet_from_snapshot_csv(
        args.snapshot_csv,
        contract_specs=contract_specs,
        broker_symbol_by_symbol=broker_map,
        include_holds=args.include_holds,
    )
    write_mt5_ticket_sheet_csv(tickets, args.output)
    _print_summary(tickets, Path(args.output), args.write_contract_spec_template)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _print_summary(tickets, output_path: Path, template_path: str | None) -> None:
    status_counts: dict[str, int] = {}
    for ticket in tickets:
        status_counts[ticket.status] = status_counts.get(ticket.status, 0) + 1
    print("MT5 Ticket Sheet")
    print(f"  Output CSV: {output_path}")
    if template_path:
        print(f"  Contract spec template: {template_path}")
    print(f"  Rows: {len(tickets)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    ready = tuple(ticket for ticket in tickets if ticket.status == "READY")
    if not ready:
        return
    print("  Ready tickets:")
    for ticket in ready[:10]:
        print(
            "   "
            f"{ticket.side} {ticket.broker_symbol} "
            f"{ticket.rounded_lots:.4f} lots "
            f"notional=${ticket.rounded_notional_usd:,.0f}"
        )
