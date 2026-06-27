from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.reporting.operator_dashboard import build_operator_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a static HTML operator dashboard from live/profile artifacts."
    )
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    parser.add_argument(
        "--snapshot-csv",
        default="outputs/research/deployment_profile_conservative_signal_snapshot.csv",
    )
    parser.add_argument(
        "--allocation-csv",
        default="outputs/research/profile_live_allocation.csv",
    )
    parser.add_argument(
        "--monitor-csv",
        default="outputs/research/profile_live_monitor.csv",
    )
    parser.add_argument(
        "--ticket-csv",
        default="outputs/research/mt5_ticket_sheet_asof_0815.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/operator_dashboard.html",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    dashboard = build_operator_dashboard(
        profile_pack_json=args.profile_pack_json,
        snapshot_csv=args.snapshot_csv,
        allocation_csv=args.allocation_csv,
        monitor_csv=args.monitor_csv,
        ticket_csv=args.ticket_csv,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard.html, encoding="utf-8")
    print(f"Dashboard: {output_path}")
    print(f"Title: {dashboard.title}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
