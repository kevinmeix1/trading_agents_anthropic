from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.reporting.dashboard import DashboardOptions, serve_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Claude Agent Trader dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backtest-dir", default="outputs/backtests")
    parser.add_argument("--live-monitor", default="outputs/live_competition_monitor.csv")
    parser.add_argument("--live-journal", default="outputs/live_dry_run_journal.jsonl")
    parser.add_argument("--dry-journal", default="outputs/dry_run_journal.jsonl")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser.")
    return parser


def run(args: argparse.Namespace) -> None:
    serve_dashboard(
        host=args.host,
        port=args.port,
        options=DashboardOptions(
            backtest_dir=Path(args.backtest_dir),
            live_monitor_path=Path(args.live_monitor),
            live_journal_path=Path(args.live_journal),
            dry_journal_path=Path(args.dry_journal),
        ),
        open_browser=args.open,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
