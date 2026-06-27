from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.trading.execution import DryRunExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the current portfolio reconstructed from the dry-run journal."
    )
    parser.add_argument("--journal", default="outputs/dry_run_journal.jsonl")
    return parser


def run(args: argparse.Namespace) -> None:
    portfolio = DryRunExecutor(Path(args.journal)).current_portfolio()

    print("Dry-Run Positions")
    print(f"  Journal: {args.journal}")
    print(f"  Gross notional: ${portfolio.gross_notional_usd:,.0f}")

    if not portfolio.positions:
        print("  Positions: none")
        return

    print("  Positions:")
    for position in portfolio.positions:
        side = "LONG" if position.notional_usd > 0 else "SHORT"
        print(f"    {position.symbol}: {side} ${abs(position.notional_usd):,.0f}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
