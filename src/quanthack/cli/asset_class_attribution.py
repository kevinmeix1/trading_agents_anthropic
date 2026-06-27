from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.asset_class_attribution import (
    build_asset_class_attribution_report,
    write_asset_class_attribution_csv,
)
from quanthack.cli._format import money


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group portfolio P&L attribution by asset class."
    )
    parser.add_argument("--pnl-csv", required=True)
    parser.add_argument(
        "--output",
        default="outputs/backtests/asset_class_attribution.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    report = build_asset_class_attribution_report(args.pnl_csv)
    write_asset_class_attribution_csv(report, args.output)

    print("Asset Class Attribution")
    print(f"  P&L CSV: {report.source_csv}")
    print(f"  Portfolio P&L: {money(report.portfolio_total_pnl_usd)}")
    print(f"  Output CSV: {args.output}")
    for row in report.asset_class_rows:
        print(
            f"  {row.asset_class.value}: pnl={money(row.total_pnl_usd)}, "
            f"fills={row.fills}, winners={row.winners}, losers={row.losers}, "
            f"net_share={row.share_of_portfolio_pnl:.1%}, "
            f"abs_share={row.share_of_gross_abs_pnl:.1%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
