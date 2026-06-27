from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.reporting.fold_trade_attribution import (
    build_fold_trade_attribution_report,
    write_fold_trade_attribution_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute realized P&L by fold, symbol, signal, UTC hour, and side."
    )
    parser.add_argument("--fills-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument(
        "--output",
        default="outputs/backtests/fold_trade_attribution.csv",
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    report = build_fold_trade_attribution_report(
        fills_csv=args.fills_csv,
        folds_csv=args.folds_csv,
    )
    write_fold_trade_attribution_csv(report, args.output)

    print("Fold Trade Attribution")
    print(f"  Fills CSV: {args.fills_csv}")
    print(f"  Folds CSV: {args.folds_csv}")
    print(f"  Output CSV: {args.output}")
    print("Weakest rows")
    for row in report.weakest_rows[: max(args.limit, 0)]:
        print(
            f"  fold {row.fold} {row.symbol} {row.primary_signal} "
            f"h{row.utc_hour:02d} {row.side}: "
            f"pnl={money(row.realized_pnl_usd)}, "
            f"fills={row.fills}, win={row.win_rate:.1%}, "
            f"fold_return={row.fold_return_pct:.3%}"
        )
    print("Strongest rows")
    for row in report.strongest_rows[: max(args.limit, 0)]:
        print(
            f"  fold {row.fold} {row.symbol} {row.primary_signal} "
            f"h{row.utc_hour:02d} {row.side}: "
            f"pnl={money(row.realized_pnl_usd)}, "
            f"fills={row.fills}, win={row.win_rate:.1%}, "
            f"fold_return={row.fold_return_pct:.3%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
