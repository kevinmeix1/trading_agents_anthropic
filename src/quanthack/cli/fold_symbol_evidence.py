from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.reporting.fold_symbol_evidence import (
    FoldSymbolEvidencePolicy,
    build_fold_symbol_evidence_report,
    write_fold_symbol_evidence_detail_csv,
    write_fold_symbol_evidence_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a past-only symbol evidence gate from fold trade attribution."
        )
    )
    parser.add_argument("--attribution-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--lookback-folds", type=int, default=2)
    parser.add_argument("--min-prior-active-folds", type=int, default=1)
    parser.add_argument("--min-prior-realized-events", type=int, default=1)
    parser.add_argument("--min-prior-pnl-usd", type=float, default=0.0)
    parser.add_argument("--min-prior-win-rate", type=float, default=0.0)
    parser.add_argument(
        "--block-without-history",
        action="store_true",
        help="Block symbols until they have prior active fold evidence.",
    )
    parser.add_argument(
        "--detail-output",
        default="outputs/backtests/fold_symbol_evidence_detail.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/backtests/fold_symbol_evidence_summary.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    report = build_fold_symbol_evidence_report(
        attribution_csv=args.attribution_csv,
        folds_csv=args.folds_csv,
        symbols=tuple(args.symbol) if args.symbol else None,
        policy=FoldSymbolEvidencePolicy(
            lookback_folds=args.lookback_folds,
            min_prior_active_folds=args.min_prior_active_folds,
            min_prior_realized_events=args.min_prior_realized_events,
            min_prior_pnl_usd=args.min_prior_pnl_usd,
            min_prior_win_rate=args.min_prior_win_rate,
            allow_without_history=not args.block_without_history,
        ),
    )
    write_fold_symbol_evidence_detail_csv(report, args.detail_output)
    write_fold_symbol_evidence_summary_csv(report, args.summary_output)

    print("Fold Symbol Evidence Gate")
    print(f"  Attribution CSV: {args.attribution_csv}")
    print(f"  Folds CSV: {args.folds_csv}")
    print(f"  Detail CSV: {args.detail_output}")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Lookback folds: {report.policy.lookback_folds}")
    print(f"  Allowed symbol-folds: {report.allowed_fraction:.1%}")
    print(f"  Ungated realized P&L: {money(report.ungated_realized_pnl_usd)}")
    print(f"  Gated realized P&L: {money(report.gated_realized_pnl_usd)}")
    print(f"  Simulated delta: {money(report.simulated_delta_usd)}")
    print(f"  Avoided loss: {money(report.avoided_loss_usd)}")
    print(f"  Missed gain: {money(report.missed_gain_usd)}")
    print("Folds")
    for row in report.fold_rows:
        print(
            f"  fold {row.fold}: allowed={row.allowed_symbols}/{row.symbols}, "
            f"ungated={money(row.ungated_realized_pnl_usd)}, "
            f"gated={money(row.gated_realized_pnl_usd)}, "
            f"delta={money(row.simulated_delta_usd)}, "
            f"avoided={money(row.avoided_loss_usd)}, "
            f"missed={money(row.missed_gain_usd)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
