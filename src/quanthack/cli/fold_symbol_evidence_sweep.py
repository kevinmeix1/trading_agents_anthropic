from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.reporting.fold_symbol_evidence import (
    sweep_fold_symbol_evidence_policies,
    write_fold_symbol_evidence_sweep_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep past-only symbol evidence gate policies."
    )
    parser.add_argument("--attribution-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--lookback-folds", action="append", type=int, default=None)
    parser.add_argument("--min-prior-pnl-usd", action="append", type=float, default=None)
    parser.add_argument("--min-prior-win-rate", action="append", type=float, default=None)
    parser.add_argument("--min-prior-active-folds", type=int, default=1)
    parser.add_argument("--min-prior-realized-events", type=int, default=1)
    parser.add_argument("--block-without-history", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/backtests/fold_symbol_evidence_sweep.csv",
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    report = sweep_fold_symbol_evidence_policies(
        attribution_csv=args.attribution_csv,
        folds_csv=args.folds_csv,
        symbols=tuple(args.symbol) if args.symbol else None,
        lookback_folds_values=tuple(args.lookback_folds or (1, 2, 3)),
        min_prior_pnl_usd_values=tuple(
            args.min_prior_pnl_usd or (-1_000.0, 0.0, 250.0, 1_000.0)
        ),
        min_prior_win_rate_values=tuple(args.min_prior_win_rate or (0.0, 0.5)),
        min_prior_active_folds=args.min_prior_active_folds,
        min_prior_realized_events=args.min_prior_realized_events,
        allow_without_history=not args.block_without_history,
    )
    write_fold_symbol_evidence_sweep_csv(report, args.output)

    print("Fold Symbol Evidence Sweep")
    print(f"  Attribution CSV: {args.attribution_csv}")
    print(f"  Folds CSV: {args.folds_csv}")
    print(f"  Output CSV: {args.output}")
    print(f"  Candidates: {len(report.candidates)}")
    print("Top policies")
    for candidate in report.candidates[: max(args.limit, 0)]:
        policy = candidate.policy
        print(
            f"  {candidate.rank}. lookback={policy.lookback_folds}, "
            f"min_pnl={money(policy.min_prior_pnl_usd)}, "
            f"win_rate={policy.min_prior_win_rate:.0%}, "
            f"allowed={candidate.allowed_fraction:.1%}, "
            f"delta={money(candidate.simulated_delta_usd)}, "
            f"gated={money(candidate.gated_realized_pnl_usd)}, "
            f"avoided={money(candidate.avoided_loss_usd)}, "
            f"missed={money(candidate.missed_gain_usd)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
