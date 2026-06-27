from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.reporting.adaptive_handoff_diagnostic import (
    build_adaptive_handoff_diagnostic,
    write_adaptive_handoff_diagnostic_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge adaptive oracle and ex-ante regime diagnostics into "
            "handoff labels."
        )
    )
    parser.add_argument(
        "--oracle-folds-csv",
        default="outputs/research/adaptive_current_top_oracle_folds.csv",
    )
    parser.add_argument(
        "--oracle-candidates-csv",
        default="outputs/research/adaptive_current_top_oracle_candidates.csv",
    )
    parser.add_argument(
        "--regime-summary-csv",
        default="outputs/research/adaptive_current_top_regime_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/research/adaptive_current_top_handoff_diagnostic.csv",
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    report = build_adaptive_handoff_diagnostic(
        oracle_folds_csv=args.oracle_folds_csv,
        oracle_candidates_csv=args.oracle_candidates_csv,
        regime_summary_csv=args.regime_summary_csv,
    )
    write_adaptive_handoff_diagnostic_csv(report, args.output)

    print("Adaptive Handoff Diagnostic")
    print(f"  Oracle folds CSV: {args.oracle_folds_csv}")
    print(f"  Oracle candidates CSV: {args.oracle_candidates_csv}")
    print(f"  Regime summary CSV: {args.regime_summary_csv}")
    print(f"  Output CSV: {args.output}")
    print(f"  Folds: {report.fold_count}")
    print(f"  Total regret: {report.total_regret_pct:.3%}")
    print("Diagnosis counts")
    for diagnosis, count in report.diagnosis_counts:
        print(f"  {diagnosis}: {count}")
    print("Largest regret rows")
    for row in report.largest_regret_rows[: max(args.limit, 0)]:
        print(
            f"  fold {row.fold}: {row.diagnosis}, "
            f"selected={row.selected_strategy} ({row.selected_return_pct:.3%}), "
            f"oracle={row.oracle_strategy} ({row.oracle_return_pct:.3%}), "
            f"regret={row.regret_pct:.3%}, "
            f"chop={row.chop_fraction:.1%}, "
            f"consensus={row.trend_consensus:.1%}, "
            f"champion_train_adj={row.champion_train_adjusted_return_pct:.3%}, "
            f"macd_train_adj={row.macd_train_adjusted_return_pct:.3%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
