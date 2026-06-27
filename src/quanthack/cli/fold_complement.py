from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.fold_complement import (
    evaluate_fold_complement,
    read_fold_returns,
    write_fold_complement_csv,
    write_fold_complement_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate whether candidate sleeves complement baseline folds."
    )
    parser.add_argument("--baseline-folds", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=FOLDS_CSV",
        help="Candidate fold CSV to compare against the baseline. Repeatable.",
    )
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--detail-output", required=True)
    return parser


def run(args: argparse.Namespace) -> None:
    baseline = read_fold_returns(args.baseline_folds)
    summaries = tuple(
        evaluate_fold_complement(
            baseline=baseline,
            candidate=read_fold_returns(candidate_path),
            label=label,
        )
        for label, candidate_path in (_parse_candidate(value) for value in args.candidate)
    )
    write_fold_complement_summary_csv(summaries, args.summary_output)
    write_fold_complement_csv(summaries, args.detail_output)

    print("Fold Complement Analysis")
    print(f"  Baseline folds: {args.baseline_folds}")
    print(f"  Candidates: {len(summaries)}")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Detail CSV: {args.detail_output}")
    for summary in summaries:
        print(
            f"  {summary.label}: combined_pos={summary.combined_positive_fraction:.1%}, "
            f"combined_nonneg={summary.combined_non_negative_fraction:.1%}, "
            f"positive_on_flat={summary.candidate_positive_on_baseline_flat}, "
            f"positive_on_losing={summary.candidate_positive_on_baseline_losing}, "
            f"hurt_positive={summary.candidate_hurt_baseline_positive}, "
            f"incremental_return={summary.incremental_return_sum_pct:.3%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw_value: str) -> tuple[str, str]:
    if "=" not in raw_value:
        raise SystemExit("--candidate must use LABEL=FOLDS_CSV")
    label, path = raw_value.split("=", 1)
    if not label.strip() or not path.strip():
        raise SystemExit("--candidate label and path cannot be empty")
    return label.strip(), path.strip()
