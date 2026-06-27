from __future__ import annotations

import argparse
from collections.abc import Sequence
from glob import glob
from pathlib import Path

from quanthack.backtesting.experiment_leaderboard import (
    build_experiment_leaderboard,
    write_experiment_leaderboard_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank walk-forward experiment summary CSVs."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help=(
            "Summary CSV path or glob. Repeat for multiple groups. "
            "Default: outputs/backtests/*summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/experiment_leaderboard.csv",
    )
    parser.add_argument("--limit", type=int, default=20)
    return parser


def run(args: argparse.Namespace) -> None:
    paths = _expand_inputs(tuple(args.input or ("outputs/backtests/*summary.csv",)))
    rows = build_experiment_leaderboard(paths)
    write_experiment_leaderboard_csv(rows, args.output)

    print("Experiment Leaderboard")
    print(f"  Inputs: {len(paths)}")
    print(f"  Ranked rows: {len(rows)}")
    print(f"  Output CSV: {args.output}")
    for rank, row in enumerate(rows[: max(args.limit, 0)], start=1):
        print(
            f"  {rank}. {row.label}: "
            f"score={row.score:.3f}, "
            f"pos={row.positive_fold_fraction:.1%}, "
            f"active_pos={row.active_positive_fold_fraction:.1%}, "
            f"nonneg={row.non_negative_fold_fraction:.1%}, "
            f"cmpd={row.compounded_return_pct:.3%}, "
            f"med_active={row.median_active_return_pct:.3%}, "
            f"dd={row.worst_drawdown_pct:.3%}, "
            f"fills={row.total_evaluation_fills}, "
            f"strategies={row.strategy_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _expand_inputs(patterns: tuple[str, ...]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            paths.update(Path(match) for match in matches)
        else:
            paths.add(Path(pattern))
    return tuple(sorted(paths))
