from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.market.market_data import load_price_history
from quanthack.reporting.fold_regime_diagnostics import (
    build_fold_regime_diagnostics_report,
    write_fold_regime_detail_csv,
    write_fold_regime_summary_csv,
)
from quanthack.strategies.time_series import KalmanTrendConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explain each fixed-warmup fold using ex-ante Kalman regime, "
            "trend, and volatility diagnostics."
        )
    )
    parser.add_argument("--price-csv", default="data/full_20gb_15m_prices.csv")
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--lookback", type=int, default=80)
    parser.add_argument("--process-noise", type=float, default=1e-6)
    parser.add_argument("--observation-noise", type=float, default=1e-4)
    parser.add_argument("--min-abs-slope-bps", type=float, default=0.75)
    parser.add_argument("--min-trend-efficiency", type=float, default=0.25)
    parser.add_argument("--max-realized-volatility-bps", type=float, default=120.0)
    parser.add_argument(
        "--detail-output",
        default="outputs/backtests/fold_regime_detail.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/backtests/fold_regime_summary.csv",
    )
    parser.add_argument("--limit", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> None:
    prices = load_price_history(args.price_csv)
    report = build_fold_regime_diagnostics_report(
        prices=prices,
        folds_csv=args.folds_csv,
        symbols=tuple(args.symbol) if args.symbol else None,
        config=KalmanTrendConfig(
            lookback=args.lookback,
            process_noise=args.process_noise,
            observation_noise=args.observation_noise,
            min_abs_slope_bps=args.min_abs_slope_bps,
            min_trend_efficiency=args.min_trend_efficiency,
            max_realized_volatility_bps=args.max_realized_volatility_bps,
        ),
    )
    write_fold_regime_detail_csv(report.detail_rows, args.detail_output)
    write_fold_regime_summary_csv(report.summary_rows, args.summary_output)

    print("Fold Regime Diagnostics")
    print(f"  Price CSV: {args.price_csv}")
    print(f"  Folds CSV: {args.folds_csv}")
    print(f"  Detail CSV: {args.detail_output}")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Folds: {len(report.summary_rows)}")
    print(f"  Symbol rows: {len(report.detail_rows)}")
    print("Weakest folds")
    for row in report.weakest_folds[: max(args.limit, 0)]:
        print(
            f"  fold {row.fold}: return={row.fold_return_pct:.3%}, "
            f"trend_up={row.trend_up_symbols}, trend_down={row.trend_down_symbols}, "
            f"chop={row.chop_symbols}, high_vol={row.high_volatility_symbols}, "
            f"consensus={row.trend_consensus:.1%}, "
            f"avg_vol={row.average_realized_volatility_bps:.1f}bps"
        )
    print("Strongest folds")
    for row in report.strongest_folds[: max(args.limit, 0)]:
        print(
            f"  fold {row.fold}: return={row.fold_return_pct:.3%}, "
            f"trend_up={row.trend_up_symbols}, trend_down={row.trend_down_symbols}, "
            f"chop={row.chop_symbols}, high_vol={row.high_volatility_symbols}, "
            f"consensus={row.trend_consensus:.1%}, "
            f"avg_vol={row.average_realized_volatility_bps:.1f}bps"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
