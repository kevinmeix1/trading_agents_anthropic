from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.trend_pullback_optimizer import (
    DEFAULT_TREND_PULLBACK_PARAMETER_SETS,
    TrendPullbackParameterSet,
    optimize_trend_pullback_parameters,
    write_trend_pullback_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize trend-pullback parameters with portfolio backtests."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Candidate as label,lookback,pullback_window,min_trend_bps,"
            "min_trend_efficiency,min_pullback_bps,max_pullback_bps,"
            "min_resume_bps,min_expected_edge_bps"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/trend_pullback_optimization.csv",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run portfolio walk-forward for every candidate.",
    )
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--test-size", type=int, default=240)
    parser.add_argument("--step-size", type=int, default=240)
    parser.add_argument("--max-baskets", type=int, default=30)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_TREND_PULLBACK_PARAMETER_SETS
    )
    result = optimize_trend_pullback_parameters(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        parameter_sets=parameter_sets,
        include_walk_forward=args.walk_forward,
        walk_forward_train_size=args.train_size,
        walk_forward_test_size=args.test_size,
        walk_forward_step_size=args.step_size,
        walk_forward_max_baskets=args.max_baskets,
    )
    write_trend_pullback_optimization_csv(result, args.output)

    print("Trend Pullback Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Walk-forward: {'yes' if args.walk_forward else 'no'}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates, start=1):
        params = candidate.parameters
        metrics = candidate.comparison_row.competition_metrics
        summary = candidate.walk_forward_summary
        walk_text = (
            ""
            if summary is None
            else (
                f", wf_stable={summary.stable_fold_fraction:.1%}, "
                f"wf_median_return={summary.median_test_return_pct:.3%}, "
                f"wf_fills={summary.total_test_fills}, "
                f"wf_eligible={summary.eligible}"
            )
        )
        print(
            f"  {rank}. {params.label}: "
            f"lookback={params.lookback}, "
            f"pullback={params.pullback_window}, "
            f"trend={params.min_trend_bps:.1f}, "
            f"eff={params.min_trend_efficiency:.2f}, "
            f"resume={params.min_resume_bps:.1f}, "
            f"edge={params.min_expected_edge_bps:.1f}, "
            f"proxy={candidate.comparison_row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
            f"{walk_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> TrendPullbackParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 9:
        raise argparse.ArgumentTypeError(
            "candidate must be label,lookback,pullback_window,min_trend_bps,"
            "min_trend_efficiency,min_pullback_bps,max_pullback_bps,"
            "min_resume_bps,min_expected_edge_bps"
        )
    (
        label,
        lookback,
        pullback_window,
        min_trend_bps,
        min_trend_efficiency,
        min_pullback_bps,
        max_pullback_bps,
        min_resume_bps,
        min_expected_edge_bps,
    ) = parts
    try:
        return TrendPullbackParameterSet(
            label=label,
            lookback=int(lookback),
            pullback_window=int(pullback_window),
            min_trend_bps=float(min_trend_bps),
            min_trend_efficiency=float(min_trend_efficiency),
            min_pullback_bps=float(min_pullback_bps),
            max_pullback_bps=float(max_pullback_bps),
            min_resume_bps=float(min_resume_bps),
            min_expected_edge_bps=float(min_expected_edge_bps),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
