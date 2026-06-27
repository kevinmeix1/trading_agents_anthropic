from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.kalman_trend_optimizer import (
    DEFAULT_KALMAN_TREND_PARAMETER_SETS,
    KalmanTrendParameterSet,
    optimize_kalman_trend_parameters,
    write_kalman_trend_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize Kalman trend parameters with portfolio backtests."
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
            "Candidate as label,lookback,min_abs_slope_bps,"
            "min_trend_efficiency,min_expected_edge_bps,"
            "expected_holding_bars,max_holding_period"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/kalman_trend_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_KALMAN_TREND_PARAMETER_SETS
    )
    result = optimize_kalman_trend_parameters(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        parameter_sets=parameter_sets,
    )
    write_kalman_trend_optimization_csv(result, args.output)

    print("Kalman Trend Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates, start=1):
        params = candidate.parameters
        metrics = candidate.comparison_row.competition_metrics
        print(
            f"  {rank}. {params.label}: "
            f"lookback={params.lookback}, "
            f"slope={params.min_abs_slope_bps:.2f}, "
            f"eff={params.min_trend_efficiency:.2f}, "
            f"edge={params.min_expected_edge_bps:.1f}, "
            f"hold={params.max_holding_period}, "
            f"proxy={candidate.comparison_row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> KalmanTrendParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "candidate must be label,lookback,min_abs_slope_bps,"
            "min_trend_efficiency,min_expected_edge_bps,"
            "expected_holding_bars,max_holding_period"
        )
    (
        label,
        lookback,
        min_abs_slope_bps,
        min_trend_efficiency,
        min_expected_edge_bps,
        expected_holding_bars,
        max_holding_period,
    ) = parts
    try:
        return KalmanTrendParameterSet(
            label=label,
            lookback=int(lookback),
            min_abs_slope_bps=float(min_abs_slope_bps),
            min_trend_efficiency=float(min_trend_efficiency),
            min_expected_edge_bps=float(min_expected_edge_bps),
            expected_holding_bars=int(expected_holding_bars),
            max_holding_period=int(max_holding_period),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
