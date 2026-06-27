from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.liquidity_sweep_reversal_optimizer import (
    DEFAULT_LIQUIDITY_SWEEP_REVERSAL_PARAMETER_SETS,
    LiquiditySweepReversalParameterSet,
    optimize_liquidity_sweep_reversal_parameters,
    write_liquidity_sweep_reversal_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize liquidity-sweep reversal parameters with portfolio backtests."
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
            "Candidate as label,lookback,min_sweep_bps,reentry_buffer_bps,"
            "min_range_width_bps,max_sweep_bps,max_trend_efficiency,"
            "min_expected_edge_bps,max_holding_period[,allowed_utc_hours]. "
            "Use hours like 10|11|12."
        ),
    )
    parser.add_argument("--include-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--output",
        default="outputs/backtests/liquidity_sweep_reversal_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_LIQUIDITY_SWEEP_REVERSAL_PARAMETER_SETS
    )
    result = optimize_liquidity_sweep_reversal_parameters(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        parameter_sets=parameter_sets,
        include_walk_forward=args.include_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_liquidity_sweep_reversal_optimization_csv(result, args.output)

    print("Liquidity Sweep Reversal Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Walk-forward: {'yes' if args.include_walk_forward else 'no'}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates, start=1):
        params = candidate.parameters
        metrics = candidate.comparison_row.competition_metrics
        wf = candidate.walk_forward
        wf_text = (
            ""
            if wf is None
            else (
                f", wf_pos={wf.positive_fold_fraction:.1%}, "
                f"wf_active={wf.active_fold_fraction:.1%}, "
                f"wf_active_pos={wf.active_positive_fold_fraction:.1%}, "
                f"wf_nonneg={wf.non_negative_fold_fraction:.1%}, "
                f"wf_active_med={wf.median_active_test_return_pct:.3%}"
            )
        )
        print(
            f"  {rank}. {params.label}: "
            f"lookback={params.lookback}, "
            f"sweep={params.min_sweep_bps:.2f}, "
            f"reentry={params.reentry_buffer_bps:.2f}, "
            f"range={params.min_range_width_bps:.2f}, "
            f"max_sweep={params.max_sweep_bps:.1f}, "
            f"max_eff={params.max_trend_efficiency:.2f}, "
            f"edge={params.min_expected_edge_bps:.2f}, "
            f"hold={params.max_holding_period}, "
            f"hours={_format_hours(params.allowed_utc_hours)}, "
            f"proxy={candidate.comparison_row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
            f"{wf_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> LiquiditySweepReversalParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {9, 10}:
        raise argparse.ArgumentTypeError(
            "candidate must be label,lookback,min_sweep_bps,reentry_buffer_bps,"
            "min_range_width_bps,max_sweep_bps,max_trend_efficiency,"
            "min_expected_edge_bps,max_holding_period[,allowed_utc_hours]"
        )
    (
        label,
        lookback,
        min_sweep_bps,
        reentry_buffer_bps,
        min_range_width_bps,
        max_sweep_bps,
        max_trend_efficiency,
        min_expected_edge_bps,
        max_holding_period,
    ) = parts[:9]
    allowed_utc_hours = _parse_hours(parts[9]) if len(parts) == 10 else None
    try:
        return LiquiditySweepReversalParameterSet(
            label=label,
            lookback=int(lookback),
            min_sweep_bps=float(min_sweep_bps),
            reentry_buffer_bps=float(reentry_buffer_bps),
            min_range_width_bps=float(min_range_width_bps),
            max_sweep_bps=float(max_sweep_bps),
            max_trend_efficiency=float(max_trend_efficiency),
            min_expected_edge_bps=float(min_expected_edge_bps),
            max_holding_period=int(max_holding_period),
            allowed_utc_hours=allowed_utc_hours,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_hours(raw: str) -> tuple[int, ...] | None:
    if not raw.strip():
        return None
    separators_normalized = raw.replace(";", "|").replace(":", "|")
    hours = tuple(
        int(part.strip())
        for part in separators_normalized.split("|")
        if part.strip()
    )
    return hours or None


def _format_hours(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return "config"
    return "|".join(str(hour) for hour in hours)
