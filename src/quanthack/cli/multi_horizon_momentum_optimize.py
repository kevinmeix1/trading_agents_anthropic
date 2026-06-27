from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.multi_horizon_momentum_optimizer import (
    DEFAULT_MULTI_HORIZON_MOMENTUM_PARAMETER_SETS,
    MultiHorizonMomentumParameterSet,
    optimize_multi_horizon_momentum_parameters,
    write_multi_horizon_momentum_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize multi-horizon momentum parameters with portfolio backtests."
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
            "Candidate as label,fast_lookback,slow_lookback,volatility_lookback,"
            "baseline_volatility_lookback,min_fast_move_bps,min_slow_move_bps,"
            "min_trend_efficiency,min_volatility_ratio,max_volatility_ratio,"
            "max_holding_period[,allowed_utc_hours]. Use hours like 10|11|12."
        ),
    )
    parser.add_argument("--include-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--output",
        default="outputs/backtests/multi_horizon_momentum_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_MULTI_HORIZON_MOMENTUM_PARAMETER_SETS
    )
    result = optimize_multi_horizon_momentum_parameters(
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
    write_multi_horizon_momentum_optimization_csv(result, args.output)

    print("Multi-Horizon Momentum Optimization")
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
            f"fast={params.fast_lookback}, "
            f"slow={params.slow_lookback}, "
            f"vol={params.volatility_lookback}, "
            f"base_vol={params.baseline_volatility_lookback}, "
            f"fast_bps={params.min_fast_move_bps:.1f}, "
            f"slow_bps={params.min_slow_move_bps:.1f}, "
            f"eff={params.min_trend_efficiency:.2f}, "
            f"vol_ratio={params.min_volatility_ratio:.2f}-{params.max_volatility_ratio:.2f}, "
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


def _parse_candidate(raw: str) -> MultiHorizonMomentumParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {11, 12}:
        raise argparse.ArgumentTypeError(
            "candidate must be label,fast_lookback,slow_lookback,"
            "volatility_lookback,baseline_volatility_lookback,"
            "min_fast_move_bps,min_slow_move_bps,min_trend_efficiency,"
            "min_volatility_ratio,max_volatility_ratio,max_holding_period"
            "[,allowed_utc_hours]"
        )
    (
        label,
        fast_lookback,
        slow_lookback,
        volatility_lookback,
        baseline_volatility_lookback,
        min_fast_move_bps,
        min_slow_move_bps,
        min_trend_efficiency,
        min_volatility_ratio,
        max_volatility_ratio,
        max_holding_period,
        *rest,
    ) = parts
    allowed_utc_hours = _parse_hours(rest[0]) if rest else None
    return MultiHorizonMomentumParameterSet(
        label=label,
        fast_lookback=int(fast_lookback),
        slow_lookback=int(slow_lookback),
        volatility_lookback=int(volatility_lookback),
        baseline_volatility_lookback=int(baseline_volatility_lookback),
        min_fast_move_bps=float(min_fast_move_bps),
        min_slow_move_bps=float(min_slow_move_bps),
        min_trend_efficiency=float(min_trend_efficiency),
        min_volatility_ratio=float(min_volatility_ratio),
        max_volatility_ratio=float(max_volatility_ratio),
        max_holding_period=int(max_holding_period),
        allowed_utc_hours=allowed_utc_hours,
    )


def _parse_hours(raw: str) -> tuple[int, ...]:
    if not raw:
        raise argparse.ArgumentTypeError("allowed_utc_hours cannot be empty")
    return tuple(int(part.strip()) for part in raw.split("|") if part.strip())


def _format_hours(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return "config"
    return "|".join(str(hour) for hour in hours)
