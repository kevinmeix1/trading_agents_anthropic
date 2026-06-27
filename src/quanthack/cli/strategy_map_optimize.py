from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.strategy_map_optimizer import (
    optimize_strategy_map,
    write_strategy_map_optimization_csv,
    write_symbol_strategy_scores_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize per-symbol strategy maps with shared-risk backtests."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--include-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--min-positive-pnl-usd",
        type=float,
        default=0.0,
        help="Minimum single-symbol P&L required for positive-only map inclusion.",
    )
    parser.add_argument(
        "--top-symbol-count",
        action="append",
        type=int,
        default=None,
        help="Top-N positive symbols to test. Repeat to test several N values.",
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/strategy_map_optimization.csv",
    )
    parser.add_argument(
        "--score-output",
        default="outputs/backtests/strategy_map_symbol_scores.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    strategies = tuple(args.strategy) if args.strategy else (
        "champion_ensemble",
        "macd_momentum",
        "kalman_trend",
    )
    result = optimize_strategy_map(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=strategies,
        symbols=tuple(args.symbol) if args.symbol else None,
        include_walk_forward=args.include_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_positive_pnl_usd=args.min_positive_pnl_usd,
        top_symbol_counts=tuple(args.top_symbol_count or (3, 4, 5, 6)),
    )
    write_strategy_map_optimization_csv(result, args.output)
    write_symbol_strategy_scores_csv(result, args.score_output)

    print("Strategy Map Optimization")
    print(f"  Symbols: {', '.join(result.available_symbols)}")
    print(f"  Strategies: {', '.join(result.strategy_names)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Walk-forward: {'yes' if args.include_walk_forward else 'no'}")
    print(f"  Output CSV: {args.output}")
    print(f"  Score CSV: {args.score_output}")
    print("Ranked maps")
    for rank, candidate in enumerate(result.candidates, start=1):
        metrics = candidate.competition_metrics
        wf = candidate.walk_forward
        wf_text = (
            ""
            if wf is None
            else (
                f", wf_pos={wf.positive_fold_fraction:.1%}, "
                f"wf_active_pos={wf.active_positive_fold_fraction:.1%}, "
                f"wf_nonneg={wf.non_negative_fold_fraction:.1%}, "
                f"wf_active_med={wf.median_active_test_return_pct:.3%}"
            )
        )
        print(
            f"  {rank}. {candidate.label}: "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}, "
            f"map={candidate.strategy_map_text}"
            f"{wf_text}"
        )

    print("Top symbol-strategy scores")
    for score in result.symbol_scores[:10]:
        print(
            f"  {score.symbol} {score.strategy_name}: "
            f"pnl={money(score.total_pnl_usd)}, "
            f"return={score.return_pct:.3%}, "
            f"drawdown={score.max_drawdown_pct:.3%}, "
            f"fills={score.fills}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
