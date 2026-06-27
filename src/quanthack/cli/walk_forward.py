from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES
from quanthack.backtesting.walk_forward import (
    run_walk_forward,
    write_walk_forward_folds_csv,
    write_walk_forward_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chronological walk-forward strategy evaluation."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--ma-fast-window", type=int, action="append", default=None)
    parser.add_argument("--ma-slow-window", type=int, action="append", default=None)
    parser.add_argument("--ma-min-separation-bps", type=float, action="append", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--folds-output", default=None)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    settings = config.walk_forward
    symbol = args.symbol or config.strategy_symbol(config.active_strategy)
    strategy_names = tuple(args.strategy or STRATEGY_NAMES)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    prices = load_price_history(price_csv)
    quotes = load_quote_history(quote_csv)

    result = run_walk_forward(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=strategy_names,
        symbol=symbol,
        train_size=args.train_size or settings.train_size,
        test_size=args.test_size or settings.test_size,
        step_size=args.step_size or settings.step_size,
        momentum_lookbacks=config.sweep.lookbacks,
        momentum_threshold_bps=config.sweep.threshold_bps,
        ma_fast_windows=tuple(args.ma_fast_window or settings.ma_fast_windows),
        ma_slow_windows=tuple(args.ma_slow_window or settings.ma_slow_windows),
        ma_min_separation_bps=tuple(
            args.ma_min_separation_bps or settings.ma_min_separation_bps
        ),
        cost_multipliers=settings.cost_multipliers,
        min_total_fills=settings.min_total_fills,
        min_profitable_fold_fraction=settings.min_profitable_fold_fraction,
        max_worst_drawdown_pct=settings.max_worst_drawdown_pct,
    )

    summary_output = args.summary_output or settings.summary_csv
    folds_output = args.folds_output or settings.folds_csv
    write_walk_forward_summary_csv(result, summary_output)
    write_walk_forward_folds_csv(result, folds_output)

    print("Walk-Forward Evaluation")
    print(f"  Symbol: {symbol}")
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Summary CSV: {summary_output}")
    print(f"  Folds CSV: {folds_output}")
    if result.best is not None:
        print(f"  Best robust rank: {result.best.strategy_name}")
    print(
        "  rank | strategy        | eligible | folds | median sharpe | "
        "lq return | worst dd | prof folds | fills"
    )
    for rank, summary in enumerate(result.summaries, start=1):
        print(
            f"  {rank:>4} | {summary.strategy_name:<15} | "
            f"{str(summary.eligible):<8} | "
            f"{len(summary.folds):>5} | "
            f"{summary.median_test_sharpe:>13.3f} | "
            f"{summary.lower_quartile_test_return:>9.3%} | "
            f"{summary.worst_test_drawdown:>8.3%} | "
            f"{summary.profitable_fold_fraction:>10.1%} | "
            f"{summary.total_test_fills:>5}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
