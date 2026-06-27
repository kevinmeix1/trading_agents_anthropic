from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES
from quanthack.backtesting.strategy_compare import compare_strategies, write_strategy_comparison_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare configured strategies by backtest.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, action="append", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--output", default="outputs/backtests/strategy_comparison.csv")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_names = tuple(args.strategy) if args.strategy else STRATEGY_NAMES
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    comparison = compare_strategies(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=strategy_names,
        symbol=args.symbol,
    )
    write_strategy_comparison_csv(comparison, args.output)

    print("Strategy Comparison")
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Results CSV: {args.output}")
    if comparison.best is not None:
        print(f"  Best by test rank: {comparison.best.strategy_name}")

    print("  rank | strategy        | final equity  | return  | sharpe  | max dd  | fills")
    for rank, row in enumerate(comparison.rows, start=1):
        metrics = row.result.metrics
        print(
            f"  {rank:>4} | "
            f"{row.strategy_name:<15} | "
            f"{money(metrics.final_equity):>13} | "
            f"{metrics.total_return_pct:>7.3%} | "
            f"{metrics.sharpe_ratio:>7.3f} | "
            f"{metrics.max_drawdown_pct:>7.3%} | "
            f"{len(row.result.fills):>5}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
