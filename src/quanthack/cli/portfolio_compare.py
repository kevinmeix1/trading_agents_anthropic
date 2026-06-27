from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.portfolio_strategy_compare import (
    compare_portfolio_strategies,
    write_portfolio_strategy_comparison_csv,
)
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare strategies on a shared-risk multi-symbol portfolio."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--output",
        default="outputs/backtests/portfolio_strategy_comparison.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    comparison = compare_portfolio_strategies(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=tuple(args.strategy) if args.strategy else STRATEGY_NAMES,
        symbols=tuple(args.symbol) if args.symbol else None,
    )
    write_portfolio_strategy_comparison_csv(comparison, args.output)

    print("Portfolio Strategy Comparison")
    print(f"  Symbols: {', '.join(comparison.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, row in enumerate(comparison.rows, start=1):
        metrics = row.competition_metrics
        print(
            f"  {rank}. {row.strategy_name}: "
            f"proxy={row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={row.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
