from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.strategy_attribution import (
    run_strategy_attribution,
    write_strategy_attribution_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write per-symbol P&L attribution for one or more strategies."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--output",
        default="outputs/backtests/strategy_attribution.csv",
    )
    parser.add_argument("--limit", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    strategy_names = tuple(args.strategy) if args.strategy else (config.active_strategy,)
    report = run_strategy_attribution(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=strategy_names,
        symbols=tuple(args.symbol) if args.symbol else None,
    )
    write_strategy_attribution_csv(report, args.output)

    print("Strategy Attribution")
    print(f"  Strategies: {', '.join(strategy_names)}")
    print(f"  Symbols: {', '.join(report.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print("Top rows")
    for row in report.rows[: args.limit]:
        print(
            f"  {row.strategy_name}/{row.symbol}: "
            f"pnl={money(row.total_pnl_usd)}, "
            f"fills={row.fills}, "
            f"win={row.win_rate:.1%}, "
            f"pf={row.profit_factor:.2f}, "
            f"portfolio_return={row.portfolio_return_pct:.3%}, "
            f"risk={row.portfolio_risk_score}/100"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
