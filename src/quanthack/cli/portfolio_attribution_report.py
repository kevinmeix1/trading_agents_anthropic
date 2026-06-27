from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.portfolio_attribution import (
    build_portfolio_attribution_report,
    write_portfolio_attribution_csv,
)
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report portfolio P&L attribution by symbol, signal, hour, and side."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--output",
        default="outputs/backtests/portfolio_attribution.csv",
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy or config.active_strategy
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    prices = load_price_history(price_csv)
    quotes = load_quote_history(quote_csv)
    symbols = tuple(args.symbol or sorted(set(prices.symbols()) & set(quotes.symbols())))
    if not symbols:
        raise SystemExit("No symbols found in both price and quote data.")

    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(strategy_name, symbol=symbol)
            for symbol in symbols
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in symbols
        },
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
    )
    result = engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )
    report = build_portfolio_attribution_report(result)
    write_portfolio_attribution_csv(report, args.output)

    print("Portfolio Attribution")
    print(f"  Strategy: {strategy_name}")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Fills: {report.total_fills}")
    print(f"  Realized P&L: {money(report.total_realized_pnl_usd)}")
    print(f"  Turnover: {money(report.total_turnover_notional_usd)}")
    print(f"  CSV: {args.output}")
    print("Weakest rows")
    for row in report.weakest_rows[: args.limit]:
        print(
            f"  {row.symbol} {row.primary_signal} h{row.utc_hour:02d} {row.side}: "
            f"pnl={money(row.realized_pnl_usd)}, "
            f"fills={row.fills}, "
            f"win={row.win_rate:.1%}"
        )
    print("Strongest rows")
    for row in report.strongest_rows[: args.limit]:
        print(
            f"  {row.symbol} {row.primary_signal} h{row.utc_hour:02d} {row.side}: "
            f"pnl={money(row.realized_pnl_usd)}, "
            f"fills={row.fills}, "
            f"win={row.win_rate:.1%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
