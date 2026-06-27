from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.backtest import BacktestEngine, FillModel
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.router_report import (
    build_router_attribution_report,
    write_router_attribution_csv,
)
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report alpha-router signal attribution.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="alpha_router")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--output", default="outputs/backtests/router_attribution.csv")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy
    symbol = args.symbol or config.strategy_symbol(strategy_name)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    engine = BacktestEngine(
        strategy=config.build_strategy(strategy_name, symbol=symbol),
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
    )
    result = engine.run(
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbol=symbol,
        starting_equity=config.competition.starting_equity,
    )
    report = build_router_attribution_report(result)
    write_router_attribution_csv(report, args.output)

    print("Router Attribution")
    print(f"  Strategy: {strategy_name}")
    print(f"  Symbol: {report.symbol}")
    print(f"  Fills: {report.total_fills}")
    print(f"  Realized P&L: {money(report.total_realized_pnl_usd)}")
    print(f"  Turnover: {money(report.total_turnover_notional_usd)}")
    print(f"  Conflict fills: {report.conflict_fills}")
    print(f"  CSV: {args.output}")
    print("By primary signal")
    if not report.rows:
        print("  none")
        return
    for row in report.rows:
        print(
            f"  {row.primary_signal}: fills={row.fills}, "
            f"realized={money(row.realized_pnl_usd)}, "
            f"win_rate={row.win_rate:.1%}, "
            f"conflicts={row.conflict_fills}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
