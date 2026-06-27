from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from quanthack.backtesting.backtest import BacktestEngine, FillModel, write_equity_curve_csv
from quanthack.cli._competition import print_competition_view
from quanthack.cli._format import money
from quanthack.backtesting.competition_score import (
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_single_symbol_equity,
)
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.pnl import write_pnl_ledger_csv
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline backtest.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--equity-output", default=None)
    parser.add_argument("--pnl-output", default=None)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy or config.active_strategy
    symbol = args.symbol or config.strategy_symbol(strategy_name)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    equity_output = args.equity_output or config.backtest.equity_curve_csv
    pnl_output = args.pnl_output or config.backtest.pnl_ledger_csv

    engine = BacktestEngine(
        strategy=config.build_strategy(strategy_name, symbol=symbol),
        risk_limits=config.risk,
        quality_limits=replace(
            config.market_quality,
            max_spread_bps=instrument_for(symbol).max_spread_bps,
        ),
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
    write_equity_curve_csv(result, equity_output)
    write_pnl_ledger_csv(result.pnl_ledger, pnl_output)
    competition_metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk_discipline = build_risk_discipline_report(
        risk_samples_from_single_symbol_equity(result.equity_curve)
    )

    metrics = result.metrics
    print("Backtest")
    print(f"  Strategy: {strategy_name}")
    print(f"  Symbol: {result.symbol}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Fills: {len(result.fills)}")
    print(f"  Observations: {metrics.observations}")
    print(f"  Final equity: {money(metrics.final_equity)}")
    print(f"  Total return: {metrics.total_return_pct:.3%}")
    print(f"  Sharpe ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max drawdown: {metrics.max_drawdown_pct:.3%}")
    print(f"  Win rate: {metrics.win_rate:.1%}")
    print(f"  Profit factor: {metrics.profit_factor:.3f}")
    print(f"  Turnover: {money(metrics.turnover_notional)}")
    print(f"  Realized P&L: {money(result.pnl_ledger.realized_pnl_usd)}")
    print(f"  Open P&L: {money(result.pnl_ledger.open_pnl_usd)}")
    print(f"  Total attributed P&L: {money(result.pnl_ledger.total_pnl_usd)}")
    print(f"  Equity curve: {equity_output}")
    print(f"  P&L ledger: {pnl_output}")
    print_competition_view(
        metrics=competition_metrics,
        risk_discipline=risk_discipline,
    )

    if result.fills:
        print("Recent fills")
        for fill in result.fills[-5:]:
            print(
                f"  {fill.timestamp} | {fill.side.value} {fill.symbol} | "
                f"fill={fill.fill_price:.5f} | adjusted={money(fill.adjusted_notional_usd)}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
