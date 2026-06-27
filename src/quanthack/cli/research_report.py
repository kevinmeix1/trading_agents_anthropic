from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.backtest import BacktestEngine, FillModel, write_equity_curve_csv
from quanthack.core.config import load_config
from quanthack.market.data_health import validate_market_data, write_market_data_health_csv
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.pnl import write_pnl_ledger_csv
from quanthack.trading.preflight import run_preflight
from quanthack.reporting.research_report import build_research_report
from quanthack.strategies.strategy import STRATEGY_NAMES
from quanthack.backtesting.strategy_compare import compare_strategies, write_strategy_comparison_csv
from quanthack.backtesting.sweep import run_parameter_sweep, write_sweep_csv
from quanthack.backtesting.walk_forward import (
    run_walk_forward,
    write_walk_forward_folds_csv,
    write_walk_forward_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a demo-ready research HTML report.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--output", default="outputs/reports/research_report.html")
    parser.add_argument("--comparison-output", default="outputs/backtests/strategy_comparison.csv")
    parser.add_argument("--data-health-output", default="outputs/backtests/data_health.csv")
    parser.add_argument("--walk-forward-summary-output", default=None)
    parser.add_argument("--walk-forward-folds-output", default=None)
    parser.add_argument("--sweep-limit", type=int, default=6)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy or config.active_strategy
    symbol = args.symbol or config.strategy_symbol(strategy_name)
    prices = load_price_history(config.backtest.price_csv)
    quotes = load_quote_history(config.backtest.quote_csv)
    fill_model = FillModel(slippage_bps=config.backtest.slippage_bps)

    data_health = validate_market_data(
        prices=prices,
        quotes=quotes,
        symbols=(symbol,),
        max_gap_seconds=300.0,
        max_spread_bps=config.market_quality.max_spread_bps,
    )
    preflight = run_preflight(config_path=args.config)
    comparison = compare_strategies(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=STRATEGY_NAMES,
        symbol=args.symbol,
    )
    sweep = run_parameter_sweep(
        prices=prices,
        quotes=quotes,
        symbol=config.simple_momentum.symbol,
        base_config=config.simple_momentum,
        lookbacks=config.sweep.lookbacks,
        threshold_bps=config.sweep.threshold_bps,
        train_fraction=config.sweep.train_fraction,
        starting_equity=config.competition.starting_equity,
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        clock=config.competition.to_clock(),
        fill_model=fill_model,
        periods_per_year=config.backtest.periods_per_year,
    )
    walk_forward = run_walk_forward(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=STRATEGY_NAMES,
        symbol=symbol,
        train_size=config.walk_forward.train_size,
        test_size=config.walk_forward.test_size,
        step_size=config.walk_forward.step_size,
        momentum_lookbacks=config.sweep.lookbacks,
        momentum_threshold_bps=config.sweep.threshold_bps,
        cost_multipliers=config.walk_forward.cost_multipliers,
        min_total_fills=config.walk_forward.min_total_fills,
        min_profitable_fold_fraction=config.walk_forward.min_profitable_fold_fraction,
        max_worst_drawdown_pct=config.walk_forward.max_worst_drawdown_pct,
    )

    engine = BacktestEngine(
        strategy=config.build_strategy(strategy_name, symbol=symbol),
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        clock=config.competition.to_clock(),
        fill_model=fill_model,
        periods_per_year=config.backtest.periods_per_year,
    )
    backtest = engine.run(
        prices=prices,
        quotes=quotes,
        symbol=symbol,
        starting_equity=config.competition.starting_equity,
    )

    write_equity_curve_csv(backtest, config.backtest.equity_curve_csv)
    write_pnl_ledger_csv(backtest.pnl_ledger, config.backtest.pnl_ledger_csv)
    write_strategy_comparison_csv(comparison, args.comparison_output)
    write_sweep_csv(sweep, config.sweep.results_csv)
    write_market_data_health_csv(data_health, args.data_health_output)
    walk_forward_summary_output = (
        args.walk_forward_summary_output or config.walk_forward.summary_csv
    )
    walk_forward_folds_output = args.walk_forward_folds_output or config.walk_forward.folds_csv
    write_walk_forward_summary_csv(walk_forward, walk_forward_summary_output)
    write_walk_forward_folds_csv(walk_forward, walk_forward_folds_output)

    report = build_research_report(
        config=config,
        preflight=preflight,
        backtest=backtest,
        comparison=comparison,
        sweep=sweep,
        strategy_name=strategy_name,
        data_health=data_health,
        walk_forward=walk_forward,
        sweep_limit=args.sweep_limit,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.html, encoding="utf-8")

    print("Research Report")
    print(f"  Report: {output}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Symbol: {symbol}")
    print(f"  Preflight: {preflight.overall}")
    print(f"  Data health: {data_health.overall}")
    if comparison.best is not None:
        print(f"  Best strategy: {comparison.best.strategy_name}")
    if sweep.best is not None:
        print(
            "  Best momentum params: "
            f"lookback={sweep.best.lookback}, "
            f"threshold={sweep.best.threshold_bps:.1f} bps"
        )
    if walk_forward.best is not None:
        print(f"  Best walk-forward strategy: {walk_forward.best.strategy_name}")
    print(f"  Equity curve: {config.backtest.equity_curve_csv}")
    print(f"  P&L ledger: {config.backtest.pnl_ledger_csv}")
    print(f"  Strategy comparison: {args.comparison_output}")
    print(f"  Sweep CSV: {config.sweep.results_csv}")
    print(f"  Data health CSV: {args.data_health_output}")
    print(f"  Walk-forward summary CSV: {walk_forward_summary_output}")
    print(f"  Walk-forward folds CSV: {walk_forward_folds_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
