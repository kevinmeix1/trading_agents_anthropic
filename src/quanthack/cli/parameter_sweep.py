from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.backtest import FillModel
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.sweep import run_parameter_sweep, write_sweep_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a momentum parameter sweep.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--limit", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    symbol = args.symbol or config.simple_momentum.symbol
    result = run_parameter_sweep(
        prices=load_price_history(config.backtest.price_csv),
        quotes=load_quote_history(config.backtest.quote_csv),
        symbol=symbol,
        base_config=config.simple_momentum,
        lookbacks=config.sweep.lookbacks,
        threshold_bps=config.sweep.threshold_bps,
        train_fraction=config.sweep.train_fraction,
        starting_equity=config.competition.starting_equity,
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
    )
    write_sweep_csv(result, config.sweep.results_csv)

    print("Parameter Sweep")
    print(f"  Symbol: {symbol}")
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Results CSV: {config.sweep.results_csv}")

    if result.best is None:
        print("  Best: none")
        return

    best = result.best
    print(
        "  Best: "
        f"lookback={best.lookback}, threshold={best.threshold_bps:.1f} bps, "
        f"test_sharpe={best.test.metrics.sharpe_ratio:.3f}, "
        f"test_return={best.test.metrics.total_return_pct:.3%}"
    )

    print("Top Candidates")
    print("  rank | lookback | threshold | train_sharpe | test_sharpe | test_return | test_dd")
    for rank, candidate in enumerate(result.candidates[: args.limit], start=1):
        print(
            f"  {rank:>4} | "
            f"{candidate.lookback:>8} | "
            f"{candidate.threshold_bps:>8.1f} | "
            f"{candidate.train.metrics.sharpe_ratio:>12.3f} | "
            f"{candidate.test.metrics.sharpe_ratio:>11.3f} | "
            f"{candidate.test.metrics.total_return_pct:>10.3%} | "
            f"{candidate.test.metrics.max_drawdown_pct:>7.3%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
