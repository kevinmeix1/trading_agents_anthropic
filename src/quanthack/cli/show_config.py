from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show configured hackathon settings.")
    parser.add_argument("--config", default="configs/default.toml")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    print("Competition")
    print(f"  timezone: {config.competition.timezone}")
    print(f"  starting equity: ${config.competition.starting_equity:,.0f}")
    print(f"  open: {config.competition.open_at.isoformat(timespec='seconds')}")
    print("  checkpoints:")
    for checkpoint in config.competition.checkpoints:
        print(f"    - {checkpoint.isoformat(timespec='seconds')}")

    print("Risk")
    print(f"  max gross leverage: {config.risk.max_gross_leverage:.2f}x")
    print(f"  max symbol notional: {config.risk.max_symbol_notional_pct:.1%} of equity")
    print(f"  max daily loss: {config.risk.max_daily_loss_pct:.1%}")
    print(f"  max drawdown: {config.risk.max_drawdown_pct:.1%}")
    print(f"  min margin level: {config.risk.min_margin_level_pct:.0f}%")

    print("Strategy")
    print(f"  active: {config.active_strategy}")
    print("  simple_momentum:")
    print(f"    symbol: {config.simple_momentum.symbol}")
    print(f"    lookback: {config.simple_momentum.lookback}")
    print(f"    threshold: {config.simple_momentum.threshold_bps:.1f} bps")
    print(f"    target notional: ${config.simple_momentum.target_notional_usd:,.0f}")
    print("  ma_crossover:")
    print(f"    symbol: {config.ma_crossover.symbol}")
    print(f"    fast window: {config.ma_crossover.fast_window}")
    print(f"    slow window: {config.ma_crossover.slow_window}")
    print(f"    min separation: {config.ma_crossover.min_separation_bps:.1f} bps")
    print(f"    target notional: ${config.ma_crossover.target_notional_usd:,.0f}")
    print("  mean_reversion:")
    print(f"    symbol: {config.mean_reversion.symbol}")
    print(f"    lookback: {config.mean_reversion.lookback}")
    print(f"    entry z-score: {config.mean_reversion.entry_zscore:.2f}")
    print(f"    target notional: ${config.mean_reversion.target_notional_usd:,.0f}")
    print("  alpha_router:")
    print(f"    symbol: {config.alpha_router.symbol}")
    print(f"    target notional: ${config.alpha_router.target_notional_usd:,.0f}")
    print(f"    moving average weight: {config.alpha_router.moving_average_weight:.2f}")
    print(f"    ml enabled: {config.alpha_router.ml_enabled}")
    print(f"    ml lookback: {config.alpha_router.ml_lookback}")
    print(f"    ml train window: {config.alpha_router.ml_train_window}")
    print(f"    ml min samples: {config.alpha_router.ml_min_train_samples}")
    print(f"    ml min samples for trade: {config.alpha_router.ml_min_samples_for_trade}")
    print(f"    ml min training accuracy: {config.alpha_router.ml_min_training_accuracy:.1%}")
    print(f"    ml min expected edge: {config.alpha_router.ml_min_expected_edge_bps:.1f} bps")
    print("  usd_pressure_router:")
    print(f"    symbol: {config.usd_pressure.symbol}")
    print(f"    lookback: {config.usd_pressure.lookback}")
    print(f"    pressure threshold: {config.usd_pressure.pressure_threshold_bps:.1f} bps")
    print(
        "    min target volatility: "
        f"{config.usd_pressure.min_target_volatility_bps:.1f} bps"
    )
    print(f"    min component symbols: {config.usd_pressure.min_component_symbols}")
    print(f"    min confirming symbols: {config.usd_pressure.min_confirming_symbols}")
    print(f"    exit on conflict: {config.usd_pressure.exit_on_conflict}")

    print("Market Data")
    print(f"  price csv: {config.market_data.price_csv}")
    print(f"  quote csv: {config.market_data.quote_csv}")

    print("Market Quality")
    print(f"  max spread: {config.market_quality.max_spread_bps:.1f} bps")
    print(f"  max quote age: {config.market_quality.max_quote_age_seconds:.1f}s")

    print("Execution")
    print(f"  route: {config.execution.route}")
    print(f"  journal: {config.execution.journal_path}")

    print("Backtest")
    print(f"  price csv: {config.backtest.price_csv}")
    print(f"  quote csv: {config.backtest.quote_csv}")
    print(f"  slippage: {config.backtest.slippage_bps:.1f} bps")
    print(f"  periods/year: {config.backtest.periods_per_year:.1f}")
    print(f"  equity curve: {config.backtest.equity_curve_csv}")
    print(f"  pnl ledger: {config.backtest.pnl_ledger_csv}")

    print("Sweep")
    print(f"  lookbacks: {list(config.sweep.lookbacks)}")
    print(f"  threshold bps: {list(config.sweep.threshold_bps)}")
    print(f"  train fraction: {config.sweep.train_fraction:.1%}")
    print(f"  results csv: {config.sweep.results_csv}")

    print("Walk Forward")
    print(f"  ma fast windows: {list(config.walk_forward.ma_fast_windows)}")
    print(f"  ma slow windows: {list(config.walk_forward.ma_slow_windows)}")
    print(
        "  ma min separation bps: "
        f"{list(config.walk_forward.ma_min_separation_bps)}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
