from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history
from quanthack.strategies.time_series import (
    KalmanTrendConfig,
    evaluate_time_series_regimes,
    write_time_series_regime_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify symbol trend/chop regimes with a Kalman-style filter."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--lookback", type=int, default=80)
    parser.add_argument("--min-abs-slope-bps", type=float, default=0.75)
    parser.add_argument("--min-trend-efficiency", type=float, default=0.25)
    parser.add_argument("--max-realized-volatility-bps", type=float, default=120.0)
    parser.add_argument("--output", default="outputs/backtests/time_series_regimes.csv")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    prices = load_price_history(price_csv)
    trend_config = KalmanTrendConfig(
        lookback=args.lookback,
        min_abs_slope_bps=args.min_abs_slope_bps,
        min_trend_efficiency=args.min_trend_efficiency,
        max_realized_volatility_bps=args.max_realized_volatility_bps,
    )
    readings = evaluate_time_series_regimes(
        prices=prices,
        symbols=tuple(args.symbol) if args.symbol else None,
        config=trend_config,
    )
    write_time_series_regime_csv(readings, args.output)

    print("Advanced Time-Series Regime Report")
    print(f"  Price CSV: {price_csv}")
    print(f"  Symbols: {', '.join(reading.symbol for reading in readings)}")
    print(f"  Lookback: {args.lookback}")
    for reading in readings:
        print(
            f"  {reading.symbol}: {reading.regime.value} "
            f"slope={reading.kalman_slope_bps:.2f} bps "
            f"eff={reading.trend_efficiency:.2f} "
            f"vol={reading.realized_volatility_bps:.2f} bps"
        )
    print(f"  Output CSV: {args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))

