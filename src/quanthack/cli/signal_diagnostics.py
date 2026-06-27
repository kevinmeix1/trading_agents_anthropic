from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.signal_diagnostics import (
    evaluate_signal_diagnostics,
    write_signal_diagnostics_csv,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast forward-return diagnostics for strategy signal sleeves."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="alpha_router")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--min-edge-after-cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="outputs/backtests/signal_diagnostics.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    report = evaluate_signal_diagnostics(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_name=args.strategy,
        symbols=tuple(args.symbol) if args.symbol else None,
        horizon_bars=args.horizon_bars,
        min_confidence=args.min_confidence,
        min_edge_after_cost_bps=args.min_edge_after_cost_bps,
    )
    write_signal_diagnostics_csv(report, args.output)

    print("Signal Diagnostics")
    print(f"  Strategy: {report.strategy_name}")
    print(f"  Horizon bars: {report.horizon_bars}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print(
        "  symbol | signal           | active | hit rate | avg signed bps | "
        "avg weight | avg edge-cost"
    )
    for row in report.ranked_rows[:20]:
        print(
            f"  {row.symbol:<6} | "
            f"{row.signal_name:<16} | "
            f"{row.active_count:>6} | "
            f"{row.hit_rate:>8.1%} | "
            f"{row.average_signed_forward_return_bps:>14.2f} | "
            f"{row.average_weight:>10.2f} | "
            f"{row.average_edge_after_cost_bps:>13.2f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
