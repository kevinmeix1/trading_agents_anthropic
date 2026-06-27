from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime

from quanthack.core.instruments import AssetClass
from quanthack.market.sample_data import (
    DEFAULT_SAMPLE_START,
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic multi-symbol sample data for the competition instruments."
    )
    parser.add_argument("--price-output", default="data/syphonix_sample_prices.csv")
    parser.add_argument("--quote-output", default="data/syphonix_sample_quotes.csv")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--asset-class", choices=["forex", "metal", "crypto"], default=None)
    parser.add_argument("--periods", type=int, default=96)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", default=DEFAULT_SAMPLE_START.isoformat(timespec="seconds"))
    return parser


def run(args: argparse.Namespace) -> None:
    asset_class = AssetClass(args.asset_class.upper()) if args.asset_class else None
    data = generate_synthetic_market_data(
        symbols=tuple(args.symbol) if args.symbol else None,
        asset_class=asset_class,
        start=datetime.fromisoformat(args.start),
        periods=args.periods,
        interval_minutes=args.interval_minutes,
        seed=args.seed,
    )
    write_price_history_csv(data.prices, args.price_output)
    write_quote_history_csv(data.quotes, args.quote_output)

    print("Synthetic Competition Data")
    print(f"  Symbols: {', '.join(data.prices.symbols())}")
    print(f"  Price bars: {len(data.prices.bars)}")
    print(f"  Quotes: {len(data.quotes.quotes)}")
    print(f"  Price CSV: {args.price_output}")
    print(f"  Quote CSV: {args.quote_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
