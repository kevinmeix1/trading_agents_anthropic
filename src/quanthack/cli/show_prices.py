from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect offline CSV price data.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--csv", default=None, help="Override configured CSV path.")
    parser.add_argument("--symbol", default=None, help="Symbol to inspect.")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    csv_path = args.csv or config.market_data.price_csv
    symbol = args.symbol or config.simple_momentum.symbol
    history = load_price_history(csv_path)
    symbol_history = history.for_symbol(symbol)
    latest = symbol_history.latest_bar(symbol)

    print(f"CSV: {csv_path}")
    print(f"Available symbols: {', '.join(history.symbols())}")
    print(f"Selected symbol: {symbol}")
    print(f"Rows for symbol: {len(symbol_history.bars)}")

    if latest is None:
        print("Latest close: none")
        return

    print(f"Latest timestamp: {latest.timestamp.isoformat(timespec='seconds')}")
    print(f"Latest close: {latest.close}")
    print(f"Close prices: {symbol_history.close_prices(symbol=symbol)}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
