from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._time import parse_datetime
from quanthack.core.config import load_config
from quanthack.market.market_data import load_quote_history
from quanthack.market.market_quality import MarketQualityChecker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect offline quote quality.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--csv", default=None, help="Override configured quote CSV path.")
    parser.add_argument("--symbol", default=None, help="Symbol to inspect.")
    parser.add_argument("--as-of", type=parse_datetime, default=None)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    csv_path = args.csv or config.market_data.quote_csv
    symbol = args.symbol or config.simple_momentum.symbol
    quotes = load_quote_history(csv_path)
    quote = quotes.latest_quote(symbol)

    print(f"CSV: {csv_path}")
    print(f"Available symbols: {', '.join(quotes.symbols())}")
    print(f"Selected symbol: {symbol}")

    if quote is None:
        print("Quote: none")
        print("Market quality: BLOCKED")
        print("Reason: no quote for symbol")
        return

    as_of = args.as_of or quote.timestamp
    decision = MarketQualityChecker(config.market_quality).evaluate(quote=quote, as_of=as_of)

    print(f"Quote timestamp: {quote.timestamp.isoformat(timespec='seconds')}")
    print(f"As of: {as_of.isoformat(timespec='seconds')}")
    print(f"Bid: {quote.bid}")
    print(f"Ask: {quote.ask}")
    print(f"Spread: {quote.spread_bps:.2f} bps")
    print(f"Quote age: {decision.quote_age_seconds:.1f}s")
    print(f"Market quality: {'OK' if decision.ok else 'BLOCKED'}")
    print(f"Reason: {decision.reason}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
