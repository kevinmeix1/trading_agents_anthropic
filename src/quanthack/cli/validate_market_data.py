from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.config import load_config
from quanthack.core.instruments import DEFAULT_INSTRUMENTS, instrument_for
from quanthack.market.data_health import (
    DataHealthSeverity,
    validate_market_data,
    write_market_data_health_csv,
)
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate price/quote CSV coverage and alignment.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument(
        "--competition-symbols",
        action="store_true",
        help="Validate coverage for every configured competition instrument.",
    )
    parser.add_argument("--max-gap-seconds", type=float, default=300.0)
    parser.add_argument("--output", default="outputs/backtests/data_health.csv")
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    prices = load_price_history(price_csv)
    quotes = load_quote_history(quote_csv)
    if args.competition_symbols:
        symbols = tuple(instrument.symbol for instrument in DEFAULT_INSTRUMENTS)
    elif args.all_symbols:
        symbols = None
    else:
        symbols = tuple(args.symbol or [config.strategy_symbol()])
    selected_symbols = tuple(
        symbols or sorted(set(prices.symbols()) | set(quotes.symbols()))
    )

    report = validate_market_data(
        prices=prices,
        quotes=quotes,
        symbols=selected_symbols,
        max_gap_seconds=args.max_gap_seconds,
        max_spread_bps=config.market_quality.max_spread_bps,
        max_spread_bps_by_symbol=_spread_limits_by_symbol(selected_symbols),
    )
    write_market_data_health_csv(report, Path(args.output))

    for line in report.summary_lines():
        print(line)
    print(f"  CSV: {args.output}")

    if report.overall == DataHealthSeverity.FAIL:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _spread_limits_by_symbol(symbols: tuple[str, ...]) -> dict[str, float]:
    limits: dict[str, float] = {}
    for symbol in symbols:
        try:
            limits[symbol] = instrument_for(symbol).max_spread_bps
        except ValueError:
            continue
    return limits
