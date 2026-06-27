from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime

from quanthack.core.instruments import instrument_for
from quanthack.market.merge_market_data import merge_market_data_csvs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge price/quote CSVs and optionally crop them to a shared window."
    )
    parser.add_argument("--price-input", action="append", required=True)
    parser.add_argument("--quote-input", action="append", required=True)
    parser.add_argument("--price-output", required=True)
    parser.add_argument("--quote-output", required=True)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--start", default=None, help="Optional timezone-aware ISO timestamp.")
    parser.add_argument("--end", default=None, help="Optional timezone-aware ISO timestamp.")
    parser.add_argument(
        "--common-window",
        action="store_true",
        help="Crop to the common first/last timestamp window across selected symbols.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    summary = merge_market_data_csvs(
        price_inputs=tuple(args.price_input),
        quote_inputs=tuple(args.quote_input),
        price_output=args.price_output,
        quote_output=args.quote_output,
        symbols=_symbols(args.symbol),
        start=_parse_optional_datetime(args.start, "--start"),
        end=_parse_optional_datetime(args.end, "--end"),
        common_window=args.common_window,
    )

    print("Market Data Merge")
    print(f"  Symbols: {', '.join(summary.symbols)}")
    print(f"  Price rows: {summary.price_rows:,}")
    print(f"  Quote rows: {summary.quote_rows:,}")
    print(f"  Window: {summary.start.isoformat()} -> {summary.end.isoformat()}")
    print(f"  Price CSV: {summary.price_csv}")
    print(f"  Quote CSV: {summary.quote_csv}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _symbols(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    return tuple(instrument_for(value).symbol for value in values)


def _parse_optional_datetime(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SystemExit(f"{label} must be timezone-aware")
    return parsed
