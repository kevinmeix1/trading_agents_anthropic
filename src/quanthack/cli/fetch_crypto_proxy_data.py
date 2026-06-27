from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime

from quanthack.core.instruments import DEFAULT_INSTRUMENTS, AssetClass, instrument_for
from quanthack.market.crypto_proxy_data import (
    DEFAULT_PROXY_SYMBOL_MAP,
    default_crypto_window,
    fetch_crypto_proxy_to_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch research-only crypto proxy candles into QuanHack CSV format. "
            "This uses Binance USDT spot klines, not the official competition feed."
        )
    )
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--all-crypto", action="store_true")
    parser.add_argument("--start", default=None, help="Timezone-aware ISO timestamp.")
    parser.add_argument("--end", default=None, help="Timezone-aware ISO timestamp.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--price-output", default="data/research_crypto_proxy_prices.csv")
    parser.add_argument("--quote-output", default="data/research_crypto_proxy_quotes.csv")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--source-symbol-map",
        action="append",
        default=None,
        metavar="SYMBOL=SOURCE",
        help="Override proxy mapping, for example BTCUSD=BTCUSDT.",
    )
    parser.add_argument(
        "--confirm-research-proxy",
        action="store_true",
        help="Required acknowledgement that this is not official competition data.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.confirm_research_proxy:
        raise SystemExit(
            "Refusing to fetch proxy crypto data without --confirm-research-proxy. "
            "This data is research-only and is not the official competition feed."
        )
    start, end = _window(args)
    symbols = _symbols(args)
    summary = fetch_crypto_proxy_to_csv(
        symbols=symbols,
        price_output=args.price_output,
        quote_output=args.quote_output,
        start=start,
        end=end,
        interval=args.interval,
        limit=args.limit,
        source_symbol_map=_parse_source_symbol_map(tuple(args.source_symbol_map or ())),
        progress_callback=print,
    )

    print("Crypto Proxy Data Fetch")
    print("  Source: Binance spot USDT klines")
    print("  Status: research-only, not official competition data")
    print(f"  Symbols: {', '.join(summary.symbols)}")
    print(f"  Source symbols: {', '.join(summary.source_symbols)}")
    print(f"  Bars written: {summary.bars_written:,}")
    print(f"  Window: {summary.start.isoformat()} -> {summary.end.isoformat()}")
    print(f"  Interval: {summary.interval}")
    print(f"  Price CSV: {summary.price_csv}")
    print(f"  Quote CSV: {summary.quote_csv}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_crypto and args.symbol:
        raise SystemExit("Use either --all-crypto or repeated --symbol, not both.")
    if args.all_crypto:
        return tuple(
            instrument.symbol
            for instrument in DEFAULT_INSTRUMENTS
            if instrument.asset_class == AssetClass.CRYPTO
        )
    if args.symbol:
        return tuple(instrument_for(symbol).symbol for symbol in args.symbol)
    return tuple(DEFAULT_PROXY_SYMBOL_MAP)


def _window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.start or args.end:
        if not args.start or not args.end:
            raise SystemExit("--start and --end must be provided together.")
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        if start.tzinfo is None or end.tzinfo is None:
            raise SystemExit("--start and --end must include timezone offsets.")
        return start, end
    return default_crypto_window(days=args.days)


def _parse_source_symbol_map(values: tuple[str, ...]) -> dict[str, str] | None:
    if not values:
        return None
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source-symbol-map must use SYMBOL=SOURCE, got {value!r}")
        raw_symbol, raw_source = value.split("=", 1)
        symbol = instrument_for(raw_symbol).symbol
        source = raw_source.strip().upper()
        if not source:
            raise SystemExit("--source-symbol-map source cannot be empty")
        parsed[symbol] = source
    return parsed
