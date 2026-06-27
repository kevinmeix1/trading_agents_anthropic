from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.config import load_config
from quanthack.market.backtest_import import import_pricer_zip_to_backtest_csv


DEFAULT_PRICE_OUTPUT = "data/downloaded_backtest_prices.csv"
DEFAULT_QUOTE_OUTPUT = "data/downloaded_backtest_quotes.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert downloaded tick-level backtest data into Claude Agent Trader backtest CSVs."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-output", default=DEFAULT_PRICE_OUTPUT)
    parser.add_argument("--quote-output", default=DEFAULT_QUOTE_OUTPUT)
    parser.add_argument("--bar-seconds", type=int, default=900)
    parser.add_argument("--source-timezone", default="UTC")
    parser.add_argument(
        "--max-files-per-symbol",
        type=int,
        default=None,
        help="Useful for quick samples, for example 2 symbol-days.",
    )
    parser.add_argument(
        "--progress-every-files",
        type=int,
        default=25,
        help="Print import progress after this many imported parquet files. Use 0 to disable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    input_path = Path(args.input).expanduser() if args.input else _latest_downloaded_archive()
    symbols = tuple(args.symbol or (config.strategy_symbol(),))

    summary = import_pricer_zip_to_backtest_csv(
        input_path=input_path,
        price_output=args.price_output,
        quote_output=args.quote_output,
        symbols=symbols,
        bar_seconds=args.bar_seconds,
        source_timezone=args.source_timezone,
        max_files_per_symbol=args.max_files_per_symbol,
        progress_every_files=(
            args.progress_every_files if args.progress_every_files > 0 else None
        ),
        progress_callback=print if args.progress_every_files > 0 else None,
    )

    print("Imported Backtest Data")
    print(f"  Input: {summary.input_path}")
    print(f"  Symbols: {', '.join(summary.symbols)}")
    print(f"  Archive parquet files seen: {summary.files_seen}")
    print(f"  Files imported: {summary.files_imported}")
    print(f"  Ticks read: {summary.ticks_seen:,}")
    print(f"  Bars written: {summary.bars_written:,}")
    print(f"  Bar size: {summary.bar_seconds} seconds")
    print(f"  Price CSV: {summary.price_csv}")
    print(f"  Quote CSV: {summary.quote_csv}")
    print("  Next:")
    print(
        "    quanthack backtest "
        f"--symbol {summary.symbols[0]} "
        f"--price-csv {summary.price_csv} "
        f"--quote-csv {summary.quote_csv}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _latest_downloaded_archive() -> Path:
    downloads = Path.home() / "Downloads"
    candidates = tuple(
        path
        for path in downloads.glob("pricer-output-*.zip")
        if path.is_file() and path.stat().st_size > 0
    )
    if not candidates:
        raise SystemExit(
            "No completed pricer-output-*.zip found in ~/Downloads. "
            "Use --input /path/to/archive.zip after the download finishes."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)
