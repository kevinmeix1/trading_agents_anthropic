from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.instruments import DEFAULT_INSTRUMENTS, AssetClass, instrument_for
from quanthack.market.backtest_import import (
    inspect_pricer_zip_archive,
    write_pricer_archive_coverage_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a downloaded pricer zip for symbol coverage without importing parquet."
    )
    parser.add_argument("--input", default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument(
        "--competition-symbols",
        action="store_true",
        help="Require every official QuanHack instrument.",
    )
    parser.add_argument(
        "--crypto-symbols",
        action="store_true",
        help="Require only the official crypto instruments.",
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/archive_data_coverage.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when expected symbols are missing.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser() if args.input else _latest_downloaded_archive()
    expected = _expected_symbols(args)
    summary = inspect_pricer_zip_archive(input_path, expected_symbols=expected)
    write_pricer_archive_coverage_csv(summary, args.output)

    print("Archive Data Coverage")
    print(f"  Input: {summary.input_path}")
    print(f"  Parquet files: {summary.files_seen}")
    print(f"  Available symbols ({len(summary.available_symbols)}): {', '.join(summary.available_symbols)}")
    if summary.expected_symbols:
        print(f"  Expected symbols ({len(summary.expected_symbols)}): {', '.join(summary.expected_symbols)}")
        print(
            f"  Present expected ({len(summary.present_expected_symbols)}): "
            f"{_list_or_none(summary.present_expected_symbols)}"
        )
        print(
            f"  Missing ({len(summary.missing_symbols)}): "
            f"{_list_or_none(summary.missing_symbols)}"
        )
        print(f"  Complete: {'yes' if summary.complete else 'no'}")
    print(f"  CSV: {args.output}")

    if args.strict and not summary.complete:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _expected_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.competition_symbols and args.crypto_symbols:
        raise SystemExit("Use only one of --competition-symbols or --crypto-symbols.")
    if args.competition_symbols:
        return tuple(instrument.symbol for instrument in DEFAULT_INSTRUMENTS)
    if args.crypto_symbols:
        return tuple(
            instrument.symbol
            for instrument in DEFAULT_INSTRUMENTS
            if instrument.asset_class == AssetClass.CRYPTO
        )
    return tuple(instrument_for(symbol).symbol for symbol in args.symbol or ())


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


def _list_or_none(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"
