from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep

from quanthack.core.clock import UTC
from quanthack.core.config import load_config
from quanthack.core.env import env_bool, env_int, env_str, load_env_file
from quanthack.market.adapters import (
    MT5AccountAdapter,
    MT5ConnectionSettings,
    MT5MarketDataAdapter,
    MT5UnavailableError,
    parse_symbol_map,
)


@dataclass(frozen=True)
class MT5CaptureResult:
    iterations: int
    quote_rows: int
    bar_rows: int
    account_rows: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture read-only MT5 quotes/account snapshots into CSV files."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--bars", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--quotes-output", default="outputs/live_mt5_quotes.csv")
    parser.add_argument("--bars-output", default="outputs/live_mt5_bars.csv")
    parser.add_argument("--account-output", default="outputs/live_mt5_account.csv")
    parser.add_argument("--mt5-terminal-path", default=None)
    parser.add_argument("--mt5-login", type=int, default=None)
    parser.add_argument("--mt5-password", default=None)
    parser.add_argument("--mt5-server", default=None)
    parser.add_argument("--mt5-timeout-ms", type=int, default=60_000)
    parser.add_argument("--mt5-portable", action="store_true")
    parser.add_argument(
        "--mt5-symbol-map",
        action="append",
        default=None,
        help="Map canonical to broker symbol, for example EURUSD=EURUSD.pro",
    )
    parser.add_argument(
        "--confirm-read-only-mt5",
        action="store_true",
        help="Required. This command reads MT5 data only and never sends orders.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.confirm_read_only_mt5:
        raise SystemExit(
            "Refusing MT5 capture without --confirm-read-only-mt5. "
            "This command is read-only, but the explicit flag keeps the workflow deliberate."
        )
    if args.bars < 0:
        raise SystemExit("--bars cannot be negative")
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.poll_seconds < 0:
        raise SystemExit("--poll-seconds cannot be negative")

    load_env_file(args.env_file)
    config = load_config(args.config)
    symbols = tuple(args.symbol or (config.strategy_symbol(),))
    timeframe = args.timeframe or config.live_dry_run.timeframe
    adapter = MT5MarketDataAdapter(_settings_from_args(args))
    account_adapter = MT5AccountAdapter(adapter)

    try:
        result = capture_mt5_data(
            adapter=adapter,
            account_adapter=account_adapter,
            symbols=symbols,
            timeframe=timeframe,
            bars=args.bars,
            iterations=args.iterations,
            poll_seconds=args.poll_seconds,
            starting_equity=config.competition.starting_equity,
            quotes_output=Path(args.quotes_output),
            bars_output=Path(args.bars_output),
            account_output=Path(args.account_output),
        )
    except MT5UnavailableError as exc:
        print("MT5 Capture")
        print("  Connection: unavailable")
        print(f"  Reason: {exc}")
        print(
            "  Mac note: native macOS Python often cannot use the official "
            "MetaTrader5 package. Run this inside a Windows/Parallels MT5 Python "
            "environment if this Mac cannot load the package."
        )
        return
    finally:
        adapter.close()

    print("MT5 Capture")
    print("  Mode: read-only, no order_send")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Quote rows written: {result.quote_rows}")
    print(f"  Account rows written: {result.account_rows}")
    print(f"  Bar rows written: {result.bar_rows}")
    print(f"  Quotes CSV: {args.quotes_output}")
    print(f"  Account CSV: {args.account_output}")
    if args.bars > 0:
        print(f"  Bars CSV: {args.bars_output}")


def capture_mt5_data(
    *,
    adapter: MT5MarketDataAdapter,
    account_adapter: MT5AccountAdapter,
    symbols: tuple[str, ...],
    timeframe: str,
    bars: int,
    iterations: int,
    poll_seconds: float,
    starting_equity: float,
    quotes_output: Path,
    bars_output: Path,
    account_output: Path,
) -> MT5CaptureResult:
    quote_rows = 0
    bar_rows = 0
    account_rows = 0
    peak_equity = starting_equity
    day_start_equity = starting_equity

    for index in range(iterations):
        captured_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
        account = account_adapter.get_account_snapshot(
            starting_equity=starting_equity,
            day_start_equity=day_start_equity,
            peak_equity=peak_equity,
        )
        peak_equity = max(peak_equity, account.equity)
        _append_account_row(
            account_output,
            {
                "captured_at": captured_at,
                "equity": account.equity,
                "daily_pnl_pct": account.daily_pnl_pct,
                "drawdown_pct": account.drawdown_pct,
                "margin_level_pct": account.margin_level_pct,
            },
        )
        account_rows += 1

        for symbol in symbols:
            quote = adapter.get_latest_quote(symbol)
            _append_quote_row(
                quotes_output,
                {
                    "captured_at": captured_at,
                    "quote_timestamp": quote.timestamp.isoformat(timespec="milliseconds"),
                    "symbol": quote.symbol,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "mid": quote.mid,
                    "spread_bps": quote.spread_bps,
                },
            )
            quote_rows += 1

            if bars > 0:
                for bar in adapter.get_recent_bars(symbol, timeframe=timeframe, count=bars):
                    _append_bar_row(
                        bars_output,
                        {
                            "captured_at": captured_at,
                            "bar_timestamp": bar.timestamp.isoformat(timespec="seconds"),
                            "symbol": bar.symbol,
                            "timeframe": timeframe,
                            "close": bar.close,
                        },
                    )
                    bar_rows += 1

        if index < iterations - 1 and poll_seconds > 0:
            sleep(poll_seconds)

    return MT5CaptureResult(
        iterations=iterations,
        quote_rows=quote_rows,
        bar_rows=bar_rows,
        account_rows=account_rows,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _settings_from_args(args: argparse.Namespace) -> MT5ConnectionSettings:
    return MT5ConnectionSettings(
        terminal_path=args.mt5_terminal_path or env_str("MT5_TERMINAL_PATH"),
        login=args.mt5_login if args.mt5_login is not None else env_int("MT5_LOGIN"),
        password=args.mt5_password or env_str("MT5_PASSWORD"),
        server=args.mt5_server or env_str("MT5_SERVER"),
        timeout_ms=(
            args.mt5_timeout_ms
            if args.mt5_timeout_ms != 60_000
            else env_int("MT5_TIMEOUT_MS", 60_000)
        ),
        portable=args.mt5_portable or env_bool("MT5_PORTABLE", False),
        symbol_map=parse_symbol_map(tuple(args.mt5_symbol_map or ())),
    )


def _append_quote_row(path: Path, row: dict[str, object]) -> None:
    _append_row(
        path,
        fieldnames=(
            "captured_at",
            "quote_timestamp",
            "symbol",
            "bid",
            "ask",
            "mid",
            "spread_bps",
        ),
        row=row,
    )


def _append_bar_row(path: Path, row: dict[str, object]) -> None:
    _append_row(
        path,
        fieldnames=("captured_at", "bar_timestamp", "symbol", "timeframe", "close"),
        row=row,
    )


def _append_account_row(path: Path, row: dict[str, object]) -> None:
    _append_row(
        path,
        fieldnames=(
            "captured_at",
            "equity",
            "daily_pnl_pct",
            "drawdown_pct",
            "margin_level_pct",
        ),
        row=row,
    )


def _append_row(
    path: Path,
    *,
    fieldnames: tuple[str, ...],
    row: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
