from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.core.env import env_bool, env_int, env_str, load_env_file
from quanthack.market.adapters import (
    MT5AccountAdapter,
    MT5ConnectionSettings,
    MT5MarketDataAdapter,
    MT5UnavailableError,
    parse_symbol_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the read-only MT5 connection without placing orders."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--bars", type=int, default=3)
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
    return parser


def run(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    config = load_config(args.config)
    settings = _settings_from_args(args)
    symbols = tuple(args.symbol or (config.strategy_symbol(),))
    timeframe = args.timeframe or config.live_dry_run.timeframe

    print("MT5 Probe")
    print(f"  Python: {sys.version.split()[0]} at {sys.executable}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Timeframe: {timeframe}")
    print("  Mode: read-only, no order_send")

    adapter = MT5MarketDataAdapter(settings)
    account_adapter = MT5AccountAdapter(adapter)
    try:
        account = account_adapter.get_account_snapshot(
            starting_equity=config.competition.starting_equity,
            day_start_equity=config.competition.starting_equity,
            peak_equity=config.competition.starting_equity,
        )
        print("  Connection: OK")
        print(f"  Account equity: {money(account.equity)}")
        margin_text = (
            f"{account.margin_level_pct:.1f}%"
            if account.margin_level_pct is not None
            else "unknown"
        )
        print(f"  Margin level: {margin_text}")

        for symbol in symbols:
            quote = adapter.get_latest_quote(symbol)
            bars = adapter.get_recent_bars(symbol, timeframe=timeframe, count=args.bars)
            print(f"  {quote.symbol} quote: bid={quote.bid} ask={quote.ask} at {quote.timestamp.isoformat()}")
            print(f"  {quote.symbol} bars: {len(bars)} latest_close={bars[-1].close}")
    except MT5UnavailableError as exc:
        print("  Connection: unavailable")
        print(f"  Reason: {exc}")
        print("  Mac note: native macOS Python often cannot install/use the official MetaTrader5 package.")
        print("  Practical path: keep strategy research on Mac, then run this probe inside Windows/Parallels or the same MT5 environment that supports the Python package.")
    finally:
        adapter.close()


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
