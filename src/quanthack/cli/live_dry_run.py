from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.deployment_profile_backtest import (
    LoadedDeploymentProfile,
    load_deployment_profile,
    session_gate_policy_for_profile,
)
from quanthack.backtesting.portfolio_allocator import write_allocation_report_csv
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.cli._competition import print_competition_view
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.core.env import env_bool, env_int, env_str, load_env_file
from quanthack.core.instruments import instrument_for
from quanthack.market.adapters import (
    CsvMarketDataAdapter,
    MT5AccountAdapter,
    MT5ConnectionSettings,
    MT5MarketDataAdapter,
    StaticAccountAdapter,
    parse_symbol_map,
)
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name
from quanthack.trading.competition_monitor import write_monitor_csv
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.live_dry_run import LiveDryRunEngine, LiveDryRunSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a read-only live dry-run loop from CSV or MT5 data."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--adapter", choices=["csv", "mt5"], default=None)
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument(
        "--profile-pack-json",
        default=None,
        help="Optional deployment_profile_pack.json for exact profile live dry-run.",
    )
    parser.add_argument(
        "--profile-slot",
        default="conservative",
    )
    parser.add_argument(
        "--strategy-map",
        action="append",
        default=None,
        metavar="SYMBOL=STRATEGY",
        help=(
            "Optional per-symbol strategy override. Repeat for hybrid live dry-runs; "
            "symbols not listed use --strategy or the configured active strategy."
        ),
    )
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--monitor-output", default=None)
    parser.add_argument(
        "--allocation-output",
        default=None,
        help="Optional CSV path for per-iteration allocation reports.",
    )
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
        help="Required for --adapter mt5. This command still does not place orders.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    config = load_config(args.config)
    live_config = config.live_dry_run
    adapter_name = args.adapter or live_config.adapter
    profile = _load_optional_profile(args.profile_pack_json, args.profile_slot)
    strategy_name = args.strategy or config.active_strategy
    strategy_by_symbol = _parse_strategy_map(args.strategy_map or ())
    journal_path = args.journal or live_config.journal_path
    monitor_output = args.monitor_output or live_config.monitor_csv

    market_adapter, account_adapter = _build_adapters(
        adapter_name=adapter_name,
        config=config,
        args=args,
    )
    try:
        supported_symbols = market_adapter.supported_symbols()
        profile_settings = _profile_settings(
            profile=profile,
            requested_symbols=tuple(args.symbol or ()),
            strategy_name=strategy_name,
            strategy_by_symbol=strategy_by_symbol,
        )
        if profile_settings is not None:
            (
                symbols,
                strategy_name,
                strategy_by_symbol,
                multipliers,
                session_gate_policy,
            ) = profile_settings
        else:
            symbols = tuple(args.symbol or supported_symbols)
            multipliers = ()
            session_gate_policy = None
        settings = LiveDryRunSettings(
            symbols=symbols,
            strategy_name=strategy_name,
            strategy_by_symbol=strategy_by_symbol,
            timeframe=args.timeframe or live_config.timeframe,
            bars=args.bars or live_config.bars,
            iterations=args.iterations or live_config.iterations,
            poll_seconds=(
                live_config.poll_seconds
                if args.poll_seconds is None
                else args.poll_seconds
            ),
            journal_path=journal_path,
            monitor_csv=monitor_output,
            target_notional_multipliers_by_symbol=multipliers,
            session_gate_policy=session_gate_policy,
            deployment_profile_slot=profile.slot if profile is not None else None,
            deployment_profile_label=profile.label if profile is not None else None,
        )
        _validate_requested_symbols(
            adapter_name=adapter_name,
            requested=settings.symbols,
            supported=supported_symbols,
        )
        engine = LiveDryRunEngine(
            config=config,
            settings=settings,
            market_data=market_adapter,
            account_adapter=account_adapter,
            executor=DryRunExecutor(Path(journal_path)),
        )
        result = engine.run()
        write_monitor_csv(result.monitor_report.snapshots, monitor_output)
        if args.allocation_output:
            write_allocation_report_csv(
                tuple(iteration.allocation for iteration in result.iterations),
                args.allocation_output,
            )
    finally:
        close = getattr(market_adapter, "close", None)
        if callable(close):
            close()

    _print_result(
        adapter_name=adapter_name,
        settings=settings,
        journal_path=journal_path,
        monitor_output=monitor_output,
        allocation_output=args.allocation_output,
        result=result,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _load_optional_profile(
    profile_pack_json: str | None,
    slot: str,
) -> LoadedDeploymentProfile | None:
    if profile_pack_json is None:
        return None
    return load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )


def _profile_settings(
    *,
    profile: LoadedDeploymentProfile | None,
    requested_symbols: tuple[str, ...],
    strategy_name: str,
    strategy_by_symbol: tuple[tuple[str, str], ...],
) -> (
    tuple[
        tuple[str, ...],
        str,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, float], ...],
        SessionGatePolicy | None,
    ]
    | None
):
    if profile is None:
        return None
    if strategy_by_symbol:
        raise SystemExit("--strategy-map cannot be combined with --profile-pack-json")
    profile_symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    symbols = _profile_symbols(profile_symbols, requested_symbols)
    strategy_map = tuple(
        (symbol, strategy)
        for symbol, strategy in profile.strategy_by_symbol
        if symbol in symbols
    )
    multipliers = tuple(
        (symbol, multiplier)
        for symbol, multiplier in profile.multipliers_by_symbol
        if symbol in symbols
    )
    return (
        symbols,
        strategy_map[0][1] if strategy_map else strategy_name,
        strategy_map,
        multipliers,
        _session_gate_policy(profile),
    )


def _profile_symbols(
    profile_symbols: tuple[str, ...],
    requested_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested_symbols:
        return profile_symbols
    requested = tuple(instrument_for(symbol).symbol for symbol in requested_symbols)
    missing = tuple(symbol for symbol in requested if symbol not in profile_symbols)
    if missing:
        raise SystemExit(
            "profile does not contain requested symbols: "
            + ", ".join(missing)
        )
    return requested


def _session_gate_policy(profile: LoadedDeploymentProfile) -> SessionGatePolicy | None:
    return session_gate_policy_for_profile(profile)


def _build_adapters(*, adapter_name: str, config, args: argparse.Namespace):
    if adapter_name == "mt5":
        if not args.confirm_read_only_mt5:
            raise SystemExit(
                "Refusing MT5 connection without --confirm-read-only-mt5. "
                "This command is read-only, but the explicit flag keeps the workflow deliberate."
            )
        settings = MT5ConnectionSettings(
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
        adapter = MT5MarketDataAdapter(settings)
        return adapter, MT5AccountAdapter(adapter)

    price_csv = args.price_csv or config.market_data.price_csv
    quote_csv = args.quote_csv or config.market_data.quote_csv
    return (
        CsvMarketDataAdapter(price_csv=price_csv, quote_csv=quote_csv),
        StaticAccountAdapter(equity=config.competition.starting_equity),
    )


def _validate_requested_symbols(
    *,
    adapter_name: str,
    requested: tuple[str, ...],
    supported: tuple[str, ...],
) -> None:
    supported_set = set(supported)
    missing = tuple(symbol for symbol in requested if symbol not in supported_set)
    if not missing:
        return
    available = ", ".join(supported) if supported else "none"
    raise SystemExit(
        f"{adapter_name} adapter does not have data for: {', '.join(missing)}. "
        f"Available symbols: {available}."
    )


def _print_result(
    *,
    adapter_name: str,
    settings: LiveDryRunSettings,
    journal_path: str,
    monitor_output: str,
    allocation_output: str | None,
    result,
) -> None:
    report = result.monitor_report
    latest = report.latest
    latest_allocation = result.iterations[-1].allocation
    print("Live Dry Run")
    print(f"  Adapter: {adapter_name}")
    if settings.deployment_profile_slot:
        print(
            "  Deployment profile: "
            f"{settings.deployment_profile_slot} ({settings.deployment_profile_label})"
        )
    print(f"  Strategy: {settings.strategy_name}")
    if settings.strategy_by_symbol:
        print(
            "  Strategy map: "
            + ", ".join(
                f"{symbol}={strategy}" for symbol, strategy in settings.strategy_by_symbol
            )
        )
    if settings.target_notional_multipliers_by_symbol:
        print(
            "  Multipliers: "
            + ", ".join(
                f"{symbol}={multiplier:.3f}"
                for symbol, multiplier in settings.target_notional_multipliers_by_symbol
            )
        )
    print(f"  Iterations: {len(result.iterations)}")
    print(f"  Journal: {journal_path}")
    print(f"  Monitor CSV: {monitor_output}")
    if allocation_output:
        print(f"  Allocation CSV: {allocation_output}")
    print(f"  Requested gross exposure: {money(latest_allocation.requested_gross_notional_usd)}")
    print(f"  Adjusted gross exposure: {money(latest_allocation.adjusted_gross_notional_usd)}")
    print(f"  Allocation status: {latest_allocation.estimated_risk_status}")
    if latest_allocation.trim_reasons:
        print(f"  Allocation trims: {'; '.join(latest_allocation.trim_reasons)}")
    print(f"  Latest equity: {money(latest.account.equity)}")
    print(f"  Latest leverage: {latest.leverage:.2f}x")
    print(f"  Latest net directional exposure: {latest.net_directional_exposure:.1%}")
    print(f"  Latest largest-symbol concentration: {latest.single_symbol_concentration:.1%}")
    print(f"  Dry-run records written: {len(result.records)}")
    print_competition_view(
        metrics=report.competition_metrics,
        risk_discipline=report.risk_discipline,
    )


def _parse_strategy_map(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    strategy_by_symbol: dict[str, str] = {}
    for raw_value in values:
        if "=" not in raw_value:
            raise SystemExit(
                "--strategy-map values must use SYMBOL=STRATEGY, "
                f"got {raw_value!r}"
            )
        raw_symbol, raw_strategy = raw_value.split("=", 1)
        symbol = instrument_for(raw_symbol).symbol
        try:
            strategy = normalize_strategy_name(raw_strategy.strip())
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        strategy_by_symbol[symbol] = strategy
    return tuple(sorted(strategy_by_symbol.items()))
