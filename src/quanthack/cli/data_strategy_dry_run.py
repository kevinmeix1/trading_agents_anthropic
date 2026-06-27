from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.clock import CompetitionMode
from quanthack.core.config import load_config
from quanthack.trading.execution import DryRunExecutor
from quanthack.market.market_data import load_price_history
from quanthack.trading.risk import AccountSnapshot, RiskEngine
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run configured strategy using offline CSV market data."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--csv", default=None, help="Override configured CSV path.")
    parser.add_argument("--symbol", default=None, help="Override configured strategy symbol.")
    parser.add_argument("--equity", type=float, default=None)
    parser.add_argument("--day-start-equity", type=float, default=None)
    parser.add_argument("--peak-equity", type=float, default=None)
    parser.add_argument("--margin-level-pct", type=float, default=2_000)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CompetitionMode],
        default=CompetitionMode.QUALIFY.value,
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy or config.active_strategy
    csv_path = args.csv or config.market_data.price_csv
    symbol = args.symbol or config.strategy_symbol(strategy_name)
    history = load_price_history(csv_path)
    closes = history.close_prices(symbol=symbol)

    strategy = config.build_strategy(strategy_name, symbol=symbol)
    request = strategy.generate_request(closes)

    print(f"Config: {args.config}")
    print(f"Strategy: {strategy_name}")
    print(f"CSV: {csv_path}")
    print(f"Symbol: {symbol}")
    print(f"Closes loaded: {len(closes)}")

    if request is None:
        print("Strategy output: NO TRADE")
        print("No risk decision or journal record was created.")
        return

    starting_equity = config.competition.starting_equity
    equity = args.equity if args.equity is not None else starting_equity
    day_start_equity = (
        args.day_start_equity if args.day_start_equity is not None else starting_equity
    )
    peak_equity = args.peak_equity if args.peak_equity is not None else starting_equity
    mode = CompetitionMode(args.mode)

    account = AccountSnapshot(
        equity=equity,
        starting_equity=starting_equity,
        day_start_equity=day_start_equity,
        peak_equity=peak_equity,
        margin_level_pct=args.margin_level_pct,
    )
    executor = DryRunExecutor(Path(config.execution.journal_path))
    portfolio = executor.current_portfolio()
    decision = RiskEngine(config.risk).evaluate(
        account=account,
        portfolio=portfolio,
        request=request,
        mode=mode,
    )
    record = executor.submit(
        account=account,
        request=request,
        decision=decision,
        mode=mode,
        portfolio_before=portfolio,
    )

    print(f"Strategy output: {request.side.value} {request.symbol}")
    print(f"Requested notional: ${request.target_notional_usd:,.0f}")
    print(f"Portfolio before: gross=${portfolio.gross_notional_usd:,.0f}")
    print(f"Risk decision: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"Reason: {decision.reason}")
    print(f"Adjusted notional: ${decision.adjusted_notional_usd:,.0f}")
    print(f"Dry-run status: {record.status}")
    print(f"Journal: {config.execution.journal_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
