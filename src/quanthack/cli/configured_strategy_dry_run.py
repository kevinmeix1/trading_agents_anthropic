from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.clock import CompetitionMode
from quanthack.core.config import load_config
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.risk import AccountSnapshot, RiskEngine
from quanthack.strategies.strategy import STRATEGY_NAMES


SCENARIOS = {
    "up": [1.1000, 1.1002, 1.1004, 1.1007, 1.1010],
    "down": [1.1000, 1.0998, 1.0996, 1.0993, 1.0990],
    "flat": [1.1000, 1.1001, 1.1000, 1.1001, 1.1000],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run configured strategy through risk and dry-run journaling."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="up")
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
    starting_equity = config.competition.starting_equity
    equity = args.equity if args.equity is not None else starting_equity
    day_start_equity = (
        args.day_start_equity if args.day_start_equity is not None else starting_equity
    )
    peak_equity = args.peak_equity if args.peak_equity is not None else starting_equity

    strategy = config.build_strategy(strategy_name)
    request = strategy.generate_request(SCENARIOS[args.scenario])

    print(f"Config: {args.config}")
    print(f"Strategy: {strategy_name}")
    print(f"Scenario: {args.scenario}")

    if request is None:
        print("Strategy output: NO TRADE")
        print("No risk decision or journal record was created.")
        return

    account = AccountSnapshot(
        equity=equity,
        starting_equity=starting_equity,
        day_start_equity=day_start_equity,
        peak_equity=peak_equity,
        margin_level_pct=args.margin_level_pct,
    )
    mode = CompetitionMode(args.mode)
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
