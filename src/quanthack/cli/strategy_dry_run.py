from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.clock import CompetitionMode
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.risk import AccountSnapshot, RiskEngine
from quanthack.strategies.strategy import MomentumConfig, SimpleMomentumStrategy


SCENARIOS = {
    "up": [1.1000, 1.1002, 1.1004, 1.1007, 1.1010],
    "down": [1.1000, 1.0998, 1.0996, 1.0993, 1.0990],
    "flat": [1.1000, 1.1001, 1.1000, 1.1001, 1.1000],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the simple strategy through risk and dry-run journaling."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="up")
    parser.add_argument("--threshold-bps", type=float, default=8.0)
    parser.add_argument("--target-notional", type=float, default=50_000)
    parser.add_argument("--equity", type=float, default=1_000_000)
    parser.add_argument("--day-start-equity", type=float, default=1_000_000)
    parser.add_argument("--peak-equity", type=float, default=1_000_000)
    parser.add_argument("--margin-level-pct", type=float, default=2_000)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CompetitionMode],
        default=CompetitionMode.QUALIFY.value,
    )
    parser.add_argument("--journal", default="outputs/dry_run_journal.jsonl")
    return parser


def run(args: argparse.Namespace) -> None:
    strategy = SimpleMomentumStrategy(
        MomentumConfig(
            threshold_bps=args.threshold_bps,
            target_notional_usd=args.target_notional,
        )
    )
    request = strategy.generate_request(SCENARIOS[args.scenario])

    if request is None:
        print("Strategy output: NO TRADE")
        print("No risk decision or journal record was created.")
        return

    account = AccountSnapshot(
        equity=args.equity,
        day_start_equity=args.day_start_equity,
        peak_equity=args.peak_equity,
        margin_level_pct=args.margin_level_pct,
    )
    mode = CompetitionMode(args.mode)
    executor = DryRunExecutor(Path(args.journal))
    portfolio = executor.current_portfolio()
    decision = RiskEngine().evaluate(
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
    print(f"Portfolio before: gross=${portfolio.gross_notional_usd:,.0f}")
    print(f"Risk decision: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"Reason: {decision.reason}")
    print(f"Adjusted notional: ${decision.adjusted_notional_usd:,.0f}")
    print(f"Dry-run status: {record.status}")
    print(f"Journal: {args.journal}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
