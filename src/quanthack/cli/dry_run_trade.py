from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.core.clock import CompetitionMode
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.risk import AccountSnapshot, RiskEngine, Side, TradeRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fake trade through risk and journal it.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--side", choices=[side.value for side in Side], default=Side.BUY.value)
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
    parser.add_argument(
        "--journal",
        default="outputs/dry_run_journal.jsonl",
        help="Where to write the JSONL dry-run journal.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    account = AccountSnapshot(
        equity=args.equity,
        day_start_equity=args.day_start_equity,
        peak_equity=args.peak_equity,
        margin_level_pct=args.margin_level_pct,
    )
    request = TradeRequest(
        symbol=args.symbol,
        side=Side(args.side),
        target_notional_usd=args.target_notional,
        reason="manual dry-run trade from VS Code terminal",
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

    print(f"Record ID: {record.record_id}")
    print(f"Status: {record.status}")
    print(f"Portfolio before: gross=${portfolio.gross_notional_usd:,.0f}")
    print(f"Risk decision: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"Reason: {decision.reason}")
    print(f"Adjusted notional: ${decision.adjusted_notional_usd:,.0f}")
    print(f"Journal: {args.journal}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
