from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.clock import CompetitionMode
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    RiskEngine,
    RiskLimits,
    Side,
    TradeRequest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple dry-run risk decision.")
    parser.add_argument("--equity", type=float, default=1_000_000)
    parser.add_argument("--day-start-equity", type=float, default=1_000_000)
    parser.add_argument("--peak-equity", type=float, default=1_000_000)
    parser.add_argument("--margin-level-pct", type=float, default=2_000)
    parser.add_argument("--target-notional", type=float, default=50_000)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CompetitionMode],
        default=CompetitionMode.QUALIFY.value,
    )
    return parser


def run(args: argparse.Namespace) -> None:
    account = AccountSnapshot(
        equity=args.equity,
        day_start_equity=args.day_start_equity,
        peak_equity=args.peak_equity,
        margin_level_pct=args.margin_level_pct,
    )
    portfolio = PortfolioSnapshot()
    request = TradeRequest(
        symbol="EURUSD",
        side=Side.BUY,
        target_notional_usd=args.target_notional,
        reason="demo request before any strategy is built",
    )

    engine = RiskEngine(RiskLimits())
    decision = engine.evaluate(
        account=account,
        portfolio=portfolio,
        request=request,
        mode=CompetitionMode(args.mode),
    )

    print(f"Equity: ${account.equity:,.0f}")
    print(f"Daily P&L: {account.daily_pnl_pct:.2%}")
    print(f"Drawdown: {account.drawdown_pct:.2%}")
    print(f"Margin level: {account.margin_level_pct:.0f}%")
    print(f"Requested: {request.side.value} {request.symbol} ${request.target_notional_usd:,.0f}")
    print(f"Decision: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"Reason: {decision.reason}")
    print(f"Adjusted notional: ${decision.adjusted_notional_usd:,.0f}")
    print(f"Risk state: {decision.state.value}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
