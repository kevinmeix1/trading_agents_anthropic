from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from quanthack.cli._format import money
from quanthack.core.clock import CompetitionMode
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.trading.deployment_profile_snapshot import (
    build_deployment_profile_signal_snapshot,
    write_deployment_profile_signal_snapshot_csv,
)
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show current read-only targets from one deployment profile slot. "
            "This previews strategy, allocation, and risk without journaling or trading."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    parser.add_argument(
        "--slot",
        default="conservative",
    )
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/research/deployment_profile_signal_snapshot.csv",
    )
    parser.add_argument("--equity", type=float, default=None)
    parser.add_argument("--day-start-equity", type=float, default=None)
    parser.add_argument("--peak-equity", type=float, default=None)
    parser.add_argument("--margin-level-pct", type=float, default=2_000.0)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CompetitionMode],
        default=CompetitionMode.QUALIFY.value,
    )
    parser.add_argument(
        "--journal",
        default=None,
        help="Optional dry-run journal used to reconstruct current local positions.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional ISO timestamp. Uses the latest common row at or before this time.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    starting_equity = config.competition.starting_equity
    equity = args.equity if args.equity is not None else starting_equity
    account = AccountSnapshot(
        equity=equity,
        starting_equity=starting_equity,
        day_start_equity=(
            args.day_start_equity
            if args.day_start_equity is not None
            else starting_equity
        ),
        peak_equity=args.peak_equity if args.peak_equity is not None else starting_equity,
        margin_level_pct=args.margin_level_pct,
    )
    portfolio = _portfolio_from_optional_journal(args.journal)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    snapshot = build_deployment_profile_signal_snapshot(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=args.slot,
        account=account,
        portfolio=portfolio,
        mode=CompetitionMode(args.mode),
        as_of=_parse_optional_as_of(args.as_of),
    )
    write_deployment_profile_signal_snapshot_csv(snapshot, args.output)
    _print_snapshot(snapshot, Path(args.output))


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _portfolio_from_optional_journal(journal: str | None) -> PortfolioSnapshot:
    if journal is None:
        return PortfolioSnapshot()
    return DryRunExecutor(Path(journal)).current_portfolio()


def _parse_optional_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise SystemExit("--as-of must include a timezone, for example 2026-06-10T10:00:00+00:00")
    return parsed


def _print_snapshot(snapshot, output_path: Path) -> None:
    profile = snapshot.profile
    allocation = snapshot.allocation
    print("Deployment Profile Signal Snapshot")
    print(f"  Slot: {profile.slot}")
    print(f"  Label: {profile.label}")
    print(f"  Evidence status: {profile.evidence_status}")
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  Account equity: {money(snapshot.account.equity)}")
    print(f"  Requested gross: {money(allocation.requested_gross_notional_usd)}")
    print(f"  Adjusted gross: {money(allocation.adjusted_gross_notional_usd)}")
    print(f"  Net directional exposure: {allocation.net_directional_exposure:.1%}")
    print(f"  Largest symbol concentration: {allocation.largest_symbol_concentration:.1%}")
    print(f"  Estimated risk status: {allocation.estimated_risk_status}")
    print(f"  Actionable rows: {len(snapshot.actionable_rows)}")
    print(f"  Output CSV: {output_path}")
    if not snapshot.actionable_rows:
        print("  Current action: HOLD")
        return
    print("  Current actions:")
    for row in snapshot.actionable_rows[:8]:
        print(
            "   "
            f"{row.order_side} {row.symbol} "
            f"change={money(row.change_notional_usd)} "
            f"target={money(row.allocated_target_notional_usd)} "
            f"risk={'OK' if row.risk_approved else 'BLOCK'} "
            f"reason={row.risk_reason}"
        )
