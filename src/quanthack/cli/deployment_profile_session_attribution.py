from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.deployment_profile_backtest import (
    run_deployment_profile_backtest,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.deployment_profile_session_attribution import (
    build_deployment_profile_session_attribution_report,
    write_deployment_profile_session_attribution_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute an exact deployment profile backtest P&L by UTC entry "
            "hour, symbol, signal, and side."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    parser.add_argument(
        "--recommendation-json",
        default="outputs/research/deployment_profile_recommendation.json",
        help="Used for the slot when --slot is omitted and the file exists.",
    )
    parser.add_argument(
        "--slot",
        default=None,
        help="Profile slot. Defaults to recommendation JSON, then conservative.",
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
        default="outputs/research/deployment_profile_session_attribution.csv",
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    slot = _resolve_slot(args.slot, args.recommendation_json)
    backtest = run_deployment_profile_backtest(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=slot,
    )
    report = build_deployment_profile_session_attribution_report(backtest)
    write_deployment_profile_session_attribution_csv(report, args.output)

    print("Deployment Profile Session Attribution")
    print(f"  Slot: {report.profile_slot}")
    print(f"  Label: {report.profile_label}")
    print(f"  Fills: {report.fills}")
    print(f"  Realized P&L: {money(report.realized_pnl_usd)}")
    print(f"  Open P&L: {money(report.open_pnl_usd)}")
    print(f"  Total P&L: {money(report.total_pnl_usd)}")
    print(f"  Output CSV: {args.output}")
    print("Weakest rows")
    for row in report.weakest_rows[: max(args.limit, 0)]:
        print(
            f"  h{row.utc_hour:02d} {row.symbol} {row.primary_signal} {row.side}: "
            f"pnl={money(row.total_pnl_usd)}, realized={money(row.realized_pnl_usd)}, "
            f"open={money(row.open_pnl_usd)}, fills={row.fills}, win={row.win_rate:.1%}"
        )
    print("Strongest rows")
    for row in report.strongest_rows[: max(args.limit, 0)]:
        print(
            f"  h{row.utc_hour:02d} {row.symbol} {row.primary_signal} {row.side}: "
            f"pnl={money(row.total_pnl_usd)}, realized={money(row.realized_pnl_usd)}, "
            f"open={money(row.open_pnl_usd)}, fills={row.fills}, win={row.win_rate:.1%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _resolve_slot(slot: str | None, recommendation_json: str) -> str:
    if slot:
        return slot
    path = Path(recommendation_json)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        recommended_slot = str(data.get("recommended_slot", "")).strip()
        if recommended_slot:
            return recommended_slot
    return "conservative"
