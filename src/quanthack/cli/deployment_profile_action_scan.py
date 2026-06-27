from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from quanthack.cli._format import money
from quanthack.core.clock import CompetitionMode
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.deployment_profile_action_scan import (
    scan_deployment_profile_actions,
    write_deployment_profile_action_events_csv,
    write_deployment_profile_action_hours_csv,
    write_deployment_profile_action_scan_summary_csv,
)
from quanthack.trading.risk import AccountSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan historical timestamps to see when a deployment profile produces "
            "risk-approved actions."
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
        help="Profile slot to scan. Defaults to recommendation JSON, then conservative.",
    )
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-timestamps",
        type=int,
        default=500,
        help="Cap scan to the most recent N selected timestamps.",
    )
    parser.add_argument(
        "--no-max-timestamps",
        action="store_true",
        help="Scan every selected timestamp. Use carefully on large data.",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Do not carry local positions forward between timestamps.",
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
        "--summary-output",
        default="outputs/research/deployment_profile_action_scan_summary.csv",
    )
    parser.add_argument(
        "--events-output",
        default="outputs/research/deployment_profile_action_scan_events.csv",
    )
    parser.add_argument(
        "--hours-output",
        default="outputs/research/deployment_profile_action_scan_hours.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    slot = _resolve_slot(args.slot, args.recommendation_json)
    result = scan_deployment_profile_actions(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=slot,
        account=_account(config, args),
        mode=CompetitionMode(args.mode),
        start=_parse_optional_timestamp(args.start, "--start"),
        end=_parse_optional_timestamp(args.end, "--end"),
        stride=args.stride,
        max_timestamps=None if args.no_max_timestamps else args.max_timestamps,
        stateful=not args.stateless,
    )
    write_deployment_profile_action_scan_summary_csv(result, args.summary_output)
    write_deployment_profile_action_events_csv(result, args.events_output)
    write_deployment_profile_action_hours_csv(result, args.hours_output)
    _print_result(result, args)


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


def _account(config, args: argparse.Namespace) -> AccountSnapshot:
    starting_equity = config.competition.starting_equity
    equity = args.equity if args.equity is not None else starting_equity
    return AccountSnapshot(
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


def _parse_optional_timestamp(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise SystemExit(f"{label} must include a timezone")
    return parsed


def _print_result(result, args: argparse.Namespace) -> None:
    print("Deployment Profile Action Scan")
    print(f"  Slot: {result.profile_slot}")
    print(f"  Label: {result.profile_label}")
    print(f"  Stateful: {result.stateful}")
    print(f"  Scanned timestamps: {result.scanned_timestamps}")
    print(f"  Window: {result.first_timestamp} -> {result.last_timestamp}")
    print(f"  Actionable timestamps: {result.actionable_timestamps}")
    print(f"  Action rows: {result.action_rows}")
    print(f"  Approved actions: {result.approved_actions}")
    print(f"  Blocked actions: {result.blocked_actions}")
    print(f"  Approved action rate: {result.approved_action_rate:.1%}")
    print(f"  Buy/Sell: {result.buy_actions}/{result.sell_actions}")
    print(
        "  Symbols: "
        f"{', '.join(result.unique_action_symbols) if result.unique_action_symbols else 'none'}"
    )
    if result.events:
        print(
            "  First action: "
            f"{result.first_action_timestamp} "
            f"{result.events[0].order_side} {result.events[0].symbol} "
            f"{money(result.events[0].change_notional_usd)}"
        )
        print(
            "  Last action: "
            f"{result.last_action_timestamp} "
            f"{result.events[-1].order_side} {result.events[-1].symbol} "
            f"{money(result.events[-1].change_notional_usd)}"
        )
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Events CSV: {args.events_output}")
    print(f"  Hours CSV: {args.hours_output}")
