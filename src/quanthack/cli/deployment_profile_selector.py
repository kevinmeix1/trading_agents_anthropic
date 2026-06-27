from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.deployment_profile_selector import (
    DEFAULT_PROFILE_SLOTS,
    run_deployment_profile_selector,
    write_deployment_profile_selector_folds_csv,
    write_deployment_profile_selector_summary_csv,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    decide_fixed_warmup_promotion,
)
from quanthack.core.config import load_config
from quanthack.cli.deployment_profile_slots import (
    resolve_profile_slots,
    resolve_single_fallback_slot,
)
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed deployment profiles with a past-evidence adaptive "
            "profile selector."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    parser.add_argument(
        "--slot",
        action="append",
        default=None,
        help=(
            "Profile slot to include. Repeat to override the default "
            "profile-pack comparison."
        ),
    )
    parser.add_argument(
        "--fallback-slot",
        default=None,
        help="Profile used before enough completed folds exist to select adaptively.",
    )
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument("--min-past-folds", type=int, default=2)
    parser.add_argument("--drawdown-penalty", type=float, default=0.50)
    parser.add_argument("--risk-score-floor", type=float, default=95.0)
    parser.add_argument(
        "--allow-inactive-past-selection",
        action="store_true",
        help=(
            "Allow profiles with no active completed folds to win by having zero "
            "drawdown. Disabled by default to avoid selecting on absence of evidence."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/research/deployment_profile_selector_summary.csv",
    )
    parser.add_argument(
        "--folds-output",
        default="outputs/research/deployment_profile_selector_folds.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    slots = resolve_profile_slots(
        requested_slots=args.slot,
        profile_pack_json=args.profile_pack_json,
        fallback_slots=DEFAULT_PROFILE_SLOTS,
    )
    fallback_slot = resolve_single_fallback_slot(
        requested_fallback_slot=args.fallback_slot,
        profile_pack_json=args.profile_pack_json,
        slots=slots,
        preferred_slots=("conservative", "survival"),
    )
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = run_deployment_profile_selector(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slots=slots,
        fallback_slot=fallback_slot,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_past_folds=args.min_past_folds,
        drawdown_penalty=args.drawdown_penalty,
        risk_score_floor=args.risk_score_floor,
        require_past_activity=not args.allow_inactive_past_selection,
    )
    write_deployment_profile_selector_summary_csv(result, args.summary_output)
    write_deployment_profile_selector_folds_csv(result, args.folds_output)
    promotion = decide_fixed_warmup_promotion(result.adaptive_result)
    selected_counts = result.selected_counts

    print("Deployment Profile Selector")
    print(f"  Slots: {', '.join(result.slots)}")
    print(f"  Fallback: {result.fallback_slot}")
    print(f"  Min past folds: {result.min_past_folds}")
    print(f"  Drawdown penalty: {result.drawdown_penalty:g}")
    print(f"  Risk score floor: {result.risk_score_floor:.1f}/100")
    print(f"  Require past activity: {result.require_past_activity}")
    print(f"  Folds: {len(result.adaptive_result.folds)}")
    print(
        "  Adaptive positive fold fraction: "
        f"{result.adaptive_result.positive_fold_fraction:.1%}"
    )
    print(
        "  Adaptive active positive fold fraction: "
        f"{result.adaptive_result.active_positive_fold_fraction:.1%}"
    )
    print(
        "  Adaptive median active return: "
        f"{result.adaptive_result.median_active_test_return_pct:.3%}"
    )
    print(
        "  Adaptive worst drawdown: "
        f"{result.adaptive_result.worst_test_drawdown_pct:.3%}"
    )
    print(
        "  Adaptive risk discipline: "
        f"{result.adaptive_result.average_risk_discipline_score:.1f}/100"
    )
    print("  Selected counts:")
    for slot in result.slots:
        print(f"    {slot}: {selected_counts.get(slot, 0)}")
    print(f"  Promotion: {promotion.status} ({promotion.reason})")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Folds CSV: {args.folds_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
