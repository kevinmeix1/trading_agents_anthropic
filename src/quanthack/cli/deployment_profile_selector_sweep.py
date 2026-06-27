from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.deployment_profile_selector import DEFAULT_PROFILE_SLOTS
from quanthack.backtesting.deployment_profile_selector_sweep import (
    DEFAULT_DRAWDOWN_PENALTIES,
    DEFAULT_FALLBACK_SLOTS,
    DEFAULT_MIN_PAST_FOLDS,
    DEFAULT_RISK_SCORE_FLOORS,
    sweep_deployment_profile_selector,
    write_deployment_profile_selector_sweep_csv,
)
from quanthack.cli.deployment_profile_slots import (
    resolve_fallback_slots,
    resolve_profile_slots,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep adaptive deployment-profile selector settings after running "
            "each fixed profile walk-forward once."
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
        help="Profile slot to include. Repeat to override the default set.",
    )
    parser.add_argument(
        "--fallback-slot",
        action="append",
        default=None,
        help="Fallback slot to test. Repeat to override conservative/survival.",
    )
    parser.add_argument(
        "--min-past-folds",
        action="append",
        type=int,
        default=None,
        help="Completed fold count required before adaptive selection. Repeat.",
    )
    parser.add_argument(
        "--drawdown-penalty",
        action="append",
        type=float,
        default=None,
        help="Penalty applied to worst past drawdown in the selector score. Repeat.",
    )
    parser.add_argument(
        "--risk-score-floor",
        action="append",
        type=float,
        default=None,
        help="Minimum average past risk-discipline score. Repeat.",
    )
    parser.add_argument(
        "--include-inactive-past-selection",
        action="store_true",
        help=(
            "Also test unsafe variants where inactive profiles can win by having "
            "zero past drawdown."
        ),
    )
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--output",
        default="outputs/research/deployment_profile_selector_sweep.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    require_past_activity_values = (
        (True, False) if args.include_inactive_past_selection else (True,)
    )
    slots = resolve_profile_slots(
        requested_slots=args.slot,
        profile_pack_json=args.profile_pack_json,
        fallback_slots=DEFAULT_PROFILE_SLOTS,
    )
    fallback_slots = resolve_fallback_slots(
        requested_fallback_slots=args.fallback_slot,
        profile_pack_json=args.profile_pack_json,
        slots=slots,
        preferred_slots=DEFAULT_FALLBACK_SLOTS,
    )
    result = sweep_deployment_profile_selector(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slots=slots,
        fallback_slots=fallback_slots,
        min_past_folds_values=tuple(args.min_past_folds or DEFAULT_MIN_PAST_FOLDS),
        drawdown_penalties=tuple(args.drawdown_penalty or DEFAULT_DRAWDOWN_PENALTIES),
        risk_score_floors=tuple(args.risk_score_floor or DEFAULT_RISK_SCORE_FLOORS),
        require_past_activity_values=require_past_activity_values,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_deployment_profile_selector_sweep_csv(result, args.output)

    print("Deployment Profile Selector Sweep")
    print(f"  Fixed profiles: {len(result.fixed_results)}")
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Output CSV: {args.output}")
    for rank, candidate in enumerate(result.candidates[: max(args.limit, 0)], start=1):
        adaptive = candidate.result.adaptive_result
        cumulative_return = sum(fold.metrics.return_pct for fold in adaptive.folds)
        print(
            f"  {rank}. status={candidate.promotion.status}, "
            f"score={candidate.selector_score:.2f}, "
            f"fallback={candidate.result.fallback_slot}, "
            f"past={candidate.result.min_past_folds}, "
            f"dd_penalty={candidate.result.drawdown_penalty:g}, "
            f"risk_floor={candidate.result.risk_score_floor:.1f}, "
            f"activity={candidate.result.require_past_activity}, "
            f"pos={adaptive.positive_fold_fraction:.1%}, "
            f"active_pos={adaptive.active_positive_fold_fraction:.1%}, "
            f"cum_return={cumulative_return:.3%}, "
            f"median_active={adaptive.median_active_test_return_pct:.3%}, "
            f"worst_dd={adaptive.worst_test_drawdown_pct:.3%}, "
            f"concentration={adaptive.largest_positive_fold_contribution:.1%}, "
            f"selected={candidate.selected_counts_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
