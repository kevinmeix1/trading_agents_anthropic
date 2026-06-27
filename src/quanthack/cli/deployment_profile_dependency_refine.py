from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.deployment_profile_dependency_refiner import (
    DEFAULT_DEPENDENCY_SCALES,
    refine_deployment_profile_dependency,
    write_dependency_refined_profile_pack_json,
    write_deployment_profile_dependency_refinement_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create research-only deployment profile variants by scaling symbols "
            "that leave-one-symbol robustness marked as fragile dependencies."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_symbol_gated_pack.json",
    )
    parser.add_argument("--slot", default="symbol_refined")
    parser.add_argument(
        "--robustness-csv",
        default="outputs/research/deployment_profile_symbol_refined_robustness.csv",
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
        "--dependency-scale",
        action="append",
        type=float,
        default=None,
        help="Dependency-symbol multiplier scale to test. Repeat to override defaults.",
    )
    parser.add_argument(
        "--dependency-threshold-pct",
        type=float,
        default=-0.003,
        help=(
            "Also treat leave-one-symbol rows at or below this return delta as "
            "dependencies, even if their robustness decision is not FRAGILE/FAIL."
        ),
    )
    parser.add_argument("--min-return-retention", type=float, default=0.75)
    parser.add_argument("--min-dependency-loss-reduction", type=float, default=0.20)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument(
        "--output",
        default="outputs/research/deployment_profile_dependency_refinement.csv",
    )
    parser.add_argument(
        "--refined-pack-json",
        default="outputs/research/deployment_profile_dependency_refined_pack.json",
    )
    parser.add_argument("--refined-slot", default="dependency_refined")
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = refine_deployment_profile_dependency(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=args.slot,
        robustness_csv=args.robustness_csv,
        dependency_scales=tuple(args.dependency_scale or DEFAULT_DEPENDENCY_SCALES),
        dependency_threshold_pct=args.dependency_threshold_pct,
        min_return_retention=args.min_return_retention,
        min_dependency_loss_reduction=args.min_dependency_loss_reduction,
        include_walk_forward=not args.no_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_deployment_profile_dependency_refinement_csv(result, args.output)
    if result.best is not None:
        write_dependency_refined_profile_pack_json(
            source_profile_pack_json=args.profile_pack_json,
            result=result,
            candidate=result.best,
            output_json=args.refined_pack_json,
            refined_slot=args.refined_slot,
        )

    print("Deployment Profile Dependency Refinement")
    print(f"  Base slot: {result.base_profile.slot}")
    print(f"  Base label: {result.base_profile.label}")
    print(
        "  Dependent symbols: "
        f"{', '.join(row.symbol for row in result.dependent_symbols) or 'none'}"
    )
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Output CSV: {args.output}")
    print(f"  Refined pack JSON: {args.refined_pack_json}")
    for rank, candidate in enumerate(result.candidates[: max(args.limit, 0)], start=1):
        metrics = candidate.competition_metrics
        wf = candidate.walk_forward
        worst = candidate.worst_dependency_row
        dependency_text = "no dependency stress"
        if worst is not None:
            dependency_text = (
                f"worst={worst.symbol} {worst.return_delta_pct:.3%} "
                f"({worst.decision}), "
                f"loss_reduction={candidate.dependency_loss_reduction:.1%}"
            )
        wf_text = ""
        if wf is not None:
            wf_text = (
                f", wf_pos={wf.positive_fold_fraction:.1%}, "
                f"wf_active_pos={wf.active_positive_fold_fraction:.1%}, "
                f"wf_nonneg={wf.non_negative_fold_fraction:.1%}, "
                f"wf_dd={wf.worst_test_drawdown_pct:.3%}"
            )
        print(
            f"  {rank}. {candidate.label}: "
            f"scale={candidate.dependency_scale:.2f}, "
            f"decision={candidate.candidate_decision}, "
            f"return={metrics.return_pct:.3%}, "
            f"retention={candidate.return_retention_vs_base:.1%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.risk_discipline.score}/100, "
            f"fills={len(candidate.result.fills)}, "
            f"pnl={money(candidate.result.total_pnl_usd)}, "
            f"promotion={'' if candidate.promotion is None else candidate.promotion.status}, "
            f"{dependency_text}"
            f"{wf_text}"
        )
        print(f"      multipliers: {candidate.multiplier_map_text}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
