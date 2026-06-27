from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.deployment_profile_session_gate_refiner import (
    refine_deployment_profile_session_gates,
    write_deployment_profile_session_gate_refinement_csv,
    write_session_gated_profile_pack_json,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create research-only deployment profile variants by removing weak "
            "asset-class UTC hours found in session attribution."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_refined_pack.json",
    )
    parser.add_argument(
        "--recommendation-json",
        default="outputs/research/deployment_profile_refined_only_recommendation.json",
        help="Used for the slot when --slot is omitted and the file exists.",
    )
    parser.add_argument("--slot", default=None)
    parser.add_argument(
        "--attribution-csv",
        default="outputs/research/deployment_profile_refined_session_attribution.csv",
    )
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument("--max-dropped-hours", type=int, default=3)
    parser.add_argument("--weak-pnl-threshold-usd", type=float, default=0.0)
    parser.add_argument("--min-hour-fills", type=int, default=1)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument(
        "--output",
        default="outputs/research/deployment_profile_session_gate_refinement.csv",
    )
    parser.add_argument(
        "--refined-pack-json",
        default="outputs/research/deployment_profile_session_gated_pack.json",
    )
    parser.add_argument("--refined-slot", default="session_refined")
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    slot = _resolve_slot(args.slot, args.recommendation_json)
    result = refine_deployment_profile_session_gates(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=slot,
        attribution_csv=args.attribution_csv,
        max_dropped_hours=args.max_dropped_hours,
        weak_pnl_threshold_usd=args.weak_pnl_threshold_usd,
        min_hour_fills=args.min_hour_fills,
        include_walk_forward=not args.no_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_deployment_profile_session_gate_refinement_csv(result, args.output)
    if result.best is not None:
        write_session_gated_profile_pack_json(
            source_profile_pack_json=args.profile_pack_json,
            result=result,
            candidate=result.best,
            output_json=args.refined_pack_json,
            refined_slot=args.refined_slot,
        )

    print("Deployment Profile Session-Gate Refinement")
    print(f"  Base slot: {result.base_profile.slot}")
    print(f"  Base label: {result.base_profile.label}")
    print(
        "  Weak asset hours: "
        + (
            ", ".join(f"{row.asset_class}:{row.utc_hour}" for row in result.weak_asset_hours)
            or "none"
        )
    )
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Output CSV: {args.output}")
    print(f"  Refined pack JSON: {args.refined_pack_json}")
    for rank, candidate in enumerate(result.candidates[: max(args.limit, 0)], start=1):
        metrics = candidate.competition_metrics
        wf = candidate.walk_forward
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
            f"dropped={candidate.dropped_asset_hours_text or 'none'}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.risk_discipline.score}/100, "
            f"fills={len(candidate.result.fills)}, "
            f"pnl={money(candidate.result.total_pnl_usd)}, "
            f"promotion={'' if candidate.promotion is None else candidate.promotion.status}"
            f"{wf_text}"
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
    return "recommended"
