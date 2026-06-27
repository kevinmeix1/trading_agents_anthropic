from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.reporting.deployment_profile_pack import (
    build_deployment_profile_pack,
    write_deployment_profile_pack_csv,
    write_deployment_profile_pack_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build aggressive/conservative/survival deployment profiles from "
            "promotion and stability evidence."
        )
    )
    parser.add_argument(
        "--promotion-summary-csv",
        default="outputs/research/crypto_promotion_pipeline_summary.csv",
    )
    parser.add_argument(
        "--asset-class-stability-csv",
        default="outputs/research/crypto_promotion_pipeline_asset_class_stability.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/research/deployment_profile_pack.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    pack = build_deployment_profile_pack(
        promotion_summary_csv=args.promotion_summary_csv,
        asset_class_stability_csv=args.asset_class_stability_csv,
    )
    write_deployment_profile_pack_csv(pack, args.output_csv)
    write_deployment_profile_pack_json(pack, args.output_json)

    print("Deployment Profile Pack")
    print(f"  Data source: {pack.data_source}")
    print(f"  Recommended slot: {pack.recommended_slot}")
    print(f"  Reason: {pack.recommendation_reason}")
    print(f"  CSV: {Path(args.output_csv)}")
    print(f"  JSON: {Path(args.output_json)}")
    for profile in pack.profiles:
        print(
            f"  {profile.slot}: {profile.label} "
            f"status={profile.evidence_status}, "
            f"return={profile.return_pct:.3%}, "
            f"drawdown={profile.max_drawdown_pct:.3%}, "
            f"sharpe15={profile.sharpe_15m:.3f}, "
            f"fold_contribution={profile.fold_contribution:.1%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
