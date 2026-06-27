from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.reporting.symbol_universe_profile_pack import (
    build_symbol_universe_profile_pack,
    write_symbol_universe_profile_pack_csv,
    write_symbol_universe_profile_pack_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a symbol-eligibility optimizer candidate into an executable "
            "deployment profile pack slot."
        )
    )
    parser.add_argument(
        "--symbol-eligibility-csv",
        default="outputs/research/macd_symbol_eligibility_full20gb.csv",
    )
    parser.add_argument(
        "--candidate",
        default="rank:1",
        help="Candidate name, or rank:N for the Nth ranked row.",
    )
    parser.add_argument("--selected-slot", default="symbol_universe")
    parser.add_argument("--selected-label", default=None)
    parser.add_argument("--baseline-slot", default="baseline")
    parser.add_argument("--data-source", default="research")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--output-json",
        default="outputs/research/deployment_profile_symbol_universe_pack.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/research/deployment_profile_symbol_universe_pack.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    pack = build_symbol_universe_profile_pack(
        symbol_eligibility_csv=args.symbol_eligibility_csv,
        candidate=args.candidate,
        selected_slot=args.selected_slot,
        selected_label=args.selected_label,
        baseline_slot=args.baseline_slot,
        data_source=args.data_source,
        include_baseline=not args.no_baseline,
    )
    write_symbol_universe_profile_pack_json(pack, args.output_json)
    write_symbol_universe_profile_pack_csv(pack, args.output_csv)

    print("Deployment Profile Symbol Universe Refinement")
    print(f"  Source CSV: {pack.source_symbol_eligibility_csv}")
    print(f"  Data source: {pack.data_source}")
    print(f"  Recommended slot: {pack.recommended_slot}")
    print(f"  Output JSON: {args.output_json}")
    print(f"  Output CSV: {args.output_csv}")
    for profile in pack.profiles:
        print(
            f"  - {profile.slot}: {profile.label}, "
            f"return={profile.return_pct:.3%}, "
            f"drawdown={profile.max_drawdown_pct:.3%}, "
            f"sharpe15={profile.sharpe_15m:.3f}, "
            f"symbols={profile.strategy_map}, "
            f"excluded={profile.excluded_symbols or 'none'}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
