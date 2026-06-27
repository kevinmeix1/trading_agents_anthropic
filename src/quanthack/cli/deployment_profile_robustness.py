from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.deployment_profile_robustness import (
    evaluate_deployment_profile_robustness,
    write_deployment_profile_robustness_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stress an exact deployment profile with higher slippage and "
            "leave-one-symbol-out robustness checks."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_symbol_gated_pack.json",
    )
    parser.add_argument("--slot", default="symbol_refined")
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument(
        "--slippage-multiplier",
        action="append",
        type=float,
        default=None,
        help="Cost stress multiplier to test. Repeat to override 1.5/2/3x defaults.",
    )
    parser.add_argument(
        "--output",
        default="outputs/research/deployment_profile_symbol_refined_robustness.csv",
    )
    parser.add_argument("--limit", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = evaluate_deployment_profile_robustness(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=args.slot,
        slippage_multipliers=tuple(args.slippage_multiplier or (1.5, 2.0, 3.0)),
    )
    write_deployment_profile_robustness_csv(result, args.output)

    baseline = result.baseline
    weakest = result.weakest_row
    dependent = result.most_dependent_symbol_row
    print("Deployment Profile Robustness")
    print(f"  Slot: {result.profile.slot}")
    print(f"  Label: {result.profile.label}")
    print(f"  Output CSV: {args.output}")
    print(
        "  Baseline: "
        f"return={baseline.return_pct:.3%}, "
        f"drawdown={baseline.max_drawdown_pct:.3%}, "
        f"sharpe15={baseline.sharpe_15m:.3f}, "
        f"risk={baseline.risk_discipline_score:.0f}/100, "
        f"fills={baseline.fills}, "
        f"pnl={money(baseline.total_pnl_usd)}"
    )
    if weakest is not None:
        print(
            "  Weakest stress: "
            f"{weakest.scenario}, return={weakest.return_pct:.3%}, "
            f"delta={weakest.return_delta_pct:+.3%}, decision={weakest.decision}"
        )
    if dependent is not None:
        print(
            "  Most dependent symbol: "
            f"{dependent.excluded_symbol}, "
            f"return delta={dependent.return_delta_pct:+.3%}, "
            f"decision={dependent.decision}"
        )
    print("  Stress rows:")
    for row in result.stress_rows[: max(args.limit, 0)]:
        print(
            f"    {row.scenario}: "
            f"return={row.return_pct:.3%} "
            f"({row.return_delta_pct:+.3%}), "
            f"dd={row.max_drawdown_pct:.3%}, "
            f"risk={row.risk_discipline_score:.0f}/100, "
            f"pnl={money(row.total_pnl_usd)}, "
            f"decision={row.decision}, "
            f"{row.note}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
