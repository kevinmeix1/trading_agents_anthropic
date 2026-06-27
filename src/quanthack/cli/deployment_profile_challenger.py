from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.deployment_profile_challenger import (
    DeploymentProfileCandidateSpec,
    compare_deployment_profile_challengers,
    parse_candidate_spec,
    write_deployment_profile_challenger_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


DEFAULT_CANDIDATES = (
    DeploymentProfileCandidateSpec(
        label="survival",
        profile_pack_json="outputs/research/deployment_profile_pack.json",
        slot="survival",
    ),
    DeploymentProfileCandidateSpec(
        label="refined",
        profile_pack_json="outputs/research/deployment_profile_refined_pack.json",
        slot="refined",
    ),
    DeploymentProfileCandidateSpec(
        label="session_refined",
        profile_pack_json="outputs/research/deployment_profile_session_gated_pack.json",
        slot="session_refined",
    ),
    DeploymentProfileCandidateSpec(
        label="symbol_refined",
        profile_pack_json="outputs/research/deployment_profile_symbol_gated_pack.json",
        slot="symbol_refined",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deployment profile challengers with exact backtests, "
            "walk-forward promotion gates, and risk discipline."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        metavar="LABEL,PROFILE_PACK_JSON,SLOT",
        help=(
            "Candidate profile to compare. Repeat to override defaults. "
            "The first candidate is the baseline for deltas."
        ),
    )
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument(
        "--output",
        default="outputs/research/deployment_profile_challenger_scorecard.csv",
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    candidates = (
        tuple(parse_candidate_spec(candidate) for candidate in args.candidate)
        if args.candidate
        else DEFAULT_CANDIDATES
    )
    result = compare_deployment_profile_challengers(
        config=config,
        prices=prices,
        quotes=quotes,
        candidates=candidates,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_deployment_profile_challenger_csv(result, args.output)

    print("Deployment Profile Challenger Scorecard")
    print(f"  Candidates: {len(result.rows)}")
    print(f"  Output CSV: {args.output}")
    if result.best is not None:
        print(
            "  Best: "
            f"{result.best.label} ({result.best.decision}, "
            f"return={result.best.backtest_return_pct:.3%}, "
            f"drawdown={result.best.backtest_max_drawdown_pct:.3%}, "
            f"risk={result.best.risk_discipline_score:.0f}/100)"
        )
    for rank, row in enumerate(result.rows[: max(args.limit, 0)], start=1):
        print(
            f"  {rank}. {row.label}: "
            f"decision={row.decision}, "
            f"return={row.backtest_return_pct:.3%} "
            f"({row.backtest_return_delta_pct:+.3%}), "
            f"drawdown={row.backtest_max_drawdown_pct:.3%}, "
            f"sharpe15={row.backtest_sharpe_15m:.3f}, "
            f"risk={row.risk_discipline_score:.0f}/100, "
            f"fills={row.fills}, "
            f"pnl={money(row.total_pnl_usd)}, "
            f"promotion={row.promotion_status}, "
            f"gates={row.gate_complexity}"
        )
        print(f"      reason: {row.reason}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
