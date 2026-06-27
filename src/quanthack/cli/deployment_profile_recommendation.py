from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.deployment_profile_selector import DEFAULT_PROFILE_SLOTS
from quanthack.backtesting.deployment_profile_selector_sweep import (
    DEFAULT_DRAWDOWN_PENALTIES,
    DEFAULT_FALLBACK_SLOTS,
    DEFAULT_MIN_PAST_FOLDS,
    DEFAULT_RISK_SCORE_FLOORS,
    write_deployment_profile_selector_sweep_csv,
)
from quanthack.cli.deployment_profile_slots import (
    resolve_fallback_slots,
    resolve_profile_slots,
)
from quanthack.core.clock import CompetitionMode
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.deployment_profile_recommendation import (
    build_deployment_profile_recommendation,
    write_deployment_profile_recommendation_csv,
    write_deployment_profile_recommendation_json,
)
from quanthack.trading.deployment_profile_snapshot import (
    build_deployment_profile_signal_snapshot,
    write_deployment_profile_signal_snapshot_csv,
)
from quanthack.trading.execution import DryRunExecutor
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend the next deployment profile slot from selector-sweep evidence."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    parser.add_argument("--slot", action="append", default=None)
    parser.add_argument(
        "--fallback-slot",
        action="append",
        default=None,
    )
    parser.add_argument("--min-past-folds", action="append", type=int, default=None)
    parser.add_argument("--drawdown-penalty", action="append", type=float, default=None)
    parser.add_argument("--risk-score-floor", action="append", type=float, default=None)
    parser.add_argument(
        "--include-inactive-past-selection",
        action="store_true",
        help="Also test variants where inactive profiles can win from zero drawdown.",
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
    parser.add_argument(
        "--recommendation-csv",
        default="outputs/research/deployment_profile_recommendation.csv",
    )
    parser.add_argument(
        "--recommendation-json",
        default="outputs/research/deployment_profile_recommendation.json",
    )
    parser.add_argument(
        "--sweep-output",
        default="outputs/research/deployment_profile_recommendation_sweep.csv",
    )
    parser.add_argument(
        "--snapshot-output",
        default=None,
        help="Optional CSV signal snapshot for the recommended slot.",
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
    parser.add_argument("--journal", default=None)
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional ISO timestamp for the recommendation snapshot.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
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
    result = build_deployment_profile_recommendation(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slots=slots,
        fallback_slots=fallback_slots,
        min_past_folds_values=tuple(args.min_past_folds or DEFAULT_MIN_PAST_FOLDS),
        drawdown_penalties=tuple(args.drawdown_penalty or DEFAULT_DRAWDOWN_PENALTIES),
        risk_score_floors=tuple(args.risk_score_floor or DEFAULT_RISK_SCORE_FLOORS),
        require_past_activity_values=(
            (True, False) if args.include_inactive_past_selection else (True,)
        ),
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_deployment_profile_recommendation_csv(result, args.recommendation_csv)
    write_deployment_profile_recommendation_json(result, args.recommendation_json)
    write_deployment_profile_selector_sweep_csv(result.sweep, args.sweep_output)
    snapshot_path = None
    if args.snapshot_output is not None:
        snapshot_path = Path(args.snapshot_output)
        snapshot = build_deployment_profile_signal_snapshot(
            config=config,
            prices=prices,
            quotes=quotes,
            profile_pack_json=args.profile_pack_json,
            slot=result.recommendation.recommended_slot,
            account=_account(config, args),
            portfolio=_portfolio_from_optional_journal(args.journal),
            mode=CompetitionMode(args.mode),
            as_of=_parse_optional_as_of(args.as_of),
        )
        write_deployment_profile_signal_snapshot_csv(snapshot, snapshot_path)

    _print_recommendation(
        result=result,
        recommendation_csv=Path(args.recommendation_csv),
        recommendation_json=Path(args.recommendation_json),
        sweep_output=Path(args.sweep_output),
        snapshot_output=snapshot_path,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


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


def _portfolio_from_optional_journal(journal: str | None) -> PortfolioSnapshot:
    if journal is None:
        return PortfolioSnapshot()
    return DryRunExecutor(Path(journal)).current_portfolio()


def _parse_optional_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise SystemExit("--as-of must include a timezone")
    return parsed


def _print_recommendation(
    *,
    result,
    recommendation_csv: Path,
    recommendation_json: Path,
    sweep_output: Path,
    snapshot_output: Path | None,
) -> None:
    recommendation = result.recommendation
    print("Deployment Profile Recommendation")
    print(f"  Recommended slot: {recommendation.recommended_slot}")
    print(f"  Label: {recommendation.recommended_label}")
    print(f"  Reason: {recommendation.recommendation_reason}")
    print(f"  Promotion: {recommendation.promotion_status}")
    print(f"  Selector score: {recommendation.selector_score:.2f}")
    print(
        "  Historical sequence: "
        f"{recommendation.historical_selected_sequence or 'none'}"
    )
    print(
        "  Fold evidence: "
        f"pos={recommendation.positive_fold_fraction:.1%}, "
        f"active_pos={recommendation.active_positive_fold_fraction:.1%}, "
        f"cum_return={recommendation.cumulative_test_return_pct:.3%}, "
        f"drawdown={recommendation.worst_test_drawdown_pct:.3%}, "
        f"risk={recommendation.average_risk_discipline_score:.1f}/100"
    )
    print(f"  Recommendation CSV: {recommendation_csv}")
    print(f"  Recommendation JSON: {recommendation_json}")
    print(f"  Sweep CSV: {sweep_output}")
    if snapshot_output is not None:
        print(f"  Snapshot CSV: {snapshot_output}")
