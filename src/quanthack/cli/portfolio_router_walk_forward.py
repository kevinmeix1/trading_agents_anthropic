from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.portfolio_router_walk_forward import (
    decide_router_promotion,
    run_portfolio_router_walk_forward,
    write_portfolio_router_walk_forward_folds_csv,
    write_portfolio_router_walk_forward_summary_csv,
)
from quanthack.backtesting.router_optimizer import (
    CONSERVATIVE_ROUTER_BEHAVIOR_PROFILES,
    DEFAULT_ROUTER_WEIGHT_SETS,
    RouterBehaviorProfile,
    RouterWeightSet,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune alpha-router weights on train windows and validate them on "
            "later portfolio test windows."
        )
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Weight tuple as momentum,ma,breakout,mean_reversion or "
            "momentum,ma,breakout,mean_reversion,session_breakout,cross_rate "
            "or momentum,ma,breakout,mean_reversion,session_breakout,cross_rate,"
            "relative_strength or momentum,ma,breakout,mean_reversion,"
            "session_breakout,cross_rate,relative_strength,volatility_squeeze "
            "or momentum,ma,breakout,mean_reversion,session_breakout,cross_rate,"
            "relative_strength,volatility_squeeze,dual_squeeze "
            "or add macd_momentum,kalman_trend as the tenth and eleventh fields"
        ),
    )
    parser.add_argument(
        "--behavior-candidate",
        action="append",
        default=None,
        help=(
            "Router behavior tuple as entry_score,min_confidence,cost_buffer,"
            "conflict_penalty,override_enabled. Repeat to validate multiple "
            "router confirmation profiles in walk-forward."
        ),
    )
    parser.add_argument(
        "--conservative-behavior-grid",
        action="store_true",
        help="Test the stricter router behavior profiles in each train/test fold.",
    )
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--min-train-fills", type=int, default=1)
    parser.add_argument("--min-train-return-pct", type=float, default=0.0)
    parser.add_argument("--min-test-fills", type=int, default=1)
    parser.add_argument("--min-stable-fold-fraction", type=float, default=0.50)
    parser.add_argument("--max-test-drawdown-pct", type=float, default=0.05)
    parser.add_argument("--min-risk-discipline-score", type=int, default=80)
    parser.add_argument(
        "--summary-output",
        default="outputs/backtests/portfolio_router_walk_forward_summary.csv",
    )
    parser.add_argument(
        "--folds-output",
        default="outputs/backtests/portfolio_router_walk_forward_folds.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    settings = config.walk_forward
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    weight_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_ROUTER_WEIGHT_SETS
    )
    behavior_profiles = _behavior_profiles(args)
    result = run_portfolio_router_walk_forward(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        weight_sets=weight_sets,
        behavior_profiles=behavior_profiles,
        train_size=args.train_size or settings.train_size,
        test_size=args.test_size or settings.test_size,
        step_size=args.step_size or settings.step_size,
        min_train_fills=args.min_train_fills,
        min_train_return_pct=args.min_train_return_pct,
        min_test_fills=args.min_test_fills,
        min_stable_fold_fraction=args.min_stable_fold_fraction,
        max_test_drawdown_pct=args.max_test_drawdown_pct,
        min_risk_discipline_score=args.min_risk_discipline_score,
    )
    write_portfolio_router_walk_forward_summary_csv(
        result,
        args.summary_output,
        min_stable_fold_fraction=args.min_stable_fold_fraction,
        max_worst_drawdown_pct=args.max_test_drawdown_pct,
        min_risk_discipline_score=args.min_risk_discipline_score,
    )
    write_portfolio_router_walk_forward_folds_csv(result, args.folds_output)

    summary = result.summary
    promotion = decide_router_promotion(
        summary,
        min_stable_fold_fraction=args.min_stable_fold_fraction,
        max_worst_drawdown_pct=args.max_test_drawdown_pct,
        min_risk_discipline_score=args.min_risk_discipline_score,
    )
    print("Portfolio Router Walk-Forward")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Weight candidates: {len(result.weight_sets)}")
    print(f"  Behavior candidates: {len(result.behavior_profiles)}")
    print(f"  Folds: {len(result.folds)}")
    print(f"  Eligible: {summary.eligible}")
    print(f"  Promotion: {promotion.status} ({promotion.reason})")
    print(f"  Stable fold fraction: {summary.stable_fold_fraction:.1%}")
    print(f"  Selected was test-best: {summary.selected_was_test_best_fraction:.1%}")
    print(f"  Median test proxy: {summary.median_test_proxy_score:.1f}")
    print(f"  Median test return: {summary.median_test_return_pct:.3%}")
    print(f"  Worst test drawdown: {summary.worst_test_drawdown_pct:.3%}")
    print(f"  Average risk discipline: {summary.average_risk_discipline_score:.1f}/100")
    print(f"  Most selected weights: {summary.most_selected_weights}")
    print(f"  Most selected behavior: {summary.most_selected_behavior}")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Folds CSV: {args.folds_output}")
    print("Folds")
    for fold in result.folds:
        metrics = fold.selected_test_candidate.competition_metrics
        print(
            f"  {fold.fold_index}. {fold.selected_weights.label}; "
            f"{fold.selected_behavior.label}: "
            f"test_proxy={fold.selected_test_candidate.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"risk={fold.selected_test_candidate.risk_discipline.score}/100, "
            f"test_best={fold.selected_was_test_best}, "
            f"stable={fold.stable_candidate}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> RouterWeightSet:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {4, 5, 6, 7, 8, 9, 10, 11}:
        raise argparse.ArgumentTypeError(
            "candidate must contain between four and eleven comma-separated weights"
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate weights must be numbers") from exc
    return RouterWeightSet(*values)


def _behavior_profiles(args: argparse.Namespace) -> tuple[RouterBehaviorProfile, ...]:
    profiles: list[RouterBehaviorProfile] = []
    if args.conservative_behavior_grid:
        profiles.extend(CONSERVATIVE_ROUTER_BEHAVIOR_PROFILES)
    if args.behavior_candidate:
        profiles.extend(_parse_behavior_candidate(value) for value in args.behavior_candidate)
    if not profiles:
        return (RouterBehaviorProfile(),)
    return tuple(_unique_behavior_profiles(tuple(profiles)))


def _parse_behavior_candidate(raw: str) -> RouterBehaviorProfile:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {5, 6}:
        raise argparse.ArgumentTypeError(
            "behavior candidate must contain five or six comma-separated values"
        )
    try:
        entry_score = float(parts[0])
        min_signal_confidence = float(parts[1])
        cost_buffer = float(parts[2])
        conflict_penalty = float(parts[3])
        override_enabled = _parse_bool(parts[4])
        exit_score = float(parts[5]) if len(parts) == 6 else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "behavior candidate numeric fields must be numbers"
        ) from exc
    return RouterBehaviorProfile(
        entry_score=entry_score,
        min_signal_confidence=min_signal_confidence,
        cost_buffer=cost_buffer,
        conflict_penalty=conflict_penalty,
        primary_signal_override_enabled=override_enabled,
        exit_score=exit_score,
    )


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "override_enabled must be true/false, yes/no, on/off, or 1/0"
    )


def _unique_behavior_profiles(
    profiles: tuple[RouterBehaviorProfile, ...],
) -> tuple[RouterBehaviorProfile, ...]:
    unique: list[RouterBehaviorProfile] = []
    seen: set[RouterBehaviorProfile] = set()
    for profile in profiles:
        if profile in seen:
            continue
        seen.add(profile)
        unique.append(profile)
    return tuple(unique)
