from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.router_optimizer import (
    CONSERVATIVE_ROUTER_BEHAVIOR_PROFILES,
    DEFAULT_ROUTER_WEIGHT_SETS,
    RouterBehaviorProfile,
    RouterWeightSet,
    optimize_router_weights,
    write_router_optimization_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize alpha-router weights with allocator-aware portfolio backtests."
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
            "conflict_penalty,override_enabled. Repeat to compare multiple "
            "confirmation profiles."
        ),
    )
    parser.add_argument(
        "--conservative-behavior-grid",
        action="store_true",
        help=(
            "Also test stricter, lower-turnover behavior profiles that disable "
            "primary-signal override and require stronger router scores."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/router_weight_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    weight_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_ROUTER_WEIGHT_SETS
    )
    behavior_profiles = _behavior_profiles(args)
    result = optimize_router_weights(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        weight_sets=weight_sets,
        behavior_profiles=behavior_profiles,
    )
    write_router_optimization_csv(result, args.output)

    print("Router Weight Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates, start=1):
        metrics = candidate.competition_metrics
        print(
            f"  {rank}. {candidate.weights.label}: "
            f"{candidate.behavior.label}: "
            f"proxy={candidate.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
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
