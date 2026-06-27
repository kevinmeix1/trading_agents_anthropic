from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.adaptive_strategy_policy_sweep import (
    DEFAULT_CASH_FALLBACK_VALUES,
    DEFAULT_LOSS_COOLDOWNS,
    DEFAULT_MIN_TRAIN_ADJUSTED_RETURNS,
    DEFAULT_TRAIN_FILL_PENALTIES,
    DEFAULT_TRANSITION_RISK_MULTIPLIERS,
    sweep_adaptive_strategy_policies,
    write_adaptive_strategy_policy_sweep_csv,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


DEFAULT_STRATEGIES = (
    "kalman_trend",
    "champion_ensemble",
    "macd_momentum",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep adaptive strategy-selector policy settings."
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--loss-cooldown-folds",
        action="append",
        type=int,
        default=None,
        help="Loss cooldown value to test. Repeat to override defaults.",
    )
    parser.add_argument(
        "--min-train-fills",
        action="append",
        type=int,
        default=None,
        help="Minimum training fills to test. Repeat to override default 0.",
    )
    parser.add_argument(
        "--min-train-adjusted-return-pct",
        action="append",
        default=None,
        help=(
            "Training drawdown-adjusted return gate to test. Use 'none' for no "
            "gate. Repeat to override defaults."
        ),
    )
    parser.add_argument(
        "--train-fill-penalty-pct",
        action="append",
        type=float,
        default=None,
        help="Per-fill training return penalty to test. Repeat.",
    )
    parser.add_argument(
        "--transition-risk-multiplier",
        action="append",
        type=float,
        default=None,
        help="Risk multiplier on folds after selector changes strategy. Repeat.",
    )
    parser.add_argument(
        "--cash-fallback",
        action="append",
        choices=("yes", "no"),
        default=None,
        help="Whether to allow cash fallback. Repeat yes/no to override defaults.",
    )
    parser.add_argument(
        "--include-stability-preference",
        action="store_true",
        help="Also test train-stability-splits=4 with stability-preferred ranking.",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--output",
        default="outputs/research/adaptive_strategy_policy_sweep.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    strategies = tuple(args.strategy or DEFAULT_STRATEGIES)
    stability_settings = (
        ((0, False), (4, True))
        if args.include_stability_preference
        else ((0, False),)
    )
    result = sweep_adaptive_strategy_policies(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=strategies,
        symbols=tuple(args.symbol) if args.symbol else None,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        loss_cooldown_values=tuple(
            args.loss_cooldown_folds or DEFAULT_LOSS_COOLDOWNS
        ),
        min_train_fills_values=tuple(args.min_train_fills or (0,)),
        min_train_adjusted_return_values=(
            _parse_optional_float_values(args.min_train_adjusted_return_pct)
            if args.min_train_adjusted_return_pct is not None
            else DEFAULT_MIN_TRAIN_ADJUSTED_RETURNS
        ),
        train_fill_penalty_values=tuple(
            args.train_fill_penalty_pct or DEFAULT_TRAIN_FILL_PENALTIES
        ),
        transition_risk_multiplier_values=tuple(
            args.transition_risk_multiplier or DEFAULT_TRANSITION_RISK_MULTIPLIERS
        ),
        cash_fallback_values=(
            _parse_cash_fallback_values(args.cash_fallback)
            if args.cash_fallback is not None
            else DEFAULT_CASH_FALLBACK_VALUES
        ),
        train_stability_settings=stability_settings,
    )
    write_adaptive_strategy_policy_sweep_csv(result, args.output)

    print("Adaptive Strategy Policy Sweep")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Output CSV: {args.output}")
    for rank, candidate in enumerate(result.candidates[: max(args.limit, 0)], start=1):
        adaptive = candidate.result
        min_train_adjusted = adaptive.min_train_drawdown_adjusted_return_pct
        print(
            f"  {rank}. status={candidate.decision.status}, "
            f"score={candidate.selector_score:.2f}, "
            f"cooldown={adaptive.loss_cooldown_folds}, "
            f"min_fills={adaptive.min_train_fills}, "
            f"min_adj={min_train_adjusted}, "
            f"fill_penalty={adaptive.train_fill_penalty_pct:g}, "
            f"risk_mult={adaptive.transition_risk_multiplier:.2f}, "
            f"cash={'yes' if adaptive.allow_cash_fallback else 'no'}, "
            f"stability={adaptive.train_stability_splits}/"
            f"{'yes' if adaptive.prefer_train_stability else 'no'}, "
            f"pos={adaptive.positive_fold_fraction:.1%}, "
            f"active_pos={adaptive.active_positive_fold_fraction:.1%}, "
            f"nonneg={adaptive.non_negative_fold_fraction:.1%}, "
            f"cmpd={adaptive.compounded_test_return_pct:.3%}, "
            f"median_active={adaptive.median_active_test_return_pct:.3%}, "
            f"dd={adaptive.worst_test_drawdown_pct:.3%}, "
            f"fills={adaptive.total_evaluation_fills}, "
            f"selected={candidate.selection_counts_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_optional_float_values(values: list[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            parsed.append(None)
        else:
            parsed.append(float(value))
    return tuple(parsed)


def _parse_cash_fallback_values(values: list[str]) -> tuple[bool, ...]:
    return tuple(value == "yes" for value in values)
