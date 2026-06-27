from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.adaptive_strategy_oracle import (
    build_adaptive_strategy_oracle_diagnostic,
    write_adaptive_strategy_oracle_candidates_csv,
    write_adaptive_strategy_oracle_folds_csv,
    write_adaptive_strategy_oracle_summary_csv,
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
        description="Diagnose adaptive selector regret versus fold-level oracle."
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument("--loss-cooldown-folds", type=int, default=1)
    parser.add_argument("--min-train-fills", type=int, default=0)
    parser.add_argument("--min-train-adjusted-return-pct", type=float, default=None)
    parser.add_argument("--train-fill-penalty-pct", type=float, default=0.0)
    parser.add_argument("--train-stability-splits", type=int, default=0)
    parser.add_argument("--prefer-train-stability", action="store_true")
    parser.add_argument("--transition-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--allow-cash-fallback", action="store_true")
    parser.add_argument(
        "--exclude-cash-oracle",
        action="store_true",
        help="Do not allow cash to count as the ex-post oracle best action.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/research/adaptive_strategy_oracle",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    strategies = tuple(args.strategy or DEFAULT_STRATEGIES)
    diagnostic = build_adaptive_strategy_oracle_diagnostic(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=strategies,
        symbols=tuple(args.symbol) if args.symbol else None,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        loss_cooldown_folds=args.loss_cooldown_folds,
        min_train_fills=args.min_train_fills,
        min_train_drawdown_adjusted_return_pct=(
            args.min_train_adjusted_return_pct
        ),
        train_fill_penalty_pct=args.train_fill_penalty_pct,
        train_stability_splits=args.train_stability_splits,
        prefer_train_stability=args.prefer_train_stability,
        transition_risk_multiplier=args.transition_risk_multiplier,
        allow_cash_fallback=args.allow_cash_fallback,
        include_cash_oracle=not args.exclude_cash_oracle,
    )
    output_prefix = Path(args.output_prefix)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    folds_path = output_prefix.with_name(f"{output_prefix.name}_folds.csv")
    candidates_path = output_prefix.with_name(f"{output_prefix.name}_candidates.csv")
    write_adaptive_strategy_oracle_summary_csv(diagnostic, summary_path)
    write_adaptive_strategy_oracle_folds_csv(diagnostic, folds_path)
    write_adaptive_strategy_oracle_candidates_csv(diagnostic, candidates_path)

    print("Adaptive Strategy Oracle Diagnostic")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Folds: {diagnostic.fold_count}")
    print(
        "  Selected was oracle: "
        f"{diagnostic.selected_was_oracle_fraction:.1%}"
    )
    print(f"  Total regret: {diagnostic.total_regret_pct:.3%}")
    print(f"  Average regret: {diagnostic.average_regret_pct:.3%}")
    print(f"  Regret folds: {len(diagnostic.regret_folds)}")
    print(f"  Negative selected folds: {diagnostic.negative_selected_folds}")
    print(f"  Cash oracle folds: {diagnostic.cash_oracle_folds}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Folds CSV: {folds_path}")
    print(f"  Candidates CSV: {candidates_path}")
    for fold in sorted(
        diagnostic.regret_folds,
        key=lambda item: item.regret_pct,
        reverse=True,
    )[:5]:
        print(
            f"  Fold {fold.fold_index}: "
            f"selected={fold.selected_strategy} ({fold.selected_return_pct:.3%}), "
            f"oracle={fold.oracle_strategy} ({fold.oracle_return_pct:.3%}), "
            f"regret={fold.regret_pct:.3%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
