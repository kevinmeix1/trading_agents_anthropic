from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.crypto_overlay_fold_diagnostic import (
    DEFAULT_FOLD_DIAGNOSTIC_SPEC,
    build_crypto_overlay_fold_diagnostic,
    write_crypto_overlay_fold_diagnostic_summary_csv,
    write_crypto_overlay_fold_symbol_summary_csv,
)
from quanthack.cli._format import money
from quanthack.cli.crypto_overlay_sizing_compare import _parse_candidate
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun one crypto overlay candidate and export fold-level fills and "
            "trade attribution."
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
    parser.add_argument("--base-strategy", choices=STRATEGY_NAMES, default="macd_momentum")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument(
        "--candidate",
        default=None,
        help=(
            "Optional candidate spec like "
            "label=demo,crypto=0.75,btc=0.75,sol=1.0,crypto_hours=7|8|9"
        ),
    )
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument(
        "--output-prefix",
        default="outputs/research/crypto_overlay_fold_diagnostic",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    spec = _parse_candidate(args.candidate) if args.candidate else DEFAULT_FOLD_DIAGNOSTIC_SPEC
    diagnostic = build_crypto_overlay_fold_diagnostic(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=args.base_strategy,
        spec=spec,
        symbols=tuple(args.symbol) if args.symbol else None,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        output_prefix=args.output_prefix,
    )
    prefix = Path(args.output_prefix)
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    folds_path = prefix.with_name(f"{prefix.name}_folds.csv")
    fills_path = prefix.with_name(f"{prefix.name}_fills.csv")
    attribution_path = prefix.with_name(f"{prefix.name}_attribution.csv")
    symbol_summary_path = prefix.with_name(f"{prefix.name}_symbol_summary.csv")
    write_crypto_overlay_fold_diagnostic_summary_csv(diagnostic, summary_path)
    write_crypto_overlay_fold_symbol_summary_csv(diagnostic, symbol_summary_path)

    print("Crypto Overlay Fold Diagnostic")
    print(f"  Candidate: {diagnostic.spec.label}")
    print(f"  Crypto hours: {_hours_text(diagnostic.spec.crypto_allowed_utc_hours)}")
    print(f"  Folds: {len(diagnostic.walk_forward.folds)}")
    print(f"  Positive folds: {diagnostic.walk_forward.positive_fold_fraction:.1%}")
    print(f"  Non-negative folds: {diagnostic.walk_forward.non_negative_fold_fraction:.1%}")
    print(
        "  Largest positive fold contribution: "
        f"{diagnostic.walk_forward.largest_positive_fold_contribution:.1%}"
    )
    print(
        f"  Strongest fold: {diagnostic.strongest_fold_index} "
        f"({diagnostic.strongest_fold_return_pct:.3%})"
    )
    print(f"  Promotion: {diagnostic.promotion.status} - {diagnostic.promotion.reason}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Folds CSV: {folds_path}")
    print(f"  Evaluation fills CSV: {fills_path}")
    print(f"  Attribution CSV: {attribution_path}")
    print(f"  Symbol summary CSV: {symbol_summary_path}")
    print("Strongest attribution rows")
    for row in diagnostic.attribution.strongest_rows[:5]:
        print(
            f"  fold {row.fold} {row.symbol} {row.primary_signal} "
            f"h{row.utc_hour:02d} {row.side}: "
            f"pnl={money(row.realized_pnl_usd)}, fills={row.fills}, "
            f"fold_return={row.fold_return_pct:.3%}"
        )
    print("Weakest attribution rows")
    for row in diagnostic.attribution.weakest_rows[:5]:
        print(
            f"  fold {row.fold} {row.symbol} {row.primary_signal} "
            f"h{row.utc_hour:02d} {row.side}: "
            f"pnl={money(row.realized_pnl_usd)}, fills={row.fills}, "
            f"fold_return={row.fold_return_pct:.3%}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _hours_text(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return "all"
    return "|".join(str(hour) for hour in hours)
