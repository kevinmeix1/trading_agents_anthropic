from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.crypto_promotion_pipeline import (
    run_crypto_promotion_pipeline,
)
from quanthack.cli.asset_class_stability_optimize import _parse_asset_candidate
from quanthack.cli.crypto_overlay_component_ablation import _parse_component
from quanthack.cli.crypto_overlay_sizing_compare import _parse_candidate
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full crypto overlay promotion pipeline: data health, sizing "
            "comparison, candidate gates, component ablation, fold diagnostics, "
            "and one go/no-go summary."
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
        "--data-source",
        default="mixed_proxy",
        choices=("official", "proxy", "mixed_proxy", "synthetic"),
        help="Evidence source type used by the research gate.",
    )
    parser.add_argument("--base-strategy", choices=STRATEGY_NAMES, default="macd_momentum")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Optional sizing candidate spec. Repeat to override defaults. "
            "Example: label=demo,crypto=0.75,btc=0.75,sol=1.0,crypto_hours=7|8"
        ),
    )
    parser.add_argument(
        "--component",
        action="append",
        default=None,
        help=(
            "Optional component ablation spec. Repeat to override defaults. "
            "Example: label=no_crypto,assets=CRYPTO"
        ),
    )
    parser.add_argument(
        "--asset-candidate",
        action="append",
        default=None,
        help=(
            "Optional asset-class stability spec. Repeat to override defaults. "
            "Example: label=demo,fx=1.0,metal=0.75,"
            "crypto_spec=label=crypto,crypto=0.75,btc=0.75,sol=1.0"
        ),
    )
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument("--max-gap-seconds", type=float, default=960.0)
    parser.add_argument("--max-live-fold-contribution", type=float, default=0.80)
    parser.add_argument(
        "--output-prefix",
        default="outputs/research/crypto_promotion_pipeline",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = run_crypto_promotion_pipeline(
        config=config,
        prices=prices,
        quotes=quotes,
        price_csv=args.price_csv,
        quote_csv=args.quote_csv,
        data_source=args.data_source,
        output_prefix=args.output_prefix,
        base_strategy=args.base_strategy,
        symbols=tuple(args.symbol) if args.symbol else None,
        sizing_specs=tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else None,
        component_specs=tuple(_parse_component(value) for value in args.component)
        if args.component
        else None,
        asset_class_specs=tuple(
            _parse_asset_candidate(value) for value in args.asset_candidate
        )
        if args.asset_candidate
        else None,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        max_gap_seconds=args.max_gap_seconds,
        max_live_fold_contribution=args.max_live_fold_contribution,
    )

    summary = result.summary
    print("Crypto Promotion Pipeline")
    print(f"  Data source: {summary.data_source.value}")
    print(f"  Data health: {summary.data_health.value} ({summary.data_health_issue_count} issues)")
    print(f"  Symbols: {len(summary.selected_symbols)}")
    print(f"  Crypto symbols: {', '.join(summary.crypto_symbols) or 'none'}")
    print(f"  Best sizing: {summary.best_sizing_label}")
    print(
        f"    return={summary.best_sizing_return_pct:.3%}, "
        f"drawdown={summary.best_sizing_drawdown_pct:.3%}, "
        f"sharpe15={summary.best_sizing_sharpe_15m:.3f}, "
        f"risk={summary.best_sizing_risk_score:.0f}/100"
    )
    print(
        f"  Fold diagnostic: strongest_fold={summary.strongest_fold}, "
        f"strongest_return={summary.strongest_fold_return_pct:.3%}, "
        f"largest_contribution={summary.largest_positive_fold_contribution:.1%}"
    )
    print(
        f"  Stable backup: {summary.stable_backup_label or 'none'} "
        f"return={summary.stable_backup_return_pct:.3%}, "
        f"fold_contribution={summary.stable_backup_fold_contribution:.1%}"
    )
    print(
        f"  Decision: {summary.promotion_readiness.value}, "
        f"live_ready={summary.live_ready}"
    )
    print(f"  Reason: {summary.promotion_reason}")
    print(f"  Summary CSV: {result.artifacts.summary_csv}")
    print(f"  Sizing CSV: {result.artifacts.sizing_csv}")
    print(f"  Component CSV: {result.artifacts.component_ablation_csv}")
    print(f"  Asset-class stability CSV: {result.artifacts.asset_class_stability_csv}")
    print(f"  Fold diagnostic prefix: {result.artifacts.fold_diagnostic_prefix}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
