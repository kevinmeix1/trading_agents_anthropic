from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.crypto_fold_stability_optimizer import (
    DEFAULT_STABILITY_SPECS,
    optimize_crypto_fold_stability,
    write_crypto_fold_stability_csv,
)
from quanthack.cli.crypto_overlay_sizing_compare import _parse_candidate
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search crypto overlay sizing/session variants with an explicit "
            "penalty for fold concentration."
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
        action="append",
        default=None,
        help=(
            "Optional candidate spec. Repeat to override the default grid. "
            "Example: label=demo,crypto=0.5,btc=0.75,sol=1.0,crypto_hours=7|8"
        ),
    )
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--output",
        default="outputs/research/crypto_fold_stability_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = optimize_crypto_fold_stability(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=args.base_strategy,
        symbols=tuple(args.symbol) if args.symbol else None,
        specs=tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_STABILITY_SPECS,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_crypto_fold_stability_csv(result, args.output)

    print("Crypto Fold Stability Optimization")
    print(f"  Official symbols: {', '.join(result.comparison.official_symbols) or 'none'}")
    print(f"  Crypto symbols: {', '.join(result.comparison.crypto_symbols) or 'none'}")
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Output CSV: {args.output}")
    for rank, candidate in enumerate(result.candidates[: max(args.limit, 0)], start=1):
        sizing = candidate.sizing
        metrics = sizing.competition_metrics
        walk_forward = sizing.walk_forward
        wf_text = "wf=n/a"
        if walk_forward is not None:
            wf_text = (
                f"wf_pos={walk_forward.positive_fold_fraction:.1%}, "
                f"wf_active_pos={walk_forward.active_positive_fold_fraction:.1%}, "
                f"wf_nonneg={walk_forward.non_negative_fold_fraction:.1%}, "
                f"wf_med_active={walk_forward.median_active_test_return_pct:.3%}, "
                f"wf_conc={walk_forward.largest_positive_fold_contribution:.1%}"
            )
        print(
            f"  {rank}. {sizing.label}: "
            f"status={candidate.stability_status}, "
            f"stable_score={candidate.stability_score:.1f}, "
            f"retention={candidate.return_retention:.1%}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"trades={metrics.trade_count}, "
            f"{wf_text}"
        )
        print(f"      multipliers: {sizing.multiplier_map_text}")
        print(f"      crypto hours: {sizing.crypto_allowed_utc_hours_text}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
