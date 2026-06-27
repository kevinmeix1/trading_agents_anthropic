from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.crypto_allocation_compare import (
    DEFAULT_CRYPTO_ALLOCATION_STRATEGIES,
    compare_crypto_allocations,
    write_crypto_allocation_comparison_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare crypto portfolio allocations by assigning each crypto symbol "
            "to one of several strategy sleeves."
        )
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--price-csv",
        default="data/research_crypto_proxy_14d_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/research_crypto_proxy_14d_quotes.csv",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGY_NAMES,
        default=None,
        help="Strategy sleeve to allocate by symbol. Repeat to override defaults.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="Crypto symbol to include. Repeat to test a subset.",
    )
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--max-maps", type=int, default=128)
    parser.add_argument(
        "--output",
        default="outputs/research/crypto_allocation_comparison.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    comparison = compare_crypto_allocations(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=(
            tuple(args.strategy)
            if args.strategy
            else DEFAULT_CRYPTO_ALLOCATION_STRATEGIES
        ),
        symbols=tuple(args.symbol) if args.symbol else None,
        run_walk_forward=not args.no_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        max_maps=args.max_maps,
    )
    write_crypto_allocation_comparison_csv(comparison, args.output)

    print("Crypto Allocation Comparison")
    print(f"  Symbols: {', '.join(comparison.symbols)}")
    print(f"  Strategies: {', '.join(comparison.strategy_names)}")
    print(f"  Price CSV: {args.price_csv}")
    print(f"  Quote CSV: {args.quote_csv}")
    print(f"  Walk-forward: {'disabled' if args.no_walk_forward else 'enabled'}")
    print(f"  Output CSV: {args.output}")
    print("Ranked allocations")
    for rank, candidate in enumerate(comparison.candidates[:12], start=1):
        metrics = candidate.competition_metrics
        wf_text = "wf=n/a"
        if candidate.walk_forward is not None:
            promotion = candidate.promotion.status if candidate.promotion else "UNKNOWN"
            wf_text = (
                f"wf_nonneg={candidate.walk_forward.non_negative_fold_fraction:.1%}, "
                f"wf_active_pos={candidate.walk_forward.active_positive_fold_fraction:.1%}, "
                f"wf_med_active={candidate.walk_forward.median_active_test_return_pct:.3%}, "
                f"promotion={promotion}"
            )
        print(
            f"  {rank}. {candidate.label}: "
            f"selection={candidate.selection_score:.1f}, "
            f"proxy={candidate.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}, "
            f"{wf_text}, "
            f"map={candidate.strategy_map_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
