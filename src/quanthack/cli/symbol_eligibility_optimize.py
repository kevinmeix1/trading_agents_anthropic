from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.symbol_eligibility_optimizer import (
    optimize_symbol_eligibility,
    write_symbol_attribution_rank_csv,
    write_symbol_eligibility_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize strategy symbol eligibility from attribution and "
            "shared-risk portfolio backtests."
        )
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="dual_squeeze")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--min-symbols", type=int, default=3)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--min-fills", type=int, default=1)
    parser.add_argument("--min-symbol-pnl", type=float, default=0.0)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    parser.add_argument("--max-exclusions", type=int, default=3)
    parser.add_argument(
        "--exclude-flat-symbols",
        action="store_true",
        help="Do not keep symbols that had no fills in attribution-derived candidates.",
    )
    parser.add_argument(
        "--include-walk-forward",
        action="store_true",
        help=(
            "Attach fixed-warmup walk-forward metrics and rank candidates by "
            "out-of-sample stability."
        ),
    )
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--include-combinations",
        action="store_true",
        help="Also evaluate combinations from the top attribution-ranked symbols.",
    )
    parser.add_argument(
        "--combination-pool-size",
        type=int,
        default=0,
        help="How many top attributed symbols to use for combination search; 0 means all.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=200,
        help="Maximum generated combination candidates before de-duplication.",
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/symbol_eligibility_optimization.csv",
    )
    parser.add_argument(
        "--attribution-output",
        default=None,
        help="Optional CSV path for the ranked source attribution rows.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    result = optimize_symbol_eligibility(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_name=args.strategy,
        symbols=tuple(args.symbol) if args.symbol else None,
        min_symbols=args.min_symbols,
        max_symbols=args.max_symbols,
        min_fills=args.min_fills,
        min_symbol_pnl_usd=args.min_symbol_pnl,
        min_profit_factor=args.min_profit_factor,
        max_exclusions=args.max_exclusions,
        include_flat_symbols=not args.exclude_flat_symbols,
        include_walk_forward=args.include_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        include_combinations=args.include_combinations,
        combination_pool_size=args.combination_pool_size,
        max_combinations=args.max_combinations,
    )
    write_symbol_eligibility_csv(result, args.output)
    attribution_output = args.attribution_output or _default_attribution_output(
        args.output
    )
    write_symbol_attribution_rank_csv(result, attribution_output)

    print("Symbol Eligibility Optimization")
    print(f"  Strategy: {result.strategy_name}")
    print(f"  Available symbols: {', '.join(result.available_symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print(f"  Attribution CSV: {attribution_output}")
    print(f"  Walk-forward ranking: {'on' if args.include_walk_forward else 'off'}")
    print("Top candidates")
    for rank, candidate in enumerate(result.candidates[:8], start=1):
        metrics = candidate.comparison_row.competition_metrics
        walk_forward = candidate.walk_forward
        wf_text = ""
        if walk_forward is not None:
            wf_text = (
                f", wf_pos={walk_forward.positive_fold_fraction:.1%}, "
                f"wf_active_pos={walk_forward.active_positive_fold_fraction:.1%}, "
                f"wf_nonneg={walk_forward.non_negative_fold_fraction:.1%}, "
                f"wf_median={walk_forward.median_test_return_pct:.3%}, "
                f"wf_active_median={walk_forward.median_active_test_return_pct:.3%}, "
                f"wf_conc={walk_forward.largest_positive_fold_contribution:.1%}"
            )
        elif candidate.walk_forward_error:
            wf_text = f", wf_error={candidate.walk_forward_error}"
        print(
            f"  {rank}. {candidate.name}: "
            f"symbols={','.join(candidate.symbols)}, "
            f"excluded={','.join(candidate.excluded_symbols) or 'none'}, "
            f"proxy={candidate.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.comparison_row.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
            f"{wf_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _default_attribution_output(output: str) -> str:
    path = Path(output)
    return str(path.with_name(f"{path.stem}_attribution.csv"))
