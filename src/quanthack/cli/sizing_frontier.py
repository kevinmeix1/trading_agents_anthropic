from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.sizing_frontier import (
    evaluate_sizing_frontier,
    write_sizing_frontier_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep strategy-map sizing caps with portfolio backtests."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument(
        "--strategy-map",
        action="append",
        required=True,
        metavar="SYMBOL=STRATEGY",
        help="Per-symbol strategy assignment. Repeat for every traded symbol.",
    )
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--symbol-notional-pct",
        action="append",
        type=float,
        default=None,
        help="Per-symbol notional cap as decimal equity percent. Repeat to sweep.",
    )
    parser.add_argument("--max-gross-leverage", type=float, default=None)
    parser.add_argument("--include-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--test-size", type=int, default=96)
    parser.add_argument("--step-size", type=int, default=96)
    parser.add_argument(
        "--output",
        default="outputs/backtests/sizing_frontier.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    strategy_by_symbol = _parse_strategy_map(args.strategy_map)
    result = evaluate_sizing_frontier(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_by_symbol=strategy_by_symbol,
        symbol_notional_pcts=tuple(args.symbol_notional_pct or (0.25, 0.40, 0.60, 0.80)),
        max_gross_leverage=args.max_gross_leverage,
        include_walk_forward=args.include_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_sizing_frontier_csv(result, args.output)

    print("Sizing Frontier")
    print(f"  Config: {args.config}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Strategy map: {_format_strategy_map(result.strategy_by_symbol)}")
    print(f"  Walk-forward: {'yes' if args.include_walk_forward else 'no'}")
    print(f"  Output CSV: {args.output}")
    print("Points")
    for point in result.points:
        metrics = point.competition_metrics
        wf = point.walk_forward
        wf_text = (
            ""
            if wf is None
            else (
                f", wf_nonneg={wf.non_negative_fold_fraction:.1%}, "
                f"wf_active_pos={wf.active_positive_fold_fraction:.1%}, "
                f"wf_active_med={wf.median_active_test_return_pct:.3%}"
            )
        )
        print(
            f"  cap={point.symbol_notional_pct:.0%}: "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={point.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}, "
            f"worst_lev={point.worst_leverage:.2f}x"
            f"{wf_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_strategy_map(values: Sequence[str]) -> dict[str, str]:
    strategy_by_symbol: dict[str, str] = {}
    for raw_value in values:
        if "=" not in raw_value:
            raise SystemExit(
                "--strategy-map values must use SYMBOL=STRATEGY, "
                f"got {raw_value!r}"
            )
        raw_symbol, raw_strategy = raw_value.split("=", 1)
        symbol = instrument_for(raw_symbol).symbol
        strategy = raw_strategy.strip()
        if strategy not in STRATEGY_NAMES:
            valid = ", ".join(STRATEGY_NAMES)
            raise SystemExit(
                f"unknown strategy {strategy!r} in --strategy-map; expected one of: {valid}"
            )
        strategy_by_symbol[symbol] = strategy
    return strategy_by_symbol


def _format_strategy_map(strategy_by_symbol: tuple[tuple[str, str], ...]) -> str:
    return " ".join(f"{symbol}={strategy}" for symbol, strategy in strategy_by_symbol)
