from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.portfolio_universe_scan import UniverseBasket
from quanthack.backtesting.portfolio_walk_forward import (
    decide_portfolio_promotion,
    run_portfolio_walk_forward,
    write_portfolio_walk_forward_folds_csv,
    write_portfolio_walk_forward_summary_csv,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run portfolio walk-forward validation on diversified baskets."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGY_NAMES,
        default=None,
        help="Strategy to consider during train-window selection. Default: alpha_router.",
    )
    parser.add_argument(
        "--basket",
        action="append",
        default=None,
        help="Custom basket as name:EURUSD,USDJPY,XAUUSD. Repeat for multiple baskets.",
    )
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--min-symbols", type=int, default=3)
    parser.add_argument("--max-symbols", type=int, default=5)
    parser.add_argument("--max-baskets", type=int, default=25)
    parser.add_argument("--min-test-fills", type=int, default=1)
    parser.add_argument("--min-stable-fold-fraction", type=float, default=0.50)
    parser.add_argument("--max-test-drawdown-pct", type=float, default=0.05)
    parser.add_argument("--min-risk-discipline-score", type=int, default=80)
    parser.add_argument(
        "--summary-output",
        default="outputs/backtests/portfolio_walk_forward_summary.csv",
    )
    parser.add_argument(
        "--folds-output",
        default="outputs/backtests/portfolio_walk_forward_folds.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    settings = config.walk_forward
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    result = run_portfolio_walk_forward(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=tuple(args.strategy) if args.strategy else ("alpha_router",),
        baskets=_parse_baskets(args.basket),
        min_symbols=args.min_symbols,
        max_symbols=args.max_symbols,
        max_baskets=args.max_baskets,
        train_size=args.train_size or settings.train_size,
        test_size=args.test_size or settings.test_size,
        step_size=args.step_size or settings.step_size,
        min_test_fills=args.min_test_fills,
        min_stable_fold_fraction=args.min_stable_fold_fraction,
        max_test_drawdown_pct=args.max_test_drawdown_pct,
        min_risk_discipline_score=args.min_risk_discipline_score,
    )
    write_portfolio_walk_forward_summary_csv(result, args.summary_output)
    write_portfolio_walk_forward_folds_csv(result, args.folds_output)

    summary = result.summary
    promotion = decide_portfolio_promotion(summary)
    print("Portfolio Walk-Forward Validation")
    print(f"  Available symbols: {', '.join(result.available_symbols)}")
    print(f"  Folds: {len(result.folds)}")
    print(f"  Eligible: {summary.eligible}")
    print(f"  Promotion: {promotion.status}")
    print(f"  Promotion reason: {promotion.reason}")
    print(f"  Stable fold fraction: {summary.stable_fold_fraction:.1%}")
    print(f"  Median test proxy: {summary.median_test_proxy_score:.1f}")
    print(f"  Median test return: {summary.median_test_return_pct:.3%}")
    print(f"  Worst test drawdown: {summary.worst_test_drawdown_pct:.3%}")
    print(f"  Average risk discipline: {summary.average_risk_discipline_score:.1f}/100")
    print(f"  Most selected basket: {summary.most_selected_basket}")
    print(f"  Most selected strategy: {summary.most_selected_strategy}")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Folds CSV: {args.folds_output}")
    print("Folds")
    for fold in result.folds:
        metrics = fold.test_row.competition_metrics
        print(
            f"  {fold.fold_index}. {fold.selected_candidate_text}: "
            f"test_proxy={fold.test_row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"risk={fold.test_row.risk_discipline.score}/100, "
            f"stable={fold.stable_candidate}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_baskets(values: list[str] | None) -> tuple[UniverseBasket, ...] | None:
    if not values:
        return None

    baskets: list[UniverseBasket] = []
    for index, value in enumerate(values, start=1):
        if ":" in value:
            name, symbol_text = value.split(":", 1)
            name = name.strip()
        else:
            name = f"custom_{index}"
            symbol_text = value
        symbols = tuple(
            symbol.strip()
            for symbol in symbol_text.replace(" ", ",").split(",")
            if symbol.strip()
        )
        baskets.append(UniverseBasket(name=name, symbols=symbols))
    return tuple(baskets)
