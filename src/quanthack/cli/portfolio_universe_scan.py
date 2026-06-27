from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.portfolio_universe_scan import (
    DEFAULT_MAX_BASKETS,
    DEFAULT_MAX_SYMBOLS,
    DEFAULT_MIN_SYMBOLS,
    UniverseBasket,
    scan_portfolio_universes,
    write_portfolio_universe_scan_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank diversified symbol baskets with shared-risk portfolio backtests."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGY_NAMES,
        default=None,
        help="Strategy to evaluate. Repeat to compare several; default is alpha_router.",
    )
    parser.add_argument(
        "--basket",
        action="append",
        default=None,
        help="Custom basket as name:EURUSD,USDJPY,XAUUSD. Repeat for multiple baskets.",
    )
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--min-symbols", type=int, default=DEFAULT_MIN_SYMBOLS)
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    parser.add_argument("--max-baskets", type=int, default=DEFAULT_MAX_BASKETS)
    parser.add_argument(
        "--output",
        default="outputs/backtests/portfolio_universe_scan.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    baskets = _parse_baskets(args.basket)
    scan = scan_portfolio_universes(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=tuple(args.strategy) if args.strategy else ("alpha_router",),
        baskets=baskets,
        min_symbols=args.min_symbols,
        max_symbols=args.max_symbols,
        max_baskets=args.max_baskets,
    )
    write_portfolio_universe_scan_csv(scan, args.output)

    print("Portfolio Universe Scan")
    print(f"  Available symbols: {', '.join(scan.available_symbols)}")
    print(f"  Strategies: {', '.join(scan.strategies)}")
    print(f"  Baskets evaluated: {len(scan.baskets)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print("Top candidates")
    for rank, row in enumerate(scan.rows[:5], start=1):
        metrics = row.competition_metrics
        print(
            f"  {rank}. {row.basket.name} / {row.strategy_name}: "
            f"symbols={','.join(row.basket.symbols)}, "
            f"proxy={row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={row.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}"
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
