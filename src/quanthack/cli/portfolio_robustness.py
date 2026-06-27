from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime

from quanthack.backtesting.portfolio_robustness import (
    evaluate_leave_one_symbol_out,
    write_portfolio_robustness_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leave-one-symbol-out robustness checks for a portfolio strategy."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument(
        "--strategy-map",
        action="append",
        default=None,
        metavar="SYMBOL=STRATEGY",
        help="Optional per-symbol strategy override. Repeat for hybrid portfolios.",
    )
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--clock-open-at",
        default=None,
        help="Optional timezone-aware ISO timestamp overriding the competition open time.",
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/portfolio_robustness.csv",
    )
    parser.add_argument("--limit", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy or config.active_strategy
    strategy_by_symbol = _parse_strategy_map(args.strategy_map or ())
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    prices = load_price_history(price_csv)
    quotes = load_quote_history(quote_csv)
    symbols = tuple(args.symbol or sorted(set(prices.symbols()) & set(quotes.symbols())))
    if len(symbols) < 2:
        raise SystemExit("portfolio robustness requires at least two symbols")

    rows = evaluate_leave_one_symbol_out(
        config=config,
        prices=prices,
        quotes=quotes,
        symbols=symbols,
        strategy_name=strategy_name,
        strategy_by_symbol=strategy_by_symbol,
        clock_open_at=_parse_datetime(args.clock_open_at) if args.clock_open_at else None,
    )
    write_portfolio_robustness_csv(rows, args.output)

    baseline = rows[0]
    weakest = min(rows[1:], key=lambda row: row.return_pct)
    most_dependent = min(rows[1:], key=lambda row: row.return_delta_pct)

    print("Portfolio Robustness")
    print(f"  Strategy: {_strategy_display(strategy_name, strategy_by_symbol)}")
    print(f"  Symbols: {baseline.symbols.replace(';', ', ')}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print(
        "  Baseline: "
        f"return={baseline.return_pct:.3%}, "
        f"drawdown={baseline.max_drawdown_pct:.3%}, "
        f"sharpe15={baseline.sharpe_15m:.3f}, "
        f"trades={baseline.trade_count}, "
        f"risk={baseline.risk_discipline_score}/100, "
        f"pnl={money(baseline.total_pnl_usd)}"
    )
    print(
        "  Weakest exclusion: "
        f"drop {weakest.excluded_symbol}, "
        f"return={weakest.return_pct:.3%}, "
        f"delta={weakest.return_delta_pct:.3%}, "
        f"note={weakest.fragility_note}"
    )
    print(
        "  Most dependent symbol: "
        f"{most_dependent.excluded_symbol}, "
        f"return delta={most_dependent.return_delta_pct:.3%}"
    )
    print("Leave-one-symbol-out")
    for row in rows[1 : 1 + max(args.limit, 0)]:
        print(
            f"  - without {row.excluded_symbol}: "
            f"return={row.return_pct:.3%}, "
            f"delta={row.return_delta_pct:.3%}, "
            f"dd={row.max_drawdown_pct:.3%}, "
            f"trades={row.trade_count}, "
            f"risk={row.risk_discipline_score}/100, "
            f"{row.fragility_note}"
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


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SystemExit("--clock-open-at must be timezone-aware")
    return parsed


def _strategy_display(
    fallback_strategy: str,
    strategy_by_symbol: Mapping[str, str],
) -> str:
    if not strategy_by_symbol:
        return fallback_strategy
    overrides = ", ".join(
        f"{symbol}={strategy}" for symbol, strategy in sorted(strategy_by_symbol.items())
    )
    return f"{fallback_strategy} with overrides ({overrides})"
