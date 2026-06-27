from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.cli._time import parse_datetime
from quanthack.core.clock import CompetitionMode
from quanthack.core.config import load_config
from quanthack.trading.execution import DryRunExecutor
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.market.market_quality import MarketQualityChecker
from quanthack.trading.risk import AccountSnapshot, RiskEngine
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CSV quotes/prices through quality, strategy, risk, and dry-run journaling."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--as-of", type=parse_datetime, default=None)
    parser.add_argument("--equity", type=float, default=None)
    parser.add_argument("--day-start-equity", type=float, default=None)
    parser.add_argument("--peak-equity", type=float, default=None)
    parser.add_argument("--margin-level-pct", type=float, default=2_000)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CompetitionMode],
        default=CompetitionMode.QUALIFY.value,
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    strategy_name = args.strategy or config.active_strategy
    symbol = args.symbol or config.strategy_symbol(strategy_name)
    price_csv = args.price_csv or config.market_data.price_csv
    quote_csv = args.quote_csv or config.market_data.quote_csv

    quote = load_quote_history(quote_csv).latest_quote(symbol)
    if quote is None:
        print(f"Symbol: {symbol}")
        print("Market quality: BLOCKED")
        print("Reason: no quote for symbol")
        print("No strategy, risk decision, or journal record was created.")
        return

    as_of = args.as_of or quote.timestamp
    quality = MarketQualityChecker(config.market_quality).evaluate(quote=quote, as_of=as_of)

    print(f"Symbol: {symbol}")
    print(f"Quote spread: {quality.spread_bps:.2f} bps")
    print(f"Quote age: {quality.quote_age_seconds:.1f}s")
    print(f"Market quality: {'OK' if quality.ok else 'BLOCKED'}")
    print(f"Quality reason: {quality.reason}")

    if not quality.ok:
        print("No strategy, risk decision, or journal record was created.")
        return

    prices = load_price_history(price_csv)
    closes = prices.close_prices(symbol=symbol)
    request = config.build_strategy(strategy_name, symbol=symbol).generate_request(closes)

    print(f"Strategy: {strategy_name}")
    print(f"Closes loaded: {len(closes)}")

    if request is None:
        print("Strategy output: NO TRADE")
        print("No risk decision or journal record was created.")
        return

    starting_equity = config.competition.starting_equity
    equity = args.equity if args.equity is not None else starting_equity
    day_start_equity = (
        args.day_start_equity if args.day_start_equity is not None else starting_equity
    )
    peak_equity = args.peak_equity if args.peak_equity is not None else starting_equity
    mode = CompetitionMode(args.mode)

    account = AccountSnapshot(
        equity=equity,
        starting_equity=starting_equity,
        day_start_equity=day_start_equity,
        peak_equity=peak_equity,
        margin_level_pct=args.margin_level_pct,
    )
    executor = DryRunExecutor(Path(config.execution.journal_path))
    portfolio = executor.current_portfolio()
    decision = RiskEngine(config.risk).evaluate(
        account=account,
        portfolio=portfolio,
        request=request,
        mode=mode,
    )
    record = executor.submit(
        account=account,
        request=request,
        decision=decision,
        mode=mode,
        portfolio_before=portfolio,
    )

    print(f"Strategy output: {request.side.value} {request.symbol}")
    print(f"Portfolio before: gross=${portfolio.gross_notional_usd:,.0f}")
    print(f"Risk decision: {'APPROVED' if decision.approved else 'BLOCKED'}")
    print(f"Reason: {decision.reason}")
    print(f"Adjusted notional: ${decision.adjusted_notional_usd:,.0f}")
    print(f"Dry-run status: {record.status}")
    print(f"Journal: {config.execution.journal_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
