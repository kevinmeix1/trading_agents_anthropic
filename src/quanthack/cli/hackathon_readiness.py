from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.hackathon_readiness import (
    ReadinessStatus,
    build_hackathon_readiness_report,
    write_hackathon_readiness_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a hackathon go/no-go readiness report."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--promotion-csv",
        default="outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_promotion.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/hackathon_readiness.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless all readiness checks pass.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    report = build_hackathon_readiness_report(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        promotion_csv=args.promotion_csv,
        summary_csv=args.summary_csv,
    )
    write_hackathon_readiness_markdown(report, args.output)

    for line in report.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")

    if args.strict and report.overall_status != ReadinessStatus.PASS:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
