from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quanthack.backtesting.deployment_profile_backtest import (
    run_deployment_profile_backtest,
    write_deployment_profile_backtest_summary_csv,
)
from quanthack.backtesting.portfolio_allocator import write_allocation_report_csv
from quanthack.backtesting.portfolio_backtest import (
    write_portfolio_equity_curve_csv,
    write_portfolio_fills_csv,
    write_portfolio_pnl_summary_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest one slot from deployment_profile_pack.json exactly."
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--profile-pack-json",
        default="outputs/research/deployment_profile_pack.json",
    )
    parser.add_argument(
        "--slot",
        default="conservative",
    )
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/research/deployment_profile_backtest",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    backtest = run_deployment_profile_backtest(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=args.profile_pack_json,
        slot=args.slot,
    )

    prefix = Path(args.output_prefix)
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    equity_path = prefix.with_name(f"{prefix.name}_equity.csv")
    pnl_path = prefix.with_name(f"{prefix.name}_pnl.csv")
    fills_path = prefix.with_name(f"{prefix.name}_fills.csv")
    allocation_path = prefix.with_name(f"{prefix.name}_allocation.csv")
    write_deployment_profile_backtest_summary_csv(backtest, summary_path)
    write_portfolio_equity_curve_csv(backtest.result, equity_path)
    write_portfolio_pnl_summary_csv(backtest.result, pnl_path)
    write_portfolio_fills_csv(backtest.result, fills_path)
    write_allocation_report_csv(backtest.result.allocation_reports, allocation_path)

    profile = backtest.profile
    metrics = backtest.competition_metrics
    print("Deployment Profile Backtest")
    print(f"  Slot: {profile.slot}")
    print(f"  Label: {profile.label}")
    print(f"  Evidence status: {profile.evidence_status}")
    print(f"  Global hours: {profile.allowed_hours_text}")
    print(f"  FX hours: {profile.forex_hours_text}")
    print(f"  Metal hours: {profile.metal_hours_text}")
    print(f"  Crypto hours: {profile.crypto_hours_text}")
    if profile.symbol_hours_text:
        print(f"  Symbol hours: {profile.symbol_hours_text}")
    print(f"  Symbols: {', '.join(backtest.result.symbols)}")
    print(f"  Return: {metrics.return_pct:.3%}")
    print(f"  Max drawdown: {metrics.max_drawdown_pct:.3%}")
    print(f"  Sharpe15: {metrics.sharpe_15m:.3f}")
    print(f"  Risk discipline: {backtest.risk_discipline.score:.0f}/100")
    print(f"  Fills: {len(backtest.result.fills)}")
    print(f"  Total P&L: {money(backtest.result.total_pnl_usd)}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Equity CSV: {equity_path}")
    print(f"  P&L CSV: {pnl_path}")
    print(f"  Fills CSV: {fills_path}")
    print(f"  Allocation CSV: {allocation_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
