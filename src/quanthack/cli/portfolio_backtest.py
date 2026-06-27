from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime

from quanthack.backtesting.backtest import FillModel
from quanthack.cli._competition import print_competition_view
from quanthack.cli._format import money
from quanthack.backtesting.competition_score import (
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.portfolio_allocator import write_allocation_report_csv
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    write_portfolio_equity_curve_csv,
    write_portfolio_fills_csv,
    write_portfolio_pnl_summary_csv,
)
from quanthack.backtesting.portfolio_regime import (
    RegimeTiltPolicy,
    write_regime_tilt_report_csv,
)
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.backtesting.portfolio_symbol_evidence import (
    SymbolEvidenceGatePolicy,
    write_symbol_evidence_gate_report_csv,
)
from quanthack.backtesting.portfolio_volatility import (
    VolatilityTargetingPolicy,
    write_volatility_targeting_report_csv,
)
from quanthack.backtesting.warmup import evaluate_portfolio_after_warmup
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a shared-risk portfolio backtest.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default=None)
    parser.add_argument(
        "--strategy-map",
        action="append",
        default=None,
        metavar="SYMBOL=STRATEGY",
        help=(
            "Optional per-symbol strategy override. Repeat for hybrid portfolios; "
            "symbols not listed use --strategy or the configured active strategy."
        ),
    )
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--equity-output",
        default="outputs/backtests/portfolio_equity_curve.csv",
    )
    parser.add_argument(
        "--pnl-output",
        default="outputs/backtests/portfolio_pnl_summary.csv",
    )
    parser.add_argument(
        "--allocation-output",
        default="outputs/backtests/portfolio_allocation_report.csv",
    )
    parser.add_argument(
        "--fills-output",
        default="outputs/backtests/portfolio_fills.csv",
    )
    parser.add_argument(
        "--vol-target-output",
        default="outputs/backtests/portfolio_volatility_targeting.csv",
    )
    parser.add_argument(
        "--regime-tilt-output",
        default="outputs/backtests/portfolio_regime_tilt.csv",
    )
    parser.add_argument(
        "--symbol-evidence-output",
        default="outputs/backtests/portfolio_symbol_evidence_gate.csv",
    )
    parser.add_argument(
        "--regime-tilt",
        action="store_true",
        help="Enable Kalman-regime tilt before allocation.",
    )
    parser.add_argument("--regime-tilt-lookback", type=int, default=80)
    parser.add_argument("--regime-chop-trend-scale", type=float, default=0.70)
    parser.add_argument("--regime-chop-reversion-scale", type=float, default=1.15)
    parser.add_argument("--regime-trend-aligned-scale", type=float, default=1.10)
    parser.add_argument("--regime-trend-counter-scale", type=float, default=0.60)
    parser.add_argument("--regime-trend-reversion-scale", type=float, default=0.80)
    parser.add_argument("--regime-high-vol-scale", type=float, default=0.75)
    parser.add_argument(
        "--entry-utc-hours",
        default=None,
        help="Optional | separated UTC hours that may open or increase positions.",
    )
    parser.add_argument("--forex-entry-utc-hours", default=None)
    parser.add_argument("--metal-entry-utc-hours", default=None)
    parser.add_argument("--crypto-entry-utc-hours", default=None)
    parser.add_argument(
        "--vol-target-bar-volatility",
        type=float,
        default=None,
        help=(
            "Enable portfolio volatility targeting with this per-bar target "
            "volatility, for example 0.00075."
        ),
    )
    parser.add_argument("--vol-target-lookback", type=int, default=32)
    parser.add_argument("--vol-target-min-observations", type=int, default=12)
    parser.add_argument("--vol-target-min-scale", type=float, default=0.25)
    parser.add_argument("--vol-target-max-scale", type=float, default=1.15)
    parser.add_argument(
        "--symbol-evidence-gate",
        action="store_true",
        help="Enable online symbol evidence gate before allocation.",
    )
    parser.add_argument("--symbol-evidence-lookback-events", type=int, default=1)
    parser.add_argument("--symbol-evidence-min-events", type=int, default=1)
    parser.add_argument("--symbol-evidence-min-pnl-usd", type=float, default=0.0)
    parser.add_argument("--symbol-evidence-min-win-rate", type=float, default=0.0)
    parser.add_argument("--symbol-evidence-stale-after-bars", type=int, default=None)
    parser.add_argument("--symbol-evidence-block-without-history", action="store_true")
    parser.add_argument(
        "--symbol-evidence-target-symbol",
        action="append",
        default=None,
        help="Only apply the online symbol evidence gate to this symbol. Repeatable.",
    )
    parser.add_argument("--symbol-evidence-no-history-multiplier", type=float, default=0.0)
    parser.add_argument("--symbol-evidence-failed-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--metrics-start",
        default=None,
        help=(
            "Optional timezone-aware ISO timestamp for competition metrics after "
            "a warmup period."
        ),
    )
    parser.add_argument(
        "--clock-open-at",
        default=None,
        help=(
            "Optional timezone-aware ISO timestamp overriding the competition clock "
            "open time for historical research replays."
        ),
    )
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
    if not symbols:
        raise SystemExit("No symbols found in both price and quote data.")

    clock = config.competition.to_clock()
    if args.clock_open_at:
        clock = replace(clock, open_at=_parse_datetime(args.clock_open_at))

    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(
                strategy_by_symbol.get(symbol, strategy_name),
                symbol=symbol,
            )
            for symbol in symbols
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in symbols
        },
        clock=clock,
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
        regime_tilt_policy=_regime_policy(args),
        session_gate_policy=_session_policy(args),
        volatility_targeting_policy=_volatility_policy(args),
        symbol_evidence_gate_policy=_symbol_evidence_policy(args),
    )
    result = engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )
    write_portfolio_equity_curve_csv(result, args.equity_output)
    write_portfolio_pnl_summary_csv(result, args.pnl_output)
    write_allocation_report_csv(result.allocation_reports, args.allocation_output)
    write_portfolio_fills_csv(result, args.fills_output)
    if result.regime_reports:
        write_regime_tilt_report_csv(result.regime_reports, args.regime_tilt_output)
    if result.volatility_reports:
        write_volatility_targeting_report_csv(
            result.volatility_reports,
            args.vol_target_output,
        )
    if result.symbol_evidence_reports:
        write_symbol_evidence_gate_report_csv(
            result.symbol_evidence_reports,
            args.symbol_evidence_output,
        )
    if args.metrics_start:
        warmup_evaluation = evaluate_portfolio_after_warmup(
            result,
            evaluation_start=args.metrics_start,
        )
        competition_metrics = warmup_evaluation.competition_metrics
        risk_discipline = warmup_evaluation.risk_discipline
    else:
        warmup_evaluation = None
        competition_metrics = build_competition_metrics(
            equity_points=result.equity_curve,
            fills=result.fills,
        )
        risk_discipline = build_risk_discipline_report(
            risk_samples_from_portfolio_equity(result.equity_curve)
        )

    metrics = result.metrics
    display_strategy = _strategy_display(strategy_name, strategy_by_symbol)
    print("Portfolio Backtest")
    print(f"  Strategy: {display_strategy}")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Fills: {len(result.fills)}")
    if warmup_evaluation is not None:
        print(f"  Metrics start: {warmup_evaluation.evaluation_start}")
        print(f"  Evaluation fills: {len(warmup_evaluation.fills)}")
    print(f"  Observations: {metrics.observations}")
    print(f"  Final equity: {money(metrics.final_equity)}")
    print(f"  Total return: {metrics.total_return_pct:.3%}")
    print(f"  Sharpe ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max drawdown: {metrics.max_drawdown_pct:.3%}")
    print(f"  Turnover: {money(metrics.turnover_notional)}")
    print(f"  Realized P&L: {money(result.realized_pnl_usd)}")
    print(f"  Open P&L: {money(result.open_pnl_usd)}")
    print(f"  Total attributed P&L: {money(result.total_pnl_usd)}")
    print(f"  Equity curve: {args.equity_output}")
    print(f"  P&L summary: {args.pnl_output}")
    print(f"  Allocation report: {args.allocation_output}")
    print(f"  Fills: {args.fills_output}")
    if result.regime_reports:
        print(f"  Regime tilt report: {args.regime_tilt_output}")
    if result.volatility_reports:
        print(f"  Volatility targeting report: {args.vol_target_output}")
    if result.symbol_evidence_reports:
        print(f"  Symbol evidence report: {args.symbol_evidence_output}")
    print_competition_view(
        metrics=competition_metrics,
        risk_discipline=risk_discipline,
    )
    _print_allocation_summary(result.allocation_reports)
    _print_regime_tilt_summary(result.regime_reports)
    _print_volatility_targeting_summary(result.volatility_reports)
    _print_symbol_evidence_summary(result.symbol_evidence_reports)

    print("By symbol")
    for row in result.pnl_by_symbol:
        print(
            f"  {row.symbol}: "
            f"realized={money(row.ledger.realized_pnl_usd)}, "
            f"open={money(row.ledger.open_pnl_usd)}, "
            f"total={money(row.ledger.total_pnl_usd)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _print_allocation_summary(reports: tuple) -> None:
    if not reports:
        return
    trimmed_periods = len([report for report in reports if report.trimmed_targets])
    worst_net = max(report.net_directional_exposure for report in reports)
    worst_concentration = max(report.largest_symbol_concentration for report in reports)
    worst_leverage = max(report.leverage for report in reports)
    statuses = sorted({report.estimated_risk_status for report in reports})

    print("Allocation guardrails")
    print(f"  Periods trimmed: {trimmed_periods}/{len(reports)}")
    print(f"  Worst leverage: {worst_leverage:.2f}x")
    print(f"  Worst net directional exposure: {worst_net:.1%}")
    print(f"  Worst largest-symbol concentration: {worst_concentration:.1%}")
    print(f"  Estimated allocation statuses: {', '.join(statuses)}")


def _print_volatility_targeting_summary(reports: tuple) -> None:
    if not reports:
        return
    applied = [report for report in reports if report.applied]
    min_scale = min(report.scale for report in reports)
    max_scale = max(report.scale for report in reports)
    print("Portfolio volatility targeting")
    print(f"  Applied periods: {len(applied)}/{len(reports)}")
    print(f"  Scale range: {min_scale:.3f}x to {max_scale:.3f}x")


def _print_regime_tilt_summary(reports: tuple) -> None:
    if not reports:
        return
    applied = [report for report in reports if report.applied]
    regimes = sorted({report.regime for report in reports})
    if applied:
        min_scale = min(report.scale for report in applied)
        max_scale = max(report.scale for report in applied)
    else:
        min_scale = max_scale = 1.0
    print("Portfolio regime tilt")
    print(f"  Applied targets: {len(applied)}/{len(reports)}")
    print(f"  Regimes seen: {', '.join(regimes)}")
    print(f"  Applied scale range: {min_scale:.3f}x to {max_scale:.3f}x")


def _print_symbol_evidence_summary(reports: tuple) -> None:
    if not reports:
        return
    applied = [report for report in reports if report.applied]
    blocked_symbols = sorted({report.symbol for report in applied})
    print("Symbol evidence gate")
    print(f"  Blocked targets: {len(applied)}/{len(reports)}")
    if blocked_symbols:
        print(f"  Blocked symbols: {', '.join(blocked_symbols)}")


def _regime_policy(args: argparse.Namespace) -> RegimeTiltPolicy | None:
    if not args.regime_tilt:
        return None
    return RegimeTiltPolicy(
        lookback=args.regime_tilt_lookback,
        chop_trend_scale=args.regime_chop_trend_scale,
        chop_reversion_scale=args.regime_chop_reversion_scale,
        trend_aligned_scale=args.regime_trend_aligned_scale,
        trend_counter_scale=args.regime_trend_counter_scale,
        trend_reversion_scale=args.regime_trend_reversion_scale,
        high_volatility_scale=args.regime_high_vol_scale,
    )


def _volatility_policy(args: argparse.Namespace) -> VolatilityTargetingPolicy | None:
    if args.vol_target_bar_volatility is None:
        return None
    return VolatilityTargetingPolicy(
        lookback=args.vol_target_lookback,
        min_observations=args.vol_target_min_observations,
        target_bar_volatility=args.vol_target_bar_volatility,
        min_scale=args.vol_target_min_scale,
        max_scale=args.vol_target_max_scale,
    )


def _symbol_evidence_policy(args: argparse.Namespace) -> SymbolEvidenceGatePolicy | None:
    if not args.symbol_evidence_gate:
        return None
    return SymbolEvidenceGatePolicy(
        lookback_closed_events=args.symbol_evidence_lookback_events,
        min_closed_events=args.symbol_evidence_min_events,
        min_realized_pnl_usd=args.symbol_evidence_min_pnl_usd,
        min_win_rate=args.symbol_evidence_min_win_rate,
        allow_without_history=not args.symbol_evidence_block_without_history,
        stale_after_bars=args.symbol_evidence_stale_after_bars,
        target_symbols=tuple(args.symbol_evidence_target_symbol or ()),
        no_history_target_multiplier=args.symbol_evidence_no_history_multiplier,
        failed_evidence_target_multiplier=args.symbol_evidence_failed_multiplier,
    )


def _session_policy(args: argparse.Namespace) -> SessionGatePolicy | None:
    if not any(
        (
            args.entry_utc_hours,
            args.forex_entry_utc_hours,
            args.metal_entry_utc_hours,
            args.crypto_entry_utc_hours,
        )
    ):
        return None
    return SessionGatePolicy(
        allowed_utc_hours=_parse_hours(args.entry_utc_hours),
        forex_allowed_utc_hours=_parse_hours(args.forex_entry_utc_hours),
        metal_allowed_utc_hours=_parse_hours(args.metal_entry_utc_hours),
        crypto_allowed_utc_hours=_parse_hours(args.crypto_entry_utc_hours),
    )


def _parse_hours(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not value.strip():
        raise SystemExit("entry hour lists cannot be empty")
    try:
        return tuple(int(part) for part in value.replace(",", "|").split("|"))
    except ValueError as exc:
        raise SystemExit(f"entry hour lists must contain integers: {value!r}") from exc


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
        if not strategy:
            raise SystemExit("--strategy-map strategy name cannot be empty")
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
