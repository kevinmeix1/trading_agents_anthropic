from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
    write_fixed_warmup_folds_csv,
    write_fixed_warmup_summary_csv,
)
from quanthack.backtesting.portfolio_regime import RegimeTiltPolicy
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.backtesting.portfolio_symbol_evidence import SymbolEvidenceGatePolicy
from quanthack.backtesting.portfolio_volatility import VolatilityTargetingPolicy
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-symbol portfolio walk-forward with warmup history."
    )
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
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument("--regime-tilt", action="store_true")
    parser.add_argument("--regime-tilt-lookback", type=int, default=80)
    parser.add_argument("--regime-chop-trend-scale", type=float, default=0.70)
    parser.add_argument("--regime-chop-reversion-scale", type=float, default=1.15)
    parser.add_argument("--regime-trend-aligned-scale", type=float, default=1.10)
    parser.add_argument("--regime-trend-counter-scale", type=float, default=0.60)
    parser.add_argument("--regime-trend-reversion-scale", type=float, default=0.80)
    parser.add_argument("--regime-high-vol-scale", type=float, default=0.75)
    parser.add_argument("--entry-utc-hours", default=None)
    parser.add_argument("--forex-entry-utc-hours", default=None)
    parser.add_argument("--metal-entry-utc-hours", default=None)
    parser.add_argument("--crypto-entry-utc-hours", default=None)
    parser.add_argument("--vol-target-bar-volatility", type=float, default=None)
    parser.add_argument("--vol-target-lookback", type=int, default=32)
    parser.add_argument("--vol-target-min-observations", type=int, default=12)
    parser.add_argument("--vol-target-min-scale", type=float, default=0.25)
    parser.add_argument("--vol-target-max-scale", type=float, default=1.15)
    parser.add_argument("--symbol-evidence-gate", action="store_true")
    parser.add_argument("--symbol-evidence-lookback-events", type=int, default=1)
    parser.add_argument("--symbol-evidence-min-events", type=int, default=1)
    parser.add_argument("--symbol-evidence-min-pnl-usd", type=float, default=0.0)
    parser.add_argument("--symbol-evidence-min-win-rate", type=float, default=0.0)
    parser.add_argument("--symbol-evidence-stale-after-bars", type=int, default=None)
    parser.add_argument("--symbol-evidence-block-without-history", action="store_true")
    parser.add_argument("--symbol-evidence-target-symbol", action="append", default=None)
    parser.add_argument("--symbol-evidence-no-history-multiplier", type=float, default=0.0)
    parser.add_argument("--symbol-evidence-failed-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--summary-output",
        default="outputs/backtests/fixed_warmup_walk_forward_summary.csv",
    )
    parser.add_argument(
        "--folds-output",
        default="outputs/backtests/fixed_warmup_walk_forward_folds.csv",
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

    result = run_fixed_warmup_portfolio_walk_forward(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_name=strategy_name,
        symbols=symbols,
        strategy_by_symbol=strategy_by_symbol,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        regime_tilt_policy=_regime_policy(args),
        session_gate_policy=_session_policy(args),
        volatility_targeting_policy=_volatility_policy(args),
        symbol_evidence_gate_policy=_symbol_evidence_policy(args),
    )
    write_fixed_warmup_summary_csv(result, args.summary_output)
    write_fixed_warmup_folds_csv(result, args.folds_output)
    promotion = decide_fixed_warmup_promotion(result)

    print("Fixed Warmup Portfolio Walk-Forward")
    print(f"  Strategy: {result.strategy_name}")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Folds: {len(result.folds)}")
    print(f"  Positive fold fraction: {result.positive_fold_fraction:.1%}")
    print(f"  Active fold fraction: {result.active_fold_fraction:.1%}")
    print(
        "  Active positive fold fraction: "
        f"{result.active_positive_fold_fraction:.1%}"
    )
    print(f"  Non-negative fold fraction: {result.non_negative_fold_fraction:.1%}")
    print(f"  Median test return: {result.median_test_return_pct:.3%}")
    print(f"  Median active test return: {result.median_active_test_return_pct:.3%}")
    print(f"  Median test Sharpe 15m: {result.median_test_sharpe_15m:.3f}")
    print(f"  Worst test drawdown: {result.worst_test_drawdown_pct:.3%}")
    print(f"  Average risk discipline: {result.average_risk_discipline_score:.1f}/100")
    print(f"  Evaluation fills: {result.total_evaluation_fills}")
    if args.regime_tilt:
        print("  Regime tilt: enabled")
    if any(
        (
            args.entry_utc_hours,
            args.forex_entry_utc_hours,
            args.metal_entry_utc_hours,
            args.crypto_entry_utc_hours,
        )
    ):
        print("  Session gate: enabled")
    if args.vol_target_bar_volatility is not None:
        print(f"  Volatility target: {args.vol_target_bar_volatility:g} per bar")
    if args.symbol_evidence_gate:
        print("  Symbol evidence gate: enabled")
    print(
        "  Largest positive fold contribution: "
        f"{result.largest_positive_fold_contribution:.1%}"
    )
    print(f"  Promotion: {promotion.status} ({promotion.reason})")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Folds CSV: {args.folds_output}")
    print("Folds")
    for fold in result.folds:
        metrics = fold.metrics
        print(
            f"  {fold.fold_index}. {fold.test_start} -> {fold.test_end}: "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"fills={len(fold.evaluation.fills)}, "
            f"risk={fold.risk_discipline.score}/100"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


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
