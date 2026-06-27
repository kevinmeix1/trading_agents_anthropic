from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.crypto_overlay_sizing_compare import (
    CryptoOverlaySizingSpec,
    compare_crypto_overlay_sizing,
    write_crypto_overlay_sizing_comparison_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare crypto overlay size multipliers inside the full portfolio."
    )
    parser.add_argument("--config", default="configs/competition.toml")
    parser.add_argument(
        "--price-csv",
        default="data/mixed_official_crypto_proxy_overlap_prices.csv",
    )
    parser.add_argument(
        "--quote-csv",
        default="data/mixed_official_crypto_proxy_overlap_quotes.csv",
    )
    parser.add_argument("--base-strategy", choices=STRATEGY_NAMES, default="macd_momentum")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/research/crypto_overlay_sizing_comparison.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    comparison = compare_crypto_overlay_sizing(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=args.base_strategy,
        symbols=tuple(args.symbol) if args.symbol else None,
        specs=tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else None,
        run_walk_forward=not args.no_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_crypto_overlay_sizing_comparison_csv(comparison, args.output)

    print("Crypto Overlay Sizing Comparison")
    print(f"  Official symbols: {', '.join(comparison.official_symbols) or 'none'}")
    print(f"  Crypto symbols: {', '.join(comparison.crypto_symbols) or 'none'}")
    print(f"  Base strategy: {comparison.base_strategy}")
    print(f"  Price CSV: {args.price_csv}")
    print(f"  Quote CSV: {args.quote_csv}")
    print(f"  Walk-forward: {'disabled' if args.no_walk_forward else 'enabled'}")
    print(f"  Output CSV: {args.output}")
    print("Ranked sizing candidates")
    for rank, candidate in enumerate(comparison.candidates, start=1):
        metrics = candidate.competition_metrics
        wf_text = "wf=n/a"
        if candidate.walk_forward is not None:
            promotion = candidate.promotion.status if candidate.promotion else "UNKNOWN"
            wf_text = (
                f"wf_nonneg={candidate.walk_forward.non_negative_fold_fraction:.1%}, "
                f"wf_active_pos={candidate.walk_forward.active_positive_fold_fraction:.1%}, "
                f"wf_contrib={candidate.walk_forward.largest_positive_fold_contribution:.1%}, "
                f"promotion={promotion}"
            )
        print(
            f"  {rank}. {candidate.label}: "
            f"selection={candidate.selection_score:.1f}, "
            f"proxy={candidate.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={candidate.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}, "
            f"{wf_text}"
        )
        print(f"      multipliers: {candidate.multiplier_map_text}")
        print(f"      crypto hours: {candidate.crypto_allowed_utc_hours_text}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> CryptoOverlaySizingSpec:
    fields: dict[str, str] = {}
    for part in raw.split(","):
        key, separator, value = part.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "candidate must use label=NAME,crypto=0.75[,btc=0.5,sol=1.0,trend=0.75,reversion=0.5,crypto_hours=8|9|10]"
            )
        fields[key.strip().lower()] = value.strip()
    label = fields.get("label", "")
    if not label:
        raise argparse.ArgumentTypeError("candidate label is required")
    crypto_multiplier = _optional_float(fields, "crypto")
    return CryptoOverlaySizingSpec(
        label=label,
        crypto_multiplier=1.0 if crypto_multiplier is None else crypto_multiplier,
        btc_multiplier=_optional_float(fields, "btc"),
        sol_multiplier=_optional_float(fields, "sol"),
        trend_crypto_multiplier=_optional_float(fields, "trend"),
        reversion_crypto_multiplier=_optional_float(fields, "reversion"),
        crypto_allowed_utc_hours=_optional_hours(fields, "crypto_hours"),
    )


def _optional_float(fields: dict[str, str], key: str) -> float | None:
    raw = fields.get(key)
    if raw in {None, ""}:
        return None
    return float(raw)


def _optional_hours(fields: dict[str, str], key: str) -> tuple[int, ...] | None:
    raw = fields.get(key)
    if raw in {None, ""}:
        return None
    hours = tuple(int(part.strip()) for part in raw.split("|") if part.strip())
    if not hours:
        raise argparse.ArgumentTypeError(f"{key} cannot be empty")
    if any(hour < 0 or hour > 23 for hour in hours):
        raise argparse.ArgumentTypeError(f"{key} must contain hours between 0 and 23")
    return hours
