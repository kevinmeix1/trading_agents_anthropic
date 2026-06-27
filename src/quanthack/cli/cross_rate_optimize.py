from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.cross_rate_optimizer import (
    DEFAULT_CROSS_RATE_PARAMETER_SETS,
    CrossRateParameterSet,
    optimize_cross_rate_parameters,
    write_cross_rate_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize FX cross-rate reversion parameters with fast diagnostics."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--horizon-bars", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-edge-after-cost-bps", type=float, default=0.0)
    parser.add_argument("--min-active-signals", type=int, default=10)
    parser.add_argument("--min-hit-rate", type=float, default=0.50)
    parser.add_argument("--min-average-signed-bps", type=float, default=0.0)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Candidate as label,lookback,entry_zscore,min_dev,slippage,cost_buffer "
            "or key=value pairs. Example: "
            "label=fast,lookback=8,entry=1,min_dev=0.5,slippage=0.5,cost=0.75"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/cross_rate_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_CROSS_RATE_PARAMETER_SETS
    )
    result = optimize_cross_rate_parameters(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        parameter_sets=parameter_sets,
        horizon_bars=args.horizon_bars,
        min_confidence=args.min_confidence,
        min_edge_after_cost_bps=args.min_edge_after_cost_bps,
        min_active_signals=args.min_active_signals,
        min_hit_rate=args.min_hit_rate,
        min_average_signed_return_bps=args.min_average_signed_bps,
    )
    write_cross_rate_optimization_csv(result, args.output)

    print("Cross-Rate Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Horizon bars: {result.horizon_bars}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Output CSV: {args.output}")
    print(
        "  Recommended symbols: "
        f"{', '.join(result.recommended_symbols) if result.recommended_symbols else 'none'}"
    )
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates[:20], start=1):
        row = candidate.row
        params = candidate.parameters
        print(
            f"  {rank}. {row.symbol} {params.label}: "
            f"eligible={candidate.eligible}, "
            f"active={row.active_count}, "
            f"hit={row.hit_rate:.1%}, "
            f"avg_signed={row.average_signed_forward_return_bps:.2f} bps, "
            f"edge_cost={row.average_edge_after_cost_bps:.2f} bps, "
            f"quality={candidate.quality_score:.2f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> CrossRateParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if any("=" in part for part in parts):
        return _parse_key_value_candidate(parts)
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "candidate must be label,lookback,entry_zscore,min_dev,slippage,cost_buffer"
        )
    label, lookback, entry_zscore, min_dev, slippage, cost_buffer = parts
    try:
        return CrossRateParameterSet(
            label=label,
            lookback=int(lookback),
            entry_zscore=float(entry_zscore),
            min_abs_deviation_bps=float(min_dev),
            slippage_bps=float(slippage),
            cost_buffer=float(cost_buffer),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_key_value_candidate(parts: list[str]) -> CrossRateParameterSet:
    values: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                "key=value candidates cannot mix positional values"
            )
        key, value = part.split("=", 1)
        values[key.strip().lower().replace("-", "_")] = value.strip()

    aliases = {
        "entry": "entry_zscore",
        "exit": "exit_zscore",
        "min_dev": "min_abs_deviation_bps",
        "max_dev": "max_abs_deviation_bps",
        "slippage": "slippage_bps",
        "fee": "fee_bps",
        "cost": "cost_buffer",
        "max_spread": "max_spread_bps",
    }
    normalized = {
        aliases.get(key, key): value
        for key, value in values.items()
    }
    required = {"label", "lookback", "entry_zscore", "min_abs_deviation_bps"}
    missing = sorted(required - set(normalized))
    if missing:
        raise argparse.ArgumentTypeError(
            f"candidate missing required keys: {', '.join(missing)}"
        )

    try:
        return CrossRateParameterSet(
            label=normalized["label"],
            lookback=int(normalized["lookback"]),
            entry_zscore=float(normalized["entry_zscore"]),
            exit_zscore=float(normalized.get("exit_zscore", 0.25)),
            min_abs_deviation_bps=float(normalized["min_abs_deviation_bps"]),
            max_abs_deviation_bps=float(
                normalized.get("max_abs_deviation_bps", 80.0)
            ),
            slippage_bps=float(normalized.get("slippage_bps", 1.0)),
            fee_bps=float(normalized.get("fee_bps", 0.0)),
            cost_buffer=float(normalized.get("cost_buffer", 1.0)),
            max_spread_bps=_parse_optional_float(
                normalized.get("max_spread_bps", "none")
            ),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_optional_float(raw: str) -> float | None:
    normalized = raw.strip().lower()
    if normalized in {"", "none", "null", "off"}:
        return None
    return float(raw)
