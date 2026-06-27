from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.volatility_squeeze_optimizer import (
    DEFAULT_VOLATILITY_SQUEEZE_PARAMETER_SETS,
    VolatilitySqueezeParameterSet,
    optimize_volatility_squeeze_parameters,
    write_volatility_squeeze_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize volatility-squeeze parameters with portfolio backtests."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Candidate as label,lookback,squeeze_window,max_squeeze_ratio,"
            "breakout_buffer_bps,band_multiplier or key=value pairs. Example: "
            "base,24,8,0.70,2.0,2.0 or "
            "label=liquid,lookback=24,window=8,ratio=0.50,buffer=2.5,"
            "fx_hours=11|12|13|14|15|16|17|18|19"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/backtests/volatility_squeeze_optimization.csv",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run portfolio walk-forward for every candidate.",
    )
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--test-size", type=int, default=240)
    parser.add_argument("--step-size", type=int, default=240)
    parser.add_argument("--max-baskets", type=int, default=30)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_VOLATILITY_SQUEEZE_PARAMETER_SETS
    )
    result = optimize_volatility_squeeze_parameters(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        parameter_sets=parameter_sets,
        include_walk_forward=args.walk_forward,
        walk_forward_train_size=args.train_size,
        walk_forward_test_size=args.test_size,
        walk_forward_step_size=args.step_size,
        walk_forward_max_baskets=args.max_baskets,
    )
    write_volatility_squeeze_optimization_csv(result, args.output)

    print("Volatility Squeeze Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Walk-forward: {'yes' if args.walk_forward else 'no'}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates, start=1):
        params = candidate.parameters
        metrics = candidate.comparison_row.competition_metrics
        summary = candidate.walk_forward_summary
        walk_text = (
            ""
            if summary is None
            else (
                f", wf_stable={summary.stable_fold_fraction:.1%}, "
                f"wf_median_return={summary.median_test_return_pct:.3%}, "
                f"wf_fills={summary.total_test_fills}, "
                f"wf_eligible={summary.eligible}"
            )
        )
        print(
            f"  {rank}. {params.label}: "
            f"lookback={params.lookback}, "
            f"window={params.squeeze_window}, "
            f"ratio={params.max_squeeze_ratio:.2f}, "
            f"buffer={params.breakout_buffer_bps:.1f}, "
            f"band={params.band_stdev_multiplier:.1f}, "
            f"proxy={candidate.comparison_row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
            f"{walk_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> VolatilitySqueezeParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if any("=" in part for part in parts):
        return _parse_key_value_candidate(parts)
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "candidate must be label,lookback,squeeze_window,"
            "max_squeeze_ratio,breakout_buffer_bps,band_multiplier"
        )
    label, lookback, squeeze_window, max_ratio, breakout_buffer, band_multiplier = parts
    try:
        return VolatilitySqueezeParameterSet(
            label=label,
            lookback=int(lookback),
            squeeze_window=int(squeeze_window),
            max_squeeze_ratio=float(max_ratio),
            breakout_buffer_bps=float(breakout_buffer),
            band_stdev_multiplier=float(band_multiplier),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_key_value_candidate(parts: list[str]) -> VolatilitySqueezeParameterSet:
    values: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                "key=value candidates cannot mix positional values"
            )
        key, value = part.split("=", 1)
        values[key.strip().lower().replace("-", "_")] = value.strip()

    aliases = {
        "window": "squeeze_window",
        "ratio": "max_squeeze_ratio",
        "buffer": "breakout_buffer_bps",
        "band": "band_stdev_multiplier",
        "prior_vol": "min_prior_volatility_bps",
        "min_band": "min_band_width_bps",
        "max_hold": "max_holding_period",
        "fx_hours": "forex_allowed_utc_hours",
        "forex_hours": "forex_allowed_utc_hours",
        "metal_hours": "metal_allowed_utc_hours",
        "crypto_hours": "crypto_allowed_utc_hours",
    }
    normalized = {
        aliases.get(key, key): value
        for key, value in values.items()
    }
    required = {
        "label",
        "lookback",
        "squeeze_window",
        "max_squeeze_ratio",
        "breakout_buffer_bps",
    }
    missing = sorted(required - set(normalized))
    if missing:
        raise argparse.ArgumentTypeError(
            f"candidate missing required keys: {', '.join(missing)}"
        )

    try:
        return VolatilitySqueezeParameterSet(
            label=normalized["label"],
            lookback=int(normalized["lookback"]),
            squeeze_window=int(normalized["squeeze_window"]),
            max_squeeze_ratio=float(normalized["max_squeeze_ratio"]),
            breakout_buffer_bps=float(normalized["breakout_buffer_bps"]),
            band_stdev_multiplier=float(
                normalized.get("band_stdev_multiplier", 2.0)
            ),
            min_prior_volatility_bps=float(
                normalized.get("min_prior_volatility_bps", 0.5)
            ),
            min_band_width_bps=float(normalized.get("min_band_width_bps", 1.0)),
            max_holding_period=int(normalized.get("max_holding_period", 24)),
            forex_allowed_utc_hours=_parse_hours(
                normalized.get("forex_allowed_utc_hours")
            ),
            metal_allowed_utc_hours=_parse_hours(
                normalized.get("metal_allowed_utc_hours")
            ),
            crypto_allowed_utc_hours=_parse_hours(
                normalized.get("crypto_allowed_utc_hours")
            ),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_hours(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    parts = [
        part.strip()
        for chunk in raw.replace(";", "|").split("|")
        for part in chunk.split()
        if part.strip()
    ]
    return tuple(int(part) for part in parts)
