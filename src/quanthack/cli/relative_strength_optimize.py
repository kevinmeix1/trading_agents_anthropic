from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.relative_strength_optimizer import (
    DEFAULT_RELATIVE_STRENGTH_PARAMETER_SETS,
    RelativeStrengthParameterSet,
    optimize_relative_strength_parameters,
    write_relative_strength_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize relative-strength parameters with portfolio backtests."
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
            "Candidate as label,lookback,entry_zscore,exit_zscore or key=value "
            "pairs. Example: base,12,0.75,0.25 or "
            "label=base,lookback=12,entry=0.75,exit=0.25,dispersion=2"
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run portfolio walk-forward for every candidate.",
    )
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--max-baskets", type=int, default=10)
    parser.add_argument(
        "--output",
        default="outputs/backtests/relative_strength_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_RELATIVE_STRENGTH_PARAMETER_SETS
    )
    result = optimize_relative_strength_parameters(
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
    write_relative_strength_optimization_csv(result, args.output)

    print("Relative Strength Optimization")
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
                f"wf_eligible={summary.eligible}"
            )
        )
        print(
            f"  {rank}. {params.label}: "
            f"lookback={params.lookback}, "
            f"entry={params.entry_zscore:.2f}, "
            f"exit={params.exit_zscore:.2f}, "
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


def _parse_candidate(raw: str) -> RelativeStrengthParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if any("=" in part for part in parts):
        return _parse_key_value_candidate(parts)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "candidate must be label,lookback,entry_zscore,exit_zscore"
        )
    label, lookback, entry_zscore, exit_zscore = parts
    try:
        return RelativeStrengthParameterSet(
            label=label,
            lookback=int(lookback),
            entry_zscore=float(entry_zscore),
            exit_zscore=float(exit_zscore),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_key_value_candidate(parts: list[str]) -> RelativeStrengthParameterSet:
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
        "min_move": "min_abs_move_bps",
        "dispersion": "min_score_dispersion",
        "trend_efficiency": "min_target_trend_efficiency",
        "asset_confirm": "require_asset_class_confirmation",
        "asset_z": "asset_class_entry_zscore",
        "metal_confirm": "require_metal_trend_confirmation",
        "metal_move": "metal_trend_min_move_bps",
        "metal_efficiency": "metal_trend_min_efficiency",
    }
    normalized = {
        aliases.get(key, key): value
        for key, value in values.items()
    }
    required = {"label", "lookback", "entry_zscore", "exit_zscore"}
    missing = sorted(required - set(normalized))
    if missing:
        raise argparse.ArgumentTypeError(
            f"candidate missing required keys: {', '.join(missing)}"
        )

    try:
        return RelativeStrengthParameterSet(
            label=normalized["label"],
            lookback=int(normalized["lookback"]),
            entry_zscore=float(normalized["entry_zscore"]),
            exit_zscore=float(normalized["exit_zscore"]),
            min_abs_move_bps=float(normalized.get("min_abs_move_bps", 0.5)),
            require_asset_class_confirmation=_parse_bool(
                normalized.get("require_asset_class_confirmation", "false")
            ),
            asset_class_entry_zscore=float(
                normalized.get("asset_class_entry_zscore", 0.35)
            ),
            require_metal_trend_confirmation=_parse_bool(
                normalized.get("require_metal_trend_confirmation", "false")
            ),
            metal_trend_min_move_bps=float(
                normalized.get("metal_trend_min_move_bps", 2.0)
            ),
            metal_trend_min_efficiency=float(
                normalized.get("metal_trend_min_efficiency", 0.20)
            ),
            min_score_dispersion=float(normalized.get("min_score_dispersion", 0.0)),
            min_target_trend_efficiency=float(
                normalized.get("min_target_trend_efficiency", 0.0)
            ),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {raw!r}")
