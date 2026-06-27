from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.asset_class_stability_optimizer import (
    DEFAULT_ASSET_CLASS_STABILITY_SPECS,
    AssetClassStabilitySpec,
    optimize_asset_class_stability,
    write_asset_class_stability_csv,
)
from quanthack.cli.crypto_overlay_sizing_compare import _parse_candidate
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search FX/metal exposure multipliers around crypto overlay profiles "
            "to reduce fold concentration."
        )
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
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Optional candidate as label=NAME,fx=1.0,metal=0.75,"
            "crypto_spec=label=CRYPTO,crypto=0.75,btc=0.75,sol=1.0"
        ),
    )
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--output",
        default="outputs/research/asset_class_stability_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = optimize_asset_class_stability(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=args.base_strategy,
        symbols=tuple(args.symbol) if args.symbol else None,
        specs=tuple(_parse_asset_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_ASSET_CLASS_STABILITY_SPECS,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_asset_class_stability_csv(result, args.output)

    print("Asset-Class Stability Optimization")
    print(f"  Official symbols: {', '.join(result.official_symbols) or 'none'}")
    print(f"  Crypto symbols: {', '.join(result.crypto_symbols) or 'none'}")
    print(f"  Candidates: {len(result.candidates)}")
    print(f"  Output CSV: {args.output}")
    for rank, candidate in enumerate(result.candidates[: max(args.limit, 0)], start=1):
        metrics = candidate.competition_metrics
        walk_forward = candidate.walk_forward
        print(
            f"  {rank}. {candidate.spec.label}: "
            f"status={candidate.stability_status}, "
            f"stable_score={candidate.stability_score:.1f}, "
            f"retention={candidate.return_retention:.1%}, "
            f"fx={candidate.spec.fx_multiplier:.2f}, "
            f"metal={candidate.spec.metal_multiplier:.2f}, "
            f"crypto={candidate.spec.crypto_spec.label}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"trades={metrics.trade_count}, "
            f"wf_pos={walk_forward.positive_fold_fraction:.1%}, "
            f"wf_nonneg={walk_forward.non_negative_fold_fraction:.1%}, "
            f"wf_conc={walk_forward.largest_positive_fold_contribution:.1%}"
        )
        print(f"      multipliers: {candidate.multiplier_map_text}")
        print(f"      crypto hours: {candidate.crypto_hours_text}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_asset_candidate(raw: str) -> AssetClassStabilitySpec:
    fields = _split_top_level_fields(raw)
    label = fields.get("label", "")
    if not label:
        raise argparse.ArgumentTypeError("candidate label is required")
    crypto_spec_raw = fields.get("crypto_spec", "")
    if not crypto_spec_raw:
        raise argparse.ArgumentTypeError("candidate crypto_spec is required")
    try:
        return AssetClassStabilitySpec(
            label=label,
            fx_multiplier=float(fields.get("fx", "1.0")),
            metal_multiplier=float(fields.get("metal", "1.0")),
            crypto_spec=_parse_candidate(crypto_spec_raw),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _split_top_level_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    crypto_marker = ",crypto_spec="
    prefix, separator, crypto_spec = raw.partition(crypto_marker)
    for part in prefix.split(","):
        if not part:
            continue
        key, field_separator, value = part.partition("=")
        if not field_separator:
            raise argparse.ArgumentTypeError(
                "candidate must use label=NAME,fx=1.0,metal=0.75,crypto_spec=..."
            )
        fields[key.strip().lower()] = value.strip()
    if separator:
        fields["crypto_spec"] = crypto_spec.strip()
    return fields
