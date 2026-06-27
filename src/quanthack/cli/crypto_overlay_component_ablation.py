from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.crypto_overlay_component_ablation import (
    CryptoOverlayComponentAblationSpec,
    compare_crypto_overlay_components,
    write_crypto_overlay_component_ablation_csv,
)
from quanthack.cli.crypto_overlay_sizing_compare import _parse_candidate
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablate components of the sized crypto overlay candidate."
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
    parser.add_argument(
        "--base-candidate",
        default=(
            "label=btc075_sol100_reversion075_london,crypto=0.75,"
            "btc=0.75,sol=1.0,crypto_hours=7|8|9|10|11|12|13|14|15|16"
        ),
    )
    parser.add_argument("--component", action="append", default=None)
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=48)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/research/crypto_overlay_component_ablation.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prices = load_price_history(args.price_csv)
    quotes = load_quote_history(args.quote_csv)
    result = compare_crypto_overlay_components(
        config=config,
        prices=prices,
        quotes=quotes,
        base_strategy=args.base_strategy,
        base_spec=_parse_candidate(args.base_candidate),
        specs=tuple(_parse_component(value) for value in args.component)
        if args.component
        else None,
        symbols=tuple(args.symbol) if args.symbol else None,
        run_walk_forward=not args.no_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_crypto_overlay_component_ablation_csv(result, args.output)

    print("Crypto Overlay Component Ablation")
    print(f"  Official symbols: {', '.join(result.official_symbols) or 'none'}")
    print(f"  Crypto symbols: {', '.join(result.crypto_symbols) or 'none'}")
    print(f"  Base strategy: {result.base_strategy}")
    print(f"  Base candidate: {result.base_spec.label}")
    print(f"  Output CSV: {args.output}")
    baseline_return = result.baseline.competition_metrics.return_pct if result.baseline else 0.0
    for rank, row in enumerate(result.rows, start=1):
        metrics = row.competition_metrics
        promotion = row.promotion.status if row.promotion else "n/a"
        print(
            f"  {rank}. {row.label}: "
            f"return={metrics.return_pct:.3%} "
            f"delta={row.return_delta_pct:.3%} "
            f"retention={_retention(metrics.return_pct, baseline_return):.1%} "
            f"drawdown={metrics.max_drawdown_pct:.3%} "
            f"sharpe15={metrics.sharpe_15m:.3f} "
            f"trades={metrics.trade_count} "
            f"promotion={promotion}"
        )
        if row.disabled_symbols or row.disabled_asset_classes:
            print(
                f"      disabled: symbols={row.disabled_symbols_text or 'none'} "
                f"assets={row.disabled_asset_classes_text or 'none'}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_component(raw: str) -> CryptoOverlayComponentAblationSpec:
    fields: dict[str, str] = {}
    for part in raw.split(","):
        key, separator, value = part.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "component must use label=NAME[,symbols=BTCUSD|SOLUSD,assets=CRYPTO|METAL]"
            )
        fields[key.strip().lower()] = value.strip()
    label = fields.get("label", "")
    if not label:
        raise argparse.ArgumentTypeError("component label is required")
    return CryptoOverlayComponentAblationSpec(
        label=label,
        disabled_symbols=_split_pipe(fields.get("symbols", "")),
        disabled_asset_classes=_split_pipe(fields.get("assets", "")),
    )


def _split_pipe(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split("|") if part.strip())


def _retention(return_pct: float, baseline_return_pct: float) -> float:
    if abs(baseline_return_pct) < 1e-12:
        return 0.0
    return return_pct / baseline_return_pct
