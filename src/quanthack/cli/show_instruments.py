from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.core.instruments import AssetClass, DEFAULT_INSTRUMENTS, instruments_by_asset_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show Syphonix competition instruments.")
    parser.add_argument(
        "--asset-class",
        choices=[asset_class.value.lower() for asset_class in AssetClass],
        default=None,
    )
    return parser


def run(args: argparse.Namespace) -> None:
    instruments = DEFAULT_INSTRUMENTS
    if args.asset_class is not None:
        instruments = instruments_by_asset_class(AssetClass(args.asset_class.upper()))

    print("Competition Instruments")
    print(f"  Count: {len(instruments)}")
    print("  symbol  | class  | min notional | max spread | slippage")
    for instrument in instruments:
        print(
            f"  {instrument.symbol:<7} | "
            f"{instrument.asset_class.value:<6} | "
            f"${instrument.min_trade_notional_usd:>11,.0f} | "
            f"{instrument.max_spread_bps:>9.1f} | "
            f"{instrument.typical_slippage_bps:>7.1f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))

