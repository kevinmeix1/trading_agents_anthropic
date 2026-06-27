from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.asset_class_stability_optimizer import (
    AssetClassStabilitySpec,
    optimize_asset_class_stability,
    write_asset_class_stability_csv,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import CryptoOverlaySizingSpec
from quanthack.cli import asset_class_stability_optimize
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class AssetClassStabilityOptimizerTest(TestCase):
    def test_optimize_asset_class_stability_ranks_candidates(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            periods=40,
            interval_minutes=15,
            seed=151,
        )

        result = optimize_asset_class_stability(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            specs=(
                _spec("full", fx=1.0, metal=1.0),
                _spec("metal_half", fx=1.0, metal=0.5),
            ),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        self.assertEqual(result.official_symbols, ("EURUSD", "XAUUSD"))
        self.assertEqual(result.crypto_symbols, ("BTCUSD", "SOLUSD"))
        self.assertTrue(
            all(candidate.stability_status for candidate in result.candidates)
        )

    def test_write_asset_class_stability_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD"),
            periods=36,
            interval_minutes=15,
            seed=152,
        )
        result = optimize_asset_class_stability(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "XAUUSD", "BTCUSD"),
            specs=(_spec("metal_half", fx=1.0, metal=0.5),),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "asset.csv"
            write_asset_class_stability_csv(result, output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("rank,label,stability_status,stability_score", text)
        self.assertIn("metal_half", text)
        self.assertIn("fx_multiplier,metal_multiplier", text)

    def test_asset_class_stability_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD"),
            periods=36,
            interval_minutes=15,
            seed=153,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "asset.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                asset_class_stability_optimize.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--symbol",
                        "EURUSD",
                        "--symbol",
                        "XAUUSD",
                        "--symbol",
                        "BTCUSD",
                        "--candidate",
                        (
                            "label=demo,fx=1.0,metal=0.5,"
                            "crypto_spec=label=crypto,crypto=0.5,btc=0.5"
                        ),
                        "--train-size",
                        "12",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Asset-Class Stability Optimization", output)
        self.assertIn("demo", output)
        self.assertIn("stability_score", text)


def _spec(label: str, *, fx: float, metal: float) -> AssetClassStabilitySpec:
    return AssetClassStabilitySpec(
        label=label,
        fx_multiplier=fx,
        metal_multiplier=metal,
        crypto_spec=CryptoOverlaySizingSpec(
            label="crypto",
            crypto_multiplier=0.5,
            btc_multiplier=0.5,
            sol_multiplier=0.75,
            crypto_allowed_utc_hours=(7, 8, 9, 10),
        ),
    )
