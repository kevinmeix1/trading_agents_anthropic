from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_overlay_component_ablation import (
    CryptoOverlayComponentAblationSpec,
    compare_crypto_overlay_components,
    write_crypto_overlay_component_ablation_csv,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import CryptoOverlaySizingSpec
from quanthack.cli import crypto_overlay_component_ablation
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoOverlayComponentAblationTest(TestCase):
    def test_compare_crypto_overlay_components_runs_ablation_specs(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            periods=36,
            interval_minutes=15,
            seed=121,
        )

        result = compare_crypto_overlay_components(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            base_spec=CryptoOverlaySizingSpec(
                label="base",
                crypto_multiplier=0.75,
                btc_multiplier=0.75,
                sol_multiplier=1.0,
            ),
            specs=(
                CryptoOverlayComponentAblationSpec(label="full"),
                CryptoOverlayComponentAblationSpec(
                    label="no_crypto",
                    disabled_asset_classes=("CRYPTO",),
                ),
                CryptoOverlayComponentAblationSpec(
                    label="no_btc",
                    disabled_symbols=("BTCUSD",),
                ),
            ),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(result.official_symbols, ("EURUSD", "XAUUSD"))
        self.assertEqual(result.crypto_symbols, ("BTCUSD", "SOLUSD"))
        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.rows[0].label, "full")
        no_crypto = next(row for row in result.rows if row.label == "no_crypto")
        self.assertIn("CRYPTO", no_crypto.disabled_asset_classes)
        self.assertNotIn("BTCUSD", no_crypto.active_symbols)
        self.assertIsNotNone(no_crypto.walk_forward)
        self.assertIsNotNone(no_crypto.promotion)

    def test_write_crypto_overlay_component_ablation_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=24,
            interval_minutes=15,
            seed=122,
        )
        result = compare_crypto_overlay_components(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "BTCUSD"),
            specs=(
                CryptoOverlayComponentAblationSpec(label="full"),
                CryptoOverlayComponentAblationSpec(
                    label="no_btc",
                    disabled_symbols=("BTCUSD",),
                ),
            ),
            run_walk_forward=False,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "components.csv"
            write_crypto_overlay_component_ablation_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,disabled_symbols", text)
        self.assertIn("return_delta_pct", text)
        self.assertIn("no_btc", text)

    def test_crypto_overlay_component_ablation_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            periods=32,
            interval_minutes=15,
            seed=123,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "components.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_overlay_component_ablation.main(
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
                        "--symbol",
                        "SOLUSD",
                        "--component",
                        "label=full",
                        "--component",
                        "label=no_crypto,assets=CRYPTO",
                        "--no-walk-forward",
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Crypto Overlay Component Ablation", output)
        self.assertIn("no_crypto", output)
        self.assertIn("CRYPTO", text)
