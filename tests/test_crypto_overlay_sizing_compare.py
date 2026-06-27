from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_overlay_sizing_compare import (
    CryptoOverlaySizingSpec,
    compare_crypto_overlay_sizing,
    write_crypto_overlay_sizing_comparison_csv,
)
from quanthack.cli import crypto_overlay_sizing_compare
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoOverlaySizingCompareTest(TestCase):
    def test_compare_crypto_overlay_sizing_ranks_multiplier_grid(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=36,
            interval_minutes=15,
            seed=101,
        )

        comparison = compare_crypto_overlay_sizing(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            specs=(
                CryptoOverlaySizingSpec(label="full", crypto_multiplier=1.0),
                CryptoOverlaySizingSpec(
                    label="half",
                    crypto_multiplier=0.5,
                    crypto_allowed_utc_hours=(10, 8, 10),
                ),
            ),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(comparison.official_symbols, ("EURUSD",))
        self.assertEqual(comparison.crypto_symbols, ("BTCUSD", "SOLUSD"))
        self.assertEqual(len(comparison.candidates), 2)
        self.assertIsNotNone(comparison.best)
        rank_keys = [candidate.rank_key for candidate in comparison.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        multiplier_maps = {
            candidate.label: candidate.multiplier_map_text
            for candidate in comparison.candidates
        }
        self.assertIn("BTCUSD=1.000", multiplier_maps["full"])
        self.assertIn("BTCUSD=0.500", multiplier_maps["half"])
        half = next(candidate for candidate in comparison.candidates if candidate.label == "half")
        self.assertEqual(half.crypto_allowed_utc_hours, (8, 10))
        self.assertEqual(half.crypto_allowed_utc_hours_text, "8|10")

    def test_write_crypto_overlay_sizing_comparison_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=24,
            interval_minutes=15,
            seed=102,
        )
        comparison = compare_crypto_overlay_sizing(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "BTCUSD"),
            specs=(CryptoOverlaySizingSpec(label="btc_half", crypto_multiplier=0.5),),
            run_walk_forward=False,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sizing.csv"
            write_crypto_overlay_sizing_comparison_csv(comparison, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,official_symbols,crypto_symbols", text)
        self.assertIn("multiplier_map", text)
        self.assertIn("crypto_allowed_utc_hours", text)
        self.assertIn("btc_half", text)

    def test_crypto_overlay_sizing_compare_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=32,
            interval_minutes=15,
            seed=103,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "sizing.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_overlay_sizing_compare.main(
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
                        "BTCUSD",
                        "--symbol",
                        "SOLUSD",
                        "--candidate",
                        "label=btc_half,crypto=0.75,btc=0.5,sol=1.0,crypto_hours=8|9|10",
                        "--no-walk-forward",
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Crypto Overlay Sizing Comparison", output)
        self.assertIn("btc_half", output)
        self.assertIn("crypto hours: 8|9|10", output)
        self.assertIn("BTCUSD=0.500", text)
        self.assertIn("8|9|10", text)
