from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_overlay_compare import (
    compare_crypto_overlays,
    write_crypto_overlay_comparison_csv,
)
from quanthack.cli import crypto_overlay_compare
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoOverlayCompareTest(TestCase):
    def test_compare_crypto_overlays_ranks_baseline_and_overlay_maps(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "BTCUSD", "SOLUSD"),
            periods=40,
            interval_minutes=15,
            seed=91,
        )

        comparison = compare_crypto_overlays(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "BTCUSD", "SOLUSD"),
            train_size=14,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(comparison.official_symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(comparison.crypto_symbols, ("BTCUSD", "SOLUSD"))
        self.assertGreaterEqual(len(comparison.candidates), 4)
        self.assertIsNotNone(comparison.best)
        rank_keys = [candidate.rank_key for candidate in comparison.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        labels = {candidate.label for candidate in comparison.candidates}
        self.assertIn("official_only_base", labels)
        self.assertIn("crypto_robust_sol_overlay", labels)
        for candidate in comparison.candidates:
            self.assertIsNotNone(candidate.walk_forward)
            self.assertIsNotNone(candidate.promotion)

    def test_write_crypto_overlay_comparison_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=30,
            interval_minutes=15,
            seed=92,
        )
        comparison = compare_crypto_overlays(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            run_walk_forward=False,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "overlay.csv"
            write_crypto_overlay_comparison_csv(comparison, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,official_symbols,crypto_symbols", text)
        self.assertIn("crypto_robust_sol_overlay", text)
        self.assertIn("crypto_map", text)

    def test_crypto_overlay_compare_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=36,
            interval_minutes=15,
            seed=93,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "overlay.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_overlay_compare.main(
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

            self.assertIn("Crypto Overlay Comparison", stdout.getvalue())
            self.assertIn("crypto_robust_sol_overlay", output_path.read_text(encoding="utf-8"))
