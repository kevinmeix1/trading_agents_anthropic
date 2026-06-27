from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_sleeve_compare import (
    compare_crypto_sleeves,
    write_crypto_sleeve_comparison_csv,
)
from quanthack.cli import crypto_sleeve_compare
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoSleeveCompareTest(TestCase):
    def test_compare_crypto_sleeves_ranks_full_and_walk_forward_evidence(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD", "SOLUSD"),
            periods=48,
            interval_minutes=15,
            seed=71,
        )

        comparison = compare_crypto_sleeves(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("macd_momentum", "asset_adaptive_macd"),
            symbols=("BTCUSD", "ETHUSD", "SOLUSD"),
            train_size=18,
            test_size=10,
            step_size=10,
        )

        self.assertEqual(comparison.symbols, ("BTCUSD", "ETHUSD", "SOLUSD"))
        self.assertTrue(comparison.walk_forward_enabled)
        self.assertEqual(len(comparison.rows), 2)
        self.assertIsNotNone(comparison.best)
        rank_keys = [row.rank_key for row in comparison.rows]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        for row in comparison.rows:
            self.assertIsNotNone(row.walk_forward)
            self.assertIsNotNone(row.promotion)
            self.assertIsNone(row.walk_forward_error)
            self.assertGreaterEqual(row.selection_score, 0.0)

    def test_compare_crypto_sleeves_rejects_non_crypto_symbols(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=24,
            interval_minutes=15,
            seed=72,
        )

        with self.assertRaisesRegex(ValueError, "not CRYPTO"):
            compare_crypto_sleeves(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("macd_momentum",),
                symbols=("EURUSD", "BTCUSD"),
                run_walk_forward=False,
            )

    def test_write_crypto_sleeve_comparison_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD"),
            periods=30,
            interval_minutes=15,
            seed=73,
        )
        comparison = compare_crypto_sleeves(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("macd_momentum",),
            run_walk_forward=False,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "crypto_sleeves.csv"
            write_crypto_sleeve_comparison_csv(comparison, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,strategy,symbols,selection_score", text)
        self.assertIn("full_proxy_score", text)
        self.assertIn("walk_forward_enabled", text)
        self.assertIn("macd_momentum", text)

    def test_crypto_sleeve_compare_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD"),
            periods=36,
            interval_minutes=15,
            seed=74,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "comparison.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_sleeve_compare.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--strategy",
                        "macd_momentum",
                        "--symbol",
                        "BTCUSD",
                        "--symbol",
                        "ETHUSD",
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

            self.assertIn("Crypto Sleeve Comparison", stdout.getvalue())
            self.assertIn("macd_momentum", output_path.read_text(encoding="utf-8"))
