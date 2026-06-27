from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_allocation_compare import (
    compare_crypto_allocations,
    write_crypto_allocation_comparison_csv,
)
from quanthack.cli import crypto_allocation_compare
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoAllocationCompareTest(TestCase):
    def test_compare_crypto_allocations_enumerates_ranked_maps(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD"),
            periods=40,
            interval_minutes=15,
            seed=81,
        )

        comparison = compare_crypto_allocations(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("macd_momentum", "crypto_mean_reversion"),
            symbols=("BTCUSD", "ETHUSD"),
            train_size=14,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(comparison.symbols, ("BTCUSD", "ETHUSD"))
        self.assertEqual(comparison.strategy_names, ("macd_momentum", "crypto_mean_reversion"))
        self.assertEqual(len(comparison.candidates), 4)
        self.assertIsNotNone(comparison.best)
        rank_keys = [candidate.rank_key for candidate in comparison.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        for candidate in comparison.candidates:
            self.assertIsNotNone(candidate.walk_forward)
            self.assertIsNotNone(candidate.promotion)
            self.assertGreaterEqual(candidate.selection_score, 0.0)
            self.assertIn("BTCUSD=", candidate.strategy_map_text)

    def test_compare_crypto_allocations_enforces_max_maps(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD", "SOLUSD"),
            periods=24,
            interval_minutes=15,
            seed=82,
        )

        with self.assertRaisesRegex(ValueError, "would produce 8 maps"):
            compare_crypto_allocations(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("macd_momentum", "crypto_mean_reversion"),
                symbols=("BTCUSD", "ETHUSD", "SOLUSD"),
                run_walk_forward=False,
                max_maps=4,
            )

    def test_write_crypto_allocation_comparison_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD"),
            periods=32,
            interval_minutes=15,
            seed=83,
        )
        comparison = compare_crypto_allocations(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("macd_momentum", "crypto_mean_reversion"),
            symbols=("BTCUSD", "ETHUSD"),
            run_walk_forward=False,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "crypto_allocations.csv"
            write_crypto_allocation_comparison_csv(comparison, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,strategies", text)
        self.assertIn("strategy_map", text)
        self.assertIn("crypto_mean_reversion", text)

    def test_crypto_allocation_compare_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("BTCUSD", "ETHUSD"),
            periods=36,
            interval_minutes=15,
            seed=84,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "allocations.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_allocation_compare.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
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

            self.assertIn("Crypto Allocation Comparison", stdout.getvalue())
            self.assertIn("strategy_map", output_path.read_text(encoding="utf-8"))
