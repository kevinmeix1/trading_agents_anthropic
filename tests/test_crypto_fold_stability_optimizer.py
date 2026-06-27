from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_fold_stability_optimizer import (
    optimize_crypto_fold_stability,
    write_crypto_fold_stability_csv,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import CryptoOverlaySizingSpec
from quanthack.cli import crypto_fold_stability_optimize
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoFoldStabilityOptimizerTest(TestCase):
    def test_optimize_crypto_fold_stability_ranks_candidates(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            periods=40,
            interval_minutes=15,
            seed=141,
        )

        result = optimize_crypto_fold_stability(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            specs=(
                CryptoOverlaySizingSpec(label="full", crypto_multiplier=1.0),
                CryptoOverlaySizingSpec(
                    label="half_london",
                    crypto_multiplier=0.5,
                    btc_multiplier=0.5,
                    sol_multiplier=0.75,
                    crypto_allowed_utc_hours=(7, 8, 9, 10),
                ),
            ),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        self.assertEqual(result.comparison.official_symbols, ("EURUSD", "XAUUSD"))
        self.assertEqual(result.comparison.crypto_symbols, ("BTCUSD", "SOLUSD"))
        self.assertTrue(
            all(candidate.stability_status for candidate in result.candidates)
        )

    def test_write_crypto_fold_stability_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=32,
            interval_minutes=15,
            seed=142,
        )
        result = optimize_crypto_fold_stability(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "BTCUSD"),
            specs=(
                CryptoOverlaySizingSpec(label="btc_half", crypto_multiplier=0.5),
            ),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stability.csv"
            write_crypto_fold_stability_csv(result, output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("rank,label,stability_status,stability_score", text)
        self.assertIn("btc_half", text)
        self.assertIn("wf_largest_positive_fold_contribution", text)

    def test_crypto_fold_stability_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=36,
            interval_minutes=15,
            seed=143,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "stability.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_fold_stability_optimize.main(
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
                        "label=demo,crypto=0.5,btc=0.5,sol=0.75,crypto_hours=7|8|9",
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

        self.assertIn("Crypto Fold Stability Optimization", output)
        self.assertIn("demo", output)
        self.assertIn("stability_score", text)
