from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.autocorrelation_regime_optimizer import (
    AutocorrelationRegimeParameterSet,
    optimize_autocorrelation_regime_parameters,
    write_autocorrelation_regime_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class AutocorrelationRegimeOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=96,
            interval_minutes=15,
            seed=611,
        )

        result = optimize_autocorrelation_regime_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                AutocorrelationRegimeParameterSet(
                    "loose",
                    16,
                    4,
                    0.10,
                    0.02,
                    2.0,
                    0.0,
                    0.5,
                    1.0,
                    1.0,
                    8,
                ),
                AutocorrelationRegimeParameterSet(
                    "strict",
                    32,
                    6,
                    0.30,
                    0.05,
                    6.0,
                    0.30,
                    1.2,
                    3.0,
                    4.0,
                    6,
                    allowed_utc_hours=(10, 11, 12),
                ),
            ),
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY"))
        self.assertEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_writes_optimizer_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=96,
            interval_minutes=15,
            seed=612,
        )
        result = optimize_autocorrelation_regime_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                AutocorrelationRegimeParameterSet(
                    "strict",
                    32,
                    6,
                    0.30,
                    0.05,
                    6.0,
                    0.30,
                    1.2,
                    3.0,
                    4.0,
                    6,
                    allowed_utc_hours=(10, 11, 12),
                ),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "autocorrelation_regime_opt.csv"
            write_autocorrelation_regime_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,lookback", text)
        self.assertIn("min_abs_autocorrelation", text)
        self.assertIn("allowed_utc_hours", text)
        self.assertIn("strict", text)
        self.assertIn("wf_active_positive_fold_fraction", text)

    def test_parameter_set_rejects_invalid_autocorrelation(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            AutocorrelationRegimeParameterSet(
                "bad",
                16,
                4,
                1.5,
                0.02,
                2.0,
                0.0,
                0.5,
                1.0,
                1.0,
                8,
            )

    def test_parameter_set_rejects_invalid_session_hour(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 23"):
            AutocorrelationRegimeParameterSet(
                "bad_hour",
                16,
                4,
                0.2,
                0.02,
                2.0,
                0.0,
                0.5,
                1.0,
                1.0,
                8,
                allowed_utc_hours=(25,),
            )
