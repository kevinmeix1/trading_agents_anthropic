from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.relative_strength_optimizer import (
    RelativeStrengthParameterSet,
    optimize_relative_strength_parameters,
    write_relative_strength_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class RelativeStrengthOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=24,
            interval_minutes=15,
            seed=41,
        )
        parameter_sets = (
            RelativeStrengthParameterSet(
                label="short",
                lookback=4,
                entry_zscore=0.50,
                exit_zscore=0.15,
            ),
            RelativeStrengthParameterSet(
                label="long",
                lookback=6,
                entry_zscore=0.75,
                exit_zscore=0.25,
            ),
        )

        result = optimize_relative_strength_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            parameter_sets=parameter_sets,
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"))
        self.assertEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        self.assertEqual(
            [candidate.rank_key for candidate in result.candidates],
            sorted([candidate.rank_key for candidate in result.candidates], reverse=True),
        )

    def test_write_relative_strength_optimization_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=24,
            interval_minutes=15,
            seed=42,
        )
        result = optimize_relative_strength_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            parameter_sets=(
                RelativeStrengthParameterSet(
                    label="baseline",
                    lookback=4,
                    entry_zscore=0.50,
                    exit_zscore=0.15,
                ),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "relative_strength_opt.csv"
            write_relative_strength_optimization_csv(result, output_path)
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,lookback", csv_text)
        self.assertIn("baseline", csv_text)
