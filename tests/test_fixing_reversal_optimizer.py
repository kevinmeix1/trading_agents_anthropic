from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.fixing_reversal_optimizer import (
    FixingReversalParameterSet,
    optimize_fixing_reversal_parameters,
    write_fixing_reversal_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class FixingReversalOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=91,
        )

        result = optimize_fixing_reversal_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                FixingReversalParameterSet("loose", 4, 1.0, 0.1, 0.0, 2, (14,)),
                FixingReversalParameterSet("strict", 4, 12.0, 1.5, 0.5, 4, (14,)),
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
            periods=72,
            interval_minutes=15,
            seed=92,
        )
        result = optimize_fixing_reversal_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                FixingReversalParameterSet("loose", 4, 1.0, 0.1, 0.0, 2, (14,)),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixing_reversal_opt.csv"
            write_fixing_reversal_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,pre_fix_lookback", text)
        self.assertIn("loose", text)
        self.assertIn("allowed_utc_hours", text)

    def test_parameter_set_rejects_invalid_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowed_utc_hours"):
            FixingReversalParameterSet("bad", 4, 1.0, 0.1, 0.0, 2, (24,))
