from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.dual_squeeze_optimizer import (
    DualSqueezeParameterSet,
    optimize_dual_squeeze_parameters,
    write_dual_squeeze_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class DualSqueezeOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=44,
            interval_minutes=15,
            seed=71,
        )
        parameter_sets = (
            DualSqueezeParameterSet(
                label="fast",
                lookback=10,
                squeeze_window=3,
                max_squeeze_ratio=0.80,
                breakout_buffer_bps=1.0,
                band_stdev_multiplier=1.5,
                confirmation_lookback=14,
                confirmation_squeeze_window=4,
                confirmation_max_squeeze_ratio=0.90,
            ),
            DualSqueezeParameterSet(
                label="slow",
                lookback=14,
                squeeze_window=4,
                max_squeeze_ratio=0.60,
                breakout_buffer_bps=2.0,
                band_stdev_multiplier=2.0,
                confirmation_lookback=18,
                confirmation_squeeze_window=5,
                confirmation_max_squeeze_ratio=0.80,
            ),
        )

        result = optimize_dual_squeeze_parameters(
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

    def test_write_dual_squeeze_optimization_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=44,
            interval_minutes=15,
            seed=72,
        )
        result = optimize_dual_squeeze_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            parameter_sets=(
                DualSqueezeParameterSet(
                    label="baseline",
                    lookback=10,
                    squeeze_window=3,
                    max_squeeze_ratio=0.80,
                    breakout_buffer_bps=1.0,
                    band_stdev_multiplier=1.5,
                    confirmation_lookback=14,
                    confirmation_squeeze_window=4,
                    confirmation_max_squeeze_ratio=0.90,
                ),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dual_squeeze_opt.csv"
            write_dual_squeeze_optimization_csv(result, output_path)
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,lookback,squeeze_window", csv_text)
        self.assertIn("confirmation_lookback", csv_text)
        self.assertIn("baseline", csv_text)

    def test_optimizer_can_attach_walk_forward_summary(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=48,
            interval_minutes=15,
            seed=73,
        )

        result = optimize_dual_squeeze_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            parameter_sets=(
                DualSqueezeParameterSet(
                    label="baseline",
                    lookback=8,
                    squeeze_window=2,
                    max_squeeze_ratio=0.90,
                    breakout_buffer_bps=1.0,
                    band_stdev_multiplier=1.5,
                    confirmation_lookback=12,
                    confirmation_squeeze_window=3,
                    confirmation_max_squeeze_ratio=0.95,
                ),
            ),
            include_walk_forward=True,
            walk_forward_train_size=18,
            walk_forward_test_size=6,
            walk_forward_step_size=6,
            walk_forward_max_baskets=2,
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertIsNotNone(result.candidates[0].walk_forward_summary)
