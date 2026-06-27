from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.multi_horizon_momentum_optimizer import (
    MultiHorizonMomentumParameterSet,
    optimize_multi_horizon_momentum_parameters,
    write_multi_horizon_momentum_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class MultiHorizonMomentumOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=128,
            interval_minutes=15,
            seed=701,
        )

        result = optimize_multi_horizon_momentum_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                MultiHorizonMomentumParameterSet(
                    "fast",
                    3,
                    10,
                    4,
                    24,
                    0.5,
                    1.0,
                    0.0,
                    0.0,
                    10.0,
                    8,
                ),
                MultiHorizonMomentumParameterSet(
                    "strict",
                    6,
                    18,
                    6,
                    36,
                    2.0,
                    4.0,
                    0.2,
                    0.2,
                    3.0,
                    12,
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
            periods=128,
            interval_minutes=15,
            seed=702,
        )
        result = optimize_multi_horizon_momentum_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                MultiHorizonMomentumParameterSet(
                    "fast",
                    3,
                    10,
                    4,
                    24,
                    0.5,
                    1.0,
                    0.0,
                    0.0,
                    10.0,
                    8,
                ),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi_horizon_momentum_opt.csv"
            write_multi_horizon_momentum_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,fast_lookback", text)
        self.assertIn("allowed_utc_hours", text)
        self.assertIn("fast", text)
        self.assertIn("wf_active_positive_fold_fraction", text)

    def test_parameter_set_rejects_invalid_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "slow_lookback"):
            MultiHorizonMomentumParameterSet(
                "bad",
                8,
                8,
                4,
                24,
                0.5,
                1.0,
                0.0,
                0.0,
                10.0,
                8,
            )

    def test_parameter_set_accepts_session_hours(self) -> None:
        parameters = MultiHorizonMomentumParameterSet(
            "london",
            6,
            24,
            12,
            48,
            2.0,
            5.0,
            0.25,
            0.35,
            2.5,
            24,
            allowed_utc_hours=(10, 11, 12, 13, 14),
        )

        self.assertEqual(parameters.allowed_utc_hours, (10, 11, 12, 13, 14))

    def test_parameter_set_rejects_invalid_session_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 23"):
            MultiHorizonMomentumParameterSet(
                "bad_hours",
                6,
                24,
                12,
                48,
                2.0,
                5.0,
                0.25,
                0.35,
                2.5,
                24,
                allowed_utc_hours=(10, 24),
            )
