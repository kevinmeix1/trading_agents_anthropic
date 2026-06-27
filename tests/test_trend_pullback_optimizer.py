from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.trend_pullback_optimizer import (
    TrendPullbackParameterSet,
    optimize_trend_pullback_parameters,
    write_trend_pullback_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class TrendPullbackOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=40,
            interval_minutes=15,
            seed=71,
        )

        result = optimize_trend_pullback_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            parameter_sets=(
                TrendPullbackParameterSet(
                    "fast",
                    12,
                    2,
                    1.0,
                    0.0,
                    0.1,
                    30.0,
                    0.1,
                    0.1,
                ),
                TrendPullbackParameterSet(
                    "strict",
                    16,
                    3,
                    10.0,
                    0.5,
                    2.0,
                    10.0,
                    1.0,
                    3.0,
                ),
            ),
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_writes_optimizer_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=40,
            interval_minutes=15,
            seed=72,
        )
        result = optimize_trend_pullback_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            parameter_sets=(
                TrendPullbackParameterSet(
                    "fast",
                    12,
                    2,
                    1.0,
                    0.0,
                    0.1,
                    30.0,
                    0.1,
                    0.1,
                ),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trend_pullback_opt.csv"
            write_trend_pullback_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,lookback,pullback_window", text)
        self.assertIn("fast", text)
        self.assertIn("walk_forward_eligible", text)

    def test_optimizer_can_attach_walk_forward_summary(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=40,
            interval_minutes=15,
            seed=73,
        )

        result = optimize_trend_pullback_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                TrendPullbackParameterSet(
                    "fast",
                    12,
                    2,
                    1.0,
                    0.0,
                    0.1,
                    30.0,
                    0.1,
                    0.1,
                ),
            ),
            include_walk_forward=True,
            walk_forward_train_size=12,
            walk_forward_test_size=8,
            walk_forward_step_size=8,
            walk_forward_max_baskets=3,
        )

        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertIsNotNone(result.best.walk_forward_summary)

    def test_parameter_set_rejects_invalid_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "forex_allowed_utc_hours"):
            TrendPullbackParameterSet(
                "bad",
                12,
                2,
                1.0,
                0.0,
                0.1,
                30.0,
                0.1,
                0.1,
                forex_allowed_utc_hours=(24,),
            )
