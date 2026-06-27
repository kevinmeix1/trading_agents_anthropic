from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.kalman_trend_optimizer import (
    KalmanTrendParameterSet,
    optimize_kalman_trend_parameters,
    write_kalman_trend_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class KalmanTrendOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=96,
            interval_minutes=15,
            seed=101,
        )

        result = optimize_kalman_trend_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                KalmanTrendParameterSet("fast", 20, 0.1, 0.0, 1.0, 4, 8),
                KalmanTrendParameterSet("strict", 40, 1.0, 0.5, 5.0, 6, 16),
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
            seed=102,
        )
        result = optimize_kalman_trend_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                KalmanTrendParameterSet("fast", 20, 0.1, 0.0, 1.0, 4, 8),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kalman_trend_opt.csv"
            write_kalman_trend_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,lookback", text)
        self.assertIn("fast", text)
        self.assertIn("min_abs_slope_bps", text)

    def test_parameter_set_rejects_invalid_lookback(self) -> None:
        with self.assertRaisesRegex(ValueError, "lookback"):
            KalmanTrendParameterSet("bad", 4, 0.1, 0.0, 1.0, 4, 8)
