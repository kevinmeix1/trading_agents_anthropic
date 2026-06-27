from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.macd_momentum_optimizer import (
    DEFAULT_MACD_MOMENTUM_PARAMETER_SETS,
    MacdMomentumParameterSet,
    _config_with_parameters,
    optimize_macd_momentum_parameters,
    write_macd_momentum_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class MacdMomentumOptimizerTest(TestCase):
    def test_default_parameter_sets_include_current_competition_baseline(self) -> None:
        current = DEFAULT_MACD_MOMENTUM_PARAMETER_SETS[0]

        self.assertEqual(
            current.label,
            "competition_current_6_18_5_h2p5_m1_eff20_hold12",
        )
        self.assertEqual(current.fast_window, 6)
        self.assertEqual(current.slow_window, 18)
        self.assertEqual(current.signal_window, 5)
        self.assertEqual(current.min_histogram_bps, 2.5)
        self.assertEqual(current.min_macd_bps, 1.0)
        self.assertEqual(current.min_trend_efficiency, 0.20)
        self.assertEqual(current.max_holding_period, 12)
        self.assertEqual(current.allowed_utc_hours, (10, 11, 12, 13, 14))

    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=96,
            interval_minutes=15,
            seed=301,
        )

        result = optimize_macd_momentum_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                MacdMomentumParameterSet("fast", 3, 8, 3, 0.5, 0.1, 0.0, 8),
                MacdMomentumParameterSet("strict", 6, 14, 5, 2.0, 1.0, 0.2, 12),
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
            seed=302,
        )
        result = optimize_macd_momentum_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                MacdMomentumParameterSet("fast", 3, 8, 3, 0.5, 0.1, 0.0, 8),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "macd_momentum_opt.csv"
            write_macd_momentum_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,fast_window", text)
        self.assertIn("min_histogram_slope_bps", text)
        self.assertIn("allowed_utc_hours", text)
        self.assertIn("fast", text)
        self.assertIn("wf_active_positive_fold_fraction", text)

    def test_parameter_set_rejects_invalid_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "slow_window"):
            MacdMomentumParameterSet("bad", 8, 8, 3, 0.5, 0.1, 0.0, 8)

    def test_parameter_set_accepts_session_hours(self) -> None:
        parameters = MacdMomentumParameterSet(
            "london_ny",
            6,
            18,
            5,
            2.0,
            1.0,
            0.2,
            12,
            allowed_utc_hours=(11, 12, 13, 14, 15, 16),
        )

        self.assertEqual(parameters.allowed_utc_hours, (11, 12, 13, 14, 15, 16))

    def test_config_with_parameters_applies_hours_to_crypto(self) -> None:
        config = load_config("configs/competition.toml")
        tuned = _config_with_parameters(
            config,
            MacdMomentumParameterSet(
                "crypto_hours",
                6,
                18,
                5,
                2.0,
                1.0,
                0.2,
                12,
                allowed_utc_hours=(1, 2, 3),
            ),
        )

        self.assertEqual(tuned.macd_momentum.forex_allowed_utc_hours, (1, 2, 3))
        self.assertEqual(tuned.macd_momentum.metal_allowed_utc_hours, (1, 2, 3))
        self.assertEqual(tuned.macd_momentum.crypto_allowed_utc_hours, (1, 2, 3))

    def test_parameter_set_accepts_histogram_slope(self) -> None:
        parameters = MacdMomentumParameterSet(
            "slope",
            6,
            18,
            5,
            2.0,
            1.0,
            0.2,
            12,
            min_histogram_slope_bps=0.25,
        )

        self.assertEqual(parameters.min_histogram_slope_bps, 0.25)

    def test_parameter_set_rejects_negative_histogram_slope(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_histogram_slope_bps"):
            MacdMomentumParameterSet(
                "bad_slope",
                6,
                18,
                5,
                2.0,
                1.0,
                0.2,
                12,
                min_histogram_slope_bps=-0.1,
            )

    def test_parameter_set_rejects_invalid_session_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 23"):
            MacdMomentumParameterSet(
                "bad_hours",
                6,
                18,
                5,
                2.0,
                1.0,
                0.2,
                12,
                allowed_utc_hours=(11, 24),
            )
