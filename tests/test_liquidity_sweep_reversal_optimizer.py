from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.liquidity_sweep_reversal_optimizer import (
    LiquiditySweepReversalParameterSet,
    _config_with_parameters,
    optimize_liquidity_sweep_reversal_parameters,
    write_liquidity_sweep_reversal_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class LiquiditySweepReversalOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=96,
            interval_minutes=15,
            seed=701,
        )

        result = optimize_liquidity_sweep_reversal_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                LiquiditySweepReversalParameterSet(
                    "fast",
                    12,
                    1.0,
                    0.0,
                    2.0,
                    60.0,
                    0.90,
                    1.0,
                    6,
                ),
                LiquiditySweepReversalParameterSet(
                    "strict",
                    24,
                    4.0,
                    0.5,
                    8.0,
                    60.0,
                    0.60,
                    3.0,
                    8,
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
            seed=702,
        )
        result = optimize_liquidity_sweep_reversal_parameters(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            parameter_sets=(
                LiquiditySweepReversalParameterSet(
                    "fast",
                    12,
                    1.0,
                    0.0,
                    2.0,
                    60.0,
                    0.90,
                    1.0,
                    6,
                ),
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "liquidity_sweep_opt.csv"
            write_liquidity_sweep_reversal_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,lookback", text)
        self.assertIn("min_sweep_bps", text)
        self.assertIn("wf_active_positive_fold_fraction", text)
        self.assertIn("fast", text)

    def test_config_with_parameters_applies_hours_to_crypto(self) -> None:
        config = load_config("configs/competition.toml")
        tuned = _config_with_parameters(
            config,
            LiquiditySweepReversalParameterSet(
                "crypto_hours",
                20,
                2.0,
                0.25,
                4.0,
                80.0,
                0.75,
                2.0,
                8,
                allowed_utc_hours=(1, 2, 3),
            ),
        )

        self.assertEqual(tuned.liquidity_sweep_reversal.forex_allowed_utc_hours, (1, 2, 3))
        self.assertEqual(tuned.liquidity_sweep_reversal.metal_allowed_utc_hours, (1, 2, 3))
        self.assertEqual(tuned.liquidity_sweep_reversal.crypto_allowed_utc_hours, (1, 2, 3))

    def test_parameter_set_rejects_invalid_sweep_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_sweep_bps"):
            LiquiditySweepReversalParameterSet("bad", 12, 5.0, 0.0, 2.0, 5.0, 0.8, 1.0, 6)
