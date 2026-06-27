from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.config import load_config
from quanthack.backtesting.router_optimizer import (
    RouterBehaviorProfile,
    RouterWeightSet,
    optimize_router_weights,
    write_router_optimization_csv,
)
from quanthack.market.sample_data import generate_synthetic_market_data


class RouterOptimizerTest(TestCase):
    def test_optimizer_returns_ranked_candidates(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=20,
            interval_minutes=5,
            seed=11,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            weight_sets=(
                RouterWeightSet(0.40, 0.20, 0.35, 0.25),
                RouterWeightSet(0.20, 0.50, 0.20, 0.10),
            ),
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        for candidate in result.candidates:
            self.assertIsInstance(candidate.behavior, RouterBehaviorProfile)
            self.assertGreaterEqual(candidate.proxy_score, 0.0)
            self.assertLessEqual(candidate.proxy_score, 100.0)
            self.assertIn("session=", candidate.weights.label)
            self.assertIn("xrate=", candidate.weights.label)
            self.assertIn("rel=", candidate.weights.label)
            self.assertIn("squeeze=", candidate.weights.label)
            self.assertIn("dual=", candidate.weights.label)
            self.assertIn("macd=", candidate.weights.label)
            self.assertIn("kalman=", candidate.weights.label)

    def test_optimizer_accepts_behavior_profiles(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=20,
            interval_minutes=5,
            seed=17,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            weight_sets=(RouterWeightSet(0.40, 0.20, 0.35, 0.25),),
            behavior_profiles=(
                RouterBehaviorProfile(),
                RouterBehaviorProfile(
                    entry_score=0.55,
                    min_signal_confidence=0.20,
                    cost_buffer=1.20,
                    conflict_penalty=0.70,
                    primary_signal_override_enabled=False,
                ),
            ),
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(
            {
                candidate.behavior.primary_signal_override_enabled
                for candidate in result.candidates
            },
            {True, False},
        )

    def test_router_weight_set_can_include_newer_alpha_sleeves(self) -> None:
        weights = RouterWeightSet(
            0.25,
            0.15,
            0.10,
            0.35,
            session_breakout_weight=0.20,
            cross_rate_weight=0.10,
            relative_strength_weight=0.15,
            volatility_squeeze_weight=0.80,
            dual_squeeze_weight=0.25,
            macd_momentum_weight=0.30,
            kalman_trend_weight=0.10,
        )

        self.assertAlmostEqual(weights.total, 2.75)
        self.assertIn("session=0.20", weights.label)
        self.assertIn("xrate=0.10", weights.label)
        self.assertIn("rel=0.15", weights.label)
        self.assertIn("squeeze=0.80", weights.label)
        self.assertIn("dual=0.25", weights.label)
        self.assertIn("macd=0.30", weights.label)
        self.assertIn("kalman=0.10", weights.label)

    def test_writes_router_optimization_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=20,
            interval_minutes=5,
            seed=12,
        )
        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            weight_sets=(RouterWeightSet(0.40, 0.20, 0.35, 0.25),),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "router_opt.csv"
            write_router_optimization_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,symbols,weights,behavior,proxy_score", text)
        self.assertIn("behavior", text)
        self.assertIn("entry_score", text)
        self.assertIn("primary_signal_override_enabled", text)
        self.assertIn("session_breakout_weight", text)
        self.assertIn("cross_rate_weight", text)
        self.assertIn("relative_strength_weight", text)
        self.assertIn("volatility_squeeze_weight", text)
        self.assertIn("dual_squeeze_weight", text)
        self.assertIn("macd_momentum_weight", text)
        self.assertIn("kalman_trend_weight", text)
        self.assertIn("trimmed_allocation_periods", text)

    def test_optimizer_accepts_cross_rate_candidate_on_mixed_universe(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "BTCUSD"),
            periods=18,
            interval_minutes=5,
            seed=13,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "BTCUSD"),
            weight_sets=(RouterWeightSet(0.25, 0.15, 0.10, 0.35, 0.20, 0.10),),
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "BTCUSD"))
        self.assertEqual(result.candidates[0].weights.cross_rate_weight, 0.10)

    def test_optimizer_accepts_relative_strength_candidate(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=20,
            interval_minutes=15,
            seed=14,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            weight_sets=(RouterWeightSet(0.20, 0.10, 0.10, 0.35, 0.15, 0.00, 0.20),),
        )

        self.assertEqual(result.candidates[0].weights.relative_strength_weight, 0.20)

    def test_optimizer_accepts_volatility_squeeze_candidate(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=20,
            interval_minutes=15,
            seed=15,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            weight_sets=(RouterWeightSet(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),),
        )

        self.assertEqual(result.candidates[0].weights.volatility_squeeze_weight, 1.00)

    def test_optimizer_accepts_dual_squeeze_candidate(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=24,
            interval_minutes=15,
            seed=16,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            weight_sets=(
                RouterWeightSet(
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    1.00,
                ),
            ),
        )

        self.assertEqual(result.candidates[0].weights.dual_squeeze_weight, 1.00)

    def test_optimizer_accepts_macd_and_kalman_candidates(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=90,
            interval_minutes=15,
            seed=18,
        )

        result = optimize_router_weights(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD"),
            weight_sets=(
                RouterWeightSet(
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.75,
                    0.25,
                ),
            ),
        )

        self.assertEqual(result.candidates[0].weights.macd_momentum_weight, 0.75)
        self.assertEqual(result.candidates[0].weights.kalman_trend_weight, 0.25)
