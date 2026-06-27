from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.strategy_map_optimizer import (
    optimize_strategy_map,
    write_strategy_map_optimization_csv,
    write_symbol_strategy_scores_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class StrategyMapOptimizerTest(TestCase):
    def test_optimizer_builds_ranked_strategy_maps(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=401,
        )

        result = optimize_strategy_map(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            top_symbol_counts=(2, 3),
        )

        self.assertEqual(result.available_symbols, ("EURUSD", "GBPUSD", "USDJPY"))
        self.assertEqual(result.strategy_names, ("simple_momentum", "macd_momentum"))
        self.assertGreaterEqual(len(result.symbol_scores), 6)
        self.assertGreaterEqual(len(result.candidates), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_writes_candidate_and_score_csvs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=402,
        )
        result = optimize_strategy_map(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            top_symbol_counts=(2,),
        )

        with TemporaryDirectory() as tmpdir:
            candidate_path = Path(tmpdir) / "maps.csv"
            score_path = Path(tmpdir) / "scores.csv"
            write_strategy_map_optimization_csv(result, candidate_path)
            write_symbol_strategy_scores_csv(result, score_path)
            candidate_text = candidate_path.read_text(encoding="utf-8")
            score_text = score_path.read_text(encoding="utf-8")

        self.assertIn("rank,label,symbols,strategy_map", candidate_text)
        self.assertIn("rank,symbol,strategy,total_pnl_usd", score_text)
        self.assertIn("simple_momentum", score_text)

    def test_optimizer_rejects_empty_strategy_list(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD",),
            periods=24,
            interval_minutes=15,
            seed=403,
        )

        with self.assertRaisesRegex(ValueError, "at least one strategy"):
            optimize_strategy_map(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=(),
                symbols=("EURUSD",),
            )
