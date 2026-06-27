from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.symbol_eligibility_optimizer import (
    optimize_symbol_eligibility,
    write_symbol_attribution_rank_csv,
    write_symbol_eligibility_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class SymbolEligibilityOptimizerTest(TestCase):
    def test_optimizer_builds_ranked_candidate_universes(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=36,
            interval_minutes=15,
            seed=81,
        )

        result = optimize_symbol_eligibility(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            min_symbols=2,
            max_symbols=4,
            min_fills=0,
        )

        self.assertEqual(result.strategy_name, "simple_momentum")
        self.assertEqual(result.available_symbols, ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"))
        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertIsNotNone(result.best)
        self.assertEqual(
            [candidate.rank_key for candidate in result.candidates],
            sorted([candidate.rank_key for candidate in result.candidates], reverse=True),
        )

    def test_writes_candidate_and_attribution_csvs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=36,
            interval_minutes=15,
            seed=82,
        )
        result = optimize_symbol_eligibility(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            min_symbols=2,
            max_symbols=4,
            min_fills=0,
        )

        with TemporaryDirectory() as tmpdir:
            candidate_path = Path(tmpdir) / "eligibility.csv"
            attribution_path = Path(tmpdir) / "attribution.csv"
            write_symbol_eligibility_csv(result, candidate_path)
            write_symbol_attribution_rank_csv(result, attribution_path)
            candidate_text = candidate_path.read_text(encoding="utf-8")
            attribution_text = attribution_path.read_text(encoding="utf-8")

        self.assertIn("rank,candidate,strategy,symbols", candidate_text)
        self.assertIn("all_symbols", candidate_text)
        self.assertIn("wf_positive_fold_fraction", candidate_text)
        self.assertIn("wf_active_positive_fold_fraction", candidate_text)
        self.assertIn("wf_non_negative_fold_fraction", candidate_text)
        self.assertIn("rank,strategy,symbol,fills", attribution_text)

    def test_optimizer_can_rank_with_fixed_warmup_walk_forward(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=83,
        )

        result = optimize_symbol_eligibility(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            min_symbols=2,
            max_symbols=3,
            min_fills=0,
            include_walk_forward=True,
            train_size=24,
            test_size=12,
            step_size=12,
        )

        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertTrue(
            all(candidate.walk_forward is not None for candidate in result.candidates)
        )
        self.assertEqual(
            [candidate.rank_key for candidate in result.candidates],
            sorted([candidate.rank_key for candidate in result.candidates], reverse=True),
        )

    def test_optimizer_can_include_combinational_candidate_search(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=36,
            interval_minutes=15,
            seed=84,
        )

        result = optimize_symbol_eligibility(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            min_symbols=2,
            max_symbols=2,
            min_fills=0,
            include_combinations=True,
            combination_pool_size=3,
            max_combinations=3,
        )

        self.assertTrue(
            any(candidate.name.startswith("combo_top3_2_") for candidate in result.candidates)
        )
