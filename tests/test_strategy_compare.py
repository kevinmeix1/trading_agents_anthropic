from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.strategy_compare import compare_strategies, write_strategy_comparison_csv


class StrategyCompareTest(TestCase):
    def test_compare_strategies_returns_ranked_rows(self) -> None:
        config = load_config("configs/default.toml")
        comparison = compare_strategies(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("simple_momentum", "mean_reversion"),
        )

        self.assertEqual(len(comparison.rows), 2)
        self.assertIsNotNone(comparison.best)
        rank_keys = [row.rank_key for row in comparison.rows]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_compare_strategies_normalizes_and_deduplicates_aliases(self) -> None:
        config = load_config("configs/default.toml")
        comparison = compare_strategies(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("momentum", "simple_momentum", "mean-reversion"),
        )

        self.assertEqual(
            sorted(row.strategy_name for row in comparison.rows),
            ["mean_reversion", "simple_momentum"],
        )

    def test_write_strategy_comparison_csv(self) -> None:
        config = load_config("configs/default.toml")
        comparison = compare_strategies(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("simple_momentum",),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.csv"
            write_strategy_comparison_csv(comparison, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,strategy,symbol", text)
        self.assertIn("simple_momentum", text)
