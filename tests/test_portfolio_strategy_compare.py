from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.config import load_config
from quanthack.backtesting.portfolio_strategy_compare import (
    compare_portfolio_strategies,
    write_portfolio_strategy_comparison_csv,
)
from quanthack.market.sample_data import generate_synthetic_market_data


class PortfolioStrategyCompareTest(TestCase):
    def test_compare_portfolio_strategies_returns_proxy_ranked_rows(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=18,
            interval_minutes=5,
            seed=3,
        )

        comparison = compare_portfolio_strategies(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "ma_crossover"),
            symbols=("EURUSD", "GBPUSD"),
        )

        self.assertEqual(comparison.symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(len(comparison.rows), 2)
        self.assertIsNotNone(comparison.best)
        rank_keys = [row.rank_key for row in comparison.rows]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))
        for row in comparison.rows:
            self.assertGreaterEqual(row.proxy_score, 0.0)
            self.assertLessEqual(row.proxy_score, 100.0)

    def test_write_portfolio_strategy_comparison_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=18,
            interval_minutes=5,
            seed=4,
        )
        comparison = compare_portfolio_strategies(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum",),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio_compare.csv"
            write_portfolio_strategy_comparison_csv(comparison, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,strategy,symbols,proxy_score", text)
        self.assertIn("risk_discipline_score", text)
        self.assertIn("simple_momentum", text)
