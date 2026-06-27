from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.strategy_attribution import (
    run_strategy_attribution,
    write_strategy_attribution_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class StrategyAttributionTest(TestCase):
    def test_attribution_returns_symbol_and_portfolio_rows(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=24,
            interval_minutes=15,
            seed=81,
        )

        report = run_strategy_attribution(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "volatility_squeeze"),
            symbols=("EURUSD", "GBPUSD"),
        )

        self.assertEqual(report.symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(len(report.rows), 6)
        self.assertIn("PORTFOLIO", {row.symbol for row in report.rows})
        self.assertIn("simple_momentum", {row.strategy_name for row in report.rows})

    def test_writes_attribution_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=24,
            interval_minutes=15,
            seed=82,
        )
        report = run_strategy_attribution(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum",),
            symbols=("EURUSD", "GBPUSD"),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attribution.csv"
            write_strategy_attribution_csv(report, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("strategy,symbol,fills,realized_pnl_usd", text)
        self.assertIn("PORTFOLIO", text)
