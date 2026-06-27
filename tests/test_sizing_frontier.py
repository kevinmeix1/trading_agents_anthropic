from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.sizing_frontier import (
    evaluate_sizing_frontier,
    write_sizing_frontier_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class SizingFrontierTest(TestCase):
    def test_frontier_evaluates_multiple_symbol_caps(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=96,
            interval_minutes=15,
            seed=501,
        )

        result = evaluate_sizing_frontier(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_by_symbol={
                "EURUSD": "macd_momentum",
                "GBPUSD": "multi_horizon_momentum",
            },
            symbol_notional_pcts=(0.25, 0.50),
        )

        self.assertEqual(len(result.points), 2)
        self.assertEqual(result.points[0].symbol_notional_pct, 0.25)
        self.assertEqual(result.points[1].symbol_notional_pct, 0.50)
        self.assertIsNotNone(result.best_full_sample)
        self.assertEqual(
            result.strategy_by_symbol,
            (
                ("EURUSD", "macd_momentum"),
                ("GBPUSD", "multi_horizon_momentum"),
            ),
        )

    def test_frontier_writes_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD",),
            periods=72,
            interval_minutes=15,
            seed=502,
        )
        result = evaluate_sizing_frontier(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_by_symbol={"EURUSD": "macd_momentum"},
            symbol_notional_pcts=(0.25,),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frontier.csv"
            write_sizing_frontier_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("symbol_notional_pct,max_gross_leverage", text)
        self.assertIn("worst_leverage", text)

    def test_frontier_rejects_bad_symbol_cap(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD",),
            periods=24,
            interval_minutes=15,
            seed=503,
        )

        with self.assertRaisesRegex(ValueError, "symbol_notional_pcts"):
            evaluate_sizing_frontier(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_by_symbol={"EURUSD": "macd_momentum"},
                symbol_notional_pcts=(0.0,),
            )
