from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.backtest import FillModel
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.sweep import (
    run_parameter_sweep,
    split_price_history,
    split_quote_history,
    write_sweep_csv,
)


class SweepTest(TestCase):
    def test_split_price_and_quote_histories(self) -> None:
        config = load_config("configs/default.toml")
        prices = load_price_history(config.backtest.price_csv)
        quotes = load_quote_history(config.backtest.quote_csv)

        train_prices, test_prices = split_price_history(
            prices=prices,
            symbol="EURUSD",
            train_fraction=0.6,
        )
        train_quotes, test_quotes = split_quote_history(
            quotes=quotes,
            symbol="EURUSD",
            train_count=len(train_prices.bars),
        )

        self.assertEqual(len(train_prices.bars), 12)
        self.assertEqual(len(test_prices.bars), 8)
        self.assertEqual(len(train_quotes.quotes), 12)
        self.assertEqual(len(test_quotes.quotes), 8)

    def test_parameter_sweep_returns_ranked_candidates(self) -> None:
        config = load_config("configs/default.toml")
        result = run_parameter_sweep(
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            symbol=config.simple_momentum.symbol,
            base_config=config.simple_momentum,
            lookbacks=(3, 5),
            threshold_bps=(4.0, 8.0),
            train_fraction=0.6,
            starting_equity=config.competition.starting_equity,
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
            periods_per_year=config.backtest.periods_per_year,
        )

        self.assertEqual(len(result.candidates), 4)
        self.assertIsNotNone(result.best)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_write_sweep_csv(self) -> None:
        config = load_config("configs/default.toml")
        result = run_parameter_sweep(
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            symbol=config.simple_momentum.symbol,
            base_config=config.simple_momentum,
            lookbacks=(3,),
            threshold_bps=(4.0,),
            train_fraction=0.6,
            starting_equity=config.competition.starting_equity,
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
            periods_per_year=config.backtest.periods_per_year,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sweep.csv"
            write_sweep_csv(result, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("rank,lookback,threshold_bps", text)
        self.assertIn("test_sharpe", text)
