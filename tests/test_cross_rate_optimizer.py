from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.cross_rate_optimizer import (
    CrossRateParameterSet,
    optimize_cross_rate_parameters,
    write_cross_rate_optimization_csv,
)
from quanthack.core.config import load_config
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot


class CrossRateOptimizerTest(TestCase):
    def test_optimizer_ranks_parameter_sets_and_recommends_symbols(self) -> None:
        config = load_config("configs/default.toml")
        prices, quotes = _cross_rate_market()

        result = optimize_cross_rate_parameters(
            config=config,
            prices=prices,
            quotes=quotes,
            symbols=("EURGBP", "EURUSD", "GBPUSD"),
            parameter_sets=(
                CrossRateParameterSet(
                    label="active",
                    lookback=4,
                    entry_zscore=1.0,
                    min_abs_deviation_bps=1.0,
                    max_abs_deviation_bps=1000.0,
                    slippage_bps=0.0,
                    cost_buffer=0.5,
                    max_spread_bps=100.0,
                ),
                CrossRateParameterSet(
                    label="too_strict",
                    lookback=4,
                    entry_zscore=99.0,
                    min_abs_deviation_bps=1.0,
                    max_abs_deviation_bps=1000.0,
                    slippage_bps=0.0,
                    cost_buffer=0.5,
                    max_spread_bps=100.0,
                ),
            ),
            horizon_bars=1,
            min_active_signals=1,
            min_hit_rate=0.50,
        )

        self.assertEqual(result.symbols, ("EURGBP", "EURUSD", "GBPUSD"))
        self.assertEqual(result.horizon_bars, 1)
        self.assertEqual(len(result.candidates), 6)
        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertTrue(result.best.eligible)
        self.assertIn("EURGBP", result.recommended_symbols)
        rank_keys = [candidate.rank_key for candidate in result.candidates]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_write_cross_rate_optimization_csv(self) -> None:
        config = load_config("configs/default.toml")
        prices, quotes = _cross_rate_market()
        result = optimize_cross_rate_parameters(
            config=config,
            prices=prices,
            quotes=quotes,
            symbols=("EURGBP", "EURUSD", "GBPUSD"),
            parameter_sets=(
                CrossRateParameterSet(
                    label="active",
                    lookback=4,
                    entry_zscore=1.0,
                    min_abs_deviation_bps=1.0,
                    max_abs_deviation_bps=1000.0,
                    slippage_bps=0.0,
                    cost_buffer=0.5,
                    max_spread_bps=100.0,
                ),
            ),
            horizon_bars=1,
            min_active_signals=1,
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cross_rate_opt.csv"
            write_cross_rate_optimization_csv(result, output_path)
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("rank,eligible,quality_score,symbol,label", csv_text)
        self.assertIn("active", csv_text)
        self.assertIn("EURGBP", csv_text)

    def test_parameter_set_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "lookback"):
            CrossRateParameterSet(label="bad", lookback=3, entry_zscore=1.0)

        with self.assertRaisesRegex(ValueError, "exit_zscore"):
            CrossRateParameterSet(
                label="bad",
                lookback=4,
                entry_zscore=1.0,
                exit_zscore=1.0,
            )


def _cross_rate_market() -> tuple[PriceHistory, QuoteHistory]:
    start = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
    series = {
        "EURUSD": [1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000],
        "GBPUSD": [1.2500, 1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
        "EURGBP": [0.8798, 0.8801, 0.8799, 0.9000, 0.8850, 0.8800],
    }
    bars: list[PriceBar] = []
    quotes: list[QuoteSnapshot] = []
    for index in range(6):
        timestamp = start + timedelta(minutes=15 * index)
        for symbol, closes in series.items():
            close = closes[index]
            bars.append(PriceBar(timestamp=timestamp, symbol=symbol, close=close))
            quotes.append(
                QuoteSnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                    bid=close - 0.00001,
                    ask=close + 0.00001,
                )
            )
    return PriceHistory(tuple(bars)), QuoteHistory(tuple(quotes))
