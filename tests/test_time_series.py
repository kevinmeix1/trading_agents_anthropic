from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.market.market_data import PriceBar, PriceHistory
from quanthack.strategies.time_series import (
    KalmanTrendConfig,
    TimeSeriesRegime,
    evaluate_time_series_regimes,
    read_kalman_regime,
    write_time_series_regime_csv,
)


UTC = ZoneInfo("UTC")


class TimeSeriesRegimeTest(TestCase):
    def test_kalman_regime_detects_clean_uptrend(self) -> None:
        reading = read_kalman_regime(
            [1.0000 + index * 0.0010 for index in range(30)],
            symbol="EURUSD",
            config=KalmanTrendConfig(lookback=20, min_abs_slope_bps=0.1),
        )

        self.assertEqual(reading.regime, TimeSeriesRegime.TREND_UP)
        self.assertGreater(reading.kalman_slope_bps, 0.0)
        self.assertGreater(reading.trend_efficiency, 0.9)

    def test_kalman_regime_detects_chop(self) -> None:
        prices = [1.0 + (0.0001 if index % 2 == 0 else -0.0001) for index in range(30)]

        reading = read_kalman_regime(
            prices,
            symbol="EURUSD",
            config=KalmanTrendConfig(lookback=20, min_abs_slope_bps=5.0),
        )

        self.assertEqual(reading.regime, TimeSeriesRegime.CHOP)

    def test_portfolio_regime_csv_is_written(self) -> None:
        readings = evaluate_time_series_regimes(
            prices=_prices(
                {
                    "EURUSD": [1.0000 + index * 0.0010 for index in range(30)],
                    "GBPUSD": [1.3000 - index * 0.0010 for index in range(30)],
                }
            ),
            config=KalmanTrendConfig(lookback=20, min_abs_slope_bps=0.1),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regimes.csv"
            write_time_series_regime_csv(readings, path)
            text = path.read_text(encoding="utf-8")

        self.assertEqual([reading.symbol for reading in readings], ["EURUSD", "GBPUSD"])
        self.assertIn("symbol,observations,latest_close", text)
        self.assertIn("TREND_UP", text)
        self.assertIn("TREND_DOWN", text)


def _prices(values_by_symbol: dict[str, list[float]]) -> PriceHistory:
    bars: list[PriceBar] = []
    for symbol, values in values_by_symbol.items():
        bars.extend(
            PriceBar(
                timestamp=datetime(2026, 5, 11, tzinfo=UTC)
                + timedelta(minutes=15 * index),
                symbol=symbol,
                close=value,
            )
            for index, value in enumerate(values)
        )
    return PriceHistory(tuple(bars))

