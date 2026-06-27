from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.market.data_health import (
    DataHealthSeverity,
    validate_market_data,
    write_market_data_health_csv,
)
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot


LONDON = ZoneInfo("Europe/London")


class MarketDataHealthTest(TestCase):
    def test_aligned_backtest_data_is_ok(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5, 10]),
            quotes=_quotes([0, 5, 10]),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        self.assertEqual(report.overall, DataHealthSeverity.OK)
        self.assertTrue(report.ok)
        self.assertEqual(report.symbols[0].price_count, 3)
        self.assertEqual(report.symbols[0].quote_count, 3)

    def test_missing_quote_for_price_bar_fails(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5, 10]),
            quotes=_quotes([0, 10]),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        self.assertEqual(report.overall, DataHealthSeverity.FAIL)
        self.assertFalse(report.ok)
        self.assertIn("missing quotes", report.issues[0].details)

    def test_duplicate_timestamps_fail(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5, 5]),
            quotes=_quotes([0, 5]),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        self.assertEqual(report.overall, DataHealthSeverity.FAIL)
        self.assertEqual(report.symbols[0].duplicate_price_timestamps, 1)

    def test_extra_quote_warns(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5, 10]),
            quotes=_quotes([0, 5, 10, 15]),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        self.assertEqual(report.overall, DataHealthSeverity.WARN)
        self.assertIn("without price bars", report.issues[0].details)

    def test_large_gap_warns(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5, 20]),
            quotes=_quotes([0, 5, 20]),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        self.assertEqual(report.overall, DataHealthSeverity.WARN)
        self.assertTrue(any(issue.category == "gaps" for issue in report.issues))

    def test_wide_spread_warns(self) -> None:
        report = validate_market_data(
            prices=_prices([0]),
            quotes=QuoteHistory(
                (
                    QuoteSnapshot(
                        timestamp=_time(0),
                        symbol="EURUSD",
                        bid=1.1000,
                        ask=1.1050,
                    ),
                )
            ),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        self.assertEqual(report.overall, DataHealthSeverity.WARN)
        self.assertTrue(any(issue.category == "spread" for issue in report.issues))

    def test_symbol_specific_spread_limit_overrides_default(self) -> None:
        report = validate_market_data(
            prices=PriceHistory((PriceBar(timestamp=_time(0), symbol="XAUUSD", close=2320.0),)),
            quotes=QuoteHistory(
                (
                    QuoteSnapshot(
                        timestamp=_time(0),
                        symbol="XAUUSD",
                        bid=2320.0,
                        ask=2324.0,
                    ),
                )
            ),
            symbols=("XAUUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
            max_spread_bps_by_symbol={"XAUUSD": 25},
        )

        self.assertEqual(report.overall, DataHealthSeverity.OK)
        self.assertEqual(report.symbols[0].max_allowed_spread_bps, 25)

    def test_spread_percentiles_and_breach_fraction_are_reported(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5, 10]),
            quotes=QuoteHistory(
                (
                    _quote_with_spread(0, 1.0),
                    _quote_with_spread(5, 2.0),
                    _quote_with_spread(10, 20.0),
                )
            ),
            symbols=("EURUSD",),
            max_gap_seconds=300,
            max_spread_bps=10,
        )

        symbol = report.symbols[0]
        self.assertEqual(report.overall, DataHealthSeverity.WARN)
        self.assertAlmostEqual(symbol.median_spread_bps, 2.0)
        self.assertAlmostEqual(symbol.p95_spread_bps, 20.0)
        self.assertAlmostEqual(symbol.p99_spread_bps, 20.0)
        self.assertAlmostEqual(symbol.spread_limit_breach_fraction, 1 / 3)

    def test_write_market_data_health_csv(self) -> None:
        report = validate_market_data(
            prices=_prices([0, 5]),
            quotes=_quotes([0, 5]),
            symbols=("EURUSD",),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "health.csv"
            write_market_data_health_csv(report, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("symbol,status,issue_count,issue_details,price_count", text)
        self.assertIn("EURUSD,OK,0,,2,2", text)
        self.assertIn("median_spread_bps", text)
        self.assertIn("spread_limit_breach_fraction", text)


def _time(minute_offset: int) -> datetime:
    return datetime(2026, 6, 22, 10, minute_offset, tzinfo=LONDON)


def _prices(minute_offsets: list[int]) -> PriceHistory:
    return PriceHistory(
        tuple(
            PriceBar(timestamp=_time(minute), symbol="EURUSD", close=1.1)
            for minute in minute_offsets
        )
    )


def _quotes(minute_offsets: list[int]) -> QuoteHistory:
    return QuoteHistory(
        tuple(
            QuoteSnapshot(
                timestamp=_time(minute),
                symbol="EURUSD",
                bid=1.09995,
                ask=1.10005,
            )
            for minute in minute_offsets
        )
    )


def _quote_with_spread(minute_offset: int, spread_bps: float) -> QuoteSnapshot:
    half_spread = spread_bps / 20_000
    return QuoteSnapshot(
        timestamp=_time(minute_offset),
        symbol="EURUSD",
        bid=1.0 - half_spread,
        ask=1.0 + half_spread,
    )
