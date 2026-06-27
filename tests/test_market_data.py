from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.market.market_data import load_price_history, load_quote_history


class MarketDataTest(TestCase):
    def test_loads_sample_price_history(self) -> None:
        history = load_price_history("data/sample_prices.csv")

        self.assertEqual(history.symbols(), ["EURUSD", "XAUUSD"])
        self.assertEqual(len(history.for_symbol("EURUSD").bars), 5)
        self.assertEqual(history.close_prices(symbol="EURUSD")[-1], 1.1010)

    def test_latest_bar_for_symbol(self) -> None:
        history = load_price_history("data/sample_prices.csv")
        latest = history.latest_bar("XAUUSD")

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.close, 2321.80)

    def test_missing_symbol_returns_empty_history(self) -> None:
        history = load_price_history("data/sample_prices.csv")

        self.assertEqual(history.for_symbol("BTCUSD").bars, ())
        self.assertEqual(history.close_prices(symbol="BTCUSD"), [])
        self.assertIsNone(history.latest_bar("BTCUSD"))

    def test_missing_required_column_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.write_text(
                "timestamp,symbol\n2026-06-22T10:00:00+01:00,EURUSD\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_price_history(csv_path)

    def test_naive_timestamp_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.write_text(
                "timestamp,symbol,close\n2026-06-22T10:00:00,EURUSD,1.1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "timezone"):
                load_price_history(csv_path)

    def test_non_positive_close_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.write_text(
                "timestamp,symbol,close\n2026-06-22T10:00:00+01:00,EURUSD,0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "positive finite"):
                load_price_history(csv_path)

    def test_loads_sample_quote_history(self) -> None:
        quotes = load_quote_history("data/sample_quotes.csv")

        self.assertEqual(quotes.symbols(), ["BTCUSD", "EURUSD", "XAUUSD"])
        eurusd = quotes.latest_quote("EURUSD")
        self.assertIsNotNone(eurusd)
        assert eurusd is not None
        self.assertAlmostEqual(eurusd.spread_bps, 0.9082652134420827)

    def test_quote_ask_below_bid_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad_quotes.csv"
            csv_path.write_text(
                "timestamp,symbol,bid,ask\n2026-06-22T10:00:00+01:00,EURUSD,1.2,1.1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ask"):
                load_quote_history(csv_path)
