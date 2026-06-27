from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.instruments import DEFAULT_INSTRUMENTS, AssetClass, instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class SampleDataTest(TestCase):
    def test_generates_all_official_instruments_by_default(self) -> None:
        data = generate_synthetic_market_data(periods=4, seed=1)

        self.assertEqual(
            data.prices.symbols(),
            sorted(item.symbol for item in DEFAULT_INSTRUMENTS),
        )
        self.assertEqual(len(data.prices.bars), 4 * len(DEFAULT_INSTRUMENTS))
        self.assertEqual(len(data.quotes.quotes), 4 * len(DEFAULT_INSTRUMENTS))

    def test_asset_class_filter_generates_crypto_only(self) -> None:
        data = generate_synthetic_market_data(asset_class=AssetClass.CRYPTO, periods=3)

        self.assertEqual(
            data.prices.symbols(),
            ["BARUSD", "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD"],
        )

    def test_generated_spreads_respect_instrument_metadata(self) -> None:
        data = generate_synthetic_market_data(symbols=("EURUSD", "BTCUSD"), periods=8)

        for quote in data.quotes.quotes:
            self.assertLessEqual(
                quote.spread_bps,
                instrument_for(quote.symbol).max_spread_bps,
            )

    def test_generation_is_deterministic_for_same_seed(self) -> None:
        left = generate_synthetic_market_data(symbols=("EURUSD",), periods=6, seed=42)
        right = generate_synthetic_market_data(symbols=("EURUSD",), periods=6, seed=42)

        self.assertEqual(left.prices.bars, right.prices.bars)
        self.assertEqual(left.quotes.quotes, right.quotes.quotes)

    def test_csv_roundtrip(self) -> None:
        data = generate_synthetic_market_data(symbols=("EURUSD", "XAUUSD"), periods=5)

        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            prices = load_price_history(price_path)
            quotes = load_quote_history(quote_path)

        self.assertEqual(prices.symbols(), ["EURUSD", "XAUUSD"])
        self.assertEqual(quotes.symbols(), ["EURUSD", "XAUUSD"])
        self.assertEqual(len(prices.bars), 10)
        self.assertEqual(len(quotes.quotes), 10)
