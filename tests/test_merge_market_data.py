from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.market.merge_market_data import merge_market_data_csvs


class MergeMarketDataTest(TestCase):
    def test_merges_inputs_and_crops_to_common_window(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fx_prices = root / "fx_prices.csv"
            fx_quotes = root / "fx_quotes.csv"
            crypto_prices = root / "crypto_prices.csv"
            crypto_quotes = root / "crypto_quotes.csv"
            merged_prices = root / "merged_prices.csv"
            merged_quotes = root / "merged_quotes.csv"
            fx_prices.write_text(
                "\n".join(
                    (
                        "timestamp,symbol,close",
                        "2026-06-01T00:00:00+00:00,EURUSD,1.1000",
                        "2026-06-01T00:15:00+00:00,EURUSD,1.1010",
                        "2026-06-01T00:30:00+00:00,EURUSD,1.1020",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fx_quotes.write_text(
                "\n".join(
                    (
                        "timestamp,symbol,bid,ask",
                        "2026-06-01T00:00:00+00:00,EURUSD,1.0999,1.1001",
                        "2026-06-01T00:15:00+00:00,EURUSD,1.1009,1.1011",
                        "2026-06-01T00:30:00+00:00,EURUSD,1.1019,1.1021",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            crypto_prices.write_text(
                "\n".join(
                    (
                        "timestamp,symbol,close",
                        "2026-06-01T00:15:00+00:00,BTCUSD,100000",
                        "2026-06-01T00:30:00+00:00,BTCUSD,100100",
                        "2026-06-01T00:45:00+00:00,BTCUSD,100200",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            crypto_quotes.write_text(
                "\n".join(
                    (
                        "timestamp,symbol,bid,ask",
                        "2026-06-01T00:15:00+00:00,BTCUSD,99990,100010",
                        "2026-06-01T00:30:00+00:00,BTCUSD,100090,100110",
                        "2026-06-01T00:45:00+00:00,BTCUSD,100190,100210",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            summary = merge_market_data_csvs(
                price_inputs=(fx_prices, crypto_prices),
                quote_inputs=(fx_quotes, crypto_quotes),
                price_output=merged_prices,
                quote_output=merged_quotes,
                symbols=("EURUSD", "BTCUSD"),
                common_window=True,
            )
            prices = load_price_history(merged_prices)
            quotes = load_quote_history(merged_quotes)

        self.assertEqual(summary.symbols, ("BTCUSD", "EURUSD"))
        self.assertEqual(summary.price_rows, 4)
        self.assertEqual(summary.quote_rows, 4)
        self.assertEqual(prices.symbols(), ["BTCUSD", "EURUSD"])
        self.assertEqual(quotes.symbols(), ["BTCUSD", "EURUSD"])
        self.assertEqual(
            [bar.timestamp.isoformat() for bar in prices.for_symbol("EURUSD").bars],
            ["2026-06-01T00:15:00+00:00", "2026-06-01T00:30:00+00:00"],
        )
