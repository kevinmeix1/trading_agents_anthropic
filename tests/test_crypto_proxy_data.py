from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.market.crypto_proxy_data import fetch_crypto_proxy_to_csv
from quanthack.market.market_data import load_price_history, load_quote_history


class CryptoProxyDataTest(TestCase):
    def test_fetch_crypto_proxy_to_csv_writes_prices_and_quotes(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=UTC)
        end = start + timedelta(minutes=45)
        calls: list[tuple[str, int]] = []
        returned = False

        def fake_request(url, params, timeout_seconds):
            nonlocal returned
            calls.append((str(params["symbol"]), int(params["startTime"])))
            if returned:
                return []
            returned = True
            return [
                [_millis(start), "1", "1", "1", "100.0"],
                [_millis(start + timedelta(minutes=15)), "1", "1", "1", "101.0"],
                [_millis(start + timedelta(minutes=30)), "1", "1", "1", "102.0"],
            ]

        with TemporaryDirectory() as tmpdir:
            price_output = Path(tmpdir) / "prices.csv"
            quote_output = Path(tmpdir) / "quotes.csv"

            summary = fetch_crypto_proxy_to_csv(
                symbols=("BTCUSD",),
                price_output=price_output,
                quote_output=quote_output,
                start=start,
                end=end,
                request_json=fake_request,
            )
            prices = load_price_history(price_output)
            quotes = load_quote_history(quote_output)

        self.assertEqual(summary.symbols, ("BTCUSD",))
        self.assertEqual(summary.source_symbols, ("BTCUSDT",))
        self.assertEqual(summary.bars_written, 3)
        self.assertEqual(prices.close_prices(symbol="BTCUSD"), [100.0, 101.0, 102.0])
        self.assertEqual(len(quotes.for_symbol("BTCUSD").quotes), 3)
        self.assertGreater(quotes.for_symbol("BTCUSD").quotes[0].ask, 100.0)
        self.assertLess(quotes.for_symbol("BTCUSD").quotes[0].bid, 100.0)
        self.assertEqual(calls[0][0], "BTCUSDT")

    def test_fetch_crypto_proxy_rejects_unknown_proxy_symbol(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=UTC)
        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "no proxy source symbol"):
                fetch_crypto_proxy_to_csv(
                    symbols=("XAUUSD",),
                    price_output=Path(tmpdir) / "prices.csv",
                    quote_output=Path(tmpdir) / "quotes.csv",
                    start=start,
                    end=start + timedelta(hours=1),
                    request_json=lambda url, params, timeout: [],
                )


def _millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)
