from datetime import datetime, timedelta
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.market.market_data import QuoteSnapshot
from quanthack.market.market_quality import MarketQualityChecker, MarketQualityLimits


NOW = datetime(2026, 6, 22, 10, 20, tzinfo=ZoneInfo("Europe/London"))


class MarketQualityCheckerTest(TestCase):
    def test_good_quote_passes(self) -> None:
        quote = QuoteSnapshot(timestamp=NOW, symbol="EURUSD", bid=1.10095, ask=1.10105)
        checker = MarketQualityChecker(MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5))

        decision = checker.evaluate(quote=quote, as_of=NOW)

        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "market quality ok")

    def test_stale_quote_blocks(self) -> None:
        quote = QuoteSnapshot(
            timestamp=NOW - timedelta(seconds=10),
            symbol="EURUSD",
            bid=1.10095,
            ask=1.10105,
        )
        checker = MarketQualityChecker(MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5))

        decision = checker.evaluate(quote=quote, as_of=NOW)

        self.assertFalse(decision.ok)
        self.assertIn("stale", decision.reason)

    def test_wide_spread_blocks(self) -> None:
        quote = QuoteSnapshot(timestamp=NOW, symbol="BTCUSD", bid=65_000, ask=65_200)
        checker = MarketQualityChecker(MarketQualityLimits(max_spread_bps=10, max_quote_age_seconds=5))

        decision = checker.evaluate(quote=quote, as_of=NOW)

        self.assertFalse(decision.ok)
        self.assertIn("spread too wide", decision.reason)

    def test_future_quote_blocks(self) -> None:
        quote = QuoteSnapshot(
            timestamp=NOW + timedelta(seconds=1),
            symbol="EURUSD",
            bid=1.10095,
            ask=1.10105,
        )

        decision = MarketQualityChecker().evaluate(quote=quote, as_of=NOW)

        self.assertFalse(decision.ok)
        self.assertIn("after as_of", decision.reason)

    def test_naive_as_of_is_rejected(self) -> None:
        quote = QuoteSnapshot(timestamp=NOW, symbol="EURUSD", bid=1.10095, ask=1.10105)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            MarketQualityChecker().evaluate(
                quote=quote,
                as_of=datetime(2026, 6, 22, 10, 20),
            )
