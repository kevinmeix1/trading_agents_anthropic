from unittest import TestCase

from quanthack.core.instruments import (
    AssetClass,
    DEFAULT_INSTRUMENTS,
    instrument_for,
    instruments_by_asset_class,
    normalize_symbol,
)


class InstrumentsTest(TestCase):
    def test_default_instruments_match_syphonix_asset_universe(self) -> None:
        self.assertEqual(len(DEFAULT_INSTRUMENTS), 15)
        self.assertEqual(len(instruments_by_asset_class(AssetClass.FOREX)), 8)
        self.assertEqual(len(instruments_by_asset_class(AssetClass.METAL)), 2)
        self.assertEqual(len(instruments_by_asset_class(AssetClass.CRYPTO)), 5)

    def test_symbol_normalization_accepts_rules_format(self) -> None:
        self.assertEqual(normalize_symbol("BTC/USD"), "BTCUSD")
        self.assertEqual(normalize_symbol("eur-usd"), "EURUSD")
        self.assertEqual(normalize_symbol("xau_usd"), "XAUUSD")

    def test_instrument_lookup_returns_metadata(self) -> None:
        btc = instrument_for("BTC/USD")

        self.assertEqual(btc.symbol, "BTCUSD")
        self.assertEqual(btc.asset_class, AssetClass.CRYPTO)
        self.assertEqual(btc.quote_currency, "USD")
        self.assertGreater(btc.max_spread_bps, 0)

    def test_unknown_instrument_fails_loudly(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown instrument"):
            instrument_for("DOGE/USD")

