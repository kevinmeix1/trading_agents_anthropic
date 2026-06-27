from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.clock import UTC
from quanthack.cli.mt5_capture import capture_mt5_data
from quanthack.market.adapters import (
    CsvMarketDataAdapter,
    MT5AccountAdapter,
    MT5ConnectionSettings,
    MT5MarketDataAdapter,
    StaticAccountAdapter,
    parse_symbol_map,
)
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class MarketAdapterTest(TestCase):
    def test_csv_adapter_reads_latest_quotes_and_recent_bars(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            data = generate_synthetic_market_data(
                symbols=("EURUSD", "BTCUSD"),
                periods=6,
                seed=11,
            )
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            adapter = CsvMarketDataAdapter(
                price_csv=str(price_path),
                quote_csv=str(quote_path),
            )

            self.assertEqual(adapter.supported_symbols(), ("BTCUSD", "EURUSD"))
            quote = adapter.get_latest_quote("EURUSD")
            bars = adapter.get_recent_bars("EURUSD", timeframe="M1", count=3)

        self.assertEqual(quote.symbol, "EURUSD")
        self.assertEqual(len(bars), 3)
        self.assertLess(bars[0].timestamp, bars[-1].timestamp)

    def test_static_account_adapter_uses_live_equity_context(self) -> None:
        adapter = StaticAccountAdapter(equity=998_000, margin_level_pct=1_500)

        account = adapter.get_account_snapshot(
            starting_equity=1_000_000,
            day_start_equity=1_010_000,
            peak_equity=1_025_000,
        )

        self.assertEqual(account.equity, 998_000)
        self.assertAlmostEqual(account.daily_pnl_pct, -0.0118811881)
        self.assertEqual(account.peak_equity, 1_025_000)
        self.assertEqual(account.margin_level_pct, 1_500)

    def test_mt5_adapter_converts_ticks_bars_and_account_info(self) -> None:
        fake_mt5 = _FakeMT5()
        settings = MT5ConnectionSettings(
            symbol_map=parse_symbol_map(("EURUSD=EURUSD.pro",)),
        )
        adapter = MT5MarketDataAdapter(settings=settings, mt5_module=fake_mt5)
        account_adapter = MT5AccountAdapter(adapter)

        self.assertEqual(adapter.supported_symbols(), ("EURUSD",))
        quote = adapter.get_latest_quote("EURUSD")
        bars = adapter.get_recent_bars("EURUSD", timeframe="M1", count=2)
        account = account_adapter.get_account_snapshot(
            starting_equity=1_000_000,
            day_start_equity=1_000_000,
            peak_equity=1_000_000,
        )
        adapter.close()

        self.assertTrue(fake_mt5.initialize_called)
        self.assertTrue(fake_mt5.shutdown_called)
        self.assertEqual(fake_mt5.last_tick_symbol, "EURUSD.pro")
        self.assertEqual(quote.symbol, "EURUSD")
        self.assertEqual(quote.bid, 1.1)
        self.assertEqual(quote.ask, 1.1002)
        self.assertEqual(
            quote.timestamp,
            datetime.fromtimestamp(1_803_000_000.123, tz=UTC),
        )
        self.assertEqual([bar.close for bar in bars], [1.101, 1.102])
        self.assertEqual(account.equity, 999_000)
        self.assertEqual(account.margin_level_pct, 1_250)

    def test_mt5_account_adapter_treats_zero_margin_level_as_unknown(self) -> None:
        fake_mt5 = _FakeMT5()
        fake_mt5.account_info_response = {"equity": 1_000_000, "margin_level": 0}
        adapter = MT5MarketDataAdapter(mt5_module=fake_mt5)

        account = MT5AccountAdapter(adapter).get_account_snapshot(
            starting_equity=1_000_000,
            day_start_equity=1_000_000,
            peak_equity=1_000_000,
        )

        self.assertIsNone(account.margin_level_pct)

    def test_mt5_adapter_logs_in_when_credentials_are_provided(self) -> None:
        fake_mt5 = _FakeMT5()
        settings = MT5ConnectionSettings(
            login=10344,
            password="secret",
            server="demo-server",
            timeout_ms=45_000,
        )
        adapter = MT5MarketDataAdapter(settings=settings, mt5_module=fake_mt5)

        adapter.get_latest_quote("EURUSD")

        self.assertTrue(fake_mt5.login_called)
        self.assertEqual(fake_mt5.login_value, 10344)
        self.assertEqual(fake_mt5.login_kwargs["password"], "secret")
        self.assertEqual(fake_mt5.login_kwargs["server"], "demo-server")
        self.assertEqual(fake_mt5.login_kwargs["timeout"], 45_000)

    def test_mt5_adapter_rejects_unknown_timeframe(self) -> None:
        adapter = MT5MarketDataAdapter(mt5_module=_FakeMT5())

        with self.assertRaisesRegex(ValueError, "unsupported MT5 timeframe"):
            adapter.get_recent_bars("EURUSD", timeframe="TICK", count=2)

    def test_mt5_capture_writes_read_only_quote_account_and_bar_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            adapter = MT5MarketDataAdapter(mt5_module=_FakeMT5())
            result = capture_mt5_data(
                adapter=adapter,
                account_adapter=MT5AccountAdapter(adapter),
                symbols=("EURUSD",),
                timeframe="M1",
                bars=2,
                iterations=2,
                poll_seconds=0.0,
                starting_equity=1_000_000,
                quotes_output=tmp_path / "quotes.csv",
                bars_output=tmp_path / "bars.csv",
                account_output=tmp_path / "account.csv",
            )

            quote_text = (tmp_path / "quotes.csv").read_text(encoding="utf-8")
            bar_text = (tmp_path / "bars.csv").read_text(encoding="utf-8")
            account_text = (tmp_path / "account.csv").read_text(encoding="utf-8")

        self.assertEqual(result.quote_rows, 2)
        self.assertEqual(result.bar_rows, 4)
        self.assertEqual(result.account_rows, 2)
        self.assertIn("quote_timestamp,symbol,bid,ask,mid,spread_bps", quote_text)
        self.assertIn("bar_timestamp,symbol,timeframe,close", bar_text)
        self.assertIn("equity,daily_pnl_pct,drawdown_pct,margin_level_pct", account_text)


class _FakeMT5:
    TIMEFRAME_M1 = 1

    def __init__(self) -> None:
        self.initialize_called = False
        self.shutdown_called = False
        self.login_called = False
        self.login_value = 0
        self.login_kwargs: dict[str, object] = {}
        self.last_tick_symbol = ""
        self.account_info_response = {"equity": 999_000, "margin_level": 1_250}

    def initialize(self, *args, **kwargs) -> bool:
        self.initialize_called = True
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def login(self, login: int, **kwargs) -> bool:
        self.login_called = True
        self.login_value = login
        self.login_kwargs = kwargs
        return True

    def last_error(self) -> tuple[int, str]:
        return (0, "ok")

    def symbol_info_tick(self, symbol: str) -> dict[str, float]:
        self.last_tick_symbol = symbol
        return {
            "bid": 1.1000,
            "ask": 1.1002,
            "time_msc": 1_803_000_000_123,
        }

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe,
        start_pos: int,
        count: int,
    ) -> tuple[dict[str, float], ...]:
        return (
            {"time": 1_803_000_060, "close": 1.1020},
            {"time": 1_803_000_000, "close": 1.1010},
        )

    def account_info(self) -> dict[str, float]:
        return self.account_info_response
