from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.backtest import BacktestFill
from quanthack.backtesting.pnl import build_pnl_ledger, write_pnl_ledger_csv
from quanthack.trading.risk import Side


class PnlLedgerTest(TestCase):
    def test_long_position_realizes_and_marks_open_pnl(self) -> None:
        ledger = build_pnl_ledger(
            (
                _fill(Side.BUY, price=100, units=10),
                _fill(Side.SELL, price=110, units=-4),
            ),
            final_mark_price=105,
        )

        self.assertEqual(ledger.realized_pnl_usd, 40)
        self.assertEqual(ledger.open_pnl_usd, 30)
        self.assertEqual(ledger.total_pnl_usd, 70)
        self.assertEqual(ledger.final_position_units, 6)
        self.assertEqual(ledger.average_entry_price, 100)

    def test_short_position_realizes_profit_when_price_falls(self) -> None:
        ledger = build_pnl_ledger(
            (
                _fill(Side.SELL, price=100, units=-10),
                _fill(Side.BUY, price=90, units=10),
            ),
            final_mark_price=90,
        )

        self.assertEqual(ledger.realized_pnl_usd, 100)
        self.assertEqual(ledger.open_pnl_usd, 0)
        self.assertEqual(ledger.final_position_units, 0)
        self.assertIsNone(ledger.average_entry_price)

    def test_position_flip_realizes_old_side_and_opens_new_side(self) -> None:
        ledger = build_pnl_ledger(
            (
                _fill(Side.BUY, price=100, units=10),
                _fill(Side.SELL, price=110, units=-15),
            ),
            final_mark_price=105,
        )

        self.assertEqual(ledger.realized_pnl_usd, 100)
        self.assertEqual(ledger.open_pnl_usd, 25)
        self.assertEqual(ledger.total_pnl_usd, 125)
        self.assertEqual(ledger.final_position_units, -5)
        self.assertEqual(ledger.average_entry_price, 110)

    def test_write_pnl_ledger_csv(self) -> None:
        ledger = build_pnl_ledger((_fill(Side.BUY, price=100, units=10),))

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pnl.csv"
            write_pnl_ledger_csv(ledger, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("timestamp,symbol,side,fill_price", text)
        self.assertIn("BUY", text)


def _fill(side: Side, *, price: float, units: float) -> BacktestFill:
    return BacktestFill(
        timestamp="2026-06-22T10:00:00+01:00",
        symbol="EURUSD",
        side=side,
        fill_price=price,
        trade_units=units,
        requested_notional_usd=abs(price * units),
        adjusted_notional_usd=abs(price * units),
        risk_reason="approved",
    )
