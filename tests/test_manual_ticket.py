from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.cli.manual_ticket import build_manual_ticket
from quanthack.core.clock import CompetitionMode
from quanthack.trading.risk import Side


class ManualTicketTest(TestCase):
    def test_eurusd_quote_usd_notional_converts_to_rounded_lots(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ticket = build_manual_ticket(
                symbol="EURUSD",
                broker_symbol=None,
                side=Side.BUY,
                target_notional_usd=50_000,
                price=1.1002,
                equity=1_000_000,
                day_start_equity=1_000_000,
                peak_equity=1_000_000,
                margin_level_pct=2_000,
                mode=CompetitionMode.QUALIFY,
                contract_size=None,
                volume_step=0.01,
                min_volume=0.01,
                quote_usd_rate=None,
                journal=Path(tmpdir) / "journal.jsonl",
            )

        self.assertTrue(ticket.risk_approved)
        self.assertAlmostEqual(ticket.raw_lots, 50_000 / (100_000 * 1.1002))
        self.assertEqual(ticket.rounded_lots, 0.45)

    def test_usdjpy_usd_base_notional_uses_contract_size_directly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ticket = build_manual_ticket(
                symbol="USDJPY",
                broker_symbol=None,
                side=Side.SELL,
                target_notional_usd=50_000,
                price=157.50,
                equity=1_000_000,
                day_start_equity=1_000_000,
                peak_equity=1_000_000,
                margin_level_pct=2_000,
                mode=CompetitionMode.QUALIFY,
                contract_size=None,
                volume_step=0.01,
                min_volume=0.01,
                quote_usd_rate=None,
                journal=Path(tmpdir) / "journal.jsonl",
            )

        self.assertEqual(ticket.rounded_lots, 0.5)
        self.assertAlmostEqual(ticket.rounded_notional_usd, 50_000)

    def test_non_fx_requires_explicit_contract_size(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "contract-size"):
                build_manual_ticket(
                    symbol="XAUUSD",
                    broker_symbol=None,
                    side=Side.BUY,
                    target_notional_usd=50_000,
                    price=2_300,
                    equity=1_000_000,
                    day_start_equity=1_000_000,
                    peak_equity=1_000_000,
                    margin_level_pct=2_000,
                    mode=CompetitionMode.QUALIFY,
                    contract_size=None,
                    volume_step=0.01,
                    min_volume=0.01,
                    quote_usd_rate=None,
                    journal=Path(tmpdir) / "journal.jsonl",
                )
