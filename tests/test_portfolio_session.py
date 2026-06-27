from __future__ import annotations

from datetime import datetime, timezone
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.backtesting.portfolio_allocator import SymbolIntent
from quanthack.backtesting.portfolio_session import (
    PortfolioSessionGate,
    SessionGatePolicy,
)


class PortfolioSessionGateTest(TestCase):
    def test_blocks_new_exposure_outside_allowed_hours(self) -> None:
        gate = PortfolioSessionGate(SessionGatePolicy(allowed_utc_hours=(16, 17)))

        (intent,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                    reason="entry",
                ),
            ),
            timestamp=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(intent.target_notional_usd, 0)
        self.assertEqual(intent.primary_signal, "session_gate")
        self.assertIn("blocked_hour=12", intent.supporting_signals)

    def test_allows_reductions_outside_allowed_hours(self) -> None:
        gate = PortfolioSessionGate(SessionGatePolicy(allowed_utc_hours=(16, 17)))

        (intent,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=20_000,
                    current_notional_usd=100_000,
                    reason="reduce",
                ),
            ),
            timestamp=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(intent.target_notional_usd, 20_000)
        self.assertEqual(intent.primary_signal, "strategy")

    def test_turns_reversal_into_flatten_outside_allowed_hours(self) -> None:
        gate = PortfolioSessionGate(SessionGatePolicy(allowed_utc_hours=(16, 17)))

        (intent,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=-100_000,
                    current_notional_usd=80_000,
                    reason="reverse",
                ),
            ),
            timestamp=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(intent.target_notional_usd, 0)
        self.assertEqual(intent.primary_signal, "session_gate")

    def test_uses_asset_class_specific_hours(self) -> None:
        gate = PortfolioSessionGate(
            SessionGatePolicy(
                allowed_utc_hours=(12,),
                metal_allowed_utc_hours=(16,),
            )
        )

        (intent,) = gate.apply(
            (
                SymbolIntent(
                    symbol="XAUUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                ),
            ),
            timestamp=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(intent.target_notional_usd, 0)

    def test_symbol_specific_hours_override_asset_class_hours(self) -> None:
        gate = PortfolioSessionGate(
            SessionGatePolicy(
                crypto_allowed_utc_hours=(0, 1, 2),
                symbol_allowed_utc_hours={"BTCUSD": (8,)},
            )
        )

        allowed, blocked = gate.apply(
            (
                SymbolIntent(
                    symbol="BTCUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                ),
                SymbolIntent(
                    symbol="ETHUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                ),
            ),
            timestamp=datetime(2026, 6, 1, 8, tzinfo=timezone.utc),
        )

        self.assertEqual(allowed.target_notional_usd, 100_000)
        self.assertEqual(blocked.target_notional_usd, 0)

    def test_rejects_empty_symbol_specific_hours(self) -> None:
        with self.assertRaisesRegex(ValueError, "symbol_allowed_utc_hours"):
            SessionGatePolicy(symbol_allowed_utc_hours={"BTCUSD": ()})

    def test_allowed_hours_are_interpreted_as_utc(self) -> None:
        gate = PortfolioSessionGate(SessionGatePolicy(allowed_utc_hours=(16,)))

        (intent,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                    reason="entry",
                ),
            ),
            timestamp=datetime(2026, 6, 1, 17, tzinfo=ZoneInfo("Europe/London")),
        )

        self.assertEqual(intent.target_notional_usd, 100_000)
        self.assertEqual(intent.primary_signal, "strategy")
