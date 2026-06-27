from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.backtest import BacktestFill
from quanthack.backtesting.portfolio_allocator import SymbolIntent
from quanthack.backtesting.portfolio_symbol_evidence import (
    PortfolioSymbolEvidenceGate,
    SymbolEvidenceGatePolicy,
    write_symbol_evidence_gate_report_csv,
)
from quanthack.trading.risk import Side


class PortfolioSymbolEvidenceGateTest(TestCase):
    def test_blocks_new_exposure_after_recent_losing_close(self) -> None:
        gate = PortfolioSymbolEvidenceGate(
            SymbolEvidenceGatePolicy(
                lookback_closed_events=1,
                min_realized_pnl_usd=0.0,
            )
        )
        gate.observe_fill(
            BacktestFill(
                timestamp="2026-06-01T10:00:00+00:00",
                symbol="EURUSD",
                side=Side.BUY,
                fill_price=1.0,
                trade_units=100_000,
                requested_notional_usd=100_000,
                adjusted_notional_usd=100_000,
                risk_reason="entry",
            )
        )
        realized = gate.observe_fill(
            BacktestFill(
                timestamp="2026-06-01T10:15:00+00:00",
                symbol="EURUSD",
                side=Side.SELL,
                fill_price=0.99,
                trade_units=-100_000,
                requested_notional_usd=0,
                adjusted_notional_usd=0,
                risk_reason="exit",
            )
        )

        (intent,), (report,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                    reason="new entry",
                ),
            ),
            timestamp="2026-06-01T10:30:00+00:00",
        )

        self.assertAlmostEqual(realized, -1_000)
        self.assertEqual(intent.target_notional_usd, 0)
        self.assertEqual(intent.primary_signal, "symbol_evidence_gate")
        self.assertTrue(report.applied)
        self.assertFalse(report.allowed)
        self.assertAlmostEqual(report.prior_realized_pnl_usd, -1_000)

    def test_allows_reductions_even_when_recent_evidence_is_bad(self) -> None:
        gate = PortfolioSymbolEvidenceGate(SymbolEvidenceGatePolicy())
        gate.observe_fill(
            BacktestFill(
                timestamp="2026-06-01T10:00:00+00:00",
                symbol="EURUSD",
                side=Side.BUY,
                fill_price=1.0,
                trade_units=100_000,
                requested_notional_usd=100_000,
                adjusted_notional_usd=100_000,
                risk_reason="entry",
            )
        )
        gate.observe_fill(
            BacktestFill(
                timestamp="2026-06-01T10:15:00+00:00",
                symbol="EURUSD",
                side=Side.SELL,
                fill_price=0.99,
                trade_units=-100_000,
                requested_notional_usd=0,
                adjusted_notional_usd=0,
                risk_reason="exit",
            )
        )

        (intent,), (report,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=20_000,
                    current_notional_usd=100_000,
                    reason="reduce",
                ),
            ),
            timestamp="2026-06-01T10:30:00+00:00",
        )

        self.assertEqual(intent.target_notional_usd, 20_000)
        self.assertFalse(report.allowed)
        self.assertFalse(report.applied)

    def test_stale_evidence_allows_new_probe(self) -> None:
        gate = PortfolioSymbolEvidenceGate(
            SymbolEvidenceGatePolicy(
                min_realized_pnl_usd=0.0,
                stale_after_bars=2,
            )
        )
        gate.observe_fill(
            BacktestFill(
                timestamp="2026-06-01T10:00:00+00:00",
                symbol="EURUSD",
                side=Side.BUY,
                fill_price=1.0,
                trade_units=100_000,
                requested_notional_usd=100_000,
                adjusted_notional_usd=100_000,
                risk_reason="entry",
            )
        )
        gate.observe_fill(
            BacktestFill(
                timestamp="2026-06-01T10:15:00+00:00",
                symbol="EURUSD",
                side=Side.SELL,
                fill_price=0.99,
                trade_units=-100_000,
                requested_notional_usd=0,
                adjusted_notional_usd=0,
                risk_reason="exit",
            )
        )

        (blocked,), _ = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                ),
            ),
            timestamp="2026-06-01T10:30:00+00:00",
        )
        gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=0,
                    current_notional_usd=0,
                ),
            ),
            timestamp="2026-06-01T10:45:00+00:00",
        )
        (probe,), (report,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                ),
            ),
            timestamp="2026-06-01T11:00:00+00:00",
        )

        self.assertEqual(blocked.target_notional_usd, 0)
        self.assertEqual(probe.target_notional_usd, 100_000)
        self.assertTrue(report.allowed)
        self.assertEqual(report.reason, "allowed: no prior closed-trade evidence")

    def test_target_symbols_leave_other_symbols_untouched(self) -> None:
        gate = PortfolioSymbolEvidenceGate(
            SymbolEvidenceGatePolicy(
                target_symbols=("USDCAD",),
                allow_without_history=False,
            )
        )

        (intent,), (report,) = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                    reason="untargeted entry",
                ),
            ),
            timestamp="2026-06-01T10:00:00+00:00",
        )

        self.assertEqual(intent.target_notional_usd, 100_000)
        self.assertTrue(report.allowed)
        self.assertFalse(report.applied)
        self.assertEqual(report.reason, "allowed: symbol not targeted by evidence gate")

    def test_no_history_probe_reduces_targeted_new_exposure(self) -> None:
        gate = PortfolioSymbolEvidenceGate(
            SymbolEvidenceGatePolicy(
                target_symbols=("USDCAD",),
                allow_without_history=False,
                no_history_target_multiplier=0.25,
            )
        )

        (intent,), (report,) = gate.apply(
            (
                SymbolIntent(
                    symbol="USDCAD",
                    target_notional_usd=200_000,
                    current_notional_usd=0,
                    reason="probe entry",
                ),
            ),
            timestamp="2026-06-01T10:00:00+00:00",
        )

        self.assertEqual(intent.target_notional_usd, 50_000)
        self.assertEqual(intent.primary_signal, "symbol_evidence_gate")
        self.assertFalse(report.allowed)
        self.assertTrue(report.applied)
        self.assertEqual(report.requested_after_usd, 50_000)

    def test_writes_report_csv(self) -> None:
        gate = PortfolioSymbolEvidenceGate(SymbolEvidenceGatePolicy())
        _, reports = gate.apply(
            (
                SymbolIntent(
                    symbol="EURUSD",
                    target_notional_usd=100_000,
                    current_notional_usd=0,
                ),
            ),
            timestamp="2026-06-01T10:00:00+00:00",
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "symbol_evidence.csv"
            write_symbol_evidence_gate_report_csv(reports, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("timestamp,symbol,primary_signal", text)
        self.assertIn("EURUSD", text)
