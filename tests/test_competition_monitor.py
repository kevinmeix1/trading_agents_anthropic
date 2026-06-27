from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.clock import UTC
from quanthack.trading.competition_monitor import CompetitionMonitor, write_monitor_csv
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot, Position


class CompetitionMonitorTest(TestCase):
    def test_monitor_tracks_competition_metrics_and_risk_discipline(self) -> None:
        monitor = CompetitionMonitor()
        start = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
        portfolio = PortfolioSnapshot(
            positions=(
                Position(symbol="EURUSD", notional_usd=200_000),
                Position(symbol="BTCUSD", notional_usd=-100_000),
            )
        )

        first = monitor.record(
            timestamp=start,
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
            portfolio=portfolio,
            accepted_trade_count=2,
        )
        monitor.record(
            timestamp=start + timedelta(minutes=15),
            account=AccountSnapshot(
                equity=1_002_000,
                day_start_equity=1_000_000,
                peak_equity=1_002_000,
                margin_level_pct=2_000,
            ),
            portfolio=portfolio,
            accepted_trade_count=2,
        )

        report = monitor.report()

        self.assertEqual(first.gross_notional_usd, 300_000)
        self.assertEqual(first.net_notional_usd, 100_000)
        self.assertEqual(first.largest_symbol_notional_usd, 200_000)
        self.assertAlmostEqual(first.leverage, 0.3)
        self.assertAlmostEqual(first.margin_usage, 0.01)
        self.assertAlmostEqual(first.single_symbol_concentration, 2 / 3)
        self.assertAlmostEqual(first.net_directional_exposure, 1 / 3)
        self.assertEqual(report.competition_metrics.trade_count, 2)
        self.assertEqual(report.competition_metrics.return_observations, 1)
        self.assertEqual(report.risk_discipline.score, 100)
        self.assertFalse(report.risk_discipline.compliance_review_required)

    def test_write_monitor_csv_outputs_dashboard_fields(self) -> None:
        monitor = CompetitionMonitor()
        timestamp = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
        monitor.record(
            timestamp=timestamp,
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
            portfolio=PortfolioSnapshot(
                positions=(Position(symbol="EURUSD", notional_usd=50_000),)
            ),
            accepted_trade_count=1,
        )

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "monitor.csv"
            write_monitor_csv(monitor.snapshots, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("timestamp,equity,daily_pnl_pct", text)
        self.assertIn("gross_notional_usd", text)
        self.assertIn("accepted_trade_count", text)
        self.assertIn("50000", text)
