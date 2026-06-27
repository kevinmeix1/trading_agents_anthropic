from datetime import datetime
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.reporting.html_report import build_journal_html_report


class HtmlReportTest(TestCase):
    def test_builds_html_report_with_summary_values(self) -> None:
        records = [
            {
                "created_at_utc": "2026-06-17T13:00:00+00:00",
                "status": "DRY_RUN_ACCEPTED",
                "mode": "QUALIFY",
                "request": {
                    "side": "BUY",
                    "symbol": "EURUSD",
                    "target_notional_usd": 50_000,
                },
                "decision": {
                    "approved": True,
                    "adjusted_notional_usd": 25_000,
                    "reason": "approved",
                },
            }
        ]

        report = build_journal_html_report(
            records=records,
            generated_at=datetime(2026, 6, 17, 14, 0, tzinfo=ZoneInfo("Europe/London")),
        )

        self.assertIn("<!doctype html>", report.html)
        self.assertIn("QuantHack Dry-Run Journal", report.html)
        self.assertIn("Trimmed By Risk", report.html)
        self.assertIn("$25,000", report.html)
        self.assertIn("EURUSD", report.html)

    def test_escapes_html_values(self) -> None:
        records = [
            {
                "created_at_utc": "<bad>",
                "status": "DRY_RUN_ACCEPTED",
                "mode": "QUALIFY",
                "request": {
                    "side": "BUY",
                    "symbol": "<EURUSD>",
                    "target_notional_usd": 50_000,
                },
                "decision": {
                    "approved": True,
                    "adjusted_notional_usd": 50_000,
                    "reason": "approved",
                },
            }
        ]

        report = build_journal_html_report(records=records)

        self.assertIn("&lt;EURUSD&gt;", report.html)
        self.assertIn("&lt;bad&gt;", report.html)
        self.assertNotIn("<EURUSD>", report.html)

