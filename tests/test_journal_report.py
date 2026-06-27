from unittest import TestCase

from quanthack.reporting.journal_report import summarize_journal


class JournalReportTest(TestCase):
    def test_empty_journal_summary(self) -> None:
        summary = summarize_journal([])

        self.assertEqual(summary.total_records, 0)
        self.assertEqual(summary.accepted_rate, 0.0)
        self.assertEqual(summary.trimmed_notional_usd, 0.0)
        self.assertEqual(summary.by_symbol, ())

    def test_summarizes_records(self) -> None:
        records = [
            {
                "status": "DRY_RUN_ACCEPTED",
                "mode": "QUALIFY",
                "request": {
                    "symbol": "EURUSD",
                    "target_notional_usd": 50_000,
                },
                "decision": {
                    "approved": True,
                    "adjusted_notional_usd": 50_000,
                },
            },
            {
                "status": "DRY_RUN_ACCEPTED",
                "mode": "CHECKPOINT_PROTECT",
                "request": {
                    "symbol": "EURUSD",
                    "target_notional_usd": 50_000,
                },
                "decision": {
                    "approved": True,
                    "adjusted_notional_usd": 25_000,
                },
            },
            {
                "status": "DRY_RUN_BLOCKED",
                "mode": "QUALIFY",
                "request": {
                    "symbol": "BTCUSD",
                    "target_notional_usd": 50_000,
                },
                "decision": {
                    "approved": False,
                    "adjusted_notional_usd": 0,
                },
            },
        ]

        summary = summarize_journal(records)

        self.assertEqual(summary.total_records, 3)
        self.assertEqual(summary.accepted, 2)
        self.assertEqual(summary.blocked, 1)
        self.assertAlmostEqual(summary.accepted_rate, 2 / 3)
        self.assertEqual(summary.requested_notional_usd, 150_000)
        self.assertEqual(summary.adjusted_notional_usd, 75_000)
        self.assertEqual(summary.trimmed_notional_usd, 75_000)
        self.assertEqual(summary.by_status["DRY_RUN_ACCEPTED"], 2)
        self.assertEqual(summary.by_mode["QUALIFY"], 2)

        eurusd = [row for row in summary.by_symbol if row.symbol == "EURUSD"][0]
        self.assertEqual(eurusd.count, 2)
        self.assertEqual(eurusd.trimmed_notional_usd, 25_000)

    def test_missing_values_do_not_crash_summary(self) -> None:
        summary = summarize_journal([{}])

        self.assertEqual(summary.total_records, 1)
        self.assertEqual(summary.by_status["UNKNOWN"], 1)
        self.assertEqual(summary.by_symbol[0].symbol, "UNKNOWN")

