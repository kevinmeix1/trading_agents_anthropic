from unittest import TestCase

from quanthack import build_status


class ProjectStatusTest(TestCase):
    def test_project_starts_in_dry_run(self) -> None:
        status = build_status()

        self.assertEqual(status.project_name, "quanthack")
        self.assertEqual(status.timezone, "Europe/London")
        self.assertEqual(status.trading_route, "dry_run")
        self.assertTrue(status.dry_run)

    def test_status_derives_dry_run_from_route(self) -> None:
        status = build_status(trading_route="live")

        self.assertEqual(status.trading_route, "live")
        self.assertFalse(status.dry_run)
