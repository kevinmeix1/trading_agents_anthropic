from datetime import datetime
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.core.clock import CompetitionClock, CompetitionMode, LONDON


class CompetitionClockTest(TestCase):
    def setUp(self) -> None:
        self.clock = CompetitionClock()

    def test_pre_live_before_open(self) -> None:
        now = datetime(2026, 6, 21, 21, 59, tzinfo=LONDON)

        self.assertEqual(self.clock.mode_at(now), CompetitionMode.PRE_LIVE)

    def test_qualify_after_open_outside_checkpoint(self) -> None:
        now = datetime(2026, 6, 22, 12, 0, tzinfo=LONDON)

        self.assertEqual(self.clock.mode_at(now), CompetitionMode.QUALIFY)

    def test_checkpoint_protect_before_cut(self) -> None:
        now = datetime(2026, 6, 22, 21, 15, tzinfo=LONDON)

        self.assertEqual(self.clock.mode_at(now), CompetitionMode.CHECKPOINT_PROTECT)

    def test_minutes_to_next_checkpoint(self) -> None:
        now = datetime(2026, 6, 22, 21, 0, tzinfo=LONDON)

        self.assertEqual(self.clock.minutes_to_next_checkpoint(now), 60.0)

    def test_finalist_modes(self) -> None:
        now = datetime(2026, 6, 25, 12, 0, tzinfo=LONDON)

        self.assertEqual(
            self.clock.mode_at(now, finalist=True),
            CompetitionMode.FINAL_RANK_PUSH,
        )
        self.assertEqual(
            self.clock.mode_at(now, finalist=True, sharpe_candidate=True),
            CompetitionMode.FINAL_SHARPE,
        )

    def test_utc_time_converts_to_london(self) -> None:
        now_utc = datetime(2026, 6, 22, 20, 15, tzinfo=ZoneInfo("UTC"))

        self.assertEqual(self.clock.mode_at(now_utc), CompetitionMode.CHECKPOINT_PROTECT)

    def test_naive_datetime_is_rejected(self) -> None:
        now = datetime(2026, 6, 22, 21, 15)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.clock.mode_at(now)

