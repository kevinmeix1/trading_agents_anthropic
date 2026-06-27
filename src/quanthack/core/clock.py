from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo


LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")


class CompetitionMode(StrEnum):
    PRE_LIVE = "PRE_LIVE"
    QUALIFY = "QUALIFY"
    CHECKPOINT_PROTECT = "CHECKPOINT_PROTECT"
    FINAL_RANK_PUSH = "FINAL_RANK_PUSH"
    FINAL_SHARPE = "FINAL_SHARPE"


@dataclass(frozen=True)
class CompetitionClock:
    open_at: datetime = datetime(2026, 6, 21, 22, 0, tzinfo=LONDON)
    checkpoints: tuple[datetime, ...] = field(
        default_factory=lambda: (
            datetime(2026, 6, 22, 22, 0, tzinfo=LONDON),
            datetime(2026, 6, 23, 22, 0, tzinfo=LONDON),
            datetime(2026, 6, 24, 22, 0, tzinfo=LONDON),
            datetime(2026, 6, 26, 17, 0, tzinfo=LONDON),
        )
    )
    protect_minutes_before: float = 90.0
    protect_minutes_after: float = 5.0

    def mode_at(
        self,
        now: datetime,
        *,
        finalist: bool = False,
        sharpe_candidate: bool = False,
    ) -> CompetitionMode:
        now_london = self.to_london(now)

        if now_london < self.open_at:
            return CompetitionMode.PRE_LIVE

        if self.in_checkpoint_window(now_london):
            return CompetitionMode.CHECKPOINT_PROTECT

        if finalist and sharpe_candidate:
            return CompetitionMode.FINAL_SHARPE

        if finalist:
            return CompetitionMode.FINAL_RANK_PUSH

        return CompetitionMode.QUALIFY

    def minutes_to_next_checkpoint(self, now: datetime) -> float | None:
        now_london = self.to_london(now)
        next_checkpoint = self.next_checkpoint(now_london)
        if next_checkpoint is None:
            return None
        return (next_checkpoint - now_london).total_seconds() / 60

    def next_checkpoint(self, now: datetime) -> datetime | None:
        now_london = self.to_london(now)
        for checkpoint in self.checkpoints:
            if checkpoint >= now_london:
                return checkpoint
        return None

    def in_checkpoint_window(self, now: datetime) -> bool:
        now_london = self.to_london(now)
        for checkpoint in self.checkpoints:
            minutes_from_checkpoint = abs((checkpoint - now_london).total_seconds() / 60)
            is_before = now_london <= checkpoint
            window = self.protect_minutes_before if is_before else self.protect_minutes_after
            if minutes_from_checkpoint <= window:
                return True
        return False

    @staticmethod
    def to_london(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("competition clock requires a timezone-aware datetime")
        return value.astimezone(LONDON)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)

