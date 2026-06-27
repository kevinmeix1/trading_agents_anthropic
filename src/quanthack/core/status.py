from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ProjectStatus:
    project_name: str
    timezone: str
    trading_route: str
    dry_run: bool
    checked_at: datetime

    def summary_lines(self) -> list[str]:
        dry_run_text = "on" if self.dry_run else "off"
        return [
            f"Project: {self.project_name}",
            f"Timezone: {self.timezone}",
            f"Trading route: {self.trading_route}",
            f"Dry run: {dry_run_text}",
            f"Checked at: {self.checked_at.isoformat(timespec='seconds')}",
        ]


def build_status(
    *,
    project_name: str = "quanthack",
    timezone: str = "Europe/London",
    trading_route: str = "dry_run",
) -> ProjectStatus:
    dry_run = trading_route == "dry_run"
    return ProjectStatus(
        project_name=project_name,
        timezone=timezone,
        trading_route=trading_route,
        dry_run=dry_run,
        checked_at=datetime.now(tz=ZoneInfo(timezone)),
    )
