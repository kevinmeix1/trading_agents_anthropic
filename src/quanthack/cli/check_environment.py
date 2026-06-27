from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo


def main(argv: Sequence[str] | None = None) -> None:
    del argv
    london = ZoneInfo("Europe/London")
    now_london = datetime.now(tz=london)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"London time: {now_london.isoformat(timespec='seconds')}")

    if sys.version_info < (3, 11):
        raise SystemExit("Status: Python 3.11+ is required")

    print("Status: environment looks ready for step 1")
