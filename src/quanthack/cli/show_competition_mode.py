from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.cli._time import parse_datetime
from quanthack.core.clock import CompetitionClock, utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the current hackathon competition mode.")
    parser.add_argument(
        "--at",
        type=parse_datetime,
        default=None,
        help="Optional timezone-aware datetime, e.g. 2026-06-22T21:15:00+01:00",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    now = args.at or utc_now()
    clock = CompetitionClock()
    london_now = clock.to_london(now)
    next_checkpoint = clock.next_checkpoint(now)
    minutes = clock.minutes_to_next_checkpoint(now)
    mode = clock.mode_at(now)

    print(f"London time: {london_now.isoformat(timespec='seconds')}")
    print(f"Mode: {mode.value}")

    if next_checkpoint is None:
        print("Next checkpoint: none")
        return

    print(f"Next checkpoint: {next_checkpoint.isoformat(timespec='seconds')}")
    print(f"Minutes to checkpoint: {minutes:.1f}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
