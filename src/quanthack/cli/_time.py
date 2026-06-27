"""Datetime parsing helpers for command-line tools."""

from __future__ import annotations

import argparse
from datetime import datetime


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("include a timezone, for example +01:00 or Z")
    return parsed
