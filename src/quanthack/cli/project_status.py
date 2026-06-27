from __future__ import annotations

from collections.abc import Sequence

from quanthack import build_status


def main(argv: Sequence[str] | None = None) -> None:
    del argv
    status = build_status()
    for line in status.summary_lines():
        print(line)
