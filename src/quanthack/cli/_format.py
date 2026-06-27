"""Small formatting helpers shared by CLI modules."""

from __future__ import annotations


def money(value: float, *, cents: bool = True) -> str:
    if cents:
        return f"${value:,.2f}"
    return f"${value:,.0f}"
