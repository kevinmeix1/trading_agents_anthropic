from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolJournalStats:
    symbol: str
    count: int
    accepted: int
    blocked: int
    requested_notional_usd: float
    adjusted_notional_usd: float

    @property
    def trimmed_notional_usd(self) -> float:
        return self.requested_notional_usd - self.adjusted_notional_usd


@dataclass(frozen=True)
class JournalSummary:
    total_records: int
    accepted: int
    blocked: int
    requested_notional_usd: float
    adjusted_notional_usd: float
    by_status: dict[str, int]
    by_mode: dict[str, int]
    by_symbol: tuple[SymbolJournalStats, ...]

    @property
    def trimmed_notional_usd(self) -> float:
        return self.requested_notional_usd - self.adjusted_notional_usd

    @property
    def accepted_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.accepted / self.total_records


def summarize_journal(records: list[dict]) -> JournalSummary:
    by_status: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    symbol_totals: dict[str, dict[str, float | int]] = defaultdict(_empty_symbol_totals)

    total_requested = 0.0
    total_adjusted = 0.0
    accepted = 0
    blocked = 0

    for record in records:
        status = str(record.get("status", "UNKNOWN"))
        mode = str(record.get("mode", "UNKNOWN"))
        request = record.get("request", {})
        decision = record.get("decision", {})

        symbol = str(request.get("symbol", "UNKNOWN"))
        requested = _as_float(request.get("target_notional_usd", 0.0))
        adjusted = _as_float(decision.get("adjusted_notional_usd", 0.0))
        approved = bool(decision.get("approved", False))

        by_status[status] += 1
        by_mode[mode] += 1
        total_requested += requested
        total_adjusted += adjusted

        if approved:
            accepted += 1
        else:
            blocked += 1

        totals = symbol_totals[symbol]
        totals["count"] += 1
        totals["requested"] += requested
        totals["adjusted"] += adjusted
        if approved:
            totals["accepted"] += 1
        else:
            totals["blocked"] += 1

    by_symbol = tuple(
        SymbolJournalStats(
            symbol=symbol,
            count=int(values["count"]),
            accepted=int(values["accepted"]),
            blocked=int(values["blocked"]),
            requested_notional_usd=float(values["requested"]),
            adjusted_notional_usd=float(values["adjusted"]),
        )
        for symbol, values in sorted(symbol_totals.items())
    )

    return JournalSummary(
        total_records=len(records),
        accepted=accepted,
        blocked=blocked,
        requested_notional_usd=total_requested,
        adjusted_notional_usd=total_adjusted,
        by_status=dict(sorted(by_status.items())),
        by_mode=dict(sorted(by_mode.items())),
        by_symbol=by_symbol,
    )


def _empty_symbol_totals() -> dict[str, float | int]:
    return {
        "count": 0,
        "accepted": 0,
        "blocked": 0,
        "requested": 0.0,
        "adjusted": 0.0,
    }


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

