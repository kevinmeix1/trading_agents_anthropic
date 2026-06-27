from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.backtest import BacktestResult


@dataclass(frozen=True)
class RouterAttributionRow:
    primary_signal: str
    fills: int
    realized_events: int
    wins: int
    losses: int
    realized_pnl_usd: float
    turnover_notional_usd: float
    requested_notional_usd: float
    adjusted_notional_usd: float
    conflict_fills: int

    @property
    def win_rate(self) -> float:
        if self.realized_events == 0:
            return 0.0
        return self.wins / self.realized_events

    @property
    def average_adjusted_notional_usd(self) -> float:
        if self.fills == 0:
            return 0.0
        return self.adjusted_notional_usd / self.fills


@dataclass(frozen=True)
class RouterAttributionReport:
    symbol: str
    rows: tuple[RouterAttributionRow, ...]
    total_fills: int
    total_realized_pnl_usd: float
    total_turnover_notional_usd: float
    conflict_fills: int

    @property
    def best_row(self) -> RouterAttributionRow | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda row: row.realized_pnl_usd)


def build_router_attribution_report(result: BacktestResult) -> RouterAttributionReport:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for index, event in enumerate(result.pnl_ledger.events):
        fill = result.fills[index]
        signal = fill.primary_signal or "unknown"
        grouped.setdefault(signal, []).append((index, event.realized_pnl_usd))

    rows: list[RouterAttributionRow] = []
    for signal, indexed_events in grouped.items():
        indexes = [index for index, _ in indexed_events]
        realized_pnls = [pnl for _, pnl in indexed_events if pnl != 0]
        fills = [result.fills[index] for index in indexes]
        rows.append(
            RouterAttributionRow(
                primary_signal=signal,
                fills=len(fills),
                realized_events=len(realized_pnls),
                wins=len([pnl for pnl in realized_pnls if pnl > 0]),
                losses=len([pnl for pnl in realized_pnls if pnl < 0]),
                realized_pnl_usd=sum(realized_pnls),
                turnover_notional_usd=sum(fill.turnover_notional_usd for fill in fills),
                requested_notional_usd=sum(fill.requested_notional_usd for fill in fills),
                adjusted_notional_usd=sum(fill.adjusted_notional_usd for fill in fills),
                conflict_fills=len([fill for fill in fills if fill.conflicting_signals]),
            )
        )

    rows.sort(key=lambda row: row.realized_pnl_usd, reverse=True)
    return RouterAttributionReport(
        symbol=result.symbol,
        rows=tuple(rows),
        total_fills=len(result.fills),
        total_realized_pnl_usd=sum(row.realized_pnl_usd for row in rows),
        total_turnover_notional_usd=sum(row.turnover_notional_usd for row in rows),
        conflict_fills=sum(row.conflict_fills for row in rows),
    )


def write_router_attribution_csv(
    report: RouterAttributionReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "primary_signal",
                "fills",
                "realized_events",
                "wins",
                "losses",
                "win_rate",
                "realized_pnl_usd",
                "turnover_notional_usd",
                "requested_notional_usd",
                "adjusted_notional_usd",
                "average_adjusted_notional_usd",
                "conflict_fills",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "primary_signal": row.primary_signal,
                    "fills": row.fills,
                    "realized_events": row.realized_events,
                    "wins": row.wins,
                    "losses": row.losses,
                    "win_rate": row.win_rate,
                    "realized_pnl_usd": row.realized_pnl_usd,
                    "turnover_notional_usd": row.turnover_notional_usd,
                    "requested_notional_usd": row.requested_notional_usd,
                    "adjusted_notional_usd": row.adjusted_notional_usd,
                    "average_adjusted_notional_usd": row.average_adjusted_notional_usd,
                    "conflict_fills": row.conflict_fills,
                }
            )
