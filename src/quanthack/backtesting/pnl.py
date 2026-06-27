from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from quanthack.backtesting.metrics import profit_factor_from_pnl
from quanthack.trading.risk import Side


EPSILON_UNITS = 1e-12


class FillLike(Protocol):
    timestamp: str
    symbol: str
    side: Side
    fill_price: float
    trade_units: float


@dataclass(frozen=True)
class PnlEvent:
    timestamp: str
    symbol: str
    side: Side
    fill_price: float
    trade_units: float
    realized_pnl_usd: float
    cumulative_realized_pnl_usd: float
    position_units_after: float
    average_entry_price_after: float | None
    open_pnl_at_fill_usd: float


@dataclass(frozen=True)
class PnlLedger:
    events: tuple[PnlEvent, ...]
    realized_pnl_usd: float
    open_pnl_usd: float
    total_pnl_usd: float
    final_position_units: float
    average_entry_price: float | None
    final_mark_price: float | None

    @property
    def closed_event_count(self) -> int:
        return len([event for event in self.events if event.realized_pnl_usd != 0])

    @property
    def realized_win_rate(self) -> float:
        closed_pnls = [
            event.realized_pnl_usd
            for event in self.events
            if event.realized_pnl_usd != 0
        ]
        if not closed_pnls:
            return 0.0
        wins = [pnl for pnl in closed_pnls if pnl > 0]
        return len(wins) / len(closed_pnls)

    @property
    def realized_profit_factor(self) -> float:
        return profit_factor_from_pnl(
            [
                event.realized_pnl_usd
                for event in self.events
                if event.realized_pnl_usd != 0
            ]
        )


def build_pnl_ledger(
    fills: Sequence[FillLike],
    *,
    final_mark_price: float | None = None,
) -> PnlLedger:
    position_units = 0.0
    average_entry_price: float | None = None
    cumulative_realized_pnl = 0.0
    events: list[PnlEvent] = []

    for fill in fills:
        trade_units = fill.trade_units
        fill_price = fill.fill_price
        realized_pnl = 0.0

        if abs(position_units) <= EPSILON_UNITS:
            position_units = trade_units
            average_entry_price = fill_price if abs(position_units) > EPSILON_UNITS else None
        elif _same_direction(position_units, trade_units):
            average_entry_price = _weighted_average_price(
                current_units=position_units,
                current_average_price=average_entry_price,
                trade_units=trade_units,
                trade_price=fill_price,
            )
            position_units += trade_units
        else:
            close_units = min(abs(position_units), abs(trade_units))
            position_direction = 1 if position_units > 0 else -1
            entry_price = average_entry_price or fill_price
            realized_pnl = close_units * position_direction * (fill_price - entry_price)
            position_units += trade_units

            if abs(position_units) <= EPSILON_UNITS:
                position_units = 0.0
                average_entry_price = None
            elif abs(trade_units) > close_units + EPSILON_UNITS:
                average_entry_price = fill_price

        cumulative_realized_pnl += realized_pnl
        events.append(
            PnlEvent(
                timestamp=fill.timestamp,
                symbol=fill.symbol,
                side=fill.side,
                fill_price=fill_price,
                trade_units=trade_units,
                realized_pnl_usd=realized_pnl,
                cumulative_realized_pnl_usd=cumulative_realized_pnl,
                position_units_after=position_units,
                average_entry_price_after=average_entry_price,
                open_pnl_at_fill_usd=_open_pnl(
                    position_units=position_units,
                    average_entry_price=average_entry_price,
                    mark_price=fill_price,
                ),
            )
        )

    open_pnl = _open_pnl(
        position_units=position_units,
        average_entry_price=average_entry_price,
        mark_price=final_mark_price,
    )
    return PnlLedger(
        events=tuple(events),
        realized_pnl_usd=cumulative_realized_pnl,
        open_pnl_usd=open_pnl,
        total_pnl_usd=cumulative_realized_pnl + open_pnl,
        final_position_units=position_units,
        average_entry_price=average_entry_price,
        final_mark_price=final_mark_price,
    )


def write_pnl_ledger_csv(ledger: PnlLedger, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "symbol",
                "side",
                "fill_price",
                "trade_units",
                "realized_pnl_usd",
                "cumulative_realized_pnl_usd",
                "position_units_after",
                "average_entry_price_after",
                "open_pnl_at_fill_usd",
            ],
        )
        writer.writeheader()
        for event in ledger.events:
            writer.writerow(
                {
                    "timestamp": event.timestamp,
                    "symbol": event.symbol,
                    "side": event.side.value,
                    "fill_price": event.fill_price,
                    "trade_units": event.trade_units,
                    "realized_pnl_usd": event.realized_pnl_usd,
                    "cumulative_realized_pnl_usd": event.cumulative_realized_pnl_usd,
                    "position_units_after": event.position_units_after,
                    "average_entry_price_after": event.average_entry_price_after,
                    "open_pnl_at_fill_usd": event.open_pnl_at_fill_usd,
                }
            )


def _same_direction(left_units: float, right_units: float) -> bool:
    return (left_units > 0 and right_units > 0) or (left_units < 0 and right_units < 0)


def _weighted_average_price(
    *,
    current_units: float,
    current_average_price: float | None,
    trade_units: float,
    trade_price: float,
) -> float:
    current_price = current_average_price or trade_price
    total_units = abs(current_units) + abs(trade_units)
    return (
        (current_price * abs(current_units)) + (trade_price * abs(trade_units))
    ) / total_units


def _open_pnl(
    *,
    position_units: float,
    average_entry_price: float | None,
    mark_price: float | None,
) -> float:
    if (
        abs(position_units) <= EPSILON_UNITS
        or average_entry_price is None
        or mark_price is None
    ):
        return 0.0
    return (mark_price - average_entry_price) * position_units
