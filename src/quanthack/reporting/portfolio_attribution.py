from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quanthack.backtesting.portfolio_backtest import PortfolioBacktestResult


@dataclass(frozen=True)
class PortfolioAttributionRow:
    symbol: str
    primary_signal: str
    utc_hour: int
    side: str
    fills: int
    realized_events: int
    wins: int
    losses: int
    realized_pnl_usd: float
    turnover_notional_usd: float
    adjusted_notional_usd: float

    @property
    def win_rate(self) -> float:
        if self.realized_events == 0:
            return 0.0
        return self.wins / self.realized_events


@dataclass(frozen=True)
class PortfolioAttributionReport:
    rows: tuple[PortfolioAttributionRow, ...]
    total_fills: int
    total_realized_pnl_usd: float
    total_turnover_notional_usd: float

    @property
    def weakest_rows(self) -> tuple[PortfolioAttributionRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.realized_pnl_usd))

    @property
    def strongest_rows(self) -> tuple[PortfolioAttributionRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.realized_pnl_usd, reverse=True))


@dataclass
class _OpenLot:
    symbol: str
    primary_signal: str
    utc_hour: int
    side: str
    entry_price: float
    units: float

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.symbol, self.primary_signal, self.utc_hour, self.side)


def build_portfolio_attribution_report(
    result: PortfolioBacktestResult,
) -> PortfolioAttributionReport:
    grouped: dict[tuple[str, str, int, str], dict[str, float | int]] = {}
    for symbol in result.symbols:
        fills = tuple(fill for fill in result.fills if fill.symbol == symbol)
        open_lots: list[_OpenLot] = []
        for fill in fills:
            remaining_units = fill.trade_units
            fill_direction = 1 if fill.trade_units > 0 else -1
            fill_key = (
                fill.symbol,
                fill.primary_signal or "unknown",
                _utc_hour(fill.timestamp),
                fill.side.value,
            )

            if not open_lots or _same_direction(open_lots[0].units, remaining_units):
                _add_open_lot(
                    open_lots,
                    key=fill_key,
                    fill_price=fill.fill_price,
                    units=remaining_units,
                )
                _add_fill_volume(grouped, fill_key, fill.turnover_notional_usd, fill.adjusted_notional_usd)
                continue

            while abs(remaining_units) > 1e-12 and open_lots:
                lot = open_lots[0]
                close_units = min(abs(lot.units), abs(remaining_units))
                lot_direction = 1 if lot.units > 0 else -1
                realized_pnl = close_units * lot_direction * (fill.fill_price - lot.entry_price)
                allocated_turnover = fill.fill_price * close_units
                allocated_notional = fill.adjusted_notional_usd * (
                    close_units / max(abs(fill.trade_units), 1e-12)
                )
                _add_realized_event(
                    grouped,
                    lot.key,
                    realized_pnl=realized_pnl,
                    turnover_notional_usd=allocated_turnover,
                    adjusted_notional_usd=allocated_notional,
                )

                lot.units -= lot_direction * close_units
                remaining_units += lot_direction * close_units
                if abs(lot.units) <= 1e-12:
                    open_lots.pop(0)

            if abs(remaining_units) > 1e-12:
                side = "BUY" if fill_direction > 0 else "SELL"
                _add_open_lot(
                    open_lots,
                    key=(fill.symbol, fill.primary_signal or "unknown", _utc_hour(fill.timestamp), side),
                    fill_price=fill.fill_price,
                    units=remaining_units,
                )
                _add_fill_volume(
                    grouped,
                    fill_key,
                    abs(remaining_units * fill.fill_price),
                    fill.adjusted_notional_usd
                    * (abs(remaining_units) / max(abs(fill.trade_units), 1e-12)),
                )

    rows = tuple(
        PortfolioAttributionRow(
            symbol=symbol,
            primary_signal=primary_signal,
            utc_hour=utc_hour,
            side=side,
            fills=int(values["fills"]),
            realized_events=int(values["realized_events"]),
            wins=int(values["wins"]),
            losses=int(values["losses"]),
            realized_pnl_usd=float(values["realized_pnl_usd"]),
            turnover_notional_usd=float(values["turnover_notional_usd"]),
            adjusted_notional_usd=float(values["adjusted_notional_usd"]),
        )
        for (symbol, primary_signal, utc_hour, side), values in grouped.items()
    )
    ranked = tuple(sorted(rows, key=lambda row: row.realized_pnl_usd, reverse=True))
    return PortfolioAttributionReport(
        rows=ranked,
        total_fills=len(result.fills),
        total_realized_pnl_usd=sum(row.realized_pnl_usd for row in rows),
        total_turnover_notional_usd=sum(row.turnover_notional_usd for row in rows),
    )


def write_portfolio_attribution_csv(
    report: PortfolioAttributionReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "primary_signal",
                "utc_hour",
                "side",
                "fills",
                "realized_events",
                "wins",
                "losses",
                "win_rate",
                "realized_pnl_usd",
                "turnover_notional_usd",
                "adjusted_notional_usd",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "symbol": row.symbol,
                    "primary_signal": row.primary_signal,
                    "utc_hour": row.utc_hour,
                    "side": row.side,
                    "fills": row.fills,
                    "realized_events": row.realized_events,
                    "wins": row.wins,
                    "losses": row.losses,
                    "win_rate": row.win_rate,
                    "realized_pnl_usd": row.realized_pnl_usd,
                    "turnover_notional_usd": row.turnover_notional_usd,
                    "adjusted_notional_usd": row.adjusted_notional_usd,
                }
            )


def _utc_hour(timestamp: str) -> int:
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).hour


def _same_direction(left_units: float, right_units: float) -> bool:
    return (left_units > 0 and right_units > 0) or (left_units < 0 and right_units < 0)


def _add_open_lot(
    open_lots: list[_OpenLot],
    *,
    key: tuple[str, str, int, str],
    fill_price: float,
    units: float,
) -> None:
    symbol, primary_signal, utc_hour, side = key
    open_lots.append(
        _OpenLot(
            symbol=symbol,
            primary_signal=primary_signal,
            utc_hour=utc_hour,
            side=side,
            entry_price=fill_price,
            units=units,
        )
    )


def _add_fill_volume(
    grouped: dict[tuple[str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, int, str],
    turnover_notional_usd: float,
    adjusted_notional_usd: float,
) -> None:
    bucket = _bucket(grouped, key)
    bucket["fills"] = int(bucket["fills"]) + 1
    bucket["turnover_notional_usd"] = (
        float(bucket["turnover_notional_usd"]) + turnover_notional_usd
    )
    bucket["adjusted_notional_usd"] = (
        float(bucket["adjusted_notional_usd"]) + adjusted_notional_usd
    )


def _add_realized_event(
    grouped: dict[tuple[str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, int, str],
    *,
    realized_pnl: float,
    turnover_notional_usd: float,
    adjusted_notional_usd: float,
) -> None:
    bucket = _bucket(grouped, key)
    bucket["fills"] = int(bucket["fills"]) + 1
    bucket["realized_events"] = int(bucket["realized_events"]) + 1
    bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + realized_pnl
    bucket["turnover_notional_usd"] = (
        float(bucket["turnover_notional_usd"]) + turnover_notional_usd
    )
    bucket["adjusted_notional_usd"] = (
        float(bucket["adjusted_notional_usd"]) + adjusted_notional_usd
    )
    if realized_pnl > 0:
        bucket["wins"] = int(bucket["wins"]) + 1
    elif realized_pnl < 0:
        bucket["losses"] = int(bucket["losses"]) + 1


def _bucket(
    grouped: dict[tuple[str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, int, str],
) -> dict[str, float | int]:
    return grouped.setdefault(
        key,
        {
            "fills": 0,
            "realized_events": 0,
            "wins": 0,
            "losses": 0,
            "realized_pnl_usd": 0.0,
            "turnover_notional_usd": 0.0,
            "adjusted_notional_usd": 0.0,
        },
    )
