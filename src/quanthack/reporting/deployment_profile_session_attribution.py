from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quanthack.backtesting.backtest import BacktestFill
from quanthack.backtesting.deployment_profile_backtest import (
    DeploymentProfileBacktestResult,
)
from quanthack.trading.risk import Side


@dataclass(frozen=True)
class DeploymentProfileSessionAttributionRow:
    profile_slot: str
    profile_label: str
    symbol: str
    primary_signal: str
    utc_hour: int
    side: str
    fills: int
    realized_events: int
    wins: int
    losses: int
    realized_pnl_usd: float
    open_pnl_usd: float
    total_pnl_usd: float
    turnover_notional_usd: float
    adjusted_notional_usd: float

    @property
    def win_rate(self) -> float:
        if self.realized_events == 0:
            return 0.0
        return self.wins / self.realized_events


@dataclass(frozen=True)
class DeploymentProfileSessionAttributionReport:
    profile_slot: str
    profile_label: str
    rows: tuple[DeploymentProfileSessionAttributionRow, ...]

    @property
    def total_pnl_usd(self) -> float:
        return sum(row.total_pnl_usd for row in self.rows)

    @property
    def realized_pnl_usd(self) -> float:
        return sum(row.realized_pnl_usd for row in self.rows)

    @property
    def open_pnl_usd(self) -> float:
        return sum(row.open_pnl_usd for row in self.rows)

    @property
    def fills(self) -> int:
        return sum(row.fills for row in self.rows)

    @property
    def strongest_rows(self) -> tuple[DeploymentProfileSessionAttributionRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.total_pnl_usd, reverse=True))

    @property
    def weakest_rows(self) -> tuple[DeploymentProfileSessionAttributionRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.total_pnl_usd))


@dataclass
class _OpenLot:
    profile_slot: str
    profile_label: str
    symbol: str
    primary_signal: str
    utc_hour: int
    side: str
    entry_price: float
    units: float

    @property
    def key(self) -> tuple[str, str, str, str, int, str]:
        return (
            self.profile_slot,
            self.profile_label,
            self.symbol,
            self.primary_signal,
            self.utc_hour,
            self.side,
        )


def build_deployment_profile_session_attribution_report(
    backtest: DeploymentProfileBacktestResult,
) -> DeploymentProfileSessionAttributionReport:
    final_mark_by_symbol = {
        row.symbol: row.ledger.final_mark_price for row in backtest.result.pnl_by_symbol
    }
    rows = build_session_attribution_from_fills(
        fills=backtest.result.fills,
        final_mark_by_symbol=final_mark_by_symbol,
        profile_slot=backtest.profile.slot,
        profile_label=backtest.profile.label,
    )
    return DeploymentProfileSessionAttributionReport(
        profile_slot=backtest.profile.slot,
        profile_label=backtest.profile.label,
        rows=rows,
    )


def build_session_attribution_from_fills(
    *,
    fills: tuple[BacktestFill, ...],
    final_mark_by_symbol: dict[str, float | None],
    profile_slot: str,
    profile_label: str,
) -> tuple[DeploymentProfileSessionAttributionRow, ...]:
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, float | int]] = {}
    open_lots_by_symbol: dict[str, list[_OpenLot]] = {}

    for fill in sorted(fills, key=lambda item: item.timestamp):
        timestamp = _parse_timestamp(fill.timestamp)
        fill_key = (
            profile_slot,
            profile_label,
            fill.symbol,
            fill.primary_signal or "unknown",
            timestamp.astimezone(UTC).hour,
            fill.side.value,
        )
        open_lots = open_lots_by_symbol.setdefault(fill.symbol, [])
        remaining_units = fill.trade_units
        if not open_lots or _same_direction(open_lots[0].units, remaining_units):
            _add_open_lot(
                open_lots,
                profile_slot=profile_slot,
                profile_label=profile_label,
                fill=fill,
                utc_hour=timestamp.astimezone(UTC).hour,
                units=remaining_units,
            )
            _add_fill_volume(
                grouped,
                fill_key,
                turnover_notional_usd=fill.turnover_notional_usd,
                adjusted_notional_usd=fill.adjusted_notional_usd,
            )
            continue

        while abs(remaining_units) > 1e-12 and open_lots:
            lot = open_lots[0]
            close_units = min(abs(lot.units), abs(remaining_units))
            lot_direction = 1 if lot.units > 0 else -1
            realized_pnl = close_units * lot_direction * (
                fill.fill_price - lot.entry_price
            )
            allocated_turnover = fill.fill_price * close_units
            allocated_notional = fill.adjusted_notional_usd * (
                close_units / max(abs(fill.trade_units), 1e-12)
            )
            _add_realized_event(
                grouped,
                lot.key,
                realized_pnl_usd=realized_pnl,
                turnover_notional_usd=allocated_turnover,
                adjusted_notional_usd=allocated_notional,
            )
            lot.units -= lot_direction * close_units
            remaining_units += lot_direction * close_units
            if abs(lot.units) <= 1e-12:
                open_lots.pop(0)

        if abs(remaining_units) > 1e-12:
            _add_open_lot(
                open_lots,
                profile_slot=profile_slot,
                profile_label=profile_label,
                fill=fill,
                utc_hour=timestamp.astimezone(UTC).hour,
                units=remaining_units,
            )
            _add_fill_volume(
                grouped,
                fill_key,
                turnover_notional_usd=abs(remaining_units * fill.fill_price),
                adjusted_notional_usd=fill.adjusted_notional_usd
                * (abs(remaining_units) / max(abs(fill.trade_units), 1e-12)),
            )

    for symbol, open_lots in open_lots_by_symbol.items():
        final_mark = final_mark_by_symbol.get(symbol)
        if final_mark is None:
            continue
        for lot in open_lots:
            open_pnl = (final_mark - lot.entry_price) * lot.units
            _add_open_pnl(grouped, lot.key, open_pnl_usd=open_pnl)

    rows = tuple(
        DeploymentProfileSessionAttributionRow(
            profile_slot=profile_slot,
            profile_label=profile_label,
            symbol=symbol,
            primary_signal=primary_signal,
            utc_hour=utc_hour,
            side=side,
            fills=int(values["fills"]),
            realized_events=int(values["realized_events"]),
            wins=int(values["wins"]),
            losses=int(values["losses"]),
            realized_pnl_usd=float(values["realized_pnl_usd"]),
            open_pnl_usd=float(values["open_pnl_usd"]),
            total_pnl_usd=float(values["realized_pnl_usd"])
            + float(values["open_pnl_usd"]),
            turnover_notional_usd=float(values["turnover_notional_usd"]),
            adjusted_notional_usd=float(values["adjusted_notional_usd"]),
        )
        for (
            profile_slot,
            profile_label,
            symbol,
            primary_signal,
            utc_hour,
            side,
        ), values in grouped.items()
    )
    return tuple(
        sorted(rows, key=lambda row: (row.utc_hour, row.symbol, row.primary_signal, row.side))
    )


def write_deployment_profile_session_attribution_csv(
    report: DeploymentProfileSessionAttributionReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_slot",
                "profile_label",
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
                "open_pnl_usd",
                "total_pnl_usd",
                "turnover_notional_usd",
                "adjusted_notional_usd",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "profile_slot": row.profile_slot,
                    "profile_label": row.profile_label,
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
                    "open_pnl_usd": row.open_pnl_usd,
                    "total_pnl_usd": row.total_pnl_usd,
                    "turnover_notional_usd": row.turnover_notional_usd,
                    "adjusted_notional_usd": row.adjusted_notional_usd,
                }
            )


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _same_direction(left_units: float, right_units: float) -> bool:
    return (left_units > 0 and right_units > 0) or (
        left_units < 0 and right_units < 0
    )


def _add_open_lot(
    open_lots: list[_OpenLot],
    *,
    profile_slot: str,
    profile_label: str,
    fill: BacktestFill,
    utc_hour: int,
    units: float,
) -> None:
    side = Side.BUY.value if units > 0 else Side.SELL.value
    open_lots.append(
        _OpenLot(
            profile_slot=profile_slot,
            profile_label=profile_label,
            symbol=fill.symbol,
            primary_signal=fill.primary_signal or "unknown",
            utc_hour=utc_hour,
            side=side,
            entry_price=fill.fill_price,
            units=units,
        )
    )


def _add_fill_volume(
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, str, str, int, str],
    *,
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
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, str, str, int, str],
    *,
    realized_pnl_usd: float,
    turnover_notional_usd: float,
    adjusted_notional_usd: float,
) -> None:
    bucket = _bucket(grouped, key)
    bucket["fills"] = int(bucket["fills"]) + 1
    bucket["realized_events"] = int(bucket["realized_events"]) + 1
    bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + realized_pnl_usd
    bucket["turnover_notional_usd"] = (
        float(bucket["turnover_notional_usd"]) + turnover_notional_usd
    )
    bucket["adjusted_notional_usd"] = (
        float(bucket["adjusted_notional_usd"]) + adjusted_notional_usd
    )
    if realized_pnl_usd > 0:
        bucket["wins"] = int(bucket["wins"]) + 1
    elif realized_pnl_usd < 0:
        bucket["losses"] = int(bucket["losses"]) + 1


def _add_open_pnl(
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, str, str, int, str],
    *,
    open_pnl_usd: float,
) -> None:
    bucket = _bucket(grouped, key)
    bucket["open_pnl_usd"] = float(bucket["open_pnl_usd"]) + open_pnl_usd


def _bucket(
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, float | int]],
    key: tuple[str, str, str, str, int, str],
) -> dict[str, float | int]:
    return grouped.setdefault(
        key,
        {
            "fills": 0,
            "realized_events": 0,
            "wins": 0,
            "losses": 0,
            "realized_pnl_usd": 0.0,
            "open_pnl_usd": 0.0,
            "turnover_notional_usd": 0.0,
            "adjusted_notional_usd": 0.0,
        },
    )
