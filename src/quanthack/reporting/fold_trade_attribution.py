from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class FoldWindow:
    fold: int
    test_start: datetime
    test_end: datetime
    return_pct: float


@dataclass(frozen=True)
class CsvFill:
    timestamp: datetime
    symbol: str
    side: str
    fill_price: float
    trade_units: float
    turnover_notional_usd: float
    adjusted_notional_usd: float
    primary_signal: str


@dataclass(frozen=True)
class FoldTradeAttributionRow:
    fold: int
    fold_return_pct: float
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
class FoldTradeAttributionReport:
    rows: tuple[FoldTradeAttributionRow, ...]

    @property
    def weakest_rows(self) -> tuple[FoldTradeAttributionRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.realized_pnl_usd))

    @property
    def strongest_rows(self) -> tuple[FoldTradeAttributionRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.realized_pnl_usd, reverse=True))


@dataclass
class _OpenLot:
    fold: int
    fold_return_pct: float
    symbol: str
    primary_signal: str
    utc_hour: int
    side: str
    entry_price: float
    units: float

    @property
    def key(self) -> tuple[int, float, str, str, int, str]:
        return (
            self.fold,
            self.fold_return_pct,
            self.symbol,
            self.primary_signal,
            self.utc_hour,
            self.side,
        )


def build_fold_trade_attribution_report(
    *,
    fills_csv: str | Path,
    folds_csv: str | Path,
) -> FoldTradeAttributionReport:
    folds = _read_folds(folds_csv)
    fills = _read_fills(fills_csv)
    grouped: dict[tuple[int, float, str, str, int, str], dict[str, float | int]] = {}
    open_lots_by_symbol: dict[str, list[_OpenLot]] = {}

    for fill in fills:
        fold = _fold_for_timestamp(fill.timestamp, folds)
        if fold is None:
            continue
        open_lots = open_lots_by_symbol.setdefault(fill.symbol, [])
        remaining_units = fill.trade_units
        fill_direction = 1 if fill.trade_units > 0 else -1
        fill_key = (
            fold.fold,
            fold.return_pct,
            fill.symbol,
            fill.primary_signal or "unknown",
            fill.timestamp.astimezone(UTC).hour,
            fill.side,
        )

        if not open_lots or _same_direction(open_lots[0].units, remaining_units):
            _add_open_lot(
                open_lots,
                fold=fold,
                fill=fill,
                units=remaining_units,
            )
            _add_fill_volume(
                grouped,
                fill_key,
                fill.turnover_notional_usd,
                fill.adjusted_notional_usd,
            )
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
            _add_open_lot(
                open_lots,
                fold=fold,
                fill=fill,
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
        FoldTradeAttributionRow(
            fold=fold,
            fold_return_pct=fold_return,
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
        for (fold, fold_return, symbol, primary_signal, utc_hour, side), values in grouped.items()
    )
    return FoldTradeAttributionReport(rows=tuple(sorted(rows, key=lambda row: (row.fold, row.symbol))))


def write_fold_trade_attribution_csv(
    report: FoldTradeAttributionReport,
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "fold",
                "fold_return_pct",
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
            ),
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "fold": row.fold,
                    "fold_return_pct": row.fold_return_pct,
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


def _read_folds(path: str | Path) -> tuple[FoldWindow, ...]:
    folds: list[FoldWindow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"fold", "test_start", "test_end", "return_pct"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"fold CSV missing required columns: {sorted(missing)}")
        for row in reader:
            folds.append(
                FoldWindow(
                    fold=int(row["fold"]),
                    test_start=_parse_timestamp(row["test_start"]),
                    test_end=_parse_timestamp(row["test_end"]),
                    return_pct=float(row["return_pct"]),
                )
            )
    if not folds:
        raise ValueError("fold CSV has no rows")
    return tuple(folds)


def _read_fills(path: str | Path) -> tuple[CsvFill, ...]:
    fills: list[CsvFill] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "timestamp",
            "symbol",
            "side",
            "fill_price",
            "trade_units",
            "turnover_notional_usd",
            "adjusted_notional_usd",
            "primary_signal",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"fills CSV missing required columns: {sorted(missing)}")
        for row in reader:
            fills.append(
                CsvFill(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    symbol=row["symbol"],
                    side=row["side"],
                    fill_price=float(row["fill_price"]),
                    trade_units=float(row["trade_units"]),
                    turnover_notional_usd=float(row["turnover_notional_usd"]),
                    adjusted_notional_usd=float(row["adjusted_notional_usd"]),
                    primary_signal=row["primary_signal"],
                )
            )
    return tuple(sorted(fills, key=lambda fill: fill.timestamp))


def _fold_for_timestamp(timestamp: datetime, folds: tuple[FoldWindow, ...]) -> FoldWindow | None:
    for fold in folds:
        if fold.test_start <= timestamp <= fold.test_end:
            return fold
    return None


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _same_direction(left_units: float, right_units: float) -> bool:
    return (left_units > 0 and right_units > 0) or (left_units < 0 and right_units < 0)


def _add_open_lot(
    open_lots: list[_OpenLot],
    *,
    fold: FoldWindow,
    fill: CsvFill,
    units: float,
) -> None:
    open_lots.append(
        _OpenLot(
            fold=fold.fold,
            fold_return_pct=fold.return_pct,
            symbol=fill.symbol,
            primary_signal=fill.primary_signal or "unknown",
            utc_hour=fill.timestamp.astimezone(UTC).hour,
            side=fill.side,
            entry_price=fill.fill_price,
            units=units,
        )
    )


def _add_fill_volume(
    grouped: dict[tuple[int, float, str, str, int, str], dict[str, float | int]],
    key: tuple[int, float, str, str, int, str],
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
    grouped: dict[tuple[int, float, str, str, int, str], dict[str, float | int]],
    key: tuple[int, float, str, str, int, str],
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
    grouped: dict[tuple[int, float, str, str, int, str], dict[str, float | int]],
    key: tuple[int, float, str, str, int, str],
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
