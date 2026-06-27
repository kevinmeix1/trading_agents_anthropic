from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.core.instruments import AssetClass, instrument_for


@dataclass(frozen=True)
class SymbolAttribution:
    symbol: str
    asset_class: AssetClass
    fills: int
    realized_pnl_usd: float
    open_pnl_usd: float
    total_pnl_usd: float


@dataclass(frozen=True)
class AssetClassAttribution:
    asset_class: AssetClass
    symbols: tuple[str, ...]
    fills: int
    realized_pnl_usd: float
    open_pnl_usd: float
    total_pnl_usd: float
    share_of_portfolio_pnl: float
    share_of_gross_abs_pnl: float
    winners: int
    losers: int


@dataclass(frozen=True)
class AssetClassAttributionReport:
    source_csv: str
    portfolio_total_pnl_usd: float
    symbol_rows: tuple[SymbolAttribution, ...]
    asset_class_rows: tuple[AssetClassAttribution, ...]


def build_asset_class_attribution_report(
    pnl_csv: str | Path,
) -> AssetClassAttributionReport:
    source = Path(pnl_csv)
    symbol_rows: list[SymbolAttribution] = []
    portfolio_total: float | None = None
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames)
        for row in reader:
            symbol = row["symbol"].strip()
            total_pnl = float(row["total_pnl_usd"])
            if symbol == "PORTFOLIO":
                portfolio_total = total_pnl
                continue
            instrument = instrument_for(symbol)
            symbol_rows.append(
                SymbolAttribution(
                    symbol=instrument.symbol,
                    asset_class=instrument.asset_class,
                    fills=int(float(row["fills"])),
                    realized_pnl_usd=float(row["realized_pnl_usd"]),
                    open_pnl_usd=float(row["open_pnl_usd"]),
                    total_pnl_usd=total_pnl,
                )
            )

    if portfolio_total is None:
        portfolio_total = sum(row.total_pnl_usd for row in symbol_rows)

    gross_abs_total = sum(abs(row.total_pnl_usd) for row in symbol_rows)
    asset_rows = tuple(
        _build_asset_row(
            asset_class=asset_class,
            rows=tuple(row for row in symbol_rows if row.asset_class == asset_class),
            portfolio_total=portfolio_total,
            gross_abs_total=gross_abs_total,
        )
        for asset_class in AssetClass
        if any(row.asset_class == asset_class for row in symbol_rows)
    )
    return AssetClassAttributionReport(
        source_csv=str(source),
        portfolio_total_pnl_usd=portfolio_total,
        symbol_rows=tuple(sorted(symbol_rows, key=lambda row: (row.asset_class, row.symbol))),
        asset_class_rows=tuple(
            sorted(asset_rows, key=lambda row: row.total_pnl_usd, reverse=True)
        ),
    )


def write_asset_class_attribution_csv(
    report: AssetClassAttributionReport,
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "asset_class",
                "symbols",
                "fills",
                "realized_pnl_usd",
                "open_pnl_usd",
                "total_pnl_usd",
                "share_of_portfolio_pnl",
                "share_of_gross_abs_pnl",
                "winners",
                "losers",
            ),
        )
        writer.writeheader()
        for row in report.asset_class_rows:
            writer.writerow(
                {
                    "asset_class": row.asset_class.value,
                    "symbols": " ".join(row.symbols),
                    "fills": row.fills,
                    "realized_pnl_usd": row.realized_pnl_usd,
                    "open_pnl_usd": row.open_pnl_usd,
                    "total_pnl_usd": row.total_pnl_usd,
                    "share_of_portfolio_pnl": row.share_of_portfolio_pnl,
                    "share_of_gross_abs_pnl": row.share_of_gross_abs_pnl,
                    "winners": row.winners,
                    "losers": row.losers,
                }
            )


def _build_asset_row(
    *,
    asset_class: AssetClass,
    rows: tuple[SymbolAttribution, ...],
    portfolio_total: float,
    gross_abs_total: float,
) -> AssetClassAttribution:
    total_pnl = sum(row.total_pnl_usd for row in rows)
    return AssetClassAttribution(
        asset_class=asset_class,
        symbols=tuple(row.symbol for row in rows),
        fills=sum(row.fills for row in rows),
        realized_pnl_usd=sum(row.realized_pnl_usd for row in rows),
        open_pnl_usd=sum(row.open_pnl_usd for row in rows),
        total_pnl_usd=total_pnl,
        share_of_portfolio_pnl=(
            0.0 if portfolio_total == 0 else total_pnl / portfolio_total
        ),
        share_of_gross_abs_pnl=(
            0.0
            if gross_abs_total == 0
            else sum(abs(row.total_pnl_usd) for row in rows) / gross_abs_total
        ),
        winners=sum(1 for row in rows if row.total_pnl_usd > 0),
        losers=sum(1 for row in rows if row.total_pnl_usd < 0),
    )


def _validate_columns(fieldnames: list[str] | None) -> None:
    required = {
        "symbol",
        "fills",
        "realized_pnl_usd",
        "open_pnl_usd",
        "total_pnl_usd",
    }
    found = set(fieldnames or ())
    missing = required - found
    if missing:
        raise ValueError(f"P&L CSV missing required columns: {sorted(missing)}")
