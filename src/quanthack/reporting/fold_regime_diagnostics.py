from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceBar, PriceHistory
from quanthack.strategies.time_series import KalmanTrendConfig, read_kalman_regime


@dataclass(frozen=True)
class FoldRegimeWindow:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    return_pct: float


@dataclass(frozen=True)
class FoldRegimeDiagnosticRow:
    fold: int
    fold_return_pct: float
    test_start: datetime
    test_end: datetime
    symbol: str
    asset_class: str
    train_observations: int
    train_return_pct: float
    regime_lookback_return_pct: float
    test_symbol_return_pct: float
    kalman_slope_bps: float
    trend_efficiency: float
    realized_volatility_bps: float
    trend_confidence: float
    regime: str

    @property
    def is_trending(self) -> bool:
        return self.regime in {"TREND_UP", "TREND_DOWN"}


@dataclass(frozen=True)
class FoldRegimeSummaryRow:
    fold: int
    fold_return_pct: float
    test_start: datetime
    test_end: datetime
    symbols: int
    trend_up_symbols: int
    trend_down_symbols: int
    chop_symbols: int
    high_volatility_symbols: int
    trend_consensus: float
    chop_fraction: float
    high_volatility_fraction: float
    average_abs_slope_bps: float
    net_slope_bps: float
    average_trend_efficiency: float
    average_realized_volatility_bps: float
    average_train_return_pct: float
    average_test_symbol_return_pct: float
    forex_net_slope_bps: float
    metal_net_slope_bps: float
    crypto_net_slope_bps: float


@dataclass(frozen=True)
class FoldRegimeDiagnosticsReport:
    detail_rows: tuple[FoldRegimeDiagnosticRow, ...]
    summary_rows: tuple[FoldRegimeSummaryRow, ...]

    @property
    def weakest_folds(self) -> tuple[FoldRegimeSummaryRow, ...]:
        return tuple(sorted(self.summary_rows, key=lambda row: row.fold_return_pct))

    @property
    def strongest_folds(self) -> tuple[FoldRegimeSummaryRow, ...]:
        return tuple(
            sorted(self.summary_rows, key=lambda row: row.fold_return_pct, reverse=True)
        )


def build_fold_regime_diagnostics_report(
    *,
    prices: PriceHistory,
    folds_csv: str | Path,
    symbols: tuple[str, ...] | list[str] | None = None,
    config: KalmanTrendConfig | None = None,
) -> FoldRegimeDiagnosticsReport:
    cfg = config or KalmanTrendConfig()
    fold_windows = _read_fold_windows(folds_csv)
    selected_symbols = tuple(symbols or prices.symbols())
    if not selected_symbols:
        raise ValueError("fold regime diagnostics require at least one symbol")

    detail_rows: list[FoldRegimeDiagnosticRow] = []
    for fold in fold_windows:
        for symbol in selected_symbols:
            instrument = instrument_for(symbol)
            symbol_bars = prices.for_symbol(instrument.symbol).bars
            train_bars = _bars_between(
                symbol_bars,
                start=fold.train_start,
                end=fold.train_end,
            )
            test_bars = _bars_between(
                symbol_bars,
                start=fold.test_start,
                end=fold.test_end,
            )
            if len(train_bars) < cfg.lookback:
                raise ValueError(
                    f"not enough train bars for {instrument.symbol} fold {fold.fold}: "
                    f"{len(train_bars)} < {cfg.lookback}"
                )
            if len(test_bars) < 2:
                raise ValueError(
                    f"not enough test bars for {instrument.symbol} fold {fold.fold}"
                )

            train_closes = tuple(bar.close for bar in train_bars)
            regime = read_kalman_regime(
                train_closes,
                symbol=instrument.symbol,
                config=cfg,
            )
            lookback_closes = train_closes[-cfg.lookback :]
            detail_rows.append(
                FoldRegimeDiagnosticRow(
                    fold=fold.fold,
                    fold_return_pct=fold.return_pct,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    symbol=instrument.symbol,
                    asset_class=instrument.asset_class.value,
                    train_observations=len(train_bars),
                    train_return_pct=_return_pct(train_closes[0], train_closes[-1]),
                    regime_lookback_return_pct=_return_pct(
                        lookback_closes[0],
                        lookback_closes[-1],
                    ),
                    test_symbol_return_pct=_return_pct(
                        test_bars[0].close,
                        test_bars[-1].close,
                    ),
                    kalman_slope_bps=regime.kalman_slope_bps,
                    trend_efficiency=regime.trend_efficiency,
                    realized_volatility_bps=regime.realized_volatility_bps,
                    trend_confidence=regime.trend_confidence,
                    regime=regime.regime.value,
                )
            )

    summary_rows = tuple(
        _summarize_fold(rows)
        for rows in _group_by_fold(tuple(detail_rows))
    )
    return FoldRegimeDiagnosticsReport(
        detail_rows=tuple(
            sorted(detail_rows, key=lambda row: (row.fold, row.asset_class, row.symbol))
        ),
        summary_rows=summary_rows,
    )


def write_fold_regime_detail_csv(
    rows: tuple[FoldRegimeDiagnosticRow, ...],
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
                "test_start",
                "test_end",
                "symbol",
                "asset_class",
                "train_observations",
                "train_return_pct",
                "regime_lookback_return_pct",
                "test_symbol_return_pct",
                "kalman_slope_bps",
                "trend_efficiency",
                "realized_volatility_bps",
                "trend_confidence",
                "regime",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "fold": row.fold,
                    "fold_return_pct": row.fold_return_pct,
                    "test_start": row.test_start.isoformat(),
                    "test_end": row.test_end.isoformat(),
                    "symbol": row.symbol,
                    "asset_class": row.asset_class,
                    "train_observations": row.train_observations,
                    "train_return_pct": row.train_return_pct,
                    "regime_lookback_return_pct": row.regime_lookback_return_pct,
                    "test_symbol_return_pct": row.test_symbol_return_pct,
                    "kalman_slope_bps": row.kalman_slope_bps,
                    "trend_efficiency": row.trend_efficiency,
                    "realized_volatility_bps": row.realized_volatility_bps,
                    "trend_confidence": row.trend_confidence,
                    "regime": row.regime,
                }
            )


def write_fold_regime_summary_csv(
    rows: tuple[FoldRegimeSummaryRow, ...],
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
                "test_start",
                "test_end",
                "symbols",
                "trend_up_symbols",
                "trend_down_symbols",
                "chop_symbols",
                "high_volatility_symbols",
                "trend_consensus",
                "chop_fraction",
                "high_volatility_fraction",
                "average_abs_slope_bps",
                "net_slope_bps",
                "average_trend_efficiency",
                "average_realized_volatility_bps",
                "average_train_return_pct",
                "average_test_symbol_return_pct",
                "forex_net_slope_bps",
                "metal_net_slope_bps",
                "crypto_net_slope_bps",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "fold": row.fold,
                    "fold_return_pct": row.fold_return_pct,
                    "test_start": row.test_start.isoformat(),
                    "test_end": row.test_end.isoformat(),
                    "symbols": row.symbols,
                    "trend_up_symbols": row.trend_up_symbols,
                    "trend_down_symbols": row.trend_down_symbols,
                    "chop_symbols": row.chop_symbols,
                    "high_volatility_symbols": row.high_volatility_symbols,
                    "trend_consensus": row.trend_consensus,
                    "chop_fraction": row.chop_fraction,
                    "high_volatility_fraction": row.high_volatility_fraction,
                    "average_abs_slope_bps": row.average_abs_slope_bps,
                    "net_slope_bps": row.net_slope_bps,
                    "average_trend_efficiency": row.average_trend_efficiency,
                    "average_realized_volatility_bps": (
                        row.average_realized_volatility_bps
                    ),
                    "average_train_return_pct": row.average_train_return_pct,
                    "average_test_symbol_return_pct": (
                        row.average_test_symbol_return_pct
                    ),
                    "forex_net_slope_bps": row.forex_net_slope_bps,
                    "metal_net_slope_bps": row.metal_net_slope_bps,
                    "crypto_net_slope_bps": row.crypto_net_slope_bps,
                }
            )


def _summarize_fold(
    rows: tuple[FoldRegimeDiagnosticRow, ...],
) -> FoldRegimeSummaryRow:
    if not rows:
        raise ValueError("cannot summarize an empty fold")
    trend_up = sum(1 for row in rows if row.regime == "TREND_UP")
    trend_down = sum(1 for row in rows if row.regime == "TREND_DOWN")
    chop = sum(1 for row in rows if row.regime == "CHOP")
    high_volatility = sum(1 for row in rows if row.regime == "HIGH_VOLATILITY")
    symbols = len(rows)
    return FoldRegimeSummaryRow(
        fold=rows[0].fold,
        fold_return_pct=rows[0].fold_return_pct,
        test_start=rows[0].test_start,
        test_end=rows[0].test_end,
        symbols=symbols,
        trend_up_symbols=trend_up,
        trend_down_symbols=trend_down,
        chop_symbols=chop,
        high_volatility_symbols=high_volatility,
        trend_consensus=abs(trend_up - trend_down) / symbols,
        chop_fraction=chop / symbols,
        high_volatility_fraction=high_volatility / symbols,
        average_abs_slope_bps=mean(abs(row.kalman_slope_bps) for row in rows),
        net_slope_bps=sum(row.kalman_slope_bps for row in rows),
        average_trend_efficiency=mean(row.trend_efficiency for row in rows),
        average_realized_volatility_bps=mean(
            row.realized_volatility_bps for row in rows
        ),
        average_train_return_pct=mean(row.train_return_pct for row in rows),
        average_test_symbol_return_pct=mean(
            row.test_symbol_return_pct for row in rows
        ),
        forex_net_slope_bps=_asset_net_slope(rows, AssetClass.FOREX),
        metal_net_slope_bps=_asset_net_slope(rows, AssetClass.METAL),
        crypto_net_slope_bps=_asset_net_slope(rows, AssetClass.CRYPTO),
    )


def _asset_net_slope(
    rows: tuple[FoldRegimeDiagnosticRow, ...],
    asset_class: AssetClass,
) -> float:
    return sum(
        row.kalman_slope_bps
        for row in rows
        if row.asset_class == asset_class.value
    )


def _group_by_fold(
    rows: tuple[FoldRegimeDiagnosticRow, ...],
) -> tuple[tuple[FoldRegimeDiagnosticRow, ...], ...]:
    grouped: dict[int, list[FoldRegimeDiagnosticRow]] = {}
    for row in rows:
        grouped.setdefault(row.fold, []).append(row)
    return tuple(
        tuple(sorted(grouped[fold], key=lambda row: row.symbol))
        for fold in sorted(grouped)
    )


def _read_fold_windows(path: str | Path) -> tuple[FoldRegimeWindow, ...]:
    folds: list[FoldRegimeWindow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "return_pct",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"fold CSV missing required columns: {sorted(missing)}")
        for row in reader:
            folds.append(
                FoldRegimeWindow(
                    fold=int(row["fold"]),
                    train_start=_parse_timestamp(row["train_start"]),
                    train_end=_parse_timestamp(row["train_end"]),
                    test_start=_parse_timestamp(row["test_start"]),
                    test_end=_parse_timestamp(row["test_end"]),
                    return_pct=float(row["return_pct"]),
                )
            )
    if not folds:
        raise ValueError("fold CSV has no rows")
    return tuple(folds)


def _bars_between(
    bars: tuple[PriceBar, ...],
    *,
    start: datetime,
    end: datetime,
) -> tuple[PriceBar, ...]:
    return tuple(bar for bar in bars if start <= bar.timestamp <= end)


def _return_pct(first: float, last: float) -> float:
    if first <= 0:
        raise ValueError("return denominator must be positive")
    return (last / first) - 1.0


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
