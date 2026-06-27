from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class AdaptiveHandoffDiagnosticRow:
    fold: int
    selected_strategy: str
    oracle_strategy: str
    selected_return_pct: float
    oracle_return_pct: float
    regret_pct: float
    selected_was_negative: bool
    oracle_was_cash: bool
    trend_consensus: float
    chop_fraction: float
    high_volatility_fraction: float
    average_realized_volatility_bps: float
    average_trend_efficiency: float
    net_slope_bps: float
    macd_train_adjusted_return_pct: float
    champion_train_adjusted_return_pct: float
    kalman_train_adjusted_return_pct: float
    macd_test_return_pct: float
    champion_test_return_pct: float
    kalman_test_return_pct: float
    champion_minus_macd_train_adjusted_pct: float
    champion_minus_macd_test_return_pct: float
    diagnosis: str


@dataclass(frozen=True)
class AdaptiveHandoffDiagnosticReport:
    rows: tuple[AdaptiveHandoffDiagnosticRow, ...]

    @property
    def fold_count(self) -> int:
        return len(self.rows)

    @property
    def total_regret_pct(self) -> float:
        return sum(row.regret_pct for row in self.rows)

    @property
    def largest_regret_rows(self) -> tuple[AdaptiveHandoffDiagnosticRow, ...]:
        return tuple(sorted(self.rows, key=lambda row: row.regret_pct, reverse=True))

    @property
    def diagnosis_counts(self) -> tuple[tuple[str, int], ...]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.diagnosis] = counts.get(row.diagnosis, 0) + 1
        return tuple(sorted(counts.items()))


def build_adaptive_handoff_diagnostic(
    *,
    oracle_folds_csv: str | Path,
    oracle_candidates_csv: str | Path,
    regime_summary_csv: str | Path,
) -> AdaptiveHandoffDiagnosticReport:
    oracle_folds = _read_rows_by_fold(oracle_folds_csv)
    regime_rows = _read_rows_by_fold(regime_summary_csv)
    candidate_rows = _read_candidate_rows(oracle_candidates_csv)
    rows: list[AdaptiveHandoffDiagnosticRow] = []
    for fold, oracle in sorted(oracle_folds.items()):
        regime = regime_rows.get(fold, {})
        candidates = candidate_rows.get(fold, {})
        macd = candidates.get("macd_momentum", {})
        champion = candidates.get("champion_ensemble", {})
        kalman = candidates.get("kalman_trend", {})
        row = AdaptiveHandoffDiagnosticRow(
            fold=fold,
            selected_strategy=oracle.get("selected_strategy", ""),
            oracle_strategy=oracle.get("oracle_strategy", ""),
            selected_return_pct=_float(oracle.get("selected_return_pct")),
            oracle_return_pct=_float(oracle.get("oracle_return_pct")),
            regret_pct=_float(oracle.get("regret_pct")),
            selected_was_negative=_yes(oracle.get("selected_was_negative")),
            oracle_was_cash=_yes(oracle.get("oracle_was_cash")),
            trend_consensus=_float(regime.get("trend_consensus")),
            chop_fraction=_float(regime.get("chop_fraction")),
            high_volatility_fraction=_float(regime.get("high_volatility_fraction")),
            average_realized_volatility_bps=_float(
                regime.get("average_realized_volatility_bps")
            ),
            average_trend_efficiency=_float(regime.get("average_trend_efficiency")),
            net_slope_bps=_float(regime.get("net_slope_bps")),
            macd_train_adjusted_return_pct=_float(
                macd.get("train_drawdown_adjusted_return_pct")
            ),
            champion_train_adjusted_return_pct=_float(
                champion.get("train_drawdown_adjusted_return_pct")
            ),
            kalman_train_adjusted_return_pct=_float(
                kalman.get("train_drawdown_adjusted_return_pct")
            ),
            macd_test_return_pct=_float(macd.get("test_return_pct")),
            champion_test_return_pct=_float(champion.get("test_return_pct")),
            kalman_test_return_pct=_float(kalman.get("test_return_pct")),
            champion_minus_macd_train_adjusted_pct=(
                _float(champion.get("train_drawdown_adjusted_return_pct"))
                - _float(macd.get("train_drawdown_adjusted_return_pct"))
            ),
            champion_minus_macd_test_return_pct=(
                _float(champion.get("test_return_pct"))
                - _float(macd.get("test_return_pct"))
            ),
            diagnosis="",
        )
        rows.append(replace(row, diagnosis=_diagnose(row)))
    return AdaptiveHandoffDiagnosticReport(rows=tuple(rows))


def write_adaptive_handoff_diagnostic_csv(
    report: AdaptiveHandoffDiagnosticReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "selected_strategy",
                "oracle_strategy",
                "selected_return_pct",
                "oracle_return_pct",
                "regret_pct",
                "selected_was_negative",
                "oracle_was_cash",
                "trend_consensus",
                "chop_fraction",
                "high_volatility_fraction",
                "average_realized_volatility_bps",
                "average_trend_efficiency",
                "net_slope_bps",
                "macd_train_adjusted_return_pct",
                "champion_train_adjusted_return_pct",
                "kalman_train_adjusted_return_pct",
                "macd_test_return_pct",
                "champion_test_return_pct",
                "kalman_test_return_pct",
                "champion_minus_macd_train_adjusted_pct",
                "champion_minus_macd_test_return_pct",
                "diagnosis",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "fold": row.fold,
                    "selected_strategy": row.selected_strategy,
                    "oracle_strategy": row.oracle_strategy,
                    "selected_return_pct": row.selected_return_pct,
                    "oracle_return_pct": row.oracle_return_pct,
                    "regret_pct": row.regret_pct,
                    "selected_was_negative": "yes" if row.selected_was_negative else "no",
                    "oracle_was_cash": "yes" if row.oracle_was_cash else "no",
                    "trend_consensus": row.trend_consensus,
                    "chop_fraction": row.chop_fraction,
                    "high_volatility_fraction": row.high_volatility_fraction,
                    "average_realized_volatility_bps": (
                        row.average_realized_volatility_bps
                    ),
                    "average_trend_efficiency": row.average_trend_efficiency,
                    "net_slope_bps": row.net_slope_bps,
                    "macd_train_adjusted_return_pct": (
                        row.macd_train_adjusted_return_pct
                    ),
                    "champion_train_adjusted_return_pct": (
                        row.champion_train_adjusted_return_pct
                    ),
                    "kalman_train_adjusted_return_pct": (
                        row.kalman_train_adjusted_return_pct
                    ),
                    "macd_test_return_pct": row.macd_test_return_pct,
                    "champion_test_return_pct": row.champion_test_return_pct,
                    "kalman_test_return_pct": row.kalman_test_return_pct,
                    "champion_minus_macd_train_adjusted_pct": (
                        row.champion_minus_macd_train_adjusted_pct
                    ),
                    "champion_minus_macd_test_return_pct": (
                        row.champion_minus_macd_test_return_pct
                    ),
                    "diagnosis": row.diagnosis,
                }
            )


def _diagnose(row: AdaptiveHandoffDiagnosticRow) -> str:
    if row.regret_pct <= 1e-12:
        return "NO_REGRET"
    if row.oracle_was_cash and row.selected_was_negative:
        return "CASH_AVOIDABLE_LOSS"
    if (
        row.oracle_strategy == "champion_ensemble"
        and row.selected_strategy == "macd_momentum"
        and row.chop_fraction >= 0.8
        and row.champion_train_adjusted_return_pct < 0
        and row.champion_minus_macd_test_return_pct > 0
    ):
        return "HINDSIGHT_CHOP_BREAKOUT"
    if (
        row.oracle_strategy == "macd_momentum"
        and row.selected_strategy != "macd_momentum"
        and row.chop_fraction >= 0.8
    ):
        return "MACD_MISSED_AFTER_CHOP"
    if (
        row.oracle_strategy == "champion_ensemble"
        and row.selected_strategy != row.oracle_strategy
    ):
        return "CHAMPION_HANDOFF_MISS"
    return "LOW_SIGNAL_REGRET"


def _read_rows_by_fold(path: str | Path) -> dict[int, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {int(row["fold"]): row for row in csv.DictReader(handle)}


def _read_candidate_rows(path: str | Path) -> dict[int, dict[str, dict[str, str]]]:
    rows: dict[int, dict[str, dict[str, str]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(int(row["fold"]), {})[row["strategy"]] = row
    return rows


def _float(raw: str | float | int | None) -> float:
    if raw is None or raw == "":
        return 0.0
    return float(raw)


def _yes(raw: str | None) -> bool:
    return str(raw).strip().lower() == "yes"
