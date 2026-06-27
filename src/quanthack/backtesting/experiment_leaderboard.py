from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentLeaderboardRow:
    label: str
    source_path: str
    strategy_text: str
    symbols: str
    folds: int
    positive_fold_fraction: float
    active_fold_fraction: float
    active_positive_fold_fraction: float
    non_negative_fold_fraction: float
    compounded_return_pct: float
    median_active_return_pct: float
    worst_drawdown_pct: float
    average_risk_discipline_score: float
    total_evaluation_fills: int
    score: float


def build_experiment_leaderboard(
    paths: tuple[str | Path, ...],
) -> tuple[ExperimentLeaderboardRow, ...]:
    rows: list[ExperimentLeaderboardRow] = []
    for path in paths:
        parsed = _read_summary(Path(path))
        if parsed is not None:
            rows.append(parsed)
    rows.sort(key=lambda row: row.score, reverse=True)
    return tuple(rows)


def write_experiment_leaderboard_csv(
    rows: tuple[ExperimentLeaderboardRow, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "label",
                "source_path",
                "strategy_text",
                "symbols",
                "folds",
                "positive_fold_fraction",
                "active_fold_fraction",
                "active_positive_fold_fraction",
                "non_negative_fold_fraction",
                "compounded_return_pct",
                "median_active_return_pct",
                "worst_drawdown_pct",
                "average_risk_discipline_score",
                "total_evaluation_fills",
                "score",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "label": row.label,
                    "source_path": row.source_path,
                    "strategy_text": row.strategy_text,
                    "symbols": row.symbols,
                    "folds": row.folds,
                    "positive_fold_fraction": row.positive_fold_fraction,
                    "active_fold_fraction": row.active_fold_fraction,
                    "active_positive_fold_fraction": row.active_positive_fold_fraction,
                    "non_negative_fold_fraction": row.non_negative_fold_fraction,
                    "compounded_return_pct": row.compounded_return_pct,
                    "median_active_return_pct": row.median_active_return_pct,
                    "worst_drawdown_pct": row.worst_drawdown_pct,
                    "average_risk_discipline_score": row.average_risk_discipline_score,
                    "total_evaluation_fills": row.total_evaluation_fills,
                    "score": row.score,
                }
            )


def _read_summary(path: Path) -> ExperimentLeaderboardRow | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw = next(reader, None)
    if raw is None:
        return None
    required = {
        "positive_fold_fraction",
        "active_fold_fraction",
        "active_positive_fold_fraction",
        "non_negative_fold_fraction",
        "median_active_test_return_pct",
        "worst_test_drawdown_pct",
        "average_risk_discipline_score",
        "total_evaluation_fills",
    }
    if not required.issubset(raw):
        return None

    strategy_text = raw.get("strategies") or raw.get("strategy") or ""
    score_inputs = {
        "positive": _float(raw["positive_fold_fraction"]),
        "active": _float(raw["active_fold_fraction"]),
        "active_positive": _float(raw["active_positive_fold_fraction"]),
        "non_negative": _float(raw["non_negative_fold_fraction"]),
        "compounded": _first_float(
            raw,
            (
                "compounded_test_return_pct",
                "compounded_return_pct",
                "stitched_return_pct",
                "total_return_pct",
            ),
        ),
        "median_active": _float(raw["median_active_test_return_pct"]),
        "drawdown": _float(raw["worst_test_drawdown_pct"]),
        "risk": _float(raw["average_risk_discipline_score"]),
    }
    return ExperimentLeaderboardRow(
        label=path.stem.replace("_summary", ""),
        source_path=str(path),
        strategy_text=strategy_text,
        symbols=raw.get("symbols", ""),
        folds=int(_float(raw.get("folds", "0"))),
        positive_fold_fraction=score_inputs["positive"],
        active_fold_fraction=score_inputs["active"],
        active_positive_fold_fraction=score_inputs["active_positive"],
        non_negative_fold_fraction=score_inputs["non_negative"],
        compounded_return_pct=score_inputs["compounded"],
        median_active_return_pct=score_inputs["median_active"],
        worst_drawdown_pct=score_inputs["drawdown"],
        average_risk_discipline_score=score_inputs["risk"],
        total_evaluation_fills=int(_float(raw["total_evaluation_fills"])),
        score=_leaderboard_score(score_inputs),
    )


def _leaderboard_score(values: dict[str, float]) -> float:
    risk_multiplier = min(max(values["risk"] / 100.0, 0.0), 1.0)
    stability = (
        (0.35 * values["non_negative"])
        + (0.30 * values["active_positive"])
        + (0.20 * values["positive"])
        + (0.15 * min(values["active"], 0.75) / 0.75)
    )
    return risk_multiplier * (
        stability
        + (4.0 * values["compounded"])
        + (20.0 * values["median_active"])
        - (0.75 * values["drawdown"])
    )


def _first_float(raw: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        if name in raw:
            return _float(raw[name])
    return 0.0


def _float(raw: str | float | int | None) -> float:
    if raw is None or raw == "":
        return 0.0
    return float(raw)
