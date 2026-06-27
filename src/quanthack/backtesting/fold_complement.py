from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


RETURN_EPSILON = 1e-12


@dataclass(frozen=True)
class FoldReturn:
    fold: int
    test_start: str
    test_end: str
    return_pct: float
    evaluation_fills: int

    @property
    def is_positive(self) -> bool:
        return self.return_pct > RETURN_EPSILON

    @property
    def is_negative(self) -> bool:
        return self.return_pct < -RETURN_EPSILON

    @property
    def is_flat(self) -> bool:
        return abs(self.return_pct) <= RETURN_EPSILON and self.evaluation_fills == 0


@dataclass(frozen=True)
class FoldComplementRow:
    label: str
    fold: int
    test_start: str
    test_end: str
    baseline_return_pct: float
    candidate_return_pct: float
    combined_return_pct: float
    baseline_fills: int
    candidate_fills: int
    baseline_status: str
    candidate_helped_flat_or_losing: bool
    candidate_hurt_positive: bool


@dataclass(frozen=True)
class FoldComplementSummary:
    label: str
    folds: int
    baseline_positive_fraction: float
    candidate_positive_fraction: float
    combined_positive_fraction: float
    baseline_non_negative_fraction: float
    candidate_non_negative_fraction: float
    combined_non_negative_fraction: float
    baseline_flat_folds: int
    baseline_losing_folds: int
    candidate_positive_on_baseline_flat: int
    candidate_positive_on_baseline_losing: int
    candidate_hurt_baseline_positive: int
    incremental_return_sum_pct: float
    rows: tuple[FoldComplementRow, ...]


def read_fold_returns(path: str | Path) -> tuple[FoldReturn, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    folds: list[FoldReturn] = []
    for row in rows:
        folds.append(
            FoldReturn(
                fold=int(row["fold"]),
                test_start=str(row["test_start"]),
                test_end=str(row["test_end"]),
                return_pct=float(row["return_pct"]),
                evaluation_fills=int(row.get("evaluation_fills") or 0),
            )
        )
    return tuple(folds)


def evaluate_fold_complement(
    *,
    baseline: tuple[FoldReturn, ...],
    candidate: tuple[FoldReturn, ...],
    label: str,
) -> FoldComplementSummary:
    if not baseline:
        raise ValueError("baseline folds cannot be empty")
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate folds must have the same length")

    rows: list[FoldComplementRow] = []
    for base_fold, candidate_fold in zip(baseline, candidate, strict=True):
        if base_fold.fold != candidate_fold.fold:
            raise ValueError("baseline and candidate fold numbers must match")
        if (
            base_fold.test_start != candidate_fold.test_start
            or base_fold.test_end != candidate_fold.test_end
        ):
            raise ValueError("baseline and candidate fold windows must match")

        baseline_status = _fold_status(base_fold)
        combined_return = base_fold.return_pct + candidate_fold.return_pct
        candidate_helped_flat_or_losing = (
            baseline_status in {"flat", "losing"} and candidate_fold.is_positive
        )
        candidate_hurt_positive = base_fold.is_positive and candidate_fold.is_negative
        rows.append(
            FoldComplementRow(
                label=label,
                fold=base_fold.fold,
                test_start=base_fold.test_start,
                test_end=base_fold.test_end,
                baseline_return_pct=base_fold.return_pct,
                candidate_return_pct=candidate_fold.return_pct,
                combined_return_pct=combined_return,
                baseline_fills=base_fold.evaluation_fills,
                candidate_fills=candidate_fold.evaluation_fills,
                baseline_status=baseline_status,
                candidate_helped_flat_or_losing=candidate_helped_flat_or_losing,
                candidate_hurt_positive=candidate_hurt_positive,
            )
        )

    return FoldComplementSummary(
        label=label,
        folds=len(rows),
        baseline_positive_fraction=_positive_fraction(
            row.baseline_return_pct for row in rows
        ),
        candidate_positive_fraction=_positive_fraction(
            row.candidate_return_pct for row in rows
        ),
        combined_positive_fraction=_positive_fraction(
            row.combined_return_pct for row in rows
        ),
        baseline_non_negative_fraction=_non_negative_fraction(
            row.baseline_return_pct for row in rows
        ),
        candidate_non_negative_fraction=_non_negative_fraction(
            row.candidate_return_pct for row in rows
        ),
        combined_non_negative_fraction=_non_negative_fraction(
            row.combined_return_pct for row in rows
        ),
        baseline_flat_folds=sum(1 for row in rows if row.baseline_status == "flat"),
        baseline_losing_folds=sum(1 for row in rows if row.baseline_status == "losing"),
        candidate_positive_on_baseline_flat=sum(
            1
            for row in rows
            if row.baseline_status == "flat" and row.candidate_return_pct > RETURN_EPSILON
        ),
        candidate_positive_on_baseline_losing=sum(
            1
            for row in rows
            if row.baseline_status == "losing" and row.candidate_return_pct > RETURN_EPSILON
        ),
        candidate_hurt_baseline_positive=sum(
            1 for row in rows if row.candidate_hurt_positive
        ),
        incremental_return_sum_pct=sum(row.candidate_return_pct for row in rows),
        rows=tuple(rows),
    )


def write_fold_complement_csv(
    summaries: tuple[FoldComplementSummary, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            for row in summary.rows:
                writer.writerow(
                    {
                        "label": row.label,
                        "fold": row.fold,
                        "test_start": row.test_start,
                        "test_end": row.test_end,
                        "baseline_return_pct": row.baseline_return_pct,
                        "candidate_return_pct": row.candidate_return_pct,
                        "combined_return_pct": row.combined_return_pct,
                        "baseline_fills": row.baseline_fills,
                        "candidate_fills": row.candidate_fills,
                        "baseline_status": row.baseline_status,
                        "candidate_helped_flat_or_losing": row.candidate_helped_flat_or_losing,
                        "candidate_hurt_positive": row.candidate_hurt_positive,
                    }
                )


def write_fold_complement_summary_csv(
    summaries: tuple[FoldComplementSummary, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "label": summary.label,
                    "folds": summary.folds,
                    "baseline_positive_fraction": summary.baseline_positive_fraction,
                    "candidate_positive_fraction": summary.candidate_positive_fraction,
                    "combined_positive_fraction": summary.combined_positive_fraction,
                    "baseline_non_negative_fraction": summary.baseline_non_negative_fraction,
                    "candidate_non_negative_fraction": summary.candidate_non_negative_fraction,
                    "combined_non_negative_fraction": summary.combined_non_negative_fraction,
                    "baseline_flat_folds": summary.baseline_flat_folds,
                    "baseline_losing_folds": summary.baseline_losing_folds,
                    "candidate_positive_on_baseline_flat": (
                        summary.candidate_positive_on_baseline_flat
                    ),
                    "candidate_positive_on_baseline_losing": (
                        summary.candidate_positive_on_baseline_losing
                    ),
                    "candidate_hurt_baseline_positive": (
                        summary.candidate_hurt_baseline_positive
                    ),
                    "incremental_return_sum_pct": summary.incremental_return_sum_pct,
                }
            )


def _fold_status(fold: FoldReturn) -> str:
    if fold.is_positive:
        return "positive"
    if fold.is_negative:
        return "losing"
    if fold.is_flat:
        return "flat"
    return "active_flat"


def _positive_fraction(values) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return sum(1 for value in values_tuple if value > RETURN_EPSILON) / len(values_tuple)


def _non_negative_fraction(values) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return sum(1 for value in values_tuple if value >= -RETURN_EPSILON) / len(values_tuple)


_CSV_FIELDS = (
    "label",
    "fold",
    "test_start",
    "test_end",
    "baseline_return_pct",
    "candidate_return_pct",
    "combined_return_pct",
    "baseline_fills",
    "candidate_fills",
    "baseline_status",
    "candidate_helped_flat_or_losing",
    "candidate_hurt_positive",
)

_SUMMARY_FIELDS = (
    "label",
    "folds",
    "baseline_positive_fraction",
    "candidate_positive_fraction",
    "combined_positive_fraction",
    "baseline_non_negative_fraction",
    "candidate_non_negative_fraction",
    "combined_non_negative_fraction",
    "baseline_flat_folds",
    "baseline_losing_folds",
    "candidate_positive_on_baseline_flat",
    "candidate_positive_on_baseline_losing",
    "candidate_hurt_baseline_positive",
    "incremental_return_sum_pct",
)
