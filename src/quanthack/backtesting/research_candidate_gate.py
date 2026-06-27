from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ResearchDataSource(StrEnum):
    OFFICIAL = "official"
    PROXY = "proxy"
    MIXED_PROXY = "mixed_proxy"
    SYNTHETIC = "synthetic"


class ResearchReadiness(StrEnum):
    LIVE_READY = "LIVE_READY"
    PAPER_ONLY = "PAPER_ONLY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ResearchCandidateSource:
    path: str
    data_source: ResearchDataSource = ResearchDataSource.PROXY


@dataclass(frozen=True)
class ResearchCandidateGateRow:
    label: str
    source_file: str
    data_source: ResearchDataSource
    readiness: ResearchReadiness
    live_ready: bool
    decision_score: float
    reason: str
    return_pct: float
    max_drawdown_pct: float
    sharpe_15m: float
    risk_discipline_score: float
    promotion_status: str
    promotion_reason: str
    crypto_allowed_utc_hours: str
    wf_positive_fold_fraction: float
    wf_active_positive_fold_fraction: float
    wf_non_negative_fold_fraction: float
    wf_median_active_test_return_pct: float
    selection_score: float
    proxy_score: float
    strategy_map: str
    crypto_map: str

    @property
    def rank_key(self) -> tuple[int, float, float, float, float]:
        readiness_rank = {
            ResearchReadiness.LIVE_READY: 2,
            ResearchReadiness.PAPER_ONLY: 1,
            ResearchReadiness.REJECT: 0,
        }[self.readiness]
        return (
            readiness_rank,
            self.decision_score,
            self.return_pct,
            -self.max_drawdown_pct,
            self.sharpe_15m,
        )


def build_research_candidate_gate(
    sources: tuple[ResearchCandidateSource, ...],
) -> tuple[ResearchCandidateGateRow, ...]:
    if not sources:
        raise ValueError("at least one research candidate source is required")

    rows: list[ResearchCandidateGateRow] = []
    for source in sources:
        rows.extend(_read_source(source))
    if not rows:
        raise ValueError("research candidate sources produced no rows")
    return tuple(sorted(rows, key=lambda row: row.rank_key, reverse=True))


def write_research_candidate_gate_csv(
    rows: tuple[ResearchCandidateGateRow, ...],
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "label",
                "source_file",
                "data_source",
                "readiness",
                "live_ready",
                "decision_score",
                "reason",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "promotion_status",
                "promotion_reason",
                "crypto_allowed_utc_hours",
                "wf_positive_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_median_active_test_return_pct",
                "selection_score",
                "proxy_score",
                "strategy_map",
                "crypto_map",
            ),
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "label": row.label,
                    "source_file": row.source_file,
                    "data_source": row.data_source.value,
                    "readiness": row.readiness.value,
                    "live_ready": row.live_ready,
                    "decision_score": row.decision_score,
                    "reason": row.reason,
                    "return_pct": row.return_pct,
                    "max_drawdown_pct": row.max_drawdown_pct,
                    "sharpe_15m": row.sharpe_15m,
                    "risk_discipline_score": row.risk_discipline_score,
                    "promotion_status": row.promotion_status,
                    "promotion_reason": row.promotion_reason,
                    "crypto_allowed_utc_hours": row.crypto_allowed_utc_hours,
                    "wf_positive_fold_fraction": row.wf_positive_fold_fraction,
                    "wf_active_positive_fold_fraction": (
                        row.wf_active_positive_fold_fraction
                    ),
                    "wf_non_negative_fold_fraction": (
                        row.wf_non_negative_fold_fraction
                    ),
                    "wf_median_active_test_return_pct": (
                        row.wf_median_active_test_return_pct
                    ),
                    "selection_score": row.selection_score,
                    "proxy_score": row.proxy_score,
                    "strategy_map": row.strategy_map,
                    "crypto_map": row.crypto_map,
                }
            )


def normalize_research_data_source(value: str | ResearchDataSource) -> ResearchDataSource:
    if isinstance(value, ResearchDataSource):
        return value
    normalized = value.strip().lower().replace("-", "_")
    try:
        return ResearchDataSource(normalized)
    except ValueError as exc:
        valid = ", ".join(source.value for source in ResearchDataSource)
        raise ValueError(f"unknown data source {value!r}; expected one of {valid}") from exc


def _read_source(source: ResearchCandidateSource) -> list[ResearchCandidateGateRow]:
    path = Path(source.path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        if "label" not in reader.fieldnames:
            raise ValueError(f"{path} missing required label column")
        return [
            _row_from_csv(
                row=row,
                source_file=str(path),
                data_source=source.data_source,
            )
            for row in reader
        ]


def _row_from_csv(
    *,
    row: dict[str, str],
    source_file: str,
    data_source: ResearchDataSource,
) -> ResearchCandidateGateRow:
    label = row["label"].strip()
    return_pct = _float(row, "return_pct")
    max_drawdown_pct = _float(row, "max_drawdown_pct")
    sharpe_15m = _float(row, "sharpe_15m")
    risk_score = _float(
        row,
        "risk_discipline_score",
        "risk_score",
        default=100.0,
    )
    promotion_status = row.get("promotion_status", "").strip().upper()
    promotion_reason = row.get("promotion_reason", "").strip()
    wf_positive = _float(row, "wf_positive_fold_fraction", default=1.0)
    wf_active_positive = _float(
        row,
        "wf_active_positive_fold_fraction",
        default=1.0,
    )
    wf_non_negative = _float(row, "wf_non_negative_fold_fraction", default=1.0)
    wf_median_active = _float(
        row,
        "wf_median_active_test_return_pct",
        default=return_pct,
    )
    selection_score = _float(row, "selection_score", default=0.0)
    proxy_score = _float(
        row,
        "proxy_score",
        "composite_score",
        default=selection_score,
    )
    readiness, live_ready, reason = _readiness(
        data_source=data_source,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        risk_score=risk_score,
        promotion_status=promotion_status,
        promotion_reason=promotion_reason,
        wf_positive_fold_fraction=wf_positive,
        wf_active_positive_fold_fraction=wf_active_positive,
        wf_non_negative_fold_fraction=wf_non_negative,
        wf_median_active_test_return_pct=wf_median_active,
    )
    return ResearchCandidateGateRow(
        label=label,
        source_file=source_file,
        data_source=data_source,
        readiness=readiness,
        live_ready=live_ready,
        decision_score=_decision_score(
            readiness=readiness,
            data_source=data_source,
            selection_score=selection_score,
            proxy_score=proxy_score,
            return_pct=return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_15m=sharpe_15m,
            risk_score=risk_score,
            wf_non_negative_fold_fraction=wf_non_negative,
            wf_active_positive_fold_fraction=wf_active_positive,
        ),
        reason=reason,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_15m=sharpe_15m,
        risk_discipline_score=risk_score,
        promotion_status=promotion_status,
        promotion_reason=promotion_reason,
        crypto_allowed_utc_hours=row.get("crypto_allowed_utc_hours", "").strip(),
        wf_positive_fold_fraction=wf_positive,
        wf_active_positive_fold_fraction=wf_active_positive,
        wf_non_negative_fold_fraction=wf_non_negative,
        wf_median_active_test_return_pct=wf_median_active,
        selection_score=selection_score,
        proxy_score=proxy_score,
        strategy_map=row.get("strategy_map", "").strip(),
        crypto_map=row.get("crypto_map", "").strip(),
    )


def _readiness(
    *,
    data_source: ResearchDataSource,
    return_pct: float,
    max_drawdown_pct: float,
    risk_score: float,
    promotion_status: str,
    promotion_reason: str,
    wf_positive_fold_fraction: float,
    wf_active_positive_fold_fraction: float,
    wf_non_negative_fold_fraction: float,
    wf_median_active_test_return_pct: float,
) -> tuple[ResearchReadiness, bool, str]:
    reject_reasons: list[str] = []
    paper_reasons: list[str] = []

    if promotion_status == "REJECT":
        reject_reasons.append(_with_detail("walk-forward rejected", promotion_reason))
    if return_pct <= 0:
        reject_reasons.append("full-sample return is not positive")
    if wf_non_negative_fold_fraction < 0.70:
        reject_reasons.append("non-negative fold fraction below 70%")
    if (
        wf_positive_fold_fraction < 0.50
        and wf_active_positive_fold_fraction < 0.50
    ):
        reject_reasons.append("positive fold evidence below 50%")
    if wf_median_active_test_return_pct <= 0:
        reject_reasons.append("median active fold return is not positive")
    if risk_score < 95:
        reject_reasons.append("risk discipline below 95/100")

    if reject_reasons:
        return ResearchReadiness.REJECT, False, "; ".join(reject_reasons)

    if data_source != ResearchDataSource.OFFICIAL:
        paper_reasons.append(f"{data_source.value} data cannot be live-ready")
    if promotion_status and promotion_status != "PROMOTE":
        paper_reasons.append(_with_detail(f"walk-forward status {promotion_status}", promotion_reason))
    if risk_score < 100:
        paper_reasons.append("risk discipline is below a perfect 100/100")
    if max_drawdown_pct > 0.03:
        paper_reasons.append("drawdown above 3% internal promotion comfort line")

    if paper_reasons:
        return ResearchReadiness.PAPER_ONLY, False, "; ".join(paper_reasons)

    return (
        ResearchReadiness.LIVE_READY,
        True,
        "official data, positive return, stable folds, and clean risk discipline",
    )


def _decision_score(
    *,
    readiness: ResearchReadiness,
    data_source: ResearchDataSource,
    selection_score: float,
    proxy_score: float,
    return_pct: float,
    max_drawdown_pct: float,
    sharpe_15m: float,
    risk_score: float,
    wf_non_negative_fold_fraction: float,
    wf_active_positive_fold_fraction: float,
) -> float:
    base_score = selection_score or proxy_score
    if base_score == 0:
        base_score = (
            1000.0 * return_pct
            - 500.0 * max_drawdown_pct
            + 100.0 * sharpe_15m
            + 20.0 * wf_non_negative_fold_fraction
            + 20.0 * wf_active_positive_fold_fraction
        )
    readiness_adjustment = {
        ResearchReadiness.LIVE_READY: 20.0,
        ResearchReadiness.PAPER_ONLY: 0.0,
        ResearchReadiness.REJECT: -50.0,
    }[readiness]
    data_penalty = {
        ResearchDataSource.OFFICIAL: 0.0,
        ResearchDataSource.PROXY: 10.0,
        ResearchDataSource.MIXED_PROXY: 15.0,
        ResearchDataSource.SYNTHETIC: 30.0,
    }[data_source]
    risk_penalty = max(0.0, 100.0 - risk_score) * 2.0
    return base_score + readiness_adjustment - data_penalty - risk_penalty


def _float(
    row: dict[str, str],
    *keys: str,
    default: float | None = None,
) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in {None, ""}:
            return float(raw)
    if default is not None:
        return default
    wanted = " or ".join(keys)
    raise ValueError(f"CSV row missing required numeric field {wanted}")


def _with_detail(prefix: str, detail: str) -> str:
    if not detail:
        return prefix
    return f"{prefix}: {detail}"
