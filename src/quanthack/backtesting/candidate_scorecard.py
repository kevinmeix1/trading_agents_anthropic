from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    official_composite_score,
    risk_samples_from_portfolio_equity,
)
from quanthack.trading.risk import Position


@dataclass(frozen=True)
class CandidateBundle:
    label: str
    equity_csv: str
    fills_csv: str
    pnl_csv: str | None = None


@dataclass(frozen=True)
class CandidateEquityPoint:
    timestamp: str
    equity: float
    gross_notional_usd: float
    net_notional_usd: float
    positions: tuple[Position, ...]


@dataclass(frozen=True)
class CandidateScorecardRow:
    label: str
    equity_csv: str
    fills_csv: str
    pnl_csv: str
    return_pct: float
    max_drawdown_pct: float
    sharpe_15m: float
    trade_count: int
    risk_discipline_score: int
    compliance_review_required: bool
    total_pnl_usd: float
    return_rank: float
    drawdown_rank: float
    sharpe_rank: float
    risk_rank: float
    composite_score: float
    sharpe_prize_trade_count_met: bool


def build_candidate_scorecard(
    bundles: tuple[CandidateBundle, ...],
) -> tuple[CandidateScorecardRow, ...]:
    if not bundles:
        raise ValueError("at least one candidate bundle is required")

    raw_rows = tuple(_score_candidate(bundle) for bundle in bundles)
    return_ranks = _percentile_ranks([row.return_pct for row in raw_rows])
    drawdown_ranks = _percentile_ranks(
        [-row.max_drawdown_pct for row in raw_rows]
    )
    sharpe_ranks = _percentile_ranks([row.sharpe_15m for row in raw_rows])
    risk_ranks = _percentile_ranks(
        [float(row.risk_discipline_score) for row in raw_rows]
    )
    ranked_rows = tuple(
        _with_ranks(
            row,
            return_rank=return_rank,
            drawdown_rank=drawdown_rank,
            sharpe_rank=sharpe_rank,
            risk_rank=risk_rank,
        )
        for row, return_rank, drawdown_rank, sharpe_rank, risk_rank in zip(
            raw_rows,
            return_ranks,
            drawdown_ranks,
            sharpe_ranks,
            risk_ranks,
            strict=True,
        )
    )
    return tuple(
        sorted(
            ranked_rows,
            key=lambda row: (
                row.composite_score,
                row.return_pct,
                -row.max_drawdown_pct,
                row.sharpe_15m,
            ),
            reverse=True,
        )
    )


def write_candidate_scorecard_csv(
    rows: tuple[CandidateScorecardRow, ...],
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
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "trade_count",
                "sharpe_prize_trade_count_met",
                "risk_discipline_score",
                "compliance_review_required",
                "total_pnl_usd",
                "return_rank",
                "drawdown_rank",
                "sharpe_rank",
                "risk_rank",
                "composite_score",
                "equity_csv",
                "fills_csv",
                "pnl_csv",
            ),
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "label": row.label,
                    "return_pct": row.return_pct,
                    "max_drawdown_pct": row.max_drawdown_pct,
                    "sharpe_15m": row.sharpe_15m,
                    "trade_count": row.trade_count,
                    "sharpe_prize_trade_count_met": row.sharpe_prize_trade_count_met,
                    "risk_discipline_score": row.risk_discipline_score,
                    "compliance_review_required": row.compliance_review_required,
                    "total_pnl_usd": row.total_pnl_usd,
                    "return_rank": row.return_rank,
                    "drawdown_rank": row.drawdown_rank,
                    "sharpe_rank": row.sharpe_rank,
                    "risk_rank": row.risk_rank,
                    "composite_score": row.composite_score,
                    "equity_csv": row.equity_csv,
                    "fills_csv": row.fills_csv,
                    "pnl_csv": row.pnl_csv,
                }
            )


def _score_candidate(bundle: CandidateBundle) -> CandidateScorecardRow:
    equity_points = _read_equity_points(bundle.equity_csv)
    fill_count = _read_fill_count(bundle.fills_csv)
    metrics = build_competition_metrics(
        equity_points=equity_points,
        fills=tuple(range(fill_count)),
    )
    risk = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(equity_points)
    )
    return _row_from_metrics(
        bundle=bundle,
        metrics=metrics,
        risk=risk,
        total_pnl_usd=_read_total_pnl(bundle.pnl_csv),
    )


def _row_from_metrics(
    *,
    bundle: CandidateBundle,
    metrics: CompetitionMetrics,
    risk: RiskDisciplineReport,
    total_pnl_usd: float,
) -> CandidateScorecardRow:
    return CandidateScorecardRow(
        label=bundle.label,
        equity_csv=bundle.equity_csv,
        fills_csv=bundle.fills_csv,
        pnl_csv=bundle.pnl_csv or "",
        return_pct=metrics.return_pct,
        max_drawdown_pct=metrics.max_drawdown_pct,
        sharpe_15m=metrics.sharpe_15m,
        trade_count=metrics.trade_count,
        risk_discipline_score=risk.score,
        compliance_review_required=risk.compliance_review_required,
        total_pnl_usd=total_pnl_usd,
        return_rank=0.0,
        drawdown_rank=0.0,
        sharpe_rank=0.0,
        risk_rank=0.0,
        composite_score=0.0,
        sharpe_prize_trade_count_met=metrics.sharpe_prize_trade_count_met,
    )


def _with_ranks(
    row: CandidateScorecardRow,
    *,
    return_rank: float,
    drawdown_rank: float,
    sharpe_rank: float,
    risk_rank: float,
) -> CandidateScorecardRow:
    return CandidateScorecardRow(
        label=row.label,
        equity_csv=row.equity_csv,
        fills_csv=row.fills_csv,
        pnl_csv=row.pnl_csv,
        return_pct=row.return_pct,
        max_drawdown_pct=row.max_drawdown_pct,
        sharpe_15m=row.sharpe_15m,
        trade_count=row.trade_count,
        risk_discipline_score=row.risk_discipline_score,
        compliance_review_required=row.compliance_review_required,
        total_pnl_usd=row.total_pnl_usd,
        return_rank=return_rank,
        drawdown_rank=drawdown_rank,
        sharpe_rank=sharpe_rank,
        risk_rank=risk_rank,
        composite_score=official_composite_score(
            return_rank=return_rank,
            drawdown_rank=drawdown_rank,
            sharpe_rank=sharpe_rank,
            risk_discipline_score=risk_rank,
        ),
        sharpe_prize_trade_count_met=row.sharpe_prize_trade_count_met,
    )


def _read_equity_points(path: str | Path) -> tuple[CandidateEquityPoint, ...]:
    points: list[CandidateEquityPoint] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "equity", "gross_notional_usd", "net_notional_usd"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"equity CSV missing required columns: {sorted(missing)}")
        for row in reader:
            points.append(
                CandidateEquityPoint(
                    timestamp=row["timestamp"],
                    equity=float(row["equity"]),
                    gross_notional_usd=float(row["gross_notional_usd"]),
                    net_notional_usd=float(row["net_notional_usd"]),
                    positions=_parse_positions(row.get("positions", "")),
                )
            )
    if not points:
        raise ValueError("equity CSV has no rows")
    return tuple(points)


def _read_fill_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _read_total_pnl(path: str | Path | None) -> float:
    if path is None:
        return 0.0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("symbol") == "PORTFOLIO":
                return float(row["total_pnl_usd"])
    return 0.0


def _parse_positions(raw: str) -> tuple[Position, ...]:
    if not raw.strip():
        return ()
    positions: list[Position] = []
    for item in raw.split(";"):
        if not item.strip():
            continue
        symbol, notional = item.split("=", 1)
        positions.append(Position(symbol=symbol, notional_usd=float(notional)))
    return tuple(positions)


def _percentile_ranks(values: list[float]) -> tuple[float, ...]:
    if not values:
        return ()
    if len(values) == 1:
        return (100.0,)
    sorted_values = sorted(values)
    ranks: list[float] = []
    for value in values:
        below = sum(1 for candidate in sorted_values if candidate < value)
        equal = sum(1 for candidate in sorted_values if candidate == value)
        percentile = (below + ((equal - 1) / 2.0)) / (len(values) - 1)
        ranks.append(percentile * 100.0)
    return tuple(ranks)
