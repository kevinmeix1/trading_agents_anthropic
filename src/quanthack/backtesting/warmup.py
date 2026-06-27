from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quanthack.backtesting.backtest import BacktestFill
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestResult,
    PortfolioEquityPoint,
)


@dataclass(frozen=True)
class WarmupPortfolioEvaluation:
    evaluation_start: str
    equity_curve: tuple[PortfolioEquityPoint, ...]
    fills: tuple[BacktestFill, ...]
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


def evaluate_portfolio_after_warmup(
    result: PortfolioBacktestResult,
    *,
    evaluation_start: datetime | str,
) -> WarmupPortfolioEvaluation:
    start = _parse_timestamp(evaluation_start)
    equity_points = tuple(
        point
        for point in result.equity_curve
        if _parse_timestamp(point.timestamp) >= start
    )
    if not equity_points:
        raise ValueError("evaluation_start is after the portfolio equity curve")

    fills = tuple(
        fill
        for fill in result.fills
        if _parse_timestamp(fill.timestamp) >= start
    )
    metrics = build_competition_metrics(
        equity_points=equity_points,
        fills=fills,
    )
    risk_discipline = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(equity_points)
    )
    return WarmupPortfolioEvaluation(
        evaluation_start=start.isoformat(),
        equity_curve=equity_points,
        fills=fills,
        competition_metrics=metrics,
        risk_discipline=risk_discipline,
    )


def _parse_timestamp(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("evaluation timestamps must include timezone information")
    return timestamp
