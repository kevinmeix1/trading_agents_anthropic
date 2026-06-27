from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from statistics import fmean, pstdev
from typing import Any


DEFAULT_MAX_LEVERAGE = 30.0


class RiskBreachSeverity(StrEnum):
    PENALTY = "PENALTY"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class CompetitionMetrics:
    starting_equity: float
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_15m: float
    sampled_equity_points: int
    return_observations: int
    sharpe_rank_cap: float
    trade_count: int

    @property
    def sharpe_rank_is_capped(self) -> bool:
        return self.sharpe_rank_cap < 100.0

    @property
    def sharpe_prize_trade_count_met(self) -> bool:
        return self.trade_count >= 30


@dataclass(frozen=True)
class RiskDisciplineSample:
    timestamp: datetime
    equity: float
    gross_notional_usd: float
    net_notional_usd: float
    largest_symbol_notional_usd: float
    max_leverage: float = DEFAULT_MAX_LEVERAGE

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("risk sample timestamp must include a timezone")
        if self.equity <= 0 or not isfinite(self.equity):
            raise ValueError("risk sample equity must be positive and finite")
        if self.gross_notional_usd < 0 or not isfinite(self.gross_notional_usd):
            raise ValueError("gross_notional_usd must be non-negative and finite")
        if not isfinite(self.net_notional_usd):
            raise ValueError("net_notional_usd must be finite")
        if self.largest_symbol_notional_usd < 0 or not isfinite(
            self.largest_symbol_notional_usd
        ):
            raise ValueError("largest_symbol_notional_usd must be non-negative and finite")
        if self.max_leverage <= 0 or not isfinite(self.max_leverage):
            raise ValueError("max_leverage must be positive and finite")

    @property
    def leverage(self) -> float:
        return self.gross_notional_usd / self.equity

    @property
    def margin_usage(self) -> float:
        used_margin = self.gross_notional_usd / self.max_leverage
        return used_margin / self.equity

    @property
    def single_instrument_exposure(self) -> float:
        if self.gross_notional_usd == 0:
            return 0.0
        return self.largest_symbol_notional_usd / self.gross_notional_usd

    @property
    def net_directional_exposure(self) -> float:
        if self.gross_notional_usd == 0:
            return 0.0
        return abs(self.net_notional_usd) / self.gross_notional_usd


@dataclass(frozen=True)
class RiskDisciplineBreach:
    rule_id: str
    category: str
    severity: RiskBreachSeverity
    started_at: datetime
    ended_at: datetime
    duration_minutes: float
    max_value: float
    threshold_value: float
    penalty_points: int
    details: str


@dataclass(frozen=True)
class RiskDisciplineReport:
    breaches: tuple[RiskDisciplineBreach, ...]
    starting_score: int = 100

    @property
    def penalty_points(self) -> int:
        return sum(
            breach.penalty_points
            for breach in self.breaches
            if breach.severity == RiskBreachSeverity.PENALTY
        )

    @property
    def score(self) -> int:
        return max(0, self.starting_score + self.penalty_points)

    @property
    def compliance_review_required(self) -> bool:
        return any(breach.severity == RiskBreachSeverity.REVIEW for breach in self.breaches)


@dataclass(frozen=True)
class _RiskRule:
    rule_id: str
    category: str
    description: str
    threshold_value: float
    min_duration_minutes: float
    penalty_points: int
    severity: RiskBreachSeverity
    metric_name: str


_RISK_RULES = (
    _RiskRule(
        rule_id="margin_usage_gt_90",
        category="margin_usage",
        description="margin usage >90%",
        threshold_value=0.90,
        min_duration_minutes=30.0,
        penalty_points=-20,
        severity=RiskBreachSeverity.PENALTY,
        metric_name="margin_usage",
    ),
    _RiskRule(
        rule_id="margin_usage_gt_95",
        category="margin_usage",
        description="margin usage >95%",
        threshold_value=0.95,
        min_duration_minutes=15.0,
        penalty_points=-30,
        severity=RiskBreachSeverity.PENALTY,
        metric_name="margin_usage",
    ),
    _RiskRule(
        rule_id="margin_usage_gt_98",
        category="margin_usage",
        description="margin usage >98%",
        threshold_value=0.98,
        min_duration_minutes=10.0,
        penalty_points=0,
        severity=RiskBreachSeverity.REVIEW,
        metric_name="margin_usage",
    ),
    _RiskRule(
        rule_id="leverage_gt_28",
        category="leverage",
        description="leverage >28x",
        threshold_value=28.0,
        min_duration_minutes=30.0,
        penalty_points=-20,
        severity=RiskBreachSeverity.PENALTY,
        metric_name="leverage",
    ),
    _RiskRule(
        rule_id="leverage_gt_29",
        category="leverage",
        description="leverage >29x",
        threshold_value=29.0,
        min_duration_minutes=15.0,
        penalty_points=-30,
        severity=RiskBreachSeverity.PENALTY,
        metric_name="leverage",
    ),
    _RiskRule(
        rule_id="leverage_approaching_30",
        category="leverage",
        description="leverage approaching 30x",
        threshold_value=29.5,
        min_duration_minutes=10.0,
        penalty_points=0,
        severity=RiskBreachSeverity.REVIEW,
        metric_name="leverage",
    ),
    _RiskRule(
        rule_id="single_instrument_concentration_gt_90",
        category="concentration",
        description="single-instrument exposure >90%",
        threshold_value=0.90,
        min_duration_minutes=30.0,
        penalty_points=-10,
        severity=RiskBreachSeverity.PENALTY,
        metric_name="single_instrument_exposure",
    ),
    _RiskRule(
        rule_id="net_directional_concentration_gt_95",
        category="concentration",
        description="net directional exposure >95%",
        threshold_value=0.95,
        min_duration_minutes=30.0,
        penalty_points=-10,
        severity=RiskBreachSeverity.PENALTY,
        metric_name="net_directional_exposure",
    ),
)


def build_competition_metrics(
    *,
    equity_points: tuple[Any, ...],
    fills: tuple[Any, ...] = (),
    interval_minutes: int = 15,
) -> CompetitionMetrics:
    points = _extract_equity_points(equity_points)
    if not points:
        raise ValueError("equity_points are required")

    sampled = _sample_equity_points(points, interval_minutes=interval_minutes)
    sampled_equities = [equity for _, equity in sampled]
    returns = _interval_returns(sampled_equities)
    starting_equity = points[0][1]
    final_equity = points[-1][1]

    return CompetitionMetrics(
        starting_equity=starting_equity,
        final_equity=final_equity,
        return_pct=(final_equity / starting_equity) - 1.0,
        max_drawdown_pct=_max_drawdown([equity for _, equity in points]),
        sharpe_15m=non_annualized_sharpe(returns),
        sampled_equity_points=len(sampled_equities),
        return_observations=len(returns),
        sharpe_rank_cap=50.0 if len(returns) < 8 else 100.0,
        trade_count=len(fills),
    )


def build_risk_discipline_report(
    samples: tuple[RiskDisciplineSample, ...],
) -> RiskDisciplineReport:
    sorted_samples = tuple(sorted(samples, key=lambda sample: sample.timestamp))
    breaches: list[RiskDisciplineBreach] = []
    for rule in _RISK_RULES:
        breaches.extend(_detect_rule_breaches(rule, sorted_samples))
    return RiskDisciplineReport(breaches=tuple(breaches))


def risk_samples_from_single_symbol_equity(
    equity_points: tuple[Any, ...],
    *,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> tuple[RiskDisciplineSample, ...]:
    samples: list[RiskDisciplineSample] = []
    for point in equity_points:
        timestamp = _parse_timestamp(getattr(point, "timestamp"))
        equity = float(getattr(point, "equity"))
        notional = float(getattr(point, "position_notional_usd"))
        samples.append(
            RiskDisciplineSample(
                timestamp=timestamp,
                equity=equity,
                gross_notional_usd=abs(notional),
                net_notional_usd=notional,
                largest_symbol_notional_usd=abs(notional),
                max_leverage=max_leverage,
            )
        )
    return tuple(samples)


def risk_samples_from_portfolio_equity(
    equity_points: tuple[Any, ...],
    *,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> tuple[RiskDisciplineSample, ...]:
    samples: list[RiskDisciplineSample] = []
    for point in equity_points:
        positions = tuple(getattr(point, "positions"))
        largest = max((abs(position.notional_usd) for position in positions), default=0.0)
        samples.append(
            RiskDisciplineSample(
                timestamp=_parse_timestamp(getattr(point, "timestamp")),
                equity=float(getattr(point, "equity")),
                gross_notional_usd=float(getattr(point, "gross_notional_usd")),
                net_notional_usd=float(getattr(point, "net_notional_usd")),
                largest_symbol_notional_usd=largest,
                max_leverage=max_leverage,
            )
        )
    return tuple(samples)


def non_annualized_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    return fmean(returns) / volatility


def official_composite_score(
    *,
    return_rank: float,
    drawdown_rank: float,
    sharpe_rank: float,
    risk_discipline_score: float,
    sharpe_rank_cap: float | None = None,
) -> float:
    _validate_rank_score("return_rank", return_rank)
    _validate_rank_score("drawdown_rank", drawdown_rank)
    _validate_rank_score("sharpe_rank", sharpe_rank)
    _validate_rank_score("risk_discipline_score", risk_discipline_score)
    if sharpe_rank_cap is not None:
        _validate_rank_score("sharpe_rank_cap", sharpe_rank_cap)
        sharpe_rank = min(sharpe_rank, sharpe_rank_cap)
    return (
        (0.70 * return_rank)
        + (0.15 * drawdown_rank)
        + (0.10 * sharpe_rank)
        + (0.05 * risk_discipline_score)
    )


def _detect_rule_breaches(
    rule: _RiskRule,
    samples: tuple[RiskDisciplineSample, ...],
) -> list[RiskDisciplineBreach]:
    breaches: list[RiskDisciplineBreach] = []
    run_start: RiskDisciplineSample | None = None
    run_end: RiskDisciplineSample | None = None
    run_max = 0.0

    for sample in samples:
        value = float(getattr(sample, rule.metric_name))
        if value > rule.threshold_value:
            if run_start is None:
                run_start = sample
                run_max = value
            run_end = sample
            run_max = max(run_max, value)
            continue

        if run_start is not None and run_end is not None:
            breach = _close_breach(rule, run_start, run_end, run_max)
            if breach is not None:
                breaches.append(breach)
        run_start = None
        run_end = None
        run_max = 0.0

    if run_start is not None and run_end is not None:
        breach = _close_breach(rule, run_start, run_end, run_max)
        if breach is not None:
            breaches.append(breach)

    return breaches


def _close_breach(
    rule: _RiskRule,
    start: RiskDisciplineSample,
    end: RiskDisciplineSample,
    max_value: float,
) -> RiskDisciplineBreach | None:
    duration_minutes = (end.timestamp - start.timestamp).total_seconds() / 60
    if duration_minutes < rule.min_duration_minutes:
        return None
    return RiskDisciplineBreach(
        rule_id=rule.rule_id,
        category=rule.category,
        severity=rule.severity,
        started_at=start.timestamp,
        ended_at=end.timestamp,
        duration_minutes=duration_minutes,
        max_value=max_value,
        threshold_value=rule.threshold_value,
        penalty_points=rule.penalty_points,
        details=(
            f"{rule.description} persisted for {duration_minutes:.1f} minutes "
            f"(max={max_value:.3f})"
        ),
    )


def _extract_equity_points(equity_points: tuple[Any, ...]) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for point in equity_points:
        timestamp = _parse_timestamp(getattr(point, "timestamp"))
        equity = float(getattr(point, "equity"))
        if equity <= 0 or not isfinite(equity):
            raise ValueError("equity values must be positive and finite")
        points.append((timestamp, equity))
    return sorted(points, key=lambda item: item[0])


def _sample_equity_points(
    points: list[tuple[datetime, float]],
    *,
    interval_minutes: int,
) -> list[tuple[datetime, float]]:
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")

    interval = timedelta(minutes=interval_minutes)
    target = points[0][0]
    final_timestamp = points[-1][0]
    index = 0
    latest: tuple[datetime, float] | None = None
    sampled: list[tuple[datetime, float]] = []

    while target <= final_timestamp:
        while index < len(points) and points[index][0] <= target:
            latest = points[index]
            index += 1
        if latest is not None and (not sampled or sampled[-1][0] != latest[0]):
            sampled.append(latest)
        target += interval

    return sampled


def _interval_returns(equities: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(equities, equities[1:]):
        if previous <= 0:
            raise ValueError("equity values must stay positive")
        returns.append((current / previous) - 1.0)
    return returns


def _max_drawdown(equities: list[float]) -> float:
    peak = equities[0]
    worst = 0.0
    for equity in equities:
        peak = max(peak, equity)
        worst = max(worst, 1.0 - (equity / peak))
    return worst


def _parse_timestamp(value: datetime | str) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return timestamp


def _validate_rank_score(name: str, value: float) -> None:
    if not 0 <= value <= 100 or not isfinite(value):
        raise ValueError(f"{name} must be between 0 and 100")

