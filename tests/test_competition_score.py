from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean, pstdev
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.backtesting.competition_score import (
    RiskDisciplineSample,
    build_competition_metrics,
    build_risk_discipline_report,
    non_annualized_sharpe,
    official_composite_score,
)


LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class _EquityPoint:
    timestamp: str
    equity: float
    position_notional_usd: float = 0.0


class CompetitionScoreTest(TestCase):
    def test_non_annualized_sharpe_matches_rules_formula(self) -> None:
        returns = [0.01, 0.02, -0.005, 0.015]

        sharpe = non_annualized_sharpe(returns)

        self.assertAlmostEqual(sharpe, fmean(returns) / pstdev(returns))

    def test_competition_metrics_sample_15_minute_equity_returns(self) -> None:
        points = tuple(
            _EquityPoint(
                timestamp=_time(index).isoformat(timespec="seconds"),
                equity=1_000_000 + (index * 1_000),
            )
            for index in range(10)
        )

        metrics = build_competition_metrics(equity_points=points, fills=("fill",))

        self.assertEqual(metrics.sampled_equity_points, 4)
        self.assertEqual(metrics.return_observations, 3)
        self.assertEqual(metrics.trade_count, 1)
        self.assertTrue(metrics.sharpe_rank_is_capped)
        self.assertEqual(metrics.sharpe_rank_cap, 50.0)

    def test_composite_score_uses_official_weights_and_sharpe_cap(self) -> None:
        score = official_composite_score(
            return_rank=80,
            drawdown_rank=60,
            sharpe_rank=90,
            risk_discipline_score=100,
            sharpe_rank_cap=50,
        )

        self.assertAlmostEqual(score, 75.0)

    def test_risk_discipline_detects_persistent_leverage_penalty(self) -> None:
        samples = tuple(
            RiskDisciplineSample(
                timestamp=_time(index * 3),
                equity=1_000_000,
                gross_notional_usd=29_000_000,
                net_notional_usd=0,
                largest_symbol_notional_usd=5_000_000,
            )
            for index in range(3)
        )

        report = build_risk_discipline_report(samples)

        self.assertTrue(any(breach.rule_id == "leverage_gt_28" for breach in report.breaches))
        self.assertLess(report.score, 100)

    def test_risk_discipline_detects_concentration_penalties(self) -> None:
        samples = (
            RiskDisciplineSample(
                timestamp=_time(0),
                equity=1_000_000,
                gross_notional_usd=50_000,
                net_notional_usd=50_000,
                largest_symbol_notional_usd=50_000,
            ),
            RiskDisciplineSample(
                timestamp=_time(6),
                equity=1_000_000,
                gross_notional_usd=50_000,
                net_notional_usd=50_000,
                largest_symbol_notional_usd=50_000,
            ),
        )

        report = build_risk_discipline_report(samples)
        rule_ids = {breach.rule_id for breach in report.breaches}

        self.assertIn("single_instrument_concentration_gt_90", rule_ids)
        self.assertIn("net_directional_concentration_gt_95", rule_ids)
        self.assertEqual(report.score, 80)

    def test_risk_discipline_marks_review_thresholds(self) -> None:
        samples = (
            RiskDisciplineSample(
                timestamp=_time(0),
                equity=1_000_000,
                gross_notional_usd=29_700_000,
                net_notional_usd=0,
                largest_symbol_notional_usd=1_000_000,
            ),
            RiskDisciplineSample(
                timestamp=_time(3),
                equity=1_000_000,
                gross_notional_usd=29_700_000,
                net_notional_usd=0,
                largest_symbol_notional_usd=1_000_000,
            ),
        )

        report = build_risk_discipline_report(samples)

        self.assertTrue(report.compliance_review_required)


def _time(index: int) -> datetime:
    return datetime(2026, 6, 22, 10, 0, tzinfo=LONDON) + timedelta(minutes=5 * index)

