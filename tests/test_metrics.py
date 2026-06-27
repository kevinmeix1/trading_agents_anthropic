from unittest import TestCase

from quanthack.backtesting.metrics import (
    MAX_PROFIT_FACTOR,
    compute_returns,
    max_drawdown,
    profit_factor_from_pnl,
    summarize_performance,
)


class MetricsTest(TestCase):
    def test_compute_returns(self) -> None:
        returns = compute_returns([100.0, 110.0, 99.0])

        self.assertAlmostEqual(returns[0], 0.1)
        self.assertAlmostEqual(returns[1], -0.1)

    def test_max_drawdown(self) -> None:
        self.assertAlmostEqual(max_drawdown([100.0, 120.0, 90.0, 110.0]), 0.25)

    def test_profit_factor(self) -> None:
        self.assertEqual(profit_factor_from_pnl([10.0, -5.0, 5.0]), 3.0)
        self.assertEqual(profit_factor_from_pnl([10.0, 5.0]), MAX_PROFIT_FACTOR)
        self.assertEqual(profit_factor_from_pnl([0.0, 0.0]), 0.0)
        self.assertEqual(profit_factor_from_pnl([10_000.0, -1.0]), MAX_PROFIT_FACTOR)

    def test_summarize_performance(self) -> None:
        metrics = summarize_performance(
            equity_curve=[100.0, 101.0, 100.0, 103.0],
            turnover_notional=1_000.0,
            periods_per_year=252.0,
        )

        self.assertEqual(metrics.observations, 4)
        self.assertEqual(metrics.final_equity, 103.0)
        self.assertAlmostEqual(metrics.total_return_pct, 0.03)
        self.assertGreater(metrics.max_drawdown_pct, 0)
        self.assertEqual(metrics.turnover_notional, 1_000.0)
        self.assertEqual(metrics.positive_return_count, 2)
        self.assertEqual(metrics.negative_return_count, 1)
        self.assertEqual(metrics.flat_return_count, 0)

    def test_summarize_reports_win_rate_with_and_without_flat_returns(self) -> None:
        metrics = summarize_performance(
            equity_curve=[100.0, 101.0, 101.0, 100.0],
            turnover_notional=1_000.0,
            periods_per_year=252.0,
        )

        self.assertEqual(metrics.win_rate, 0.5)
        self.assertAlmostEqual(metrics.win_rate_including_flat, 1 / 3)
        self.assertAlmostEqual(metrics.positive_period_rate, 1 / 3)
        self.assertEqual(metrics.flat_return_count, 1)
