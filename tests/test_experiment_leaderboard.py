from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.experiment_leaderboard import (
    build_experiment_leaderboard,
    write_experiment_leaderboard_csv,
)


class ExperimentLeaderboardTest(TestCase):
    def test_builds_and_writes_ranked_leaderboard(self) -> None:
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            adaptive = directory / "adaptive_summary.csv"
            fixed = directory / "fixed_summary.csv"
            adaptive.write_text(
                "\n".join(
                    [
                        "strategies,symbols,folds,positive_fold_fraction,"
                        "active_fold_fraction,active_positive_fold_fraction,"
                        "non_negative_fold_fraction,median_active_test_return_pct,"
                        "worst_test_drawdown_pct,average_risk_discipline_score,"
                        "total_evaluation_fills",
                        "a b,EURUSD GBPUSD,4,0.5,0.75,0.67,0.75,0.001,0.01,100,12",
                    ]
                ),
                encoding="utf-8",
            )
            fixed.write_text(
                "\n".join(
                    [
                        "strategy,symbols,folds,positive_fold_fraction,"
                        "active_fold_fraction,active_positive_fold_fraction,"
                        "non_negative_fold_fraction,median_active_test_return_pct,"
                        "worst_test_drawdown_pct,average_risk_discipline_score,"
                        "total_evaluation_fills",
                        "slow,EURUSD,4,0.25,0.5,0.5,1.0,0.0,0.002,100,4",
                    ]
                ),
                encoding="utf-8",
            )

            rows = build_experiment_leaderboard((adaptive, fixed))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].source_path, str(adaptive))
            output = directory / "leaderboard.csv"
            write_experiment_leaderboard_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("rank,label,source_path", text)
        self.assertIn("compounded_return_pct", text)
        self.assertIn("adaptive", text)
        self.assertIn("fixed", text)

    def test_compounded_return_can_break_close_stability_ties(self) -> None:
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            clean_idle = directory / "clean_idle_summary.csv"
            earning = directory / "earning_summary.csv"
            header = (
                "strategy,symbols,folds,positive_fold_fraction,"
                "active_fold_fraction,active_positive_fold_fraction,"
                "non_negative_fold_fraction,compounded_test_return_pct,"
                "median_active_test_return_pct,worst_test_drawdown_pct,"
                "average_risk_discipline_score,total_evaluation_fills"
            )
            clean_idle.write_text(
                "\n".join(
                    [
                        header,
                        "idle,EURUSD,10,0.5,0.5,1.0,1.0,0.001,0.0001,0.001,100,2",
                    ]
                ),
                encoding="utf-8",
            )
            earning.write_text(
                "\n".join(
                    [
                        header,
                        "earning,EURUSD,10,0.5,0.5,0.95,0.95,0.03,0.0001,0.001,100,20",
                    ]
                ),
                encoding="utf-8",
            )

            rows = build_experiment_leaderboard((clean_idle, earning))

        self.assertEqual(rows[0].label, "earning")
        self.assertAlmostEqual(rows[0].compounded_return_pct, 0.03)
