from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.reporting.adaptive_handoff_diagnostic import (
    build_adaptive_handoff_diagnostic,
    write_adaptive_handoff_diagnostic_csv,
)


class AdaptiveHandoffDiagnosticTest(TestCase):
    def test_labels_hindsight_chop_breakout_and_cash_loss(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oracle_folds = root / "oracle_folds.csv"
            oracle_candidates = root / "oracle_candidates.csv"
            regime_summary = root / "regime_summary.csv"
            output = root / "handoff.csv"
            oracle_folds.write_text(
                "\n".join(
                    [
                        "fold,test_start,test_end,selected_strategy,oracle_strategy,"
                        "selected_return_pct,oracle_return_pct,regret_pct,"
                        "selected_max_drawdown_pct,oracle_max_drawdown_pct,"
                        "selected_fills,oracle_fills,selected_was_oracle,"
                        "selected_was_negative,oracle_was_cash",
                        "1,a,b,macd_momentum,champion_ensemble,0.01,0.05,"
                        "0.04,0.01,0.02,4,5,no,no,no",
                        "2,a,b,macd_momentum,cash,-0.01,0,0.01,0.01,0,3,0,"
                        "no,yes,yes",
                    ]
                ),
                encoding="utf-8",
            )
            oracle_candidates.write_text(
                "\n".join(
                    [
                        "fold,strategy,train_rank,selected_by_policy,oracle_best,"
                        "train_return_pct,train_drawdown_adjusted_return_pct,"
                        "train_max_drawdown_pct,train_sharpe_15m,train_fills,"
                        "test_return_pct,test_max_drawdown_pct,test_sharpe_15m,"
                        "test_fills",
                        "1,macd_momentum,1,yes,no,0.01,0.01,0.01,0.1,10,"
                        "0.01,0.01,0.1,4",
                        "1,champion_ensemble,2,no,yes,-0.01,-0.02,0.01,-0.1,8,"
                        "0.05,0.02,0.2,5",
                        "1,kalman_trend,3,no,no,0,0,0,0,0,0,0,0,0",
                        "2,macd_momentum,1,yes,no,0.01,0.01,0.01,0.1,10,"
                        "-0.01,0.01,-0.1,3",
                        "2,champion_ensemble,2,no,no,0,0,0,0,0,-0.02,0.02,-0.2,4",
                        "2,kalman_trend,3,no,no,0,0,0,0,0,-0.01,0.01,-0.1,3",
                    ]
                ),
                encoding="utf-8",
            )
            regime_summary.write_text(
                "\n".join(
                    [
                        "fold,trend_consensus,chop_fraction,high_volatility_fraction,"
                        "average_realized_volatility_bps,average_trend_efficiency,"
                        "net_slope_bps",
                        "1,0,1,0,8,0.1,0",
                        "2,0.2,0.6,0,10,0.2,1",
                    ]
                ),
                encoding="utf-8",
            )

            report = build_adaptive_handoff_diagnostic(
                oracle_folds_csv=oracle_folds,
                oracle_candidates_csv=oracle_candidates,
                regime_summary_csv=regime_summary,
            )
            write_adaptive_handoff_diagnostic_csv(report, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(report.fold_count, 2)
        self.assertEqual(report.rows[0].diagnosis, "HINDSIGHT_CHOP_BREAKOUT")
        self.assertEqual(report.rows[1].diagnosis, "CASH_AVOIDABLE_LOSS")
        self.assertIn("champion_minus_macd_train_adjusted_pct", text)
        self.assertIn("HINDSIGHT_CHOP_BREAKOUT", text)
