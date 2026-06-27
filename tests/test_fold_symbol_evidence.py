from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.reporting.fold_symbol_evidence import (
    FoldSymbolEvidencePolicy,
    build_fold_symbol_evidence_report,
    sweep_fold_symbol_evidence_policies,
    write_fold_symbol_evidence_detail_csv,
    write_fold_symbol_evidence_summary_csv,
    write_fold_symbol_evidence_sweep_csv,
)


class FoldSymbolEvidenceTest(TestCase):
    def test_simulates_prior_only_symbol_gate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            attribution = root / "attribution.csv"
            folds = root / "folds.csv"
            detail = root / "detail.csv"
            summary = root / "summary.csv"
            folds.write_text(
                "fold,test_start,test_end,return_pct,evaluation_fills\n"
                "1,a,b,-0.01,2\n"
                "2,b,c,-0.01,2\n"
                "3,c,d,0.02,2\n",
                encoding="utf-8",
            )
            attribution.write_text(
                "fold,fold_return_pct,symbol,primary_signal,utc_hour,side,"
                "fills,realized_events,wins,losses,win_rate,realized_pnl_usd,"
                "turnover_notional_usd,adjusted_notional_usd\n"
                "1,-0.01,EURUSD,kalman_trend,12,BUY,2,1,0,1,0,-100,1000,1000\n"
                "2,-0.01,EURUSD,kalman_trend,12,BUY,2,1,0,1,0,-50,1000,1000\n"
                "3,0.02,EURUSD,kalman_trend,12,BUY,2,1,1,0,1,200,1000,1000\n",
                encoding="utf-8",
            )

            report = build_fold_symbol_evidence_report(
                attribution_csv=attribution,
                folds_csv=folds,
                symbols=("EURUSD",),
                policy=FoldSymbolEvidencePolicy(
                    lookback_folds=1,
                    min_prior_pnl_usd=0.0,
                    allow_without_history=True,
                ),
            )
            write_fold_symbol_evidence_detail_csv(report, detail)
            write_fold_symbol_evidence_summary_csv(report, summary)
            detail_text = detail.read_text(encoding="utf-8")
            summary_text = summary.read_text(encoding="utf-8")

        rows = report.rows
        self.assertTrue(rows[0].allowed)
        self.assertFalse(rows[1].allowed)
        self.assertTrue(rows[2].allowed)
        self.assertEqual(rows[2].prior_realized_events, 0)
        self.assertAlmostEqual(report.ungated_realized_pnl_usd, 50.0)
        self.assertAlmostEqual(report.gated_realized_pnl_usd, 100.0)
        self.assertAlmostEqual(report.avoided_loss_usd, 50.0)
        self.assertAlmostEqual(report.missed_gain_usd, 0.0)
        self.assertIn("fold,fold_return_pct,symbol", detail_text)
        self.assertIn("scope,fold,fold_return_pct", summary_text)

    def test_can_block_symbols_without_history(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            attribution = root / "attribution.csv"
            folds = root / "folds.csv"
            folds.write_text(
                "fold,test_start,test_end,return_pct,evaluation_fills\n"
                "1,a,b,0.01,2\n",
                encoding="utf-8",
            )
            attribution.write_text(
                "fold,fold_return_pct,symbol,primary_signal,utc_hour,side,"
                "fills,realized_events,wins,losses,win_rate,realized_pnl_usd,"
                "turnover_notional_usd,adjusted_notional_usd\n"
                "1,0.01,EURUSD,kalman_trend,12,BUY,2,1,1,0,1,100,1000,1000\n",
                encoding="utf-8",
            )

            report = build_fold_symbol_evidence_report(
                attribution_csv=attribution,
                folds_csv=folds,
                symbols=("EURUSD",),
                policy=FoldSymbolEvidencePolicy(allow_without_history=False),
            )

        self.assertFalse(report.rows[0].allowed)
        self.assertAlmostEqual(report.missed_gain_usd, 100.0)

    def test_sweeps_and_ranks_policy_candidates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            attribution = root / "attribution.csv"
            folds = root / "folds.csv"
            output = root / "sweep.csv"
            folds.write_text(
                "fold,test_start,test_end,return_pct,evaluation_fills\n"
                "1,a,b,-0.01,2\n"
                "2,b,c,-0.01,2\n",
                encoding="utf-8",
            )
            attribution.write_text(
                "fold,fold_return_pct,symbol,primary_signal,utc_hour,side,"
                "fills,realized_events,wins,losses,win_rate,realized_pnl_usd,"
                "turnover_notional_usd,adjusted_notional_usd\n"
                "1,-0.01,EURUSD,kalman_trend,12,BUY,2,1,1,0,1,100,1000,1000\n"
                "2,-0.01,EURUSD,kalman_trend,12,BUY,2,1,0,1,0,-75,1000,1000\n",
                encoding="utf-8",
            )

            report = sweep_fold_symbol_evidence_policies(
                attribution_csv=attribution,
                folds_csv=folds,
                symbols=("EURUSD",),
                lookback_folds_values=(1, 2),
                min_prior_pnl_usd_values=(0.0, 150.0),
                min_prior_win_rate_values=(0.0,),
            )
            write_fold_symbol_evidence_sweep_csv(report, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(len(report.candidates), 4)
        self.assertIsNotNone(report.best)
        self.assertEqual(
            [candidate.rank_key for candidate in report.candidates],
            sorted([candidate.rank_key for candidate in report.candidates], reverse=True),
        )
        self.assertIn("rank,lookback_folds", text)
