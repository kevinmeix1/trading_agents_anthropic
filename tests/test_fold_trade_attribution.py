from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.reporting.fold_trade_attribution import (
    build_fold_trade_attribution_report,
    write_fold_trade_attribution_csv,
)


class FoldTradeAttributionTest(TestCase):
    def test_attributes_realized_pnl_to_entry_fold_signal_hour(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fills = root / "fills.csv"
            folds = root / "folds.csv"
            output = root / "attr.csv"
            fills.write_text(
                "timestamp,symbol,side,fill_price,trade_units,turnover_notional_usd,"
                "requested_notional_usd,adjusted_notional_usd,risk_reason,"
                "primary_signal,supporting_signals,conflicting_signals\n"
                "2026-06-01T10:00:00+00:00,EURUSD,BUY,1.0,1000,1000,1000,1000,"
                "approved,kalman_trend,,\n"
                "2026-06-01T10:15:00+00:00,EURUSD,SELL,1.1,-1000,1100,0,0,"
                "exit,none,,\n"
                "2026-06-02T11:00:00+00:00,EURUSD,BUY,1.2,1000,1200,1200,1200,"
                "approved,macd_momentum,,\n"
                "2026-06-02T11:15:00+00:00,EURUSD,SELL,1.1,-1000,1100,0,0,"
                "exit,none,,\n",
                encoding="utf-8",
            )
            folds.write_text(
                "fold,train_start,train_end,test_start,test_end,return_pct,"
                "max_drawdown_pct,sharpe_15m,risk_discipline_score,"
                "evaluation_fills,full_run_fills,final_equity\n"
                "1,2026-05-01T00:00:00+00:00,2026-05-31T00:00:00+00:00,"
                "2026-06-01T00:00:00+00:00,2026-06-01T23:59:00+00:00,"
                "0.01,0,0,100,2,2,1010000\n"
                "2,2026-05-02T00:00:00+00:00,2026-06-01T00:00:00+00:00,"
                "2026-06-02T00:00:00+00:00,2026-06-02T23:59:00+00:00,"
                "-0.01,0,0,100,2,2,990000\n",
                encoding="utf-8",
            )

            report = build_fold_trade_attribution_report(
                fills_csv=fills,
                folds_csv=folds,
            )
            write_fold_trade_attribution_csv(report, output)
            text = output.read_text(encoding="utf-8")

        by_signal = {row.primary_signal: row for row in report.rows}
        self.assertAlmostEqual(by_signal["kalman_trend"].realized_pnl_usd, 100.0)
        self.assertAlmostEqual(by_signal["macd_momentum"].realized_pnl_usd, -100.0)
        self.assertEqual(by_signal["kalman_trend"].fold, 1)
        self.assertEqual(by_signal["macd_momentum"].fold, 2)
        self.assertIn("fold,fold_return_pct,symbol", text)
