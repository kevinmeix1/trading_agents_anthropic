from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.market.market_data import PriceBar, PriceHistory
from quanthack.reporting.fold_regime_diagnostics import (
    build_fold_regime_diagnostics_report,
    write_fold_regime_detail_csv,
    write_fold_regime_summary_csv,
)
from quanthack.strategies.time_series import KalmanTrendConfig


class FoldRegimeDiagnosticsTest(TestCase):
    def test_summarizes_ex_ante_regime_by_fold_and_asset_class(self) -> None:
        start = datetime(2026, 6, 1, tzinfo=UTC)
        timestamps = tuple(start + timedelta(minutes=15 * index) for index in range(70))
        bars: list[PriceBar] = []
        for index, timestamp in enumerate(timestamps):
            bars.append(
                PriceBar(
                    timestamp=timestamp,
                    symbol="EURUSD",
                    close=1.0 + (0.001 * index),
                )
            )
            bars.append(
                PriceBar(
                    timestamp=timestamp,
                    symbol="XAUUSD",
                    close=2_000.0 - float(index),
                )
            )
        prices = PriceHistory(tuple(bars))

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            folds = root / "folds.csv"
            detail = root / "detail.csv"
            summary = root / "summary.csv"
            folds.write_text(
                "fold,train_start,train_end,test_start,test_end,return_pct,"
                "max_drawdown_pct,sharpe_15m,risk_discipline_score,"
                "evaluation_fills,full_run_fills,final_equity\n"
                f"1,{timestamps[0].isoformat()},{timestamps[39].isoformat()},"
                f"{timestamps[40].isoformat()},{timestamps[49].isoformat()},"
                "0.01,0,0,100,2,2,1010000\n"
                f"2,{timestamps[10].isoformat()},{timestamps[49].isoformat()},"
                f"{timestamps[50].isoformat()},{timestamps[59].isoformat()},"
                "-0.02,0,0,100,2,2,980000\n",
                encoding="utf-8",
            )

            report = build_fold_regime_diagnostics_report(
                prices=prices,
                folds_csv=folds,
                symbols=("EURUSD", "XAUUSD"),
                config=KalmanTrendConfig(lookback=20),
            )
            write_fold_regime_detail_csv(report.detail_rows, detail)
            write_fold_regime_summary_csv(report.summary_rows, summary)
            detail_text = detail.read_text(encoding="utf-8")
            summary_text = summary.read_text(encoding="utf-8")

        self.assertEqual(len(report.detail_rows), 4)
        self.assertEqual(len(report.summary_rows), 2)
        first = report.summary_rows[0]
        self.assertEqual(first.trend_up_symbols, 1)
        self.assertEqual(first.trend_down_symbols, 1)
        self.assertGreater(first.forex_net_slope_bps, 0.0)
        self.assertLess(first.metal_net_slope_bps, 0.0)
        self.assertEqual(report.weakest_folds[0].fold, 2)
        self.assertEqual(report.strongest_folds[0].fold, 1)
        self.assertIn("fold,fold_return_pct,test_start", detail_text)
        self.assertIn("trend_consensus", summary_text)
