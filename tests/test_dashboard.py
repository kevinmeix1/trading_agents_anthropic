from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.reporting.dashboard import DashboardOptions, build_dashboard_payload


class DashboardTest(TestCase):
    def test_dashboard_payload_reads_backtest_and_live_sources(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            backtests = root / "backtests"
            backtests.mkdir()
            comparison = backtests / "strategy_comparison.csv"
            comparison.write_text(
                "\n".join(
                    [
                        "rank,strategy,symbol,final_equity,total_return_pct,sharpe_ratio,max_drawdown_pct,fills,turnover_notional",
                        "1,breakout,EURUSD,1000100,0.0001,1.5,0.0002,4,200000",
                        "2,mean_reversion,EURUSD,999900,-0.0001,-1.0,0.0004,3,150000",
                    ]
                ),
                encoding="utf-8",
            )
            equity = backtests / "equity_curve.csv"
            equity.write_text(
                "\n".join(
                    [
                        "timestamp,equity,cash,position_units,position_notional_usd,drawdown_pct",
                        "2026-06-22T12:00:00+00:00,1000000,1000000,0,0,0",
                        "2026-06-22T12:15:00+00:00,1000100,1000000,1,100,0",
                    ]
                ),
                encoding="utf-8",
            )
            monitor = root / "live_monitor.csv"
            monitor.write_text(
                "\n".join(
                    [
                        "timestamp,equity,daily_pnl_pct,drawdown_pct,margin_level_pct,gross_notional_usd,net_notional_usd,leverage,margin_usage,single_symbol_concentration,net_directional_exposure,accepted_trade_count",
                        "2026-06-22T12:00:00+00:00,1000000,0,0,2000,0,0,0,0,0,0,0",
                    ]
                ),
                encoding="utf-8",
            )
            live_journal = root / "live.jsonl"
            live_journal.write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-06-22T12:00:00+00:00",
                        "mode": "QUALIFY",
                        "status": "DRY_RUN_ACCEPTED",
                        "request": {
                            "symbol": "EURUSD",
                            "side": "BUY",
                            "target_notional_usd": 50_000,
                        },
                        "decision": {
                            "approved": True,
                            "adjusted_notional_usd": 50_000,
                            "reason": "approved",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            payload = build_dashboard_payload(
                DashboardOptions(
                    backtest_dir=backtests,
                    live_monitor_path=monitor,
                    live_journal_path=live_journal,
                    dry_journal_path=root / "missing_dry.jsonl",
                )
            )

        self.assertEqual(len(payload["backtests"]["comparisons"]), 1)
        self.assertEqual(
            payload["backtests"]["comparisons"][0]["best"]["strategy"],
            "breakout",
        )
        self.assertEqual(len(payload["backtests"]["equity_curves"]), 1)
        self.assertTrue(payload["live"]["ready"])
        self.assertEqual(payload["live"]["journal"]["accepted"], 1)
        self.assertEqual(payload["live"]["journal"]["positions"][0]["symbol"], "EURUSD")

    def test_dashboard_payload_handles_missing_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = build_dashboard_payload(
                DashboardOptions(
                    backtest_dir=root / "missing_backtests",
                    live_monitor_path=root / "missing_monitor.csv",
                    live_journal_path=root / "missing_live.jsonl",
                    dry_journal_path=root / "missing_dry.jsonl",
                )
            )

        self.assertEqual(payload["backtests"]["comparisons"], [])
        self.assertFalse(payload["live"]["ready"])
        self.assertEqual(payload["dry_run"]["records"], 0)
