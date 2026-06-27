from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
import importlib
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from quanthack import cli
from quanthack.cli import (
    adaptive_handoff_diagnostic,
    adaptive_strategy_policy_sweep,
    adaptive_strategy_oracle,
    adaptive_strategy_select,
    archive_data_coverage,
    asset_class_attribution,
    autocorrelation_regime_optimize,
    backtest,
    candidate_scorecard,
    champion_ensemble_optimize,
    compare_strategies,
    dual_squeeze_optimize,
    fold_complement,
    fixing_reversal_optimize,
    fetch_crypto_proxy_data,
    generate_sample_data,
    hackathon_readiness,
    import_backtest_data,
    kalman_trend_optimize,
    live_dry_run,
    manual_ticket,
    merge_market_data,
    ml_alpha_report,
    mt5_capture,
    mt5_probe,
    portfolio_attribution_report,
    portfolio_backtest,
    portfolio_compare,
    portfolio_fixed_warmup_walk_forward,
    portfolio_robustness,
    portfolio_router_walk_forward,
    portfolio_universe_scan,
    portfolio_walk_forward,
    relative_strength_optimize,
    router_optimize,
    session_momentum_optimize,
    show_instruments,
    sizing_frontier,
    strategy_map_optimize,
    strategy_attribution,
    strategy_demo,
    symbol_eligibility_optimize,
    tech_prize_demo,
    tech_prize_dashboard,
    tech_prize_guardrails,
    tech_prize_judge_packet,
    tech_prize_pack,
    tech_prize_red_team,
    tech_prize_rehearse,
    tech_prize_rubric,
    tech_prize_runbook,
    tech_prize_submit,
    tech_prize_topology,
    tech_prize_trace,
    tech_prize_workflow,
    time_series_report,
    trend_pullback_optimize,
    validate_market_data,
    volatility_squeeze_optimize,
    walk_forward,
)


class CliTest(TestCase):
    def test_root_cli_prints_command_index(self) -> None:
        output = _capture(cli.main, ["--help"])

        self.assertIn("QuanHack CLI", output)
        self.assertIn("backtest", output)
        self.assertIn("walk-forward", output)
        self.assertIn("tech-prize-demo", output)
        self.assertIn("tech-prize-pack", output)
        self.assertIn("tech-prize-dashboard", output)
        self.assertIn("tech-prize-workflow", output)
        self.assertIn("tech-prize-trace", output)
        self.assertIn("tech-prize-guardrails", output)
        self.assertIn("tech-prize-topology", output)
        self.assertIn("tech-prize-runbook", output)
        self.assertIn("tech-prize-rehearse", output)
        self.assertIn("tech-prize-rubric", output)
        self.assertIn("tech-prize-red-team", output)
        self.assertIn("tech-prize-submit", output)
        self.assertIn("tech-prize-judge-packet", output)

    def test_all_registered_cli_modules_are_importable(self) -> None:
        for module_name, _ in cli.COMMANDS.values():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(module.main))

    def test_fold_complement_cli_compares_fold_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.csv"
            candidate_path = Path(tmpdir) / "candidate.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            detail_path = Path(tmpdir) / "detail.csv"
            baseline_path.write_text(
                "fold,test_start,test_end,return_pct,evaluation_fills\n"
                "1,a,b,0.0,0\n"
                "2,b,c,-0.01,2\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                "fold,test_start,test_end,return_pct,evaluation_fills\n"
                "1,a,b,0.02,2\n"
                "2,b,c,0.01,2\n",
                encoding="utf-8",
            )

            output = _capture(
                fold_complement.main,
                [
                    "--baseline-folds",
                    str(baseline_path),
                    "--candidate",
                    f"sleeve={candidate_path}",
                    "--summary-output",
                    str(summary_path),
                    "--detail-output",
                    str(detail_path),
                ],
            )

            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertIn("Fold Complement Analysis", output)
        self.assertIn("sleeve", summary_text)

    def test_archive_data_coverage_cli_reports_missing_crypto(self) -> None:
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "pricer-output-sample.zip"
            output_path = Path(tmpdir) / "coverage.csv"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("EURUSD_2026_05_11.parquet", b"dummy")
                zip_file.writestr("XAUUSD_2026_05_11.parquet", b"dummy")

            output = _capture(
                archive_data_coverage.main,
                [
                    "--input",
                    str(archive),
                    "--crypto-symbols",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Archive Data Coverage", output)
        self.assertIn("Missing (5): BARUSD, BTCUSD, ETHUSD, SOLUSD, XRPUSD", output)
        self.assertIn("BTCUSD,MISSING,0,,", csv_text)

    def test_fetch_crypto_proxy_data_requires_research_confirmation(self) -> None:
        with self.assertRaises(SystemExit) as context:
            _capture(fetch_crypto_proxy_data.main, ["--symbol", "BTCUSD"])

        self.assertIn("research-only", str(context.exception))

    def test_merge_market_data_cli_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_a = root / "prices_a.csv"
            quote_a = root / "quotes_a.csv"
            price_b = root / "prices_b.csv"
            quote_b = root / "quotes_b.csv"
            merged_prices = root / "merged_prices.csv"
            merged_quotes = root / "merged_quotes.csv"
            price_a.write_text(
                "timestamp,symbol,close\n"
                "2026-06-01T00:00:00+00:00,EURUSD,1.1000\n"
                "2026-06-01T00:15:00+00:00,EURUSD,1.1010\n",
                encoding="utf-8",
            )
            quote_a.write_text(
                "timestamp,symbol,bid,ask\n"
                "2026-06-01T00:00:00+00:00,EURUSD,1.0999,1.1001\n"
                "2026-06-01T00:15:00+00:00,EURUSD,1.1009,1.1011\n",
                encoding="utf-8",
            )
            price_b.write_text(
                "timestamp,symbol,close\n"
                "2026-06-01T00:15:00+00:00,BTCUSD,100000\n",
                encoding="utf-8",
            )
            quote_b.write_text(
                "timestamp,symbol,bid,ask\n"
                "2026-06-01T00:15:00+00:00,BTCUSD,99990,100010\n",
                encoding="utf-8",
            )

            output = _capture(
                merge_market_data.main,
                [
                    "--price-input",
                    str(price_a),
                    "--price-input",
                    str(price_b),
                    "--quote-input",
                    str(quote_a),
                    "--quote-input",
                    str(quote_b),
                    "--price-output",
                    str(merged_prices),
                    "--quote-output",
                    str(merged_quotes),
                    "--common-window",
                ],
            )
            price_text = merged_prices.read_text(encoding="utf-8")
            quote_text = merged_quotes.read_text(encoding="utf-8")

        self.assertIn("Market Data Merge", output)
        self.assertIn("BTCUSD", price_text)
        self.assertIn("EURUSD", quote_text)

    def test_asset_class_attribution_cli_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            pnl_path = Path(tmpdir) / "pnl.csv"
            output_path = Path(tmpdir) / "asset.csv"
            pnl_path.write_text(
                "symbol,fills,realized_pnl_usd,open_pnl_usd,total_pnl_usd,"
                "final_position_units,average_entry_price,final_mark_price\n"
                "EURUSD,1,100.0,0.0,100.0,0,,1.1\n"
                "BTCUSD,2,-40.0,0.0,-40.0,0,,100000\n"
                "PORTFOLIO,3,60.0,0.0,60.0,,,\n",
                encoding="utf-8",
            )

            output = _capture(
                asset_class_attribution.main,
                [
                    "--pnl-csv",
                    str(pnl_path),
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Asset Class Attribution", output)
        self.assertIn("FOREX", csv_text)
        self.assertIn("CRYPTO", csv_text)

    def test_candidate_scorecard_cli_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            equity_path = root / "equity.csv"
            fills_path = root / "fills.csv"
            pnl_path = root / "pnl.csv"
            output_path = root / "scorecard.csv"
            equity_path.write_text(
                "timestamp,equity,cash,gross_notional_usd,net_notional_usd,"
                "drawdown_pct,positions\n"
                "2026-06-01T00:00:00+00:00,1000000,1000000,0,0,0,\n"
                "2026-06-01T00:15:00+00:00,1001000,1001000,0,0,0,\n",
                encoding="utf-8",
            )
            fills_path.write_text(
                "timestamp,symbol,side,fill_price,trade_units,turnover_notional_usd,"
                "requested_notional_usd,adjusted_notional_usd,risk_reason,"
                "primary_signal,supporting_signals,conflicting_signals\n"
                "2026-06-01T00:01:00+00:00,EURUSD,BUY,1.1,1000,1100,"
                "1100,1100,approved,unit,,\n",
                encoding="utf-8",
            )
            pnl_path.write_text(
                "symbol,fills,realized_pnl_usd,open_pnl_usd,total_pnl_usd,"
                "final_position_units,average_entry_price,final_mark_price\n"
                "PORTFOLIO,1,1000,0,1000,,,\n",
                encoding="utf-8",
            )

            output = _capture(
                candidate_scorecard.main,
                [
                    "--candidate",
                    f"demo:{equity_path}:{fills_path}:{pnl_path}",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Candidate Scorecard", output)
        self.assertIn("demo", csv_text)

    def test_candidate_scorecard_parser_accepts_key_value_paths(self) -> None:
        bundle = candidate_scorecard._parse_candidate(
            "label=demo,equity=C:\\tmp\\equity.csv,"
            "fills=C:\\tmp\\fills.csv,pnl=C:\\tmp\\pnl.csv"
        )

        self.assertEqual(bundle.label, "demo")
        self.assertEqual(bundle.equity_csv, "C:\\tmp\\equity.csv")
        self.assertEqual(bundle.fills_csv, "C:\\tmp\\fills.csv")
        self.assertEqual(bundle.pnl_csv, "C:\\tmp\\pnl.csv")

    def test_candidate_scorecard_parser_accepts_legacy_windows_paths(self) -> None:
        bundle = candidate_scorecard._parse_candidate(
            "demo:C:\\tmp\\equity.csv:C:\\tmp\\fills.csv:C:\\tmp\\pnl.csv"
        )

        self.assertEqual(bundle.label, "demo")
        self.assertEqual(bundle.equity_csv, "C:\\tmp\\equity.csv")
        self.assertEqual(bundle.fills_csv, "C:\\tmp\\fills.csv")
        self.assertEqual(bundle.pnl_csv, "C:\\tmp\\pnl.csv")

    def test_portfolio_robustness_cli_writes_leave_one_symbol_out_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "robustness.csv"
            price_lines = ["timestamp,symbol,close"]
            quote_lines = ["timestamp,symbol,bid,ask"]
            start = datetime(2026, 6, 1, tzinfo=timezone.utc)
            for index in range(8):
                timestamp = (start + timedelta(minutes=15 * index)).isoformat()
                eur = 1.1000 + index * 0.0010
                gbp = 1.3000 + index * 0.0015
                price_lines.append(f"{timestamp},EURUSD,{eur:.5f}")
                price_lines.append(f"{timestamp},GBPUSD,{gbp:.5f}")
                quote_lines.append(f"{timestamp},EURUSD,{eur - 0.00001:.5f},{eur + 0.00001:.5f}")
                quote_lines.append(f"{timestamp},GBPUSD,{gbp - 0.00001:.5f},{gbp + 0.00001:.5f}")
            price_path.write_text("\n".join(price_lines) + "\n", encoding="utf-8")
            quote_path.write_text("\n".join(quote_lines) + "\n", encoding="utf-8")

            output = _capture(
                portfolio_robustness.main,
                [
                    "--config",
                    "configs/default.toml",
                    "--strategy",
                    "simple_momentum",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--clock-open-at",
                    "2026-06-01T00:00:00+00:00",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Portfolio Robustness", output)
        self.assertIn("baseline", csv_text)
        self.assertIn("exclude_EURUSD", csv_text)
        self.assertIn("exclude_GBPUSD", csv_text)

    def test_strategy_demo_runs_as_importable_module(self) -> None:
        output = _capture(
            strategy_demo.main,
            ["--strategy", "breakout", "--scenario", "spike_up"],
        )

        self.assertIn("Strategy: breakout", output)
        self.assertIn("Upper band:", output)
        self.assertIn("Strategy output:", output)

    def test_ml_alpha_report_accepts_research_overrides(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            price_path.write_text(
                "timestamp,symbol,close\n"
                + "\n".join(
                    (
                        f"2026-05-11T00:{index:02d}:00+00:00,EURUSD,"
                        f"{1.0000 + (index * 0.0005):.5f}"
                    )
                    for index in range(20)
                )
                + "\n",
                encoding="utf-8",
            )

            output = _capture(
                ml_alpha_report.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--price-csv",
                    str(price_path),
                    "--output",
                    str(output_path),
                    "--ml-lookback",
                    "4",
                    "--ml-train-window",
                    "8",
                    "--ml-min-train-samples",
                    "2",
                    "--ml-epochs",
                    "1",
                    "--ml-label-threshold-bps",
                    "0",
                ],
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("ML Alpha Evaluation", output)
        self.assertIn("Scored predictions:", output)
        self.assertIn("timestamp,symbol,close,next_close,probability_up", text)

    def test_ml_alpha_report_can_target_symbol_subset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            output_path = Path(tmpdir) / "predictions.csv"
            rows = ["timestamp,symbol,close"]
            for symbol, base in (("EURUSD", 1.0000), ("XAUUSD", 2300.0)):
                rows.extend(
                    (
                        f"2026-05-11T00:{index:02d}:00+00:00,{symbol},"
                        f"{base + (index * 0.0005):.5f}"
                    )
                    for index in range(20)
                )
            price_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            output = _capture(
                ml_alpha_report.main,
                [
                    "--all-symbols",
                    "--symbols",
                    "XAUUSD",
                    "--price-csv",
                    str(price_path),
                    "--output",
                    str(output_path),
                    "--ml-lookback",
                    "4",
                    "--ml-train-window",
                    "8",
                    "--ml-min-train-samples",
                    "2",
                    "--ml-epochs",
                    "1",
                    "--ml-label-threshold-bps",
                    "0",
                ],
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Symbols: XAUUSD", output)
        self.assertIn("XAUUSD", text)
        self.assertNotIn("EURUSD", text)

    def test_strategy_demo_runs_moving_average_crossover(self) -> None:
        output = _capture(
            strategy_demo.main,
            ["--strategy", "ma_crossover", "--scenario", "up"],
        )

        self.assertIn("Strategy: ma_crossover", output)
        self.assertIn("Fast average", output)
        self.assertIn("Separation:", output)

    def test_strategy_demo_runs_session_breakout(self) -> None:
        output = _capture(
            strategy_demo.main,
            ["--strategy", "session_breakout", "--scenario", "spike_up"],
        )

        self.assertIn("Strategy: session_breakout", output)
        self.assertIn("Allowed UTC hours:", output)
        self.assertIn("Strategy output:", output)

    def test_strategy_demo_runs_volatility_squeeze(self) -> None:
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "volatility_squeeze",
                "--prices",
                "1.0000,1.0010,0.9990,1.0010,0.9990,1.0000,1.00005,0.99995,1.0000,1.0025",
                "--lookback",
                "10",
            ],
        )

        self.assertIn("Strategy: volatility_squeeze", output)
        self.assertIn("Squeeze ratio:", output)
        self.assertIn("Strategy output:", output)

    def test_strategy_demo_runs_exhaustion_reversal(self) -> None:
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "exhaustion_reversal",
                "--prices",
                "1.0000,1.0002,1.0001,1.0000,0.9980,0.9960,0.9940,0.9950",
                "--lookback",
                "8",
            ],
        )

        self.assertIn("Strategy: exhaustion_reversal", output)
        self.assertIn("Shock z-score:", output)
        self.assertIn("Signal direction:", output)

    def test_strategy_demo_runs_fixing_reversal(self) -> None:
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "fixing_reversal",
                "--prices",
                "1.0000,1.0010,1.0020,1.0030,1.0040,1.0035",
                "--lookback",
                "4",
            ],
        )

        self.assertIn("Strategy: fixing_reversal", output)
        self.assertIn("Pre-fix move:", output)
        self.assertIn("Signal direction:", output)

    def test_strategy_demo_runs_kalman_trend(self) -> None:
        prices = ",".join(f"{1.0000 + index * 0.0010:.4f}" for index in range(30))
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "kalman_trend",
                "--prices",
                prices,
                "--lookback",
                "20",
            ],
        )

        self.assertIn("Strategy: kalman_trend", output)
        self.assertIn("Regime:", output)
        self.assertIn("Signal direction:", output)

    def test_strategy_demo_runs_champion_ensemble(self) -> None:
        prices = ",".join(f"{1.0000 + index * 0.0010:.4f}" for index in range(30))
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "champion_ensemble",
                "--prices",
                prices,
                "--lookback",
                "20",
            ],
        )

        self.assertIn("Strategy: champion_ensemble", output)
        self.assertIn("kalman_trend:", output)
        self.assertIn("Decision:", output)

    def test_strategy_demo_runs_dual_squeeze(self) -> None:
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "dual_squeeze",
                "--prices",
                "1.0000,1.0005,0.9995,1.0002,0.9998,1.0001,0.9999,1.0000,1.0001,1.0040",
                "--lookback",
                "8",
            ],
        )

        self.assertIn("Strategy: dual_squeeze", output)
        self.assertIn("Fast breakout:", output)
        self.assertIn("Confirmation passed:", output)

    def test_strategy_demo_runs_asset_adaptive_dual_squeeze(self) -> None:
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "asset_adaptive_dual_squeeze",
                "--symbol",
                "XAUUSD",
                "--prices",
                "1.0000,1.0005,0.9995,1.0002,0.9998,1.0001,0.9999,1.0000,1.0001,1.0040",
            ],
        )

        self.assertIn("Strategy: asset_adaptive_dual_squeeze", output)
        self.assertIn("Selected profile: metal_fast", output)
        self.assertIn("Decision:", output)

    def test_strategy_demo_runs_range_expansion_trend(self) -> None:
        output = _capture(
            strategy_demo.main,
            [
                "--strategy",
                "range_expansion_trend",
                "--prices",
                "1.0000,1.0001,0.9999,1.0002,1.0000,1.0002,0.9999,1.0001,1.0002,1.0008,1.0014,1.0020",
                "--lookback",
                "12",
            ],
        )

        self.assertIn("Strategy: range_expansion_trend", output)
        self.assertIn("Expansion z-score:", output)
        self.assertIn("Signal direction:", output)

    def test_strategy_demo_explains_cross_rate_context_requirement(self) -> None:
        output = _capture(
            strategy_demo.main,
            ["--strategy", "cross_rate_reversion"],
        )

        self.assertIn("requires multi-symbol portfolio context", output)
        self.assertIn("Strategy output: NO TRADE", output)

    def test_adaptive_strategy_select_help_does_not_crash(self) -> None:
        with self.assertRaises(SystemExit) as context:
            _capture(adaptive_strategy_select.main, ["--help"])

        self.assertEqual(context.exception.code, 0)

    def test_adaptive_strategy_policy_sweep_writes_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "adaptive_policy_sweep.csv"
            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "64",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )

            output = _capture(
                adaptive_strategy_policy_sweep.main,
                [
                    "--config",
                    "configs/default.toml",
                    "--strategy",
                    "simple_momentum",
                    "--strategy",
                    "macd_momentum",
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "32",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--loss-cooldown-folds",
                    "0",
                    "--min-train-adjusted-return-pct",
                    "none",
                    "--transition-risk-multiplier",
                    "1.0",
                    "--cash-fallback",
                    "no",
                    "--output",
                    str(output_path),
                ],
            )
            sweep_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Adaptive Strategy Policy Sweep", output)
        self.assertIn("Candidates: 1", output)
        self.assertIn("compounded_test_return_pct", sweep_text)
        self.assertIn("allow_cash_fallback", sweep_text)

    def test_adaptive_strategy_oracle_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_prefix = root / "oracle"
            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "64",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )

            output = _capture(
                adaptive_strategy_oracle.main,
                [
                    "--config",
                    "configs/default.toml",
                    "--strategy",
                    "simple_momentum",
                    "--strategy",
                    "macd_momentum",
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "32",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--output-prefix",
                    str(output_prefix),
                ],
            )
            summary_text = output_prefix.with_name("oracle_summary.csv").read_text(
                encoding="utf-8"
            )
            folds_text = output_prefix.with_name("oracle_folds.csv").read_text(
                encoding="utf-8"
            )
            candidates_text = output_prefix.with_name(
                "oracle_candidates.csv"
            ).read_text(encoding="utf-8")

        self.assertIn("Adaptive Strategy Oracle Diagnostic", output)
        self.assertIn("Selected was oracle:", output)
        self.assertIn("selected_was_oracle_fraction", summary_text)
        self.assertIn("oracle_strategy", folds_text)
        self.assertIn("oracle_best", candidates_text)

    def test_adaptive_handoff_diagnostic_writes_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oracle_folds = root / "oracle_folds.csv"
            oracle_candidates = root / "oracle_candidates.csv"
            regime_summary = root / "regime_summary.csv"
            output_path = root / "handoff.csv"
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
                    ]
                ),
                encoding="utf-8",
            )

            output = _capture(
                adaptive_handoff_diagnostic.main,
                [
                    "--oracle-folds-csv",
                    str(oracle_folds),
                    "--oracle-candidates-csv",
                    str(oracle_candidates),
                    "--regime-summary-csv",
                    str(regime_summary),
                    "--output",
                    str(output_path),
                ],
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Adaptive Handoff Diagnostic", output)
        self.assertIn("HINDSIGHT_CHOP_BREAKOUT", output)
        self.assertIn("diagnosis", text)

    def test_hackathon_readiness_writes_markdown(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "readiness.md"
            promotion_path = Path(tmpdir) / "promotion.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            _write_mini_market_csv(price_path, quote_path)
            promotion_path.write_text(
                "status,live_ready,decision_reason,gate_id,category,passed,value,threshold,comparator,gap,details\n"
                "PAPER_ONLY,no,test paper only,live_positive_fold_fraction,live,no,0.5,0.67,>=,-0.17,needs more folds\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                "strategies,symbols,folds,positive_fold_fraction,active_positive_fold_fraction,"
                "non_negative_fold_fraction,median_active_test_return_pct,worst_test_drawdown_pct,"
                "average_risk_discipline_score,total_evaluation_fills\n"
                "test,EURUSD,2,0.5,0.5,1.0,0.001,0.01,100,10\n",
                encoding="utf-8",
            )

            output = _capture(
                hackathon_readiness.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--promotion-csv",
                    str(promotion_path),
                    "--summary-csv",
                    str(summary_path),
                    "--output",
                    str(output_path),
                ],
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Hackathon Readiness", output)
        self.assertIn("Overall: FAIL", output)
        self.assertIn("Missing asset classes", text)
        self.assertIn("PAPER_ONLY", text)

    def test_adaptive_strategy_select_accepts_candidate_map(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"
            scores_path = Path(tmpdir) / "scores.csv"
            equity_path = Path(tmpdir) / "equity.csv"
            promotion_path = Path(tmpdir) / "promotion.csv"
            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "56",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )

            output = _capture(
                adaptive_strategy_select.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--candidate-map",
                    (
                        "hybrid_map:"
                        "EURUSD=simple_momentum,"
                        "GBPUSD=macd_momentum,"
                        "USDJPY=macd_momentum"
                    ),
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "24",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--min-train-fills",
                    "1",
                    "--min-train-adjusted-return-pct",
                    "-1.0",
                    "--train-fill-penalty-pct",
                    "0.000002",
                    "--train-stability-splits",
                    "4",
                    "--prefer-train-stability",
                    "--transition-risk-multiplier",
                    "0.5",
                    "--allow-cash-fallback",
                    "--per-symbol-selection",
                    "--summary-output",
                    str(summary_path),
                    "--folds-output",
                    str(folds_path),
                    "--scores-output",
                    str(scores_path),
                    "--stitched-equity-output",
                    str(equity_path),
                    "--promotion-output",
                    str(promotion_path),
                ],
            )
            scores_text = scores_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")
            promotion_text = promotion_path.read_text(encoding="utf-8")

        self.assertIn("Adaptive Strategy Selection", output)
        self.assertIn("Candidates: simple_momentum, hybrid_map", output)
        self.assertIn("Min train fills: 1", output)
        self.assertIn("Train fill penalty: 2e-06", output)
        self.assertIn("Train stability splits: 4", output)
        self.assertIn("Prefer train stability: yes", output)
        self.assertIn("Transition risk multiplier: 0.5", output)
        self.assertIn("Cash fallback: yes", output)
        self.assertIn("Per-symbol selection: yes", output)
        self.assertIn("Per-symbol only: no", output)
        self.assertIn("Promotion audit CSV:", output)
        self.assertIn("min_train_fills", summary_text)
        self.assertIn("train_fill_penalty_pct", summary_text)
        self.assertIn("train_stability_splits", summary_text)
        self.assertIn("prefer_train_stability", summary_text)
        self.assertIn("transition_risk_multiplier", summary_text)
        self.assertIn("allow_cash_fallback", summary_text)
        self.assertIn("train_gate_blocked_strategies", folds_text)
        self.assertIn("selected_train_stability_active_positive_fraction", folds_text)
        self.assertIn("evaluation_risk_multiplier", folds_text)
        self.assertIn("hybrid_map", scores_text)
        self.assertIn("train_gate_passed", scores_text)
        self.assertIn("train_stability_active_positive_fraction", scores_text)
        self.assertIn("GBPUSD=macd_momentum", scores_text)
        self.assertIn("live_positive_fold_fraction", promotion_text)

    def test_adaptive_strategy_select_accepts_recipe_map_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"
            scores_path = Path(tmpdir) / "scores.csv"
            equity_path = Path(tmpdir) / "equity.csv"
            promotion_path = Path(tmpdir) / "promotion.csv"
            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "56",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )

            output = _capture(
                adaptive_strategy_select.main,
                [
                    "--no-default-strategies",
                    "--recipe-map",
                    "fx_pair:EURUSD=simple_momentum,GBPUSD=simple_momentum",
                    "--recipe-map",
                    "jpy_only:USDJPY=macd_momentum",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "24",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--summary-output",
                    str(summary_path),
                    "--folds-output",
                    str(folds_path),
                    "--scores-output",
                    str(scores_path),
                    "--stitched-equity-output",
                    str(equity_path),
                    "--promotion-output",
                    str(promotion_path),
                ],
            )
            scores_text = scores_path.read_text(encoding="utf-8")

        self.assertIn("Candidates: fx_pair, jpy_only", output)
        self.assertIn("USDJPY=macd_momentum", scores_text)

    def test_adaptive_strategy_select_reports_bad_candidate_map_cleanly(self) -> None:
        with self.assertRaises(SystemExit) as context:
            _capture(
                adaptive_strategy_select.main,
                ["--candidate-map", "not-a-map"],
            )

        self.assertIn("candidate map must use LABEL", str(context.exception))

    def test_volatility_squeeze_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "volatility_squeeze_opt.csv"

            output = _capture(
                volatility_squeeze_optimize.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--candidate",
                    "fast,10,3,0.80,1.0,1.5",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Volatility Squeeze Optimization", output)
        self.assertIn("fast", csv_text)

    def test_dual_squeeze_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dual_squeeze_opt.csv"

            output = _capture(
                dual_squeeze_optimize.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--candidate",
                    (
                        "label=fast,lookback=10,window=3,ratio=0.80,"
                        "buffer=1.0,band=1.5,confirm_lookback=14,"
                        "confirm_window=4,confirm_ratio=0.90"
                    ),
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Dual Squeeze Optimization", output)
        self.assertIn("fast", csv_text)
        self.assertIn("confirmation_lookback", csv_text)

    def test_session_momentum_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "session_momentum_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--periods",
                    "48",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                session_momentum_optimize.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "late,5,8,3,0.4,17|18|19",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Session Momentum Optimization", output)
        self.assertIn("late", csv_text)

    def test_trend_pullback_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "trend_pullback_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "40",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                trend_pullback_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "fast,12,2,1.0,0.0,0.1,30.0,0.1,0.1",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Trend Pullback Optimization", output)
        self.assertIn("fast", csv_text)

    def test_fixing_reversal_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "fixing_reversal_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "72",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                fixing_reversal_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "loose,4,1.0,0.1,0.0,2,14",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Fixing Reversal Optimization", output)
        self.assertIn("loose", csv_text)

    def test_kalman_trend_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "kalman_trend_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "96",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                kalman_trend_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "fast,20,0.1,0.0,1.0,4,8",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Kalman Trend Optimization", output)
        self.assertIn("fast", csv_text)

    def test_autocorrelation_regime_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "autocorrelation_regime_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "96",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                autocorrelation_regime_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "strict,32,6,0.30,0.05,6.0,0.30,1.2,3.0,4.0,6,10|11|12",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Autocorrelation Regime Optimization", output)
        self.assertIn("strict", csv_text)
        self.assertIn("min_abs_autocorrelation", csv_text)

    def test_champion_ensemble_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "champion_ensemble_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "96",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                champion_ensemble_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "strict,0.70,0.30,0.0,0.0,0.50,0.50,0.70",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Champion Ensemble Optimization", output)
        self.assertIn("strict", csv_text)

    def test_strategy_demo_runs_usd_pressure_router(self) -> None:
        output = _capture(
            strategy_demo.main,
            ["--strategy", "usd_pressure_router", "--scenario", "up"],
        )

        self.assertIn("Strategy: usd_pressure_router", output)
        self.assertIn("USD pressure:", output)
        self.assertIn("Router decision:", output)

    def test_strategy_demo_runs_relative_strength(self) -> None:
        output = _capture(
            strategy_demo.main,
            ["--strategy", "relative_strength", "--scenario", "up"],
        )

        self.assertIn("Strategy: relative_strength", output)
        self.assertIn("Relative z-score:", output)
        self.assertIn("Relative-strength decision:", output)

    def test_show_instruments_cli_lists_syphonix_assets(self) -> None:
        output = _capture(show_instruments.main, [])

        self.assertIn("Competition Instruments", output)
        self.assertIn("EURUSD", output)
        self.assertIn("BTCUSD", output)

    def test_backtest_cli_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            equity_path = Path(tmpdir) / "equity.csv"
            pnl_path = Path(tmpdir) / "pnl.csv"

            output = _capture(
                backtest.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--equity-output",
                    str(equity_path),
                    "--pnl-output",
                    str(pnl_path),
                ],
            )

            self.assertTrue(equity_path.exists())
            self.assertTrue(pnl_path.exists())

        self.assertIn("Backtest", output)
        self.assertIn("Strategy: simple_momentum", output)

    def test_compare_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.csv"

            output = _capture(
                compare_strategies.main,
                ["--strategy", "simple_momentum", "--output", str(output_path)],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Strategy Comparison", output)
        self.assertIn("rank,strategy,symbol", csv_text)
        self.assertIn("simple_momentum", csv_text)

    def test_portfolio_backtest_cli_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            equity_path = Path(tmpdir) / "portfolio_equity.csv"
            pnl_path = Path(tmpdir) / "portfolio_pnl.csv"
            allocation_path = Path(tmpdir) / "portfolio_allocation.csv"
            fills_path = Path(tmpdir) / "portfolio_fills.csv"
            vol_target_path = Path(tmpdir) / "portfolio_vol_target.csv"
            regime_path = Path(tmpdir) / "portfolio_regime.csv"

            output = _capture(
                portfolio_backtest.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--equity-output",
                    str(equity_path),
                    "--pnl-output",
                    str(pnl_path),
                    "--allocation-output",
                    str(allocation_path),
                    "--fills-output",
                    str(fills_path),
                    "--regime-tilt",
                    "--regime-tilt-output",
                    str(regime_path),
                    "--regime-tilt-lookback",
                    "5",
                    "--vol-target-output",
                    str(vol_target_path),
                    "--vol-target-bar-volatility",
                    "0.00075",
                    "--vol-target-lookback",
                    "3",
                    "--vol-target-min-observations",
                    "2",
                    "--metrics-start",
                    "2026-06-22T10:10:00+01:00",
                    "--clock-open-at",
                    "2026-06-22T00:00:00+01:00",
                ],
            )
            equity_exists = equity_path.exists()
            pnl_text = pnl_path.read_text(encoding="utf-8")
            fills_text = fills_path.read_text(encoding="utf-8")
            allocation_text = allocation_path.read_text(encoding="utf-8")
            vol_target_text = vol_target_path.read_text(encoding="utf-8")
            regime_text = regime_path.read_text(encoding="utf-8")

        self.assertIn("Portfolio Backtest", output)
        self.assertIn("Symbols: EURUSD", output)
        self.assertIn("Allocation guardrails", output)
        self.assertIn("Portfolio regime tilt", output)
        self.assertIn("Portfolio volatility targeting", output)
        self.assertIn("Metrics start: 2026-06-22T10:10:00+01:00", output)
        self.assertTrue(equity_exists)
        self.assertIn("PORTFOLIO", pnl_text)
        self.assertIn("timestamp,symbol,side,fill_price", fills_text)
        self.assertIn("estimated_risk_status", allocation_text)
        self.assertIn("timestamp,scale,target_bar_volatility", vol_target_text)
        self.assertIn("timestamp,symbol,primary_signal,regime", regime_text)

    def test_portfolio_backtest_rejects_naive_clock_override(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--clock-open-at must be timezone-aware"):
            portfolio_backtest._parse_datetime("2026-06-22T00:00:00")

    def test_portfolio_backtest_cli_accepts_strategy_map(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            equity_path = Path(tmpdir) / "portfolio_equity.csv"
            pnl_path = Path(tmpdir) / "portfolio_pnl.csv"
            allocation_path = Path(tmpdir) / "portfolio_allocation.csv"
            fills_path = Path(tmpdir) / "portfolio_fills.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "32",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                portfolio_backtest.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--strategy-map",
                    "GBPUSD=macd_momentum",
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--equity-output",
                    str(equity_path),
                    "--pnl-output",
                    str(pnl_path),
                    "--allocation-output",
                    str(allocation_path),
                    "--fills-output",
                    str(fills_path),
                ],
            )

        self.assertIn("Strategy: simple_momentum with overrides", output)
        self.assertIn("GBPUSD=macd_momentum", output)

    def test_strategy_map_optimize_cli_writes_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "strategy_maps.csv"
            score_path = Path(tmpdir) / "strategy_scores.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "72",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                strategy_map_optimize.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--strategy",
                    "macd_momentum",
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--top-symbol-count",
                    "2",
                    "--output",
                    str(output_path),
                    "--score-output",
                    str(score_path),
                ],
            )
            map_text = output_path.read_text(encoding="utf-8")
            score_text = score_path.read_text(encoding="utf-8")

        self.assertIn("Strategy Map Optimization", output)
        self.assertIn("Ranked maps", output)
        self.assertIn("strategy_map", map_text)
        self.assertIn("symbol,strategy,total_pnl_usd", score_text)

    def test_sizing_frontier_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "sizing_frontier.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "96",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                sizing_frontier.main,
                [
                    "--config",
                    "configs/competition.toml",
                    "--strategy-map",
                    "EURUSD=macd_momentum",
                    "--strategy-map",
                    "GBPUSD=multi_horizon_momentum",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--symbol-notional-pct",
                    "0.25",
                    "--symbol-notional-pct",
                    "0.50",
                    "--output",
                    str(output_path),
                ],
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Sizing Frontier", output)
        self.assertIn("cap=25%", output)
        self.assertIn("symbol_notional_pct,max_gross_leverage", text)

    def test_portfolio_attribution_report_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "portfolio_attribution.csv"

            output = _capture(
                portfolio_attribution_report.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--output",
                    str(output_path),
                    "--limit",
                    "2",
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Portfolio Attribution", output)
        self.assertIn("Weakest rows", output)
        self.assertIn("symbol,primary_signal,utc_hour,side", csv_text)

    def test_generate_sample_data_and_portfolio_compare_cli(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            comparison_path = Path(tmpdir) / "comparison.csv"

            data_output = _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "BTCUSD",
                    "--periods",
                    "16",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            compare_output = _capture(
                portfolio_compare.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--strategy",
                    "ma_crossover",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--output",
                    str(comparison_path),
                ],
            )
            comparison_text = comparison_path.read_text(encoding="utf-8")

        self.assertIn("Synthetic Competition Data", data_output)
        self.assertIn("BTCUSD", data_output)
        self.assertIn("Portfolio Strategy Comparison", compare_output)
        self.assertIn("rank,strategy,symbols,proxy_score", comparison_text)
        self.assertIn("simple_momentum", comparison_text)

    def test_portfolio_universe_scan_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            scan_path = Path(tmpdir) / "universe_scan.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--symbol",
                    "XAUUSD",
                    "--periods",
                    "24",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                portfolio_universe_scan.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--basket",
                    "fx_gold:EURUSD,USDJPY,XAUUSD",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--output",
                    str(scan_path),
                ],
            )
            scan_text = scan_path.read_text(encoding="utf-8")

        self.assertIn("Portfolio Universe Scan", output)
        self.assertIn("Top candidates", output)
        self.assertIn("rank,basket,strategy,symbols,asset_mix,proxy_score", scan_text)
        self.assertIn("fx_gold", scan_text)

    def test_portfolio_walk_forward_cli_writes_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            summary_path = Path(tmpdir) / "portfolio_wf_summary.csv"
            folds_path = Path(tmpdir) / "portfolio_wf_folds.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--symbol",
                    "XAUUSD",
                    "--periods",
                    "32",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                portfolio_walk_forward.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--basket",
                    "core_fx:EURUSD,GBPUSD,USDJPY",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "12",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--min-test-fills",
                    "0",
                    "--min-stable-fold-fraction",
                    "0",
                    "--summary-output",
                    str(summary_path),
                    "--folds-output",
                    str(folds_path),
                ],
            )
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("Portfolio Walk-Forward Validation", output)
        self.assertIn("Stable fold fraction", output)
        self.assertIn("eligible,folds,available_symbols", summary_text)
        self.assertIn("selected_basket", folds_text)

    def test_portfolio_fixed_warmup_walk_forward_cli_writes_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            summary_path = Path(tmpdir) / "fixed_warmup_summary.csv"
            folds_path = Path(tmpdir) / "fixed_warmup_folds.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "32",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                portfolio_fixed_warmup_walk_forward.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--strategy-map",
                    "USDJPY=macd_momentum",
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "12",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--summary-output",
                    str(summary_path),
                    "--folds-output",
                    str(folds_path),
                ],
            )
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("Fixed Warmup Portfolio Walk-Forward", output)
        self.assertIn("Positive fold fraction", output)
        self.assertIn("Largest positive fold contribution", output)
        self.assertIn("Promotion:", output)
        self.assertIn("strategy,symbols,folds", summary_text)
        self.assertIn("evaluation_fills", folds_text)

    def test_validate_market_data_competition_symbols_flags_missing_crypto(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "data_health.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "12",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            buffer = StringIO()
            with self.assertRaises(SystemExit), redirect_stdout(buffer):
                validate_market_data.main(
                    [
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--competition-symbols",
                        "--output",
                        str(output_path),
                    ]
                )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("BTCUSD", buffer.getvalue())
        self.assertIn("no price bars", buffer.getvalue())
        self.assertIn("BTCUSD", csv_text)
        self.assertIn("BTCUSD,FAIL", csv_text)
        self.assertIn("no price bars", csv_text)

    def test_portfolio_router_walk_forward_cli_writes_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            summary_path = Path(tmpdir) / "router_wf_summary.csv"
            folds_path = Path(tmpdir) / "router_wf_folds.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "32",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                portfolio_router_walk_forward.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--candidate",
                    "0.4,0.2,0.35,0.25,0.20,0.05,0.10,0.00",
                    "--candidate",
                    "0,0,0,0,0,0,0,1",
                    "--candidate",
                    "0.2,0.5,0.2,0.1",
                    "--candidate",
                    "0,0,0,0,0,0,0,0,0,0.75,0.25",
                    "--behavior-candidate",
                    "0.55,0.20,1.20,0.70,false",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--train-size",
                    "12",
                    "--test-size",
                    "8",
                    "--step-size",
                    "8",
                    "--min-test-fills",
                    "0",
                    "--min-stable-fold-fraction",
                    "0",
                    "--summary-output",
                    str(summary_path),
                    "--folds-output",
                    str(folds_path),
                ],
            )
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("Portfolio Router Walk-Forward", output)
        self.assertIn("Most selected weights", output)
        self.assertIn("Most selected behavior", output)
        self.assertIn("Promotion:", output)
        self.assertIn("eligible,folds,symbols,candidate_weight_sets", summary_text)
        self.assertIn("candidate_behavior_profiles", summary_text)
        self.assertIn("promotion_status", summary_text)
        self.assertIn("selected_weights", folds_text)
        self.assertIn("selected_behavior", folds_text)

    def test_router_optimize_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "router_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "20",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                router_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--candidate",
                    "0.4,0.2,0.35,0.25,0.20,0.05,0.10,0.00",
                    "--candidate",
                    "0,0,0,0,0,0,0,1",
                    "--candidate",
                    "0.2,0.5,0.2,0.1",
                    "--behavior-candidate",
                    "0.55,0.20,1.20,0.70,false",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Router Weight Optimization", output)
        self.assertIn("Ranked candidates", output)
        self.assertIn("rank,symbols,weights,behavior,proxy_score", csv_text)
        self.assertIn("override=off", output)
        self.assertIn("volatility_squeeze_weight", csv_text)
        self.assertIn("dual_squeeze_weight", csv_text)
        self.assertIn("macd_momentum_weight", csv_text)
        self.assertIn("kalman_trend_weight", csv_text)

    def test_strategy_attribution_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "attribution.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--periods",
                    "24",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                strategy_attribution.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Strategy Attribution", output)
        self.assertIn("PORTFOLIO", csv_text)

    def test_symbol_eligibility_optimizer_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "eligibility.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "36",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                symbol_eligibility_optimize.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--min-symbols",
                    "2",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")
            attribution_text = output_path.with_name(
                "eligibility_attribution.csv"
            ).read_text(encoding="utf-8")

        self.assertIn("Symbol Eligibility Optimization", output)
        self.assertIn("Walk-forward ranking: off", output)
        self.assertIn("rank,candidate,strategy,symbols", csv_text)
        self.assertIn("wf_positive_fold_fraction", csv_text)
        self.assertIn("rank,strategy,symbol,fills", attribution_text)

    def test_relative_strength_optimize_cli_writes_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "relative_strength_opt.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--symbol",
                    "XAUUSD",
                    "--periods",
                    "24",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                relative_strength_optimize.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "GBPUSD",
                    "--symbol",
                    "USDJPY",
                    "--symbol",
                    "XAUUSD",
                    "--candidate",
                    "short,4,0.5,0.15",
                    "--candidate",
                    "long,6,0.75,0.25",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Relative Strength Optimization", output)
        self.assertIn("Ranked candidates", output)
        self.assertIn("rank,label,symbols,lookback", csv_text)
        self.assertIn("short", csv_text)

    def test_time_series_report_cli_writes_regime_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            output_path = Path(tmpdir) / "regimes.csv"

            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "XAUUSD",
                    "--periods",
                    "20",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )
            output = _capture(
                time_series_report.main,
                [
                    "--price-csv",
                    str(price_path),
                    "--lookback",
                    "10",
                    "--output",
                    str(output_path),
                ],
            )
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Advanced Time-Series Regime Report", output)
        self.assertIn("symbol,observations,latest_close", csv_text)
        self.assertIn("EURUSD", csv_text)

    def test_import_backtest_data_cli_writes_normalized_csvs(self) -> None:
        if importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pyarrow is needed for Parquet importer CLI test")
        try:
            from test_backtest_import import _write_sample_pricer_zip
        except ImportError:
            from tests.test_backtest_import import _write_sample_pricer_zip

        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "sample.zip"
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            _write_sample_pricer_zip(archive)

            output = _capture(
                import_backtest_data.main,
                [
                    "--input",
                    str(archive),
                    "--symbol",
                    "EURUSD",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                    "--max-files-per-symbol",
                    "1",
                ],
            )
            price_text = price_path.read_text(encoding="utf-8")
            quote_text = quote_path.read_text(encoding="utf-8")

        self.assertIn("Imported Backtest Data", output)
        self.assertIn("Bars written: 3", output)
        self.assertIn("timestamp,symbol,close", price_text)
        self.assertIn("timestamp,symbol,bid,ask", quote_text)

    def test_live_dry_run_cli_writes_monitor_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            journal_path = Path(tmpdir) / "journal.jsonl"
            monitor_path = Path(tmpdir) / "monitor.csv"
            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "USDJPY",
                    "--periods",
                    "16",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )

            output = _capture(
                live_dry_run.main,
                [
                    "--adapter",
                    "csv",
                    "--strategy",
                    "simple_momentum",
                    "--strategy-map",
                    "USDJPY=macd_momentum",
                    "--symbol",
                    "EURUSD",
                    "--symbol",
                    "USDJPY",
                    "--bars",
                    "5",
                    "--iterations",
                    "1",
                    "--price-csv",
                    str(price_path),
                    "--quote-csv",
                    str(quote_path),
                    "--journal",
                    str(journal_path),
                    "--monitor-output",
                    str(monitor_path),
                ],
            )
            monitor_text = monitor_path.read_text(encoding="utf-8")

        self.assertIn("Live Dry Run", output)
        self.assertIn("Adapter: csv", output)
        self.assertIn("Strategy map: USDJPY=macd_momentum", output)
        self.assertIn("Allocation status:", output)
        self.assertIn("Competition scoring view", output)
        self.assertIn("gross_notional_usd", monitor_text)

    def test_live_dry_run_mt5_requires_explicit_read_only_confirmation(self) -> None:
        with self.assertRaises(SystemExit) as context:
            _capture(live_dry_run.main, ["--adapter", "mt5"])

        self.assertIn("Refusing MT5 connection", str(context.exception))

    def test_mt5_capture_requires_explicit_read_only_confirmation(self) -> None:
        with self.assertRaises(SystemExit) as context:
            _capture(mt5_capture.main, ["--symbol", "EURUSD"])

        self.assertIn("Refusing MT5 capture", str(context.exception))

    def test_manual_ticket_prints_mt5_volume_without_sending_orders(self) -> None:
        output = _capture(
            manual_ticket.main,
            [
                "--symbol",
                "EURUSD",
                "--side",
                "BUY",
                "--target-notional",
                "50000",
                "--price",
                "1.1002",
            ],
        )

        self.assertIn("Manual MT5 Order Ticket", output)
        self.assertIn("Mode: manual desktop entry, no Python order_send", output)
        self.assertIn("Risk decision: APPROVED", output)
        self.assertIn("MT5 volume to enter:", output)

    def test_mt5_probe_prints_read_only_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("", encoding="utf-8")

            output = _capture(
                mt5_probe.main,
                ["--env-file", str(env_path), "--symbol", "EURUSD"],
            )

        self.assertIn("MT5 Probe", output)
        self.assertIn("Mode: read-only, no order_send", output)
        self.assertIn("Connection:", output)

    def test_live_dry_run_cli_reports_missing_csv_symbol_cleanly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            quote_path = Path(tmpdir) / "quotes.csv"
            _capture(
                generate_sample_data.main,
                [
                    "--symbol",
                    "EURUSD",
                    "--periods",
                    "8",
                    "--price-output",
                    str(price_path),
                    "--quote-output",
                    str(quote_path),
                ],
            )

            with self.assertRaises(SystemExit) as context:
                _capture(
                    live_dry_run.main,
                    [
                        "--adapter",
                        "csv",
                        "--symbol",
                        "USDJPY",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                    ],
                )

        self.assertIn("csv adapter does not have data for: USDJPY", str(context.exception))

    def test_walk_forward_cli_writes_csvs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"

            output = _capture(
                walk_forward.main,
                [
                    "--strategy",
                    "simple_momentum",
                    "--train-size",
                    "10",
                    "--test-size",
                    "5",
                    "--step-size",
                    "5",
                    "--summary-output",
                    str(summary_path),
                    "--folds-output",
                    str(folds_path),
                ],
            )
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("Walk-Forward Evaluation", output)
        self.assertIn("rank,strategy,eligible", summary_text)
        self.assertIn("strategy,fold,symbol", folds_text)


def _capture(function, argv: list[str]) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        function(argv)
    return buffer.getvalue()


def _write_mini_market_csv(price_path: Path, quote_path: Path) -> None:
    price_path.write_text(
        "timestamp,symbol,close\n"
        "2026-06-22T10:00:00+01:00,EURUSD,1.1000\n"
        "2026-06-22T10:15:00+01:00,EURUSD,1.1010\n"
        "2026-06-22T10:00:00+01:00,XAUUSD,2300.0\n"
        "2026-06-22T10:15:00+01:00,XAUUSD,2301.0\n",
        encoding="utf-8",
    )
    quote_path.write_text(
        "timestamp,symbol,bid,ask\n"
        "2026-06-22T10:00:00+01:00,EURUSD,1.0999,1.1001\n"
        "2026-06-22T10:15:00+01:00,EURUSD,1.1009,1.1011\n"
        "2026-06-22T10:00:00+01:00,XAUUSD,2299.9,2300.1\n"
        "2026-06-22T10:15:00+01:00,XAUUSD,2300.9,2301.1\n",
        encoding="utf-8",
    )
