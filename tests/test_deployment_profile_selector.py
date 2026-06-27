from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_selector import (
    run_deployment_profile_selector,
    select_next_deployment_profile,
    write_deployment_profile_selector_folds_csv,
    write_deployment_profile_selector_summary_csv,
)
from quanthack.backtesting.deployment_profile_selector_sweep import (
    sweep_deployment_profile_selector,
    write_deployment_profile_selector_sweep_csv,
)
from quanthack.cli import (
    deployment_profile_recommendation,
    deployment_profile_selector,
    deployment_profile_selector_sweep,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileSelectorTest(TestCase):
    def test_selector_builds_adaptive_folds_from_past_profile_evidence(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=171,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            summary_path = root / "summary.csv"
            folds_path = root / "folds.csv"
            _write_profile_pack(profile_path)

            result = run_deployment_profile_selector(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slots=("aggressive", "conservative", "survival"),
                fallback_slot="conservative",
                train_size=12,
                test_size=8,
                step_size=8,
                min_past_folds=1,
            )
            write_deployment_profile_selector_summary_csv(result, summary_path)
            write_deployment_profile_selector_folds_csv(result, folds_path)

            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertEqual(len(result.fixed_results), 3)
        self.assertEqual(result.selections[0].selected_slot, "conservative")
        self.assertEqual(
            len(result.adaptive_result.folds),
            len(result.fixed_results[0].walk_forward.folds),
        )
        self.assertEqual(sum(result.selected_counts.values()), len(result.selections))
        self.assertIn("mode,slot,label,selected_count", summary_text)
        self.assertIn("adaptive", summary_text)
        self.assertIn("selected_slot,selected_label,selection_score", folds_text)
        self.assertIn("aggressive_return_pct", folds_text)

    def test_selector_rejects_fallback_not_in_slot_set(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=28,
            interval_minutes=15,
            seed=172,
        )

        with TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "pack.json"
            _write_profile_pack(profile_path)

            with self.assertRaisesRegex(ValueError, "fallback_slot"):
                run_deployment_profile_selector(
                    config=config,
                    prices=data.prices,
                    quotes=data.quotes,
                    profile_pack_json=profile_path,
                    slots=("aggressive", "survival"),
                    fallback_slot="conservative",
                    train_size=10,
                    test_size=6,
                    step_size=6,
                )

    def test_deployment_profile_selector_cli_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=173,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            summary_path = root / "selector_summary.csv"
            folds_path = root / "selector_folds.csv"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_selector.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
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
                        "--min-past-folds",
                        "1",
                        "--summary-output",
                        str(summary_path),
                        "--folds-output",
                        str(folds_path),
                    ]
                )

            output = stdout.getvalue()
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Selector", output)
        self.assertIn("Adaptive positive fold fraction", output)
        self.assertIn("adaptive", summary_text)
        self.assertIn("selected_slot", folds_text)

    def test_deployment_profile_selector_cli_accepts_research_slot(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=178,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "refined_pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            summary_path = root / "selector_summary.csv"
            folds_path = root / "selector_folds.csv"
            _write_refined_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_selector.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--slot",
                        "refined",
                        "--fallback-slot",
                        "refined",
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
                        "--min-past-folds",
                        "1",
                        "--summary-output",
                        str(summary_path),
                        "--folds-output",
                        str(folds_path),
                    ]
                )

            output = stdout.getvalue()
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Selector", output)
        self.assertIn("refined", output)
        self.assertIn("demo_refined", summary_text)
        self.assertIn("refined_return_pct", folds_text)

    def test_next_selection_uses_all_completed_folds(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=176,
        )

        with TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "pack.json"
            _write_profile_pack(profile_path)
            result = run_deployment_profile_selector(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                fallback_slot="conservative",
                train_size=12,
                test_size=8,
                step_size=8,
                min_past_folds=1,
            )
            next_selection = select_next_deployment_profile(
                fixed_results=result.fixed_results,
                fallback_slot="conservative",
                min_past_folds=1,
                drawdown_penalty=0.5,
                risk_score_floor=95.0,
            )

        self.assertIn(
            next_selection.selected_slot,
            {"aggressive", "conservative", "survival"},
        )
        self.assertEqual(next_selection.completed_folds, len(result.adaptive_result.folds))
        self.assertEqual(len(next_selection.past_scores), 3)

    def test_selector_sweep_reuses_profile_walk_forward_results(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=174,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            output_path = root / "sweep.csv"
            _write_profile_pack(profile_path)

            result = sweep_deployment_profile_selector(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                fallback_slots=("conservative",),
                min_past_folds_values=(1, 2),
                drawdown_penalties=(0.0, 0.5),
                risk_score_floors=(95.0,),
                require_past_activity_values=(True, False),
                train_size=12,
                test_size=8,
                step_size=8,
            )
            write_deployment_profile_selector_sweep_csv(result, output_path)
            output_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(len(result.fixed_results), 3)
        self.assertEqual(len(result.candidates), 8)
        self.assertIsNotNone(result.best)
        self.assertIn("rank,promotion_status,selector_score", output_text)
        self.assertIn("selected_sequence", output_text)

    def test_deployment_profile_selector_sweep_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=175,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "selector_sweep.csv"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_selector_sweep.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--fallback-slot",
                        "conservative",
                        "--min-past-folds",
                        "1",
                        "--drawdown-penalty",
                        "0.5",
                        "--risk-score-floor",
                        "95",
                        "--train-size",
                        "12",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Selector Sweep", output)
        self.assertIn("Candidates: 1", output)
        self.assertIn("selector_score", csv_text)

    def test_deployment_profile_recommendation_cli_writes_artifacts(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=177,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            recommendation_csv = root / "recommendation.csv"
            recommendation_json = root / "recommendation.json"
            sweep_path = root / "recommendation_sweep.csv"
            snapshot_path = root / "snapshot.csv"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_recommendation.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--fallback-slot",
                        "conservative",
                        "--min-past-folds",
                        "1",
                        "--drawdown-penalty",
                        "0.5",
                        "--risk-score-floor",
                        "95",
                        "--train-size",
                        "12",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--recommendation-csv",
                        str(recommendation_csv),
                        "--recommendation-json",
                        str(recommendation_json),
                        "--sweep-output",
                        str(sweep_path),
                        "--snapshot-output",
                        str(snapshot_path),
                    ]
                )

            output = stdout.getvalue()
            csv_text = recommendation_csv.read_text(encoding="utf-8")
            json_text = recommendation_json.read_text(encoding="utf-8")
            sweep_text = sweep_path.read_text(encoding="utf-8")
            snapshot_text = snapshot_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Recommendation", output)
        self.assertIn("Recommended slot", output)
        self.assertIn("recommended_slot,recommended_label", csv_text)
        self.assertIn('"recommended_slot"', json_text)
        self.assertIn("selector_score", sweep_text)
        self.assertIn("profile_slot,profile_label,timestamp,symbol", snapshot_text)

    def test_deployment_profile_recommendation_cli_defaults_to_pack_research_slot(
        self,
    ) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=179,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "refined_pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            recommendation_csv = root / "recommendation.csv"
            recommendation_json = root / "recommendation.json"
            sweep_path = root / "recommendation_sweep.csv"
            _write_refined_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_recommendation.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--min-past-folds",
                        "1",
                        "--drawdown-penalty",
                        "0.5",
                        "--risk-score-floor",
                        "95",
                        "--train-size",
                        "12",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--recommendation-csv",
                        str(recommendation_csv),
                        "--recommendation-json",
                        str(recommendation_json),
                        "--sweep-output",
                        str(sweep_path),
                    ]
                )

            output = stdout.getvalue()
            csv_text = recommendation_csv.read_text(encoding="utf-8")
            json_text = recommendation_json.read_text(encoding="utf-8")
            sweep_text = sweep_path.read_text(encoding="utf-8")

        self.assertIn("Recommended slot: refined", output)
        self.assertIn("refined", csv_text)
        self.assertIn('"recommended_slot": "refined"', json_text)
        self.assertIn("refined=", sweep_text)


def _write_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": "conservative",
        "recommendation_reason": "unit test",
        "profiles": [
            {
                "slot": "aggressive",
                "label": "demo_aggressive",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test aggressive profile",
                "reason": "larger size",
                "return_pct": 0.02,
                "max_drawdown_pct": 0.006,
                "sharpe_15m": 0.08,
                "fold_contribution": 0.70,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=macd_momentum",
                "multiplier_map": "EURUSD=1.000 BTCUSD=1.000",
                "crypto_allowed_utc_hours": "all",
            },
            {
                "slot": "conservative",
                "label": "demo_conservative",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test conservative profile",
                "reason": "balanced size",
                "return_pct": 0.01,
                "max_drawdown_pct": 0.004,
                "sharpe_15m": 0.05,
                "fold_contribution": 0.65,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.500 BTCUSD=0.500",
                "crypto_allowed_utc_hours": "0|1|2|3|4|5|6|7|8",
            },
            {
                "slot": "survival",
                "label": "demo_survival",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test survival profile",
                "reason": "smaller size",
                "return_pct": 0.005,
                "max_drawdown_pct": 0.002,
                "sharpe_15m": 0.03,
                "fold_contribution": 0.50,
                "strategy_map": "EURUSD=simple_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.250 BTCUSD=0.250",
                "crypto_allowed_utc_hours": "0|1|2|3|4|5|6|7|8",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_refined_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": "refined",
        "recommendation_reason": "unit-test research refinement",
        "profiles": [
            {
                "slot": "refined",
                "label": "demo_refined",
                "evidence_status": "PROMOTE",
                "use_case": "test refined profile",
                "reason": "scaled weak symbols",
                "return_pct": 0.012,
                "max_drawdown_pct": 0.003,
                "sharpe_15m": 0.06,
                "fold_contribution": 0.60,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.500 BTCUSD=0.375",
                "crypto_allowed_utc_hours": "0|1|2|3|4|5|6|7|8",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
