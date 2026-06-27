from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.portfolio_router_walk_forward import (
    PortfolioRouterWalkForwardSummary,
    decide_router_promotion,
    run_portfolio_router_walk_forward,
    write_portfolio_router_walk_forward_folds_csv,
    write_portfolio_router_walk_forward_summary_csv,
)
from quanthack.backtesting.router_optimizer import RouterBehaviorProfile, RouterWeightSet
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class PortfolioRouterWalkForwardTest(TestCase):
    def test_router_walk_forward_tunes_weights_on_train_and_tests_later(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=40,
            interval_minutes=15,
            seed=31,
        )

        result = run_portfolio_router_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            weight_sets=(
                RouterWeightSet(0.40, 0.20, 0.35, 0.25, 0.20, 0.05, 0.10),
                RouterWeightSet(0.20, 0.50, 0.20, 0.10),
            ),
            behavior_profiles=(
                RouterBehaviorProfile(),
                RouterBehaviorProfile(
                    entry_score=0.55,
                    min_signal_confidence=0.20,
                    cost_buffer=1.20,
                    conflict_penalty=0.70,
                    primary_signal_override_enabled=False,
                ),
            ),
            train_size=14,
            test_size=8,
            step_size=8,
            min_test_fills=0,
            min_stable_fold_fraction=0.0,
        )

        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY"))
        self.assertEqual(len(result.weight_sets), 2)
        self.assertEqual(len(result.behavior_profiles), 2)
        self.assertEqual(len(result.folds), 3)
        self.assertEqual(result.summary.folds, result.folds)
        self.assertGreaterEqual(result.summary.median_test_proxy_score, 0.0)
        self.assertLessEqual(result.summary.median_test_proxy_score, 100.0)
        for fold in result.folds:
            self.assertLess(fold.train_end, fold.test_start)
            self.assertIn(fold.selected_weights, result.weight_sets)
            self.assertIn(fold.selected_behavior, result.behavior_profiles)
            self.assertIn(fold.test_best_candidate.weights, result.weight_sets)
            self.assertIn(fold.test_best_candidate.behavior, result.behavior_profiles)
            self.assertGreaterEqual(fold.selected_test_candidate.proxy_score, 0.0)
            self.assertLessEqual(fold.selected_test_candidate.proxy_score, 100.0)

    def test_router_walk_forward_rejects_too_little_data(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=10,
            interval_minutes=15,
            seed=32,
        )

        with self.assertRaisesRegex(ValueError, "not enough aligned timestamps"):
            run_portfolio_router_walk_forward(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                symbols=("EURUSD", "GBPUSD"),
                train_size=8,
                test_size=5,
                step_size=1,
            )

    def test_router_walk_forward_csv_outputs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=32,
            interval_minutes=15,
            seed=33,
        )
        result = run_portfolio_router_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            weight_sets=(
                RouterWeightSet(0.40, 0.20, 0.35, 0.25, 0.20, 0.05, 0.10),
                RouterWeightSet(0.20, 0.50, 0.20, 0.10),
            ),
            train_size=12,
            test_size=8,
            step_size=8,
            min_test_fills=0,
            min_stable_fold_fraction=0.0,
        )

        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "router_wf_summary.csv"
            folds_path = Path(tmpdir) / "router_wf_folds.csv"
            write_portfolio_router_walk_forward_summary_csv(result, summary_path)
            write_portfolio_router_walk_forward_folds_csv(result, folds_path)
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("eligible,folds,symbols,candidate_weight_sets", summary_text)
        self.assertIn("candidate_behavior_profiles", summary_text)
        self.assertIn("promotion_status", summary_text)
        self.assertIn("fold,train_start,train_end,test_start", folds_text)
        self.assertIn("selected_weights", folds_text)
        self.assertIn("selected_behavior", folds_text)

    def test_router_promotion_decision_rejects_unstable_summary(self) -> None:
        summary = PortfolioRouterWalkForwardSummary(
            folds=(None,),  # type: ignore[arg-type]
            stable_fold_fraction=0.20,
            median_test_proxy_score=80.0,
            median_test_return_pct=0.01,
            lower_quartile_test_return_pct=0.001,
            median_test_sharpe_15m=1.0,
            worst_test_drawdown_pct=0.01,
            average_risk_discipline_score=100.0,
            total_test_fills=10,
            total_test_turnover=100_000.0,
            most_selected_weights="mom=0.30",
            most_selected_behavior="entry=0.35",
            selected_was_test_best_fraction=0.80,
            eligible=True,
        )

        decision = decide_router_promotion(summary)

        self.assertEqual(decision.status, "REJECT")
        self.assertFalse(decision.live_ready)
        self.assertIn("stable fold fraction", decision.reason)

    def test_router_promotion_decision_promotes_strong_summary(self) -> None:
        summary = PortfolioRouterWalkForwardSummary(
            folds=(None,),  # type: ignore[arg-type]
            stable_fold_fraction=0.75,
            median_test_proxy_score=85.0,
            median_test_return_pct=0.01,
            lower_quartile_test_return_pct=0.001,
            median_test_sharpe_15m=1.2,
            worst_test_drawdown_pct=0.01,
            average_risk_discipline_score=100.0,
            total_test_fills=20,
            total_test_turnover=250_000.0,
            most_selected_weights="mom=0.30",
            most_selected_behavior="entry=0.35",
            selected_was_test_best_fraction=0.75,
            eligible=True,
        )

        decision = decide_router_promotion(summary)

        self.assertEqual(decision.status, "PROMOTE")
        self.assertTrue(decision.live_ready)

    def test_router_promotion_decision_rejects_economically_tiny_return(self) -> None:
        summary = PortfolioRouterWalkForwardSummary(
            folds=(None,),  # type: ignore[arg-type]
            stable_fold_fraction=0.75,
            median_test_proxy_score=85.0,
            median_test_return_pct=0.000001,
            lower_quartile_test_return_pct=0.0,
            median_test_sharpe_15m=0.5,
            worst_test_drawdown_pct=0.001,
            average_risk_discipline_score=100.0,
            total_test_fills=20,
            total_test_turnover=250_000.0,
            most_selected_weights="squeeze=1.00",
            most_selected_behavior="entry=0.35",
            selected_was_test_best_fraction=0.75,
            eligible=True,
        )

        decision = decide_router_promotion(summary)

        self.assertEqual(decision.status, "REJECT")
        self.assertFalse(decision.live_ready)
        self.assertIn("median test return", decision.reason)
