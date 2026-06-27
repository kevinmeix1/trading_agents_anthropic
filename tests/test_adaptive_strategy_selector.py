from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.adaptive_strategy_selector import (
    AdaptiveStrategyCandidate,
    AdaptiveStrategySelectionResult,
    AdaptiveStrategyTrainStability,
    AdaptiveStrategyTrainScore,
    build_adaptive_strategy_promotion_audit,
    build_adaptive_strategy_stitched_equity_curve,
    decide_adaptive_strategy_selection_promotion,
    run_adaptive_strategy_selection,
    write_adaptive_strategy_selection_folds_csv,
    write_adaptive_strategy_selection_scores_csv,
    write_adaptive_strategy_selection_summary_csv,
    write_adaptive_strategy_promotion_audit_csv,
    write_adaptive_strategy_stitched_equity_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class AdaptiveStrategySelectorTest(TestCase):
    def test_selects_strategy_from_past_window_and_scores_forward_window(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=501,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(result.strategy_names, ("simple_momentum", "macd_momentum"))
        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY"))
        self.assertEqual(result.loss_cooldown_folds, 0)
        self.assertEqual(result.transition_risk_multiplier, 1.0)
        self.assertEqual(len(result.folds), 4)
        self.assertGreaterEqual(result.positive_fold_fraction, 0.0)
        self.assertLessEqual(result.positive_fold_fraction, 1.0)
        self.assertGreaterEqual(result.active_positive_fold_fraction, 0.0)
        self.assertLessEqual(result.active_positive_fold_fraction, 1.0)
        self.assertTrue(result.selection_counts)
        for fold in result.folds:
            self.assertLess(fold.train_end, fold.test_start)
            self.assertIn(fold.selected_strategy, result.strategy_names)
            self.assertEqual(len(fold.train_scores), 2)
            self.assertEqual(fold.evaluation.evaluation_start, fold.test_start)
            self.assertEqual(fold.evaluation_risk_multiplier, 1.0)
            self.assertEqual(fold.metrics.sampled_equity_points, 8)
            self.assertGreaterEqual(fold.risk_discipline.score, 0)

    def test_builds_stitched_out_of_sample_equity_curve(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=505,
        )
        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
        )

        curve = build_adaptive_strategy_stitched_equity_curve(
            result,
            starting_equity=1_000_000,
        )

        self.assertTrue(curve)
        self.assertEqual(curve[0].fold_index, 1)
        self.assertGreater(curve[0].equity, 0)
        self.assertTrue(all(point.drawdown_pct >= 0 for point in curve))
        self.assertEqual(
            [point.timestamp for point in curve],
            sorted(point.timestamp for point in curve),
        )

    def test_loss_cooldown_is_recorded_on_result_and_folds(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=504,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            loss_cooldown_folds=1,
        )

        self.assertEqual(result.loss_cooldown_folds, 1)
        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            self.assertIsInstance(fold.cooldown_blocked_strategies, tuple)

    def test_train_fill_penalty_is_recorded_and_can_change_rank_key(self) -> None:
        light = AdaptiveStrategyTrainScore(
            strategy_name="light",
            strategy_map=(("EURUSD", "simple_momentum"),),
            return_pct=0.0010,
            max_drawdown_pct=0.0,
            sharpe_15m=0.1,
            risk_discipline_score=100,
            fills=1,
            final_equity=1_001_000,
            stability=AdaptiveStrategyTrainStability(
                splits=4,
                active_fraction=0.75,
                positive_fraction=0.50,
                active_positive_fraction=0.67,
                non_negative_fraction=0.75,
                median_return_pct=0.0001,
                median_active_return_pct=0.0002,
            ),
        )
        busy = AdaptiveStrategyTrainScore(
            strategy_name="busy",
            strategy_map=(("EURUSD", "macd_momentum"),),
            return_pct=0.0011,
            max_drawdown_pct=0.0,
            sharpe_15m=0.1,
            risk_discipline_score=100,
            fills=100,
            final_equity=1_001_100,
            stability=AdaptiveStrategyTrainStability(
                splits=4,
                active_fraction=1.0,
                positive_fraction=0.25,
                active_positive_fraction=0.25,
                non_negative_fraction=0.50,
                median_return_pct=-0.0001,
                median_active_return_pct=-0.0001,
            ),
        )

        self.assertGreater(busy.rank_key, light.rank_key)
        self.assertGreater(
            light.rank_key_with_fill_penalty(0.000002),
            busy.rank_key_with_fill_penalty(0.000002),
        )
        self.assertGreater(
            light.rank_key_with_preferences(
                fill_penalty_pct=0.0,
                prefer_train_stability=True,
            ),
            busy.rank_key_with_preferences(
                fill_penalty_pct=0.0,
                prefer_train_stability=True,
            ),
        )

        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=510,
        )
        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            train_fill_penalty_pct=0.000002,
        )

        self.assertEqual(result.train_fill_penalty_pct, 0.000002)

    def test_training_stability_is_recorded_and_can_be_preferred(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=514,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            train_stability_splits=4,
            prefer_train_stability=True,
        )

        self.assertEqual(result.train_stability_splits, 4)
        self.assertTrue(result.prefer_train_stability)
        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            for score in fold.train_scores:
                self.assertEqual(score.stability.splits, 4)
                self.assertGreaterEqual(score.stability.non_negative_fraction, 0.0)
                self.assertLessEqual(score.stability.non_negative_fraction, 1.0)

    def test_prefer_training_stability_requires_splits(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=48,
            interval_minutes=15,
            seed=515,
        )

        with self.assertRaisesRegex(ValueError, "requires train_stability_splits"):
            run_adaptive_strategy_selection(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("simple_momentum", "macd_momentum"),
                symbols=("EURUSD", "GBPUSD"),
                train_size=24,
                test_size=8,
                step_size=8,
                prefer_train_stability=True,
            )

    def test_transition_risk_multiplier_is_recorded_and_validated(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=516,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            transition_risk_multiplier=0.5,
        )

        self.assertEqual(result.transition_risk_multiplier, 0.5)
        self.assertTrue(
            all(
                fold.evaluation_risk_multiplier in {0.5, 1.0}
                for fold in result.folds
            )
        )

        with self.assertRaisesRegex(ValueError, "transition_risk_multiplier"):
            run_adaptive_strategy_selection(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("simple_momentum", "macd_momentum"),
                symbols=("EURUSD", "GBPUSD", "USDJPY"),
                train_size=40,
                test_size=8,
                step_size=8,
                transition_risk_multiplier=0.0,
            )

    def test_training_gate_blocks_candidates_but_keeps_fallback_selection(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=509,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            min_train_fills=10_000,
        )

        self.assertEqual(result.min_train_fills, 10_000)
        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            self.assertEqual(
                set(fold.train_gate_blocked_strategies),
                {"simple_momentum", "macd_momentum"},
            )
            self.assertIn(fold.selected_strategy, result.strategy_names)

    def test_cash_fallback_sits_out_when_all_training_gates_fail(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=511,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            min_train_fills=10_000,
            allow_cash_fallback=True,
        )

        self.assertTrue(result.allow_cash_fallback)
        self.assertIn("cash", result.strategy_names)
        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            self.assertEqual(fold.selected_strategy, "cash")
            self.assertEqual(len(fold.evaluation.fills), 0)
            self.assertEqual(fold.metrics.return_pct, 0.0)
            self.assertEqual(fold.metrics.max_drawdown_pct, 0.0)
            self.assertEqual(
                set(fold.train_gate_blocked_strategies),
                {"simple_momentum", "macd_momentum"},
            )
            self.assertIn("EURUSD=cash", fold.selected_train_score.strategy_map_text)

    def test_can_select_between_single_strategy_and_candidate_map(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=506,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum",),
            candidate_maps=(
                AdaptiveStrategyCandidate(
                    label="hybrid_map",
                    strategy_by_symbol=(
                        ("EURUSD", "simple_momentum"),
                        ("GBPUSD", "macd_momentum"),
                        ("USDJPY", "macd_momentum"),
                    ),
                ),
            ),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(result.strategy_names, ("simple_momentum", "hybrid_map"))
        self.assertEqual(len(result.folds), 4)
        self.assertTrue(
            any(
                score.strategy_name == "hybrid_map"
                and "GBPUSD=macd_momentum" in score.strategy_map_text
                for fold in result.folds
                for score in fold.train_scores
            )
        )

    def test_per_symbol_selection_adds_dynamic_candidate(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=511,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            per_symbol_selection=True,
        )

        self.assertTrue(result.per_symbol_selection)
        self.assertIn("per_symbol_adaptive", result.strategy_names)
        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            self.assertTrue(fold.selected_strategy_map)
            self.assertEqual(
                {symbol for symbol, _ in fold.selected_strategy_map},
                {"EURUSD", "GBPUSD", "USDJPY"},
            )
            self.assertTrue(
                any(score.strategy_name == "per_symbol_adaptive" for score in fold.train_scores)
            )

    def test_per_symbol_only_forces_dynamic_candidate(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=512,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=40,
            test_size=8,
            step_size=8,
            per_symbol_selection=True,
            per_symbol_only=True,
        )

        self.assertTrue(result.per_symbol_selection)
        self.assertTrue(result.per_symbol_only)
        self.assertEqual(result.strategy_names, ("per_symbol_adaptive",))
        for fold in result.folds:
            self.assertEqual(fold.selected_strategy, "per_symbol_adaptive")
            self.assertEqual(
                [score.strategy_name for score in fold.train_scores],
                ["per_symbol_adaptive"],
            )

    def test_per_symbol_only_requires_per_symbol_selection(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=513,
        )

        with self.assertRaisesRegex(ValueError, "requires per_symbol_selection"):
            run_adaptive_strategy_selection(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("simple_momentum", "macd_momentum"),
                symbols=("EURUSD", "GBPUSD", "USDJPY"),
                train_size=40,
                test_size=8,
                step_size=8,
                per_symbol_only=True,
            )

    def test_can_select_between_partial_recipe_maps(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=72,
            interval_minutes=15,
            seed=508,
        )

        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=(),
            recipe_maps=(
                AdaptiveStrategyCandidate(
                    label="eur_gbp_recipe",
                    strategy_by_symbol=(
                        ("EURUSD", "simple_momentum"),
                        ("GBPUSD", "simple_momentum"),
                    ),
                ),
                AdaptiveStrategyCandidate(
                    label="jpy_recipe",
                    strategy_by_symbol=(("USDJPY", "macd_momentum"),),
                ),
            ),
            train_size=40,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(result.strategy_names, ("eur_gbp_recipe", "jpy_recipe"))
        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY"))
        self.assertEqual(len(result.folds), 4)
        self.assertTrue(
            any(
                "USDJPY=macd_momentum" in score.strategy_map_text
                for fold in result.folds
                for score in fold.train_scores
            )
        )

    def test_candidate_map_must_cover_selected_symbols(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=48,
            interval_minutes=15,
            seed=507,
        )

        with self.assertRaisesRegex(ValueError, "missing symbols"):
            run_adaptive_strategy_selection(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("simple_momentum",),
                candidate_maps=(
                    AdaptiveStrategyCandidate(
                        label="incomplete_map",
                        strategy_by_symbol=(("EURUSD", "simple_momentum"),),
                    ),
                ),
                symbols=("EURUSD", "GBPUSD", "USDJPY"),
                train_size=24,
                test_size=8,
                step_size=8,
            )

    def test_rejects_empty_strategy_list(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=48,
            interval_minutes=15,
            seed=502,
        )

        with self.assertRaisesRegex(ValueError, "at least one strategy"):
            run_adaptive_strategy_selection(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=(),
                symbols=("EURUSD", "GBPUSD"),
                train_size=24,
                test_size=8,
                step_size=8,
            )

    def test_promotion_rejects_empty_result(self) -> None:
        result = AdaptiveStrategySelectionResult(
            strategy_names=("simple_momentum",),
            symbols=("EURUSD",),
            folds=(),
        )
        decision = decide_adaptive_strategy_selection_promotion(result)
        audit = build_adaptive_strategy_promotion_audit(result)

        self.assertEqual(decision.status, "REJECT")
        self.assertFalse(decision.live_ready)
        self.assertEqual(audit.decision.status, "REJECT")
        self.assertTrue(audit.failed_gates)
        self.assertTrue(
            any(gate.gate_id == "folds_present" for gate in audit.failed_gates)
        )

    def test_writes_summary_folds_and_scores_csvs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=56,
            interval_minutes=15,
            seed=503,
        )
        result = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=32,
            test_size=8,
            step_size=8,
        )

        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"
            scores_path = Path(tmpdir) / "scores.csv"
            promotion_path = Path(tmpdir) / "promotion.csv"
            write_adaptive_strategy_selection_summary_csv(result, summary_path)
            write_adaptive_strategy_selection_folds_csv(result, folds_path)
            write_adaptive_strategy_selection_scores_csv(result, scores_path)
            write_adaptive_strategy_promotion_audit_csv(
                build_adaptive_strategy_promotion_audit(result),
                promotion_path,
            )
            equity_path = Path(tmpdir) / "adaptive_equity.csv"
            write_adaptive_strategy_stitched_equity_csv(
                result,
                equity_path,
                starting_equity=1_000_000,
            )

            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")
            scores_text = scores_path.read_text(encoding="utf-8")
            promotion_text = promotion_path.read_text(encoding="utf-8")
            equity_text = equity_path.read_text(encoding="utf-8")

        self.assertIn("strategies,symbols,folds", summary_text)
        self.assertIn("loss_cooldown_folds", summary_text)
        self.assertIn("min_train_fills", summary_text)
        self.assertIn("train_fill_penalty_pct", summary_text)
        self.assertIn("train_stability_splits", summary_text)
        self.assertIn("prefer_train_stability", summary_text)
        self.assertIn("transition_risk_multiplier", summary_text)
        self.assertIn("per_symbol_selection", summary_text)
        self.assertIn("per_symbol_only", summary_text)
        self.assertIn("compounded_test_return_pct", summary_text)
        self.assertIn("selection_counts", summary_text)
        self.assertIn("selected_strategy", folds_text)
        self.assertIn("selected_strategy_map", folds_text)
        self.assertIn("cooldown_blocked_strategies", folds_text)
        self.assertIn("train_gate_blocked_strategies", folds_text)
        self.assertIn("selected_train_drawdown_adjusted_return_pct", folds_text)
        self.assertIn("selected_train_stability_active_positive_fraction", folds_text)
        self.assertIn("evaluation_risk_multiplier", folds_text)
        self.assertIn("strategy,strategy_map,selected,train_return_pct", scores_text)
        self.assertIn("train_gate_passed", scores_text)
        self.assertIn("train_stability_non_negative_fraction", scores_text)
        self.assertIn("status,live_ready,decision_reason,gate_id", promotion_text)
        self.assertIn("live_positive_fold_fraction", promotion_text)
        self.assertIn("gap", promotion_text)
        self.assertIn("timestamp,equity,drawdown_pct", equity_text)
        self.assertIn("source_fold_start_equity", equity_text)
