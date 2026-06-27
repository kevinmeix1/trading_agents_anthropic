from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
    write_fixed_warmup_folds_csv,
    write_fixed_warmup_summary_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class FixedWarmupPortfolioWalkForwardTest(TestCase):
    def test_fixed_warmup_walk_forward_scores_forward_windows(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=36,
            interval_minutes=15,
            seed=31,
        )

        result = run_fixed_warmup_portfolio_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=12,
            test_size=8,
            step_size=8,
        )

        self.assertEqual(result.strategy_name, "simple_momentum")
        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY"))
        self.assertEqual(len(result.folds), 3)
        self.assertGreaterEqual(result.active_fold_fraction, 0.0)
        self.assertLessEqual(result.active_fold_fraction, 1.0)
        self.assertGreaterEqual(result.active_positive_fold_fraction, 0.0)
        self.assertLessEqual(result.active_positive_fold_fraction, 1.0)
        self.assertGreaterEqual(result.non_negative_fold_fraction, 0.0)
        self.assertLessEqual(result.non_negative_fold_fraction, 1.0)
        for fold in result.folds:
            self.assertLess(fold.train_end, fold.test_start)
            self.assertEqual(fold.evaluation.evaluation_start, fold.test_start)
            self.assertEqual(fold.metrics.sampled_equity_points, 8)
            self.assertGreaterEqual(fold.risk_discipline.score, 0)

    def test_fixed_warmup_walk_forward_rejects_too_little_data(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=10,
            interval_minutes=15,
            seed=32,
        )

        with self.assertRaisesRegex(ValueError, "not enough aligned timestamps"):
            run_fixed_warmup_portfolio_walk_forward(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_name="simple_momentum",
                symbols=("EURUSD", "GBPUSD"),
                train_size=8,
                test_size=8,
                step_size=1,
            )

    def test_fixed_warmup_walk_forward_csv_outputs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=28,
            interval_minutes=15,
            seed=33,
        )
        result = run_fixed_warmup_portfolio_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            train_size=10,
            test_size=6,
            step_size=6,
        )

        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"
            write_fixed_warmup_summary_csv(result, summary_path)
            write_fixed_warmup_folds_csv(result, folds_path)
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("strategy,symbols,folds", summary_text)
        self.assertIn("positive_fold_fraction", summary_text)
        self.assertIn("active_positive_fold_fraction", summary_text)
        self.assertIn("non_negative_fold_fraction", summary_text)
        self.assertIn("median_active_test_return_pct", summary_text)
        self.assertIn("largest_positive_fold_contribution", summary_text)
        self.assertIn("fold,train_start,train_end,test_start", folds_text)
        self.assertIn("evaluation_fills", folds_text)

    def test_fixed_warmup_walk_forward_accepts_strategy_overrides(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=36,
            interval_minutes=15,
            seed=34,
        )

        result = run_fixed_warmup_portfolio_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_name="simple_momentum",
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            strategy_by_symbol={"GBPUSD": "macd_momentum"},
            train_size=12,
            test_size=8,
            step_size=8,
        )

        self.assertIn("GBPUSD=macd_momentum", result.strategy_name)
        self.assertEqual(result.symbols, ("EURUSD", "GBPUSD", "USDJPY"))

    def test_promotion_rejects_empty_walk_forward(self) -> None:
        decision = decide_fixed_warmup_promotion(
            FixedWarmupPortfolioWalkForwardResult(
                strategy_name="simple_momentum",
                symbols=("EURUSD",),
                folds=(),
            )
        )

        self.assertEqual(decision.status, "REJECT")
        self.assertFalse(decision.live_ready)
