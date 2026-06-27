from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.portfolio_universe_scan import UniverseBasket
from quanthack.backtesting.portfolio_walk_forward import (
    PortfolioWalkForwardSummary,
    decide_portfolio_promotion,
    run_portfolio_walk_forward,
    write_portfolio_walk_forward_folds_csv,
    write_portfolio_walk_forward_summary_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class PortfolioWalkForwardTest(TestCase):
    def test_portfolio_walk_forward_selects_and_tests_unseen_windows(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=40,
            interval_minutes=15,
            seed=21,
        )

        result = run_portfolio_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "ma_crossover"),
            baskets=(
                UniverseBasket("core_fx", ("EURUSD", "GBPUSD", "USDJPY")),
                UniverseBasket("fx_gold", ("EURUSD", "USDJPY", "XAUUSD")),
            ),
            train_size=14,
            test_size=8,
            step_size=8,
            min_test_fills=0,
            min_stable_fold_fraction=0.0,
        )

        self.assertEqual(result.available_symbols, ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"))
        self.assertEqual(len(result.folds), 3)
        self.assertEqual(result.summary.folds, result.folds)
        self.assertGreaterEqual(result.summary.median_test_proxy_score, 0.0)
        self.assertLessEqual(result.summary.median_test_proxy_score, 100.0)
        for fold in result.folds:
            self.assertLess(fold.train_end, fold.test_start)
            self.assertIn(fold.selected_strategy, {"simple_momentum", "ma_crossover"})
            self.assertIn(fold.selected_basket.name, {"core_fx", "fx_gold"})
            self.assertGreaterEqual(fold.test_row.proxy_score, 0.0)
            self.assertLessEqual(fold.test_row.proxy_score, 100.0)

    def test_portfolio_walk_forward_rejects_too_little_data(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY"),
            periods=10,
            interval_minutes=15,
            seed=22,
        )

        with self.assertRaisesRegex(ValueError, "not enough aligned timestamps"):
            run_portfolio_walk_forward(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=("simple_momentum",),
                train_size=8,
                test_size=5,
                step_size=1,
            )

    def test_portfolio_walk_forward_csv_outputs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"),
            periods=32,
            interval_minutes=15,
            seed=23,
        )
        result = run_portfolio_walk_forward(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum",),
            baskets=(UniverseBasket("core_fx", ("EURUSD", "GBPUSD", "USDJPY")),),
            train_size=12,
            test_size=8,
            step_size=8,
            min_test_fills=0,
            min_stable_fold_fraction=0.0,
        )

        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"
            write_portfolio_walk_forward_summary_csv(result, summary_path)
            write_portfolio_walk_forward_folds_csv(result, folds_path)
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("eligible,folds,available_symbols", summary_text)
        self.assertIn("promotion_status", summary_text)
        self.assertIn("fold,train_start,train_end,test_start", folds_text)
        self.assertIn("selected_basket", folds_text)

    def test_promotion_decision_rejects_non_eligible_summary(self) -> None:
        decision = decide_portfolio_promotion(
            PortfolioWalkForwardSummary(
                folds=(),
                stable_fold_fraction=0.0,
                median_test_proxy_score=0.0,
                median_test_return_pct=0.0,
                lower_quartile_test_return_pct=0.0,
                median_test_sharpe_15m=0.0,
                worst_test_drawdown_pct=0.0,
                average_risk_discipline_score=0.0,
                total_test_fills=0,
                total_test_turnover=0.0,
                most_selected_basket="",
                most_selected_strategy="",
                eligible=False,
            )
        )

        self.assertEqual(decision.status, "REJECT")
        self.assertFalse(decision.live_ready)
        self.assertIn("no walk-forward folds", decision.reason)
