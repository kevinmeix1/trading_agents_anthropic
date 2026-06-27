from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.adaptive_strategy_policy_sweep import (
    sweep_adaptive_strategy_policies,
    write_adaptive_strategy_policy_sweep_csv,
)
from quanthack.backtesting.adaptive_strategy_selector import (
    run_adaptive_strategy_selection,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class AdaptiveStrategyPolicySweepTest(TestCase):
    def test_sweeps_policy_grid_and_writes_ranked_csv(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=64,
            interval_minutes=15,
            seed=731,
        )

        result = sweep_adaptive_strategy_policies(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD"),
            train_size=32,
            test_size=8,
            step_size=8,
            loss_cooldown_values=(0, 1),
            min_train_adjusted_return_values=(None, 0.0),
            train_fill_penalty_values=(0.0,),
            transition_risk_multiplier_values=(1.0,),
            cash_fallback_values=(False, True),
            train_stability_settings=((0, False),),
        )

        self.assertEqual(len(result.candidates), 8)
        self.assertIsNotNone(result.best)
        self.assertGreaterEqual(result.candidates[0].selector_score, 0.0)
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "adaptive_policy_sweep.csv"
            write_adaptive_strategy_policy_sweep_csv(result, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("rank,promotion_status,live_ready,selector_score", text)
        self.assertIn("compounded_test_return_pct", text)
        self.assertIn("allow_cash_fallback", text)
        self.assertIn("selection_counts", text)

    def test_cached_sweep_matches_direct_selector_for_same_policy(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=64,
            interval_minutes=15,
            seed=733,
        )

        direct = run_adaptive_strategy_selection(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD"),
            train_size=32,
            test_size=8,
            step_size=8,
            loss_cooldown_folds=1,
            min_train_drawdown_adjusted_return_pct=0.0,
            transition_risk_multiplier=0.5,
            allow_cash_fallback=True,
        )
        sweep = sweep_adaptive_strategy_policies(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD"),
            train_size=32,
            test_size=8,
            step_size=8,
            loss_cooldown_values=(1,),
            min_train_adjusted_return_values=(0.0,),
            train_fill_penalty_values=(0.0,),
            transition_risk_multiplier_values=(0.5,),
            cash_fallback_values=(True,),
            train_stability_settings=((0, False),),
        )
        cached = sweep.candidates[0].result

        self.assertEqual(
            [fold.selected_strategy for fold in cached.folds],
            [fold.selected_strategy for fold in direct.folds],
        )
        self.assertAlmostEqual(
            cached.compounded_test_return_pct,
            direct.compounded_test_return_pct,
        )
        self.assertEqual(
            cached.total_evaluation_fills,
            direct.total_evaluation_fills,
        )

    def test_rejects_empty_strategy_list(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD",),
            periods=48,
            interval_minutes=15,
            seed=732,
        )

        with self.assertRaisesRegex(ValueError, "at least one strategy"):
            sweep_adaptive_strategy_policies(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                strategy_names=(),
                symbols=("EURUSD",),
                train_size=24,
                test_size=8,
                step_size=8,
            )
