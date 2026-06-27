from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.backtesting.walk_forward import (
    run_walk_forward,
    write_walk_forward_folds_csv,
    write_walk_forward_summary_csv,
)


class WalkForwardTest(TestCase):
    def test_walk_forward_returns_ranked_summaries(self) -> None:
        config = load_config("configs/default.toml")

        result = run_walk_forward(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("momentum", "simple_momentum", "mean_reversion"),
            symbol="EURUSD",
            train_size=10,
            test_size=5,
            step_size=5,
            momentum_lookbacks=(3, 5),
            momentum_threshold_bps=(4.0, 8.0),
        )

        self.assertEqual(len(result.summaries), 2)
        self.assertIsNotNone(result.best)
        rank_keys = [summary.rank_key for summary in result.summaries]
        self.assertEqual(rank_keys, sorted(rank_keys, reverse=True))

    def test_walk_forward_folds_do_not_overlap_train_and_test(self) -> None:
        config = load_config("configs/default.toml")

        result = run_walk_forward(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("simple_momentum",),
            symbol="EURUSD",
            train_size=10,
            test_size=5,
            step_size=5,
            momentum_lookbacks=(3,),
            momentum_threshold_bps=(4.0,),
        )

        summary = result.summaries[0]
        self.assertEqual(len(summary.folds), 2)
        for fold in summary.folds:
            self.assertLess(fold.train_end, fold.test_start)
            self.assertIn("lookback=", fold.selected_parameters)

    def test_walk_forward_tunes_moving_average_crossover_parameters(self) -> None:
        config = load_config("configs/default.toml")

        result = run_walk_forward(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("ma_crossover",),
            symbol="EURUSD",
            train_size=10,
            test_size=5,
            step_size=5,
            momentum_lookbacks=(3,),
            momentum_threshold_bps=(4.0,),
            ma_fast_windows=(2, 3),
            ma_slow_windows=(5, 8),
            ma_min_separation_bps=(1.0, 2.0),
        )

        summary = result.summaries[0]
        self.assertEqual(summary.strategy_name, "ma_crossover")
        self.assertEqual(len(summary.folds), 2)
        for fold in summary.folds:
            self.assertIn("fast_window=", fold.selected_parameters)
            self.assertIn("slow_window=", fold.selected_parameters)
            self.assertIn("min_separation_bps=", fold.selected_parameters)

    def test_walk_forward_rejects_invalid_moving_average_grid(self) -> None:
        config = load_config("configs/default.toml")

        with self.assertRaisesRegex(ValueError, "fast < slow"):
            run_walk_forward(
                config=config,
                prices=load_price_history(config.backtest.price_csv),
                quotes=load_quote_history(config.backtest.quote_csv),
                strategy_names=("ma_crossover",),
                symbol="EURUSD",
                train_size=10,
                test_size=5,
                step_size=5,
                momentum_lookbacks=(3,),
                momentum_threshold_bps=(4.0,),
                ma_fast_windows=(8,),
                ma_slow_windows=(5,),
                ma_min_separation_bps=(1.0,),
            )

    def test_walk_forward_rejects_too_little_data(self) -> None:
        config = load_config("configs/default.toml")

        with self.assertRaisesRegex(ValueError, "not enough bars"):
            run_walk_forward(
                config=config,
                prices=load_price_history(config.backtest.price_csv),
                quotes=load_quote_history(config.backtest.quote_csv),
                strategy_names=("simple_momentum",),
                symbol="EURUSD",
                train_size=19,
                test_size=5,
                step_size=1,
                momentum_lookbacks=(3,),
                momentum_threshold_bps=(4.0,),
            )

    def test_walk_forward_csv_outputs(self) -> None:
        config = load_config("configs/default.toml")
        result = run_walk_forward(
            config=config,
            prices=load_price_history(config.backtest.price_csv),
            quotes=load_quote_history(config.backtest.quote_csv),
            strategy_names=("simple_momentum",),
            symbol="EURUSD",
            train_size=10,
            test_size=5,
            step_size=5,
            momentum_lookbacks=(3,),
            momentum_threshold_bps=(4.0,),
        )

        with TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            folds_path = Path(tmpdir) / "folds.csv"
            write_walk_forward_summary_csv(result, summary_path)
            write_walk_forward_folds_csv(result, folds_path)
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")

        self.assertIn("rank,strategy,eligible", summary_text)
        self.assertIn("strategy,fold,symbol", folds_text)
