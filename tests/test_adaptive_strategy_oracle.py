from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.adaptive_strategy_oracle import (
    build_adaptive_strategy_oracle_diagnostic,
    write_adaptive_strategy_oracle_candidates_csv,
    write_adaptive_strategy_oracle_folds_csv,
    write_adaptive_strategy_oracle_summary_csv,
)
from quanthack.core.config import load_config
from quanthack.market.sample_data import generate_synthetic_market_data


class AdaptiveStrategyOracleDiagnosticTest(TestCase):
    def test_builds_oracle_diagnostic_and_writes_csvs(self) -> None:
        config = load_config("configs/default.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "GBPUSD"),
            periods=64,
            interval_minutes=15,
            seed=741,
        )

        diagnostic = build_adaptive_strategy_oracle_diagnostic(
            config=config,
            prices=data.prices,
            quotes=data.quotes,
            strategy_names=("simple_momentum", "macd_momentum"),
            symbols=("EURUSD", "GBPUSD"),
            train_size=32,
            test_size=8,
            step_size=8,
            loss_cooldown_folds=1,
            include_cash_oracle=True,
        )

        self.assertEqual(diagnostic.fold_count, 4)
        self.assertGreater(len(diagnostic.candidates), 0)
        self.assertGreaterEqual(diagnostic.selected_was_oracle_fraction, 0.0)
        self.assertLessEqual(diagnostic.selected_was_oracle_fraction, 1.0)
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.csv"
            folds_path = root / "folds.csv"
            candidates_path = root / "candidates.csv"
            write_adaptive_strategy_oracle_summary_csv(diagnostic, summary_path)
            write_adaptive_strategy_oracle_folds_csv(diagnostic, folds_path)
            write_adaptive_strategy_oracle_candidates_csv(
                diagnostic,
                candidates_path,
            )
            summary_text = summary_path.read_text(encoding="utf-8")
            folds_text = folds_path.read_text(encoding="utf-8")
            candidates_text = candidates_path.read_text(encoding="utf-8")

        self.assertIn("selected_was_oracle_fraction", summary_text)
        self.assertIn("total_regret_pct", summary_text)
        self.assertIn("selected_strategy,oracle_strategy", folds_text)
        self.assertIn("oracle_was_cash", folds_text)
        self.assertIn("strategy,train_rank", candidates_text)
        self.assertIn("oracle_best", candidates_text)
