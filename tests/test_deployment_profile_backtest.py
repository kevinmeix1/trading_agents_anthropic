from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_backtest import (
    load_deployment_profile,
    run_deployment_profile_backtest,
    write_deployment_profile_backtest_summary_csv,
)
from quanthack.cli import deployment_profile_backtest
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileBacktestTest(TestCase):
    def test_load_deployment_profile_parses_maps_and_hours(self) -> None:
        with TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "pack.json"
            _write_profile_pack(profile_path)

            profile = load_deployment_profile(
                profile_pack_json=profile_path,
                slot="conservative",
            )

        self.assertEqual(profile.slot, "conservative")
        self.assertEqual(
            profile.strategy_by_symbol,
            (("EURUSD", "macd_momentum"), ("BTCUSD", "crypto_mean_reversion")),
        )
        self.assertEqual(profile.forex_allowed_utc_hours, (13, 14))
        self.assertEqual(profile.metal_allowed_utc_hours, (14,))
        self.assertEqual(profile.crypto_allowed_utc_hours, (0, 1, 2))
        self.assertEqual(profile.symbol_allowed_utc_hours, (("BTCUSD", (1, 2)),))
        self.assertIn("BTCUSD=0.500", profile.multiplier_map_text)

    def test_recommended_paper_only_slot_requires_explicit_choice(self) -> None:
        with TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "pack.json"
            _write_profile_pack(profile_path)

            with self.assertRaisesRegex(ValueError, "not executable"):
                load_deployment_profile(
                    profile_pack_json=profile_path,
                    slot="recommended",
                )

    def test_recommended_slot_can_resolve_research_profile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "pack.json"
            _write_refined_profile_pack(profile_path)

            profile = load_deployment_profile(
                profile_pack_json=profile_path,
                slot="recommended",
            )

        self.assertEqual(profile.slot, "refined")
        self.assertEqual(profile.label, "demo_refined")

    def test_run_deployment_profile_backtest_and_write_summary(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=36,
            interval_minutes=15,
            seed=161,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            summary_path = root / "summary.csv"
            _write_profile_pack(profile_path)

            result = run_deployment_profile_backtest(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="conservative",
            )
            write_deployment_profile_backtest_summary_csv(result, summary_path)
            text = summary_path.read_text(encoding="utf-8")

        self.assertEqual(result.profile.label, "demo_conservative")
        self.assertEqual(result.result.symbols, ("EURUSD", "BTCUSD"))
        self.assertIn("slot,label,evidence_status", text)
        self.assertIn("demo_conservative", text)

    def test_deployment_profile_backtest_cli_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=36,
            interval_minutes=15,
            seed=162,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_prefix = root / "profile_run"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_backtest.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--slot",
                        "conservative",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--output-prefix",
                        str(output_prefix),
                    ]
                )

            output = stdout.getvalue()
            summary_text = (root / "profile_run_summary.csv").read_text(
                encoding="utf-8"
            )
            equity_exists = (root / "profile_run_equity.csv").exists()
            fills_exists = (root / "profile_run_fills.csv").exists()

        self.assertIn("Deployment Profile Backtest", output)
        self.assertIn("demo_conservative", output)
        self.assertIn("demo_conservative", summary_text)
        self.assertTrue(equity_exists)
        self.assertTrue(fills_exists)


def _write_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "mixed_proxy",
        "recommended_slot": "paper_only",
        "recommendation_reason": "research-only data",
        "profiles": [
            {
                "slot": "conservative",
                "label": "demo_conservative",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test profile",
                "reason": "test",
                "return_pct": 0.01,
                "max_drawdown_pct": 0.005,
                "sharpe_15m": 0.04,
                "fold_contribution": 0.75,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=1.000 BTCUSD=0.500",
                "forex_allowed_utc_hours": "13|14",
                "metal_allowed_utc_hours": "14",
                "crypto_allowed_utc_hours": "0|1|2",
                "symbol_allowed_utc_hours": "BTCUSD=1|2",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_refined_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "mixed_proxy",
        "recommended_slot": "refined",
        "recommendation_reason": "unit-test research refinement",
        "profiles": [
            {
                "slot": "refined",
                "label": "demo_refined",
                "evidence_status": "PROMOTE",
                "use_case": "refined research profile",
                "reason": "scaled weak symbols",
                "return_pct": 0.012,
                "max_drawdown_pct": 0.003,
                "sharpe_15m": 0.06,
                "fold_contribution": 0.60,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.500 BTCUSD=0.375",
                "crypto_allowed_utc_hours": "0|1|2|3",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
