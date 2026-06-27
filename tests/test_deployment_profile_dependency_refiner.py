from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_dependency_refiner import (
    dependent_symbols_from_robustness_csv,
    refine_deployment_profile_dependency,
    write_dependency_refined_profile_pack_json,
    write_deployment_profile_dependency_refinement_csv,
)
from quanthack.backtesting.deployment_profile_backtest import load_deployment_profile
from quanthack.cli import deployment_profile_dependency_refine
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileDependencyRefinerTest(TestCase):
    def test_refiner_scales_dependent_symbols_and_writes_outputs(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "USDCAD", "BTCUSD"),
            periods=56,
            interval_minutes=15,
            seed=241,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            robustness_path = root / "robustness.csv"
            output_path = root / "dependency.csv"
            refined_pack_path = root / "dependency_pack.json"
            _write_profile_pack(profile_path)
            _write_robustness(robustness_path)

            profile = load_deployment_profile(
                profile_pack_json=profile_path,
                slot="symbol_refined",
            )
            dependent = dependent_symbols_from_robustness_csv(
                robustness_csv=robustness_path,
                profile=profile,
            )
            result = refine_deployment_profile_dependency(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="symbol_refined",
                robustness_csv=robustness_path,
                dependency_scales=(1.0, 0.5),
                include_walk_forward=True,
                train_size=16,
                test_size=8,
                step_size=8,
            )
            write_deployment_profile_dependency_refinement_csv(result, output_path)
            self.assertIsNotNone(result.best)
            write_dependency_refined_profile_pack_json(
                source_profile_pack_json=profile_path,
                result=result,
                candidate=result.best,
                output_json=refined_pack_path,
            )
            output_text = output_path.read_text(encoding="utf-8")
            pack = json.loads(refined_pack_path.read_text(encoding="utf-8"))

        self.assertEqual(tuple(row.symbol for row in dependent), ("USDCAD",))
        self.assertEqual(tuple(row.symbol for row in result.dependent_symbols), ("USDCAD",))
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(
            any(
                ("USDCAD", 0.5) in candidate.multipliers_by_symbol
                for candidate in result.candidates
            )
        )
        self.assertIn("candidate_decision", output_text)
        self.assertIn("dependency_loss_reduction", output_text)
        self.assertEqual(pack["recommended_slot"], "dependency_refined")
        self.assertTrue(
            any(profile["slot"] == "dependency_refined" for profile in pack["profiles"])
        )

    def test_dependency_refine_cli_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "USDCAD", "BTCUSD"),
            periods=56,
            interval_minutes=15,
            seed=242,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            robustness_path = root / "robustness.csv"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "dependency.csv"
            refined_pack_path = root / "dependency_pack.json"
            _write_profile_pack(profile_path)
            _write_robustness(robustness_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_dependency_refine.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--slot",
                        "symbol_refined",
                        "--robustness-csv",
                        str(robustness_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--dependency-scale",
                        "1.0",
                        "--dependency-scale",
                        "0.0",
                        "--train-size",
                        "16",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--output",
                        str(output_path),
                        "--refined-pack-json",
                        str(refined_pack_path),
                    ]
                )

            output = stdout.getvalue()
            csv_text = output_path.read_text(encoding="utf-8")
            pack_text = refined_pack_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Dependency Refinement", output)
        self.assertIn("Dependent symbols: USDCAD", output)
        self.assertIn("dependency_scale", csv_text)
        self.assertIn('"recommended_slot": "dependency_refined"', pack_text)


def _write_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": "symbol_refined",
        "recommendation_reason": "unit test",
        "profiles": [
            {
                "slot": "symbol_refined",
                "label": "demo_symbol_refined",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test profile",
                "reason": "test",
                "return_pct": 0.01,
                "max_drawdown_pct": 0.005,
                "sharpe_15m": 0.04,
                "fold_contribution": 0.75,
                "strategy_map": (
                    "EURUSD=macd_momentum "
                    "USDCAD=macd_momentum "
                    "BTCUSD=crypto_mean_reversion"
                ),
                "multiplier_map": "EURUSD=0.500 USDCAD=1.000 BTCUSD=0.375",
                "crypto_allowed_utc_hours": "all",
                "symbol_allowed_utc_hours": "BTCUSD=0|1|2|3|4|5|6|7|8",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_robustness(path: Path) -> None:
    path.write_text(
        "scenario_type,scenario,excluded_symbol,slippage_multiplier,slippage_bps,"
        "symbols,symbol_count,return_pct,return_delta_pct,max_drawdown_pct,"
        "drawdown_delta_pct,sharpe_15m,sharpe_delta,fills,risk_discipline_score,"
        "total_pnl_usd,total_pnl_delta_usd,decision,note\n"
        "baseline,baseline,,1,1,EURUSD;USDCAD;BTCUSD,3,0.010,0,0.005,0,"
        "0.05,0,4,100,10000,0,BASELINE,reference\n"
        "leave_one_symbol,exclude_USDCAD,USDCAD,1,1,EURUSD;BTCUSD,2,"
        "0.002,-0.008,0.003,-0.002,0.02,-0.03,2,100,2000,-8000,"
        "FRAGILE,excluding USDCAD removes too much return\n"
        "leave_one_symbol,exclude_EURUSD,EURUSD,1,1,USDCAD;BTCUSD,2,"
        "0.009,-0.001,0.004,-0.001,0.04,-0.01,3,100,9000,-1000,"
        "PASS_WEAKER,still positive\n",
        encoding="utf-8",
    )
