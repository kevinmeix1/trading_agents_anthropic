from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_backtest import load_deployment_profile
from quanthack.backtesting.deployment_profile_dependency_refiner import (
    dependent_symbols_from_robustness_csv,
)
from quanthack.backtesting.deployment_profile_dependency_replacement_refiner import (
    refine_deployment_profile_dependency_replacement,
    replacement_pool_from_robustness_csv,
    write_dependency_replacement_profile_pack_json,
    write_deployment_profile_dependency_replacement_csv,
)
from quanthack.cli import deployment_profile_dependency_replace
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileDependencyReplacementRefinerTest(TestCase):
    def test_replacement_refiner_refills_dependent_multiplier_capacity(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "USDCAD", "XAUUSD", "XRPUSD"),
            periods=56,
            interval_minutes=15,
            seed=251,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            robustness_path = root / "robustness.csv"
            output_path = root / "replacement.csv"
            refined_pack_path = root / "replacement_pack.json"
            _write_profile_pack(profile_path)
            _write_robustness(robustness_path)

            profile = load_deployment_profile(
                profile_pack_json=profile_path,
                slot="symbol_refined",
            )
            dependent = dependent_symbols_from_robustness_csv(
                robustness_csv=robustness_path,
                profile=profile,
                dependency_threshold_pct=-0.005,
            )
            pool = replacement_pool_from_robustness_csv(
                robustness_csv=robustness_path,
                profile=profile,
                dependent_symbols=dependent,
            )
            result = refine_deployment_profile_dependency_replacement(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="symbol_refined",
                robustness_csv=robustness_path,
                dependency_scales=(0.0,),
                refill_fractions=(1.0,),
                basket_sizes=(2,),
                dependency_threshold_pct=-0.005,
                include_walk_forward=True,
                train_size=16,
                test_size=8,
                step_size=8,
            )
            write_deployment_profile_dependency_replacement_csv(result, output_path)
            self.assertIsNotNone(result.best)
            write_dependency_replacement_profile_pack_json(
                source_profile_pack_json=profile_path,
                result=result,
                candidate=result.best,
                output_json=refined_pack_path,
            )
            csv_text = output_path.read_text(encoding="utf-8")
            pack = json.loads(refined_pack_path.read_text(encoding="utf-8"))

        self.assertEqual(tuple(row.symbol for row in dependent), ("USDCAD",))
        self.assertEqual(tuple(row.symbol for row in pool[:2]), ("XAUUSD", "XRPUSD"))
        replacement_candidates = [
            candidate for candidate in result.candidates if candidate.dependency_scale == 0.0
        ]
        self.assertEqual(len(replacement_candidates), 1)
        multipliers = dict(replacement_candidates[0].multipliers_by_symbol)
        self.assertEqual(multipliers["USDCAD"], 0.0)
        self.assertGreater(multipliers["XAUUSD"], 0.25)
        self.assertGreater(multipliers["XRPUSD"], 0.25)
        self.assertIn("candidate_decision", csv_text)
        self.assertIn("refilled_multiplier", csv_text)
        self.assertEqual(pack["recommended_slot"], "dependency_replacement")
        self.assertTrue(
            any(profile["slot"] == "dependency_replacement" for profile in pack["profiles"])
        )

    def test_dependency_replace_cli_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "USDCAD", "XAUUSD", "XRPUSD"),
            periods=56,
            interval_minutes=15,
            seed=252,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            robustness_path = root / "robustness.csv"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "replacement.csv"
            refined_pack_path = root / "replacement_pack.json"
            _write_profile_pack(profile_path)
            _write_robustness(robustness_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_dependency_replace.main(
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
                        "0.0",
                        "--dependency-threshold-pct",
                        "-0.005",
                        "--refill-fraction",
                        "1.0",
                        "--basket-size",
                        "2",
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

        self.assertIn("Deployment Profile Dependency Replacement", output)
        self.assertIn("Dependent symbols: USDCAD", output)
        self.assertIn("Replacement pool: XAUUSD, XRPUSD", output)
        self.assertIn("refill_fraction", csv_text)
        self.assertIn('"recommended_slot": "dependency_replacement"', pack_text)


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
                    "XAUUSD=macd_momentum "
                    "XRPUSD=crypto_mean_reversion"
                ),
                "multiplier_map": (
                    "EURUSD=1.000 USDCAD=1.000 XAUUSD=0.250 XRPUSD=0.250"
                ),
                "crypto_allowed_utc_hours": "all",
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
        "baseline,baseline,,1,1,EURUSD;USDCAD;XAUUSD;XRPUSD,4,0.010,0,"
        "0.005,0,0.05,0,4,100,10000,0,BASELINE,reference\n"
        "leave_one_symbol,exclude_USDCAD,USDCAD,1,1,EURUSD;XAUUSD;XRPUSD,3,"
        "0.002,-0.008,0.003,-0.002,0.02,-0.03,2,100,2000,-8000,"
        "FRAGILE,excluding USDCAD removes too much return\n"
        "leave_one_symbol,exclude_XAUUSD,XAUUSD,1,1,EURUSD;USDCAD;XRPUSD,3,"
        "0.006,-0.004,0.004,-0.001,0.04,-0.01,3,100,6000,-4000,"
        "PASS_WEAKER,still positive\n"
        "leave_one_symbol,exclude_XRPUSD,XRPUSD,1,1,EURUSD;USDCAD;XAUUSD,3,"
        "0.007,-0.003,0.004,-0.001,0.04,-0.01,3,100,7000,-3000,"
        "PASS_WEAKER,still positive\n"
        "leave_one_symbol,exclude_EURUSD,EURUSD,1,1,USDCAD;XAUUSD;XRPUSD,3,"
        "0.009,-0.001,0.004,-0.001,0.04,-0.01,3,100,9000,-1000,"
        "PASS_WEAKER,still positive\n",
        encoding="utf-8",
    )
