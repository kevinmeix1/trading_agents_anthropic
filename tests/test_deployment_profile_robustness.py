from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_robustness import (
    evaluate_deployment_profile_robustness,
    write_deployment_profile_robustness_csv,
)
from quanthack.cli import deployment_profile_robustness
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileRobustnessTest(TestCase):
    def test_evaluates_cost_and_leave_one_symbol_stress(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=231,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            output = root / "robustness.csv"
            _write_profile_pack(profile_path)

            result = evaluate_deployment_profile_robustness(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="symbol_refined",
                slippage_multipliers=(2.0,),
            )
            write_deployment_profile_robustness_csv(result, output)
            csv_text = output.read_text(encoding="utf-8")

        self.assertEqual(result.baseline.scenario_type, "baseline")
        self.assertEqual(len(result.rows), 4)
        self.assertTrue(any(row.scenario == "slippage_2x" for row in result.rows))
        self.assertEqual(
            {row.excluded_symbol for row in result.rows if row.scenario_type == "leave_one_symbol"},
            {"EURUSD", "BTCUSD"},
        )
        self.assertIn("scenario_type,scenario,excluded_symbol", csv_text)
        self.assertIn("decision", csv_text)

    def test_robustness_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=232,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output = root / "robustness.csv"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_robustness.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--slot",
                        "symbol_refined",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--slippage-multiplier",
                        "2.0",
                        "--output",
                        str(output),
                    ]
                )

            printed = stdout.getvalue()
            csv_text = output.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Robustness", printed)
        self.assertIn("Baseline", printed)
        self.assertIn("slippage_2x", csv_text)


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
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.250 BTCUSD=0.375",
                "crypto_allowed_utc_hours": "all",
                "symbol_allowed_utc_hours": "BTCUSD=0|1|2|3|4|5|6|7|8",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
