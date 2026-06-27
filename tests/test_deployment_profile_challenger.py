from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_challenger import (
    DeploymentProfileCandidateSpec,
    compare_deployment_profile_challengers,
    parse_candidate_spec,
    write_deployment_profile_challenger_csv,
)
from quanthack.cli import deployment_profile_challenger
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileChallengerTest(TestCase):
    def test_challenger_scorecard_compares_candidate_profiles(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=221,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_pack = root / "baseline.json"
            challenger_pack = root / "challenger.json"
            output = root / "challengers.csv"
            _write_profile_pack(
                baseline_pack,
                slot="survival",
                label="demo_survival",
                multiplier_map="EURUSD=0.250 BTCUSD=0.250",
            )
            _write_profile_pack(
                challenger_pack,
                slot="symbol_refined",
                label="demo_symbol_refined",
                multiplier_map="EURUSD=0.250 BTCUSD=0.375",
                symbol_hours="BTCUSD=0|1|2|3|4|5|6|7|8",
            )

            result = compare_deployment_profile_challengers(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                candidates=(
                    DeploymentProfileCandidateSpec(
                        label="baseline",
                        profile_pack_json=str(baseline_pack),
                        slot="survival",
                    ),
                    DeploymentProfileCandidateSpec(
                        label="symbol_refined",
                        profile_pack_json=str(challenger_pack),
                        slot="symbol_refined",
                    ),
                ),
                train_size=12,
                test_size=8,
                step_size=8,
            )
            write_deployment_profile_challenger_csv(result, output)
            csv_text = output.read_text(encoding="utf-8")

        self.assertEqual(len(result.rows), 2)
        self.assertEqual({row.label for row in result.rows}, {"baseline", "symbol_refined"})
        self.assertTrue(any(row.decision == "BASELINE" for row in result.rows))
        self.assertIn("rank,label,slot,profile_label", csv_text)
        self.assertIn("symbol_allowed_utc_hours", csv_text)

    def test_parse_candidate_spec_requires_three_fields(self) -> None:
        spec = parse_candidate_spec("label,path.json,slot")

        self.assertEqual(spec.label, "label")
        self.assertEqual(spec.profile_pack_json, "path.json")
        self.assertEqual(spec.slot, "slot")
        with self.assertRaisesRegex(ValueError, "LABEL"):
            parse_candidate_spec("bad")

    def test_challenger_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=222,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_pack = root / "baseline.json"
            challenger_pack = root / "challenger.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output = root / "challengers.csv"
            _write_profile_pack(
                baseline_pack,
                slot="survival",
                label="demo_survival",
                multiplier_map="EURUSD=0.250 BTCUSD=0.250",
            )
            _write_profile_pack(
                challenger_pack,
                slot="symbol_refined",
                label="demo_symbol_refined",
                multiplier_map="EURUSD=0.250 BTCUSD=0.375",
                symbol_hours="BTCUSD=0|1|2|3|4|5|6|7|8",
            )
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_challenger.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--candidate",
                        f"baseline,{baseline_pack},survival",
                        "--candidate",
                        f"symbol_refined,{challenger_pack},symbol_refined",
                        "--train-size",
                        "12",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--output",
                        str(output),
                    ]
                )

            printed = stdout.getvalue()
            csv_text = output.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Challenger Scorecard", printed)
        self.assertIn("Candidates: 2", printed)
        self.assertIn("rank_score", csv_text)


def _write_profile_pack(
    path: Path,
    *,
    slot: str,
    label: str,
    multiplier_map: str,
    symbol_hours: str = "",
) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": slot,
        "recommendation_reason": "unit test",
        "profiles": [
            {
                "slot": slot,
                "label": label,
                "evidence_status": "PAPER_ONLY",
                "use_case": "test profile",
                "reason": "test",
                "return_pct": 0.01,
                "max_drawdown_pct": 0.005,
                "sharpe_15m": 0.04,
                "fold_contribution": 0.75,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": multiplier_map,
                "crypto_allowed_utc_hours": "all",
                "symbol_allowed_utc_hours": symbol_hours,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
