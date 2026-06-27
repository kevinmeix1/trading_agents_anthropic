from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_backtest import load_deployment_profile
from quanthack.backtesting.deployment_profile_session_gate_refiner import (
    refine_deployment_profile_session_gates,
    write_deployment_profile_session_gate_refinement_csv,
    write_session_gated_profile_pack_json,
)
from quanthack.cli import deployment_profile_session_gate_refine
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileSessionGateRefinerTest(TestCase):
    def test_refiner_builds_candidates_from_weak_asset_hours(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=201,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            attribution_path = root / "attribution.csv"
            output_path = root / "session_refine.csv"
            pack_output = root / "session_pack.json"
            _write_profile_pack(profile_path)
            _write_attribution_csv(attribution_path)

            result = refine_deployment_profile_session_gates(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="refined",
                attribution_csv=attribution_path,
                max_dropped_hours=2,
                include_walk_forward=False,
            )
            write_deployment_profile_session_gate_refinement_csv(result, output_path)
            two_drop = next(
                candidate
                for candidate in result.candidates
                if len(candidate.dropped_asset_hours) == 2
            )
            write_session_gated_profile_pack_json(
                source_profile_pack_json=profile_path,
                result=result,
                candidate=two_drop,
                output_json=pack_output,
            )
            refined = load_deployment_profile(
                profile_pack_json=pack_output,
                slot="session_refined",
            )
            csv_text = output_path.read_text(encoding="utf-8")
            json_text = pack_output.read_text(encoding="utf-8")

        self.assertEqual(len(result.weak_asset_hours), 3)
        self.assertGreaterEqual(len(result.candidates), 3)
        self.assertNotIn(13, refined.forex_allowed_utc_hours or ())
        self.assertNotIn(13, refined.metal_allowed_utc_hours or ())
        self.assertIn("dropped_asset_hours", csv_text)
        self.assertIn('"recommended_slot": "session_refined"', json_text)

    def test_session_gate_refiner_cli_writes_artifacts(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=202,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            attribution_path = root / "attribution.csv"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "session_refine.csv"
            pack_output = root / "session_pack.json"
            _write_profile_pack(profile_path)
            _write_attribution_csv(attribution_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_session_gate_refine.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--slot",
                        "refined",
                        "--attribution-csv",
                        str(attribution_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--max-dropped-hours",
                        "1",
                        "--no-walk-forward",
                        "--output",
                        str(output_path),
                        "--refined-pack-json",
                        str(pack_output),
                    ]
                )

            output = stdout.getvalue()
            csv_text = output_path.read_text(encoding="utf-8")
            json_text = pack_output.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Session-Gate Refinement", output)
        self.assertIn("Weak asset hours", output)
        self.assertIn("session_refined", json_text)
        self.assertIn("dropped_asset_hours", csv_text)


def _write_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": "refined",
        "recommendation_reason": "unit test",
        "profiles": [
            {
                "slot": "refined",
                "label": "demo_refined",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test refined profile",
                "reason": "test",
                "return_pct": 0.01,
                "max_drawdown_pct": 0.005,
                "sharpe_15m": 0.04,
                "fold_contribution": 0.75,
                "strategy_map": (
                    "EURUSD=macd_momentum XAUUSD=macd_momentum "
                    "BTCUSD=crypto_mean_reversion"
                ),
                "multiplier_map": "EURUSD=0.500 XAUUSD=0.250 BTCUSD=0.375",
                "crypto_allowed_utc_hours": "all",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_attribution_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "symbol,utc_hour,fills,total_pnl_usd",
                "EURUSD,13,2,-100",
                "XAUUSD,13,2,-50",
                "BTCUSD,0,2,-20",
                "EURUSD,14,2,200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
