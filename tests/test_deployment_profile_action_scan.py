from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.cli import deployment_profile_action_scan
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)
from quanthack.reporting.deployment_profile_action_scan import (
    scan_deployment_profile_actions,
    write_deployment_profile_action_events_csv,
    write_deployment_profile_action_hours_csv,
    write_deployment_profile_action_scan_summary_csv,
)
from quanthack.trading.risk import AccountSnapshot


class DeploymentProfileActionScanTest(TestCase):
    def test_action_scan_writes_summary_events_and_hours(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=191,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            summary_path = root / "summary.csv"
            events_path = root / "events.csv"
            hours_path = root / "hours.csv"
            _write_profile_pack(profile_path)

            result = scan_deployment_profile_actions(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="conservative",
                account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
                max_timestamps=24,
            )
            write_deployment_profile_action_scan_summary_csv(result, summary_path)
            write_deployment_profile_action_events_csv(result, events_path)
            write_deployment_profile_action_hours_csv(result, hours_path)
            summary_text = summary_path.read_text(encoding="utf-8")
            events_text = events_path.read_text(encoding="utf-8")
            hours_text = hours_path.read_text(encoding="utf-8")

        self.assertEqual(result.profile_slot, "conservative")
        self.assertLessEqual(result.scanned_timestamps, 24)
        self.assertIn("profile_slot,profile_label,stateful", summary_text)
        self.assertIn("action_rows", summary_text)
        self.assertIn("timestamp,profile_slot,profile_label,symbol", events_text)
        self.assertIn("hour_utc,action_rows,approved_actions", hours_text)

    def test_action_scan_cli_uses_recommendation_slot_and_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=192,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            recommendation_path = root / "recommendation.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            summary_path = root / "summary.csv"
            events_path = root / "events.csv"
            hours_path = root / "hours.csv"
            _write_profile_pack(profile_path)
            recommendation_path.write_text(
                json.dumps({"recommended_slot": "survival"}),
                encoding="utf-8",
            )
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_action_scan.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--recommendation-json",
                        str(recommendation_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--max-timestamps",
                        "24",
                        "--summary-output",
                        str(summary_path),
                        "--events-output",
                        str(events_path),
                        "--hours-output",
                        str(hours_path),
                    ]
                )

            output = stdout.getvalue()
            summary_text = summary_path.read_text(encoding="utf-8")
            events_text = events_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Action Scan", output)
        self.assertIn("Slot: survival", output)
        self.assertIn("survival", summary_text)
        self.assertIn("timestamp,profile_slot,profile_label,symbol", events_text)


def _write_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": "conservative",
        "recommendation_reason": "unit test",
        "profiles": [
            {
                "slot": "conservative",
                "label": "demo_conservative",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test conservative profile",
                "reason": "balanced",
                "return_pct": 0.01,
                "max_drawdown_pct": 0.004,
                "sharpe_15m": 0.05,
                "fold_contribution": 0.65,
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.500 BTCUSD=0.500",
                "crypto_allowed_utc_hours": "all",
            },
            {
                "slot": "survival",
                "label": "demo_survival",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test survival profile",
                "reason": "smaller",
                "return_pct": 0.005,
                "max_drawdown_pct": 0.002,
                "sharpe_15m": 0.03,
                "fold_contribution": 0.50,
                "strategy_map": "EURUSD=simple_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.250 BTCUSD=0.250",
                "crypto_allowed_utc_hours": "all",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
