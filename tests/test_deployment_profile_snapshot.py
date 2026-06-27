from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.cli import deployment_profile_snapshot
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)
from quanthack.trading.deployment_profile_snapshot import (
    build_deployment_profile_signal_snapshot,
    write_deployment_profile_signal_snapshot_csv,
)
from quanthack.trading.risk import AccountSnapshot


class DeploymentProfileSnapshotTest(TestCase):
    def test_build_signal_snapshot_and_write_csv(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=181,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            output_path = root / "snapshot.csv"
            _write_profile_pack(profile_path)
            as_of = data.prices.for_symbol("EURUSD").bars[24].timestamp

            snapshot = build_deployment_profile_signal_snapshot(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="conservative",
                account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
                as_of=as_of,
            )
            write_deployment_profile_signal_snapshot_csv(snapshot, output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertEqual(snapshot.profile.label, "demo_conservative")
        self.assertEqual(snapshot.timestamp, as_of.isoformat(timespec="seconds"))
        self.assertEqual({row.symbol for row in snapshot.rows}, {"EURUSD", "BTCUSD"})
        self.assertEqual(snapshot.allocation.equity, 1_000_000)
        self.assertIn("profile_slot,profile_label,timestamp,symbol", text)
        self.assertIn("demo_conservative", text)

    def test_deployment_profile_snapshot_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=182,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "snapshot.csv"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)
            as_of = data.prices.for_symbol("EURUSD").bars[30].timestamp

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_snapshot.main(
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
                        "--output",
                        str(output_path),
                        "--as-of",
                        as_of.isoformat(timespec="seconds"),
                    ]
                )

            output = stdout.getvalue()
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Signal Snapshot", output)
        self.assertIn("demo_conservative", output)
        self.assertIn("Estimated risk status", output)
        self.assertIn("demo_conservative", csv_text)


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
                "crypto_allowed_utc_hours": "all",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
