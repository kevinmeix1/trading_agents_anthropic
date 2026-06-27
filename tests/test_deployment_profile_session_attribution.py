from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.backtest import BacktestFill
from quanthack.cli import deployment_profile_session_attribution
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)
from quanthack.reporting.deployment_profile_session_attribution import (
    build_session_attribution_from_fills,
    write_deployment_profile_session_attribution_csv,
    DeploymentProfileSessionAttributionReport,
)
from quanthack.trading.risk import Side


class DeploymentProfileSessionAttributionTest(TestCase):
    def test_session_attribution_allocates_realized_and_open_pnl_to_entry_hour(self) -> None:
        fills = (
            BacktestFill(
                timestamp="2026-06-01T00:00:00+00:00",
                symbol="BTCUSD",
                side=Side.BUY,
                fill_price=100.0,
                trade_units=10.0,
                requested_notional_usd=1_000.0,
                adjusted_notional_usd=1_000.0,
                risk_reason="approved",
                primary_signal="entry_signal",
            ),
            BacktestFill(
                timestamp="2026-06-01T02:00:00+00:00",
                symbol="BTCUSD",
                side=Side.SELL,
                fill_price=110.0,
                trade_units=-4.0,
                requested_notional_usd=440.0,
                adjusted_notional_usd=440.0,
                risk_reason="approved",
                primary_signal="exit_signal",
            ),
        )

        rows = build_session_attribution_from_fills(
            fills=fills,
            final_mark_by_symbol={"BTCUSD": 120.0},
            profile_slot="demo",
            profile_label="Demo",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.utc_hour, 0)
        self.assertEqual(row.side, "BUY")
        self.assertEqual(row.fills, 2)
        self.assertEqual(row.realized_events, 1)
        self.assertAlmostEqual(row.realized_pnl_usd, 40.0)
        self.assertAlmostEqual(row.open_pnl_usd, 120.0)
        self.assertAlmostEqual(row.total_pnl_usd, 160.0)

    def test_write_session_attribution_csv(self) -> None:
        rows = build_session_attribution_from_fills(
            fills=(),
            final_mark_by_symbol={},
            profile_slot="demo",
            profile_label="Demo",
        )
        report = DeploymentProfileSessionAttributionReport(
            profile_slot="demo",
            profile_label="Demo",
            rows=rows,
        )

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "attribution.csv"
            write_deployment_profile_session_attribution_csv(report, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("profile_slot,profile_label,symbol", text)
        self.assertIn("total_pnl_usd", text)

    def test_session_attribution_cli_writes_output(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=48,
            interval_minutes=15,
            seed=193,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            recommendation_path = root / "recommendation.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "attribution.csv"
            _write_profile_pack(profile_path)
            recommendation_path.write_text(
                json.dumps({"recommended_slot": "survival"}),
                encoding="utf-8",
            )
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_session_attribution.main(
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
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            csv_text = output_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Session Attribution", output)
        self.assertIn("Slot: survival", output)
        self.assertIn("profile_slot,profile_label,symbol", csv_text)


def _write_profile_pack(path: Path) -> None:
    payload = {
        "data_source": "synthetic",
        "recommended_slot": "survival",
        "recommendation_reason": "unit test",
        "profiles": [
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
