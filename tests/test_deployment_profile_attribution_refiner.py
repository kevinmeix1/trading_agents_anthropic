from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_attribution_refiner import (
    refine_deployment_profile_from_attribution,
    write_deployment_profile_refinement_csv,
    write_refined_profile_pack_json,
)
from quanthack.cli import deployment_profile_attribution_refine
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class DeploymentProfileAttributionRefinerTest(TestCase):
    def test_refiner_scales_weak_symbols_and_writes_outputs(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=201,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            attribution_path = root / "attribution.csv"
            output_path = root / "refinement.csv"
            refined_pack_path = root / "refined_pack.json"
            _write_profile_pack(profile_path)
            _write_attribution(attribution_path)

            result = refine_deployment_profile_from_attribution(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                profile_pack_json=profile_path,
                slot="survival",
                attribution_csv=attribution_path,
                weak_symbol_scales=(1.0, 0.5, 0.0),
                include_walk_forward=True,
                train_size=12,
                test_size=8,
                step_size=8,
            )
            write_deployment_profile_refinement_csv(result, output_path)
            self.assertIsNotNone(result.best)
            write_refined_profile_pack_json(
                source_profile_pack_json=profile_path,
                result=result,
                candidate=result.best,
                output_json=refined_pack_path,
            )
            output_text = output_path.read_text(encoding="utf-8")
            pack = json.loads(refined_pack_path.read_text(encoding="utf-8"))

        self.assertEqual({row.symbol for row in result.weak_symbols}, {"BTCUSD"})
        self.assertEqual(len(result.candidates), 3)
        self.assertIn("rank,label,base_slot,weak_symbol_scale", output_text)
        self.assertIn("BTCUSD", output_text)
        self.assertEqual(pack["recommended_slot"], "refined")
        self.assertTrue(any(profile["slot"] == "refined" for profile in pack["profiles"]))

    def test_attribution_refine_cli_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD"),
            periods=44,
            interval_minutes=15,
            seed=202,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            recommendation_path = root / "recommendation.json"
            attribution_path = root / "attribution.csv"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            output_path = root / "refinement.csv"
            refined_pack_path = root / "refined_pack.json"
            _write_profile_pack(profile_path)
            _write_attribution(attribution_path)
            recommendation_path.write_text(
                json.dumps({"recommended_slot": "survival"}),
                encoding="utf-8",
            )
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_attribution_refine.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--profile-pack-json",
                        str(profile_path),
                        "--recommendation-json",
                        str(recommendation_path),
                        "--attribution-csv",
                        str(attribution_path),
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--weak-scale",
                        "1.0",
                        "--weak-scale",
                        "0.0",
                        "--train-size",
                        "12",
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

        self.assertIn("Deployment Profile Attribution Refinement", output)
        self.assertIn("Weak symbols: BTCUSD", output)
        self.assertIn("weak_symbol_scale", csv_text)
        self.assertIn('"recommended_slot": "refined"', pack_text)


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
                "risk_discipline_score": 100,
                "fold_contribution": 0.50,
                "promotion_status": "PROMOTE",
                "promotion_reason": "test",
                "strategy_map": "EURUSD=macd_momentum BTCUSD=crypto_mean_reversion",
                "multiplier_map": "EURUSD=0.500 BTCUSD=0.500",
                "crypto_allowed_utc_hours": "all",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_attribution(path: Path) -> None:
    path.write_text(
        "profile_slot,profile_label,symbol,primary_signal,utc_hour,side,"
        "fills,realized_events,wins,losses,win_rate,realized_pnl_usd,"
        "open_pnl_usd,total_pnl_usd,turnover_notional_usd,adjusted_notional_usd\n"
        "survival,demo,BTCUSD,strategy,0,BUY,2,1,0,1,0,-100,0,-100,1000,500\n"
        "survival,demo,EURUSD,macd_momentum,14,BUY,2,1,1,0,1,50,0,50,1000,500\n",
        encoding="utf-8",
    )
