from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.asset_class_stability_optimizer import (
    AssetClassStabilitySpec,
)
from quanthack.backtesting.crypto_overlay_component_ablation import (
    CryptoOverlayComponentAblationSpec,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import CryptoOverlaySizingSpec
from quanthack.backtesting.crypto_promotion_pipeline import (
    run_crypto_promotion_pipeline,
)
from quanthack.backtesting.research_candidate_gate import (
    ResearchDataSource,
    ResearchReadiness,
)
from quanthack.cli import crypto_promotion_pipeline
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoPromotionPipelineTest(TestCase):
    def test_run_crypto_promotion_pipeline_writes_evidence_bundle(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
            periods=40,
            interval_minutes=15,
            seed=131,
        )

        with TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "crypto_pipeline"
            result = run_crypto_promotion_pipeline(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                price_csv="synthetic_prices.csv",
                quote_csv="synthetic_quotes.csv",
                data_source=ResearchDataSource.SYNTHETIC,
                output_prefix=prefix,
                symbols=("EURUSD", "XAUUSD", "BTCUSD", "SOLUSD"),
                sizing_specs=(
                    CryptoOverlaySizingSpec(
                        label="sized_london",
                        crypto_multiplier=0.75,
                        btc_multiplier=0.75,
                        sol_multiplier=1.0,
                        crypto_allowed_utc_hours=(7, 8, 9, 10),
                    ),
                ),
                component_specs=(
                    CryptoOverlayComponentAblationSpec(label="full"),
                    CryptoOverlayComponentAblationSpec(
                        label="no_crypto",
                        disabled_asset_classes=("CRYPTO",),
                    ),
                ),
                asset_class_specs=(
                    _asset_spec("asset_demo", fx=1.0, metal=0.75),
                ),
                train_size=12,
                test_size=8,
                step_size=8,
                max_gap_seconds=960,
            )

            summary_text = result.artifacts.summary_csv.read_text(encoding="utf-8")
            sizing_gate_text = result.artifacts.sizing_gate_csv.read_text(encoding="utf-8")

            self.assertTrue(result.artifacts.data_health_csv.exists())
            self.assertTrue(result.artifacts.sizing_csv.exists())
            self.assertTrue(result.artifacts.component_ablation_csv.exists())
            self.assertTrue(result.artifacts.asset_class_stability_csv.exists())
            self.assertTrue(result.artifacts.fold_diagnostic_summary_csv.exists())
            self.assertTrue(result.artifacts.fold_diagnostic_symbol_summary_csv.exists())

        self.assertEqual(result.summary.data_source, ResearchDataSource.SYNTHETIC)
        self.assertEqual(result.summary.best_sizing_label, "sized_london")
        self.assertFalse(result.summary.live_ready)
        self.assertIn(
            result.summary.promotion_readiness,
            {ResearchReadiness.PAPER_ONLY, ResearchReadiness.REJECT},
        )
        self.assertIn("data_source,price_csv,quote_csv", summary_text)
        self.assertIn("stable_backup_label", summary_text)
        self.assertIn("sized_london", summary_text)
        self.assertIn("synthetic", sizing_gate_text)

    def test_crypto_promotion_pipeline_cli_writes_summary(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=36,
            interval_minutes=15,
            seed=132,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            prefix = root / "pipeline"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_promotion_pipeline.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--data-source",
                        "synthetic",
                        "--symbol",
                        "EURUSD",
                        "--symbol",
                        "BTCUSD",
                        "--symbol",
                        "SOLUSD",
                        "--candidate",
                        "label=demo,crypto=0.75,btc=0.75,sol=1.0,crypto_hours=7|8|9",
                        "--component",
                        "label=full",
                        "--component",
                        "label=no_crypto,assets=CRYPTO",
                        "--asset-candidate",
                        (
                            "label=asset_demo,fx=1.0,metal=0.75,"
                            "crypto_spec=label=crypto,crypto=0.75,btc=0.75,sol=1.0"
                        ),
                        "--train-size",
                        "12",
                        "--test-size",
                        "8",
                        "--step-size",
                        "8",
                        "--output-prefix",
                        str(prefix),
                    ]
                )

            output = stdout.getvalue()
            summary_text = (root / "pipeline_summary.csv").read_text(encoding="utf-8")

        self.assertIn("Crypto Promotion Pipeline", output)
        self.assertIn("Best sizing: demo", output)
        self.assertIn("Stable backup:", output)
        self.assertIn("Decision:", output)
        self.assertIn("demo", summary_text)
        self.assertIn("promotion_readiness", summary_text)


def _asset_spec(label: str, *, fx: float, metal: float) -> AssetClassStabilitySpec:
    return AssetClassStabilitySpec(
        label=label,
        fx_multiplier=fx,
        metal_multiplier=metal,
        crypto_spec=CryptoOverlaySizingSpec(
            label="crypto",
            crypto_multiplier=0.75,
            btc_multiplier=0.75,
            sol_multiplier=1.0,
            crypto_allowed_utc_hours=(7, 8, 9, 10),
        ),
    )
