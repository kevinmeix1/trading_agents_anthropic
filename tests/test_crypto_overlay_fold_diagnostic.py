from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.crypto_overlay_fold_diagnostic import (
    build_crypto_overlay_fold_diagnostic,
    write_crypto_overlay_fold_diagnostic_summary_csv,
    write_crypto_overlay_fold_symbol_summary_csv,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import CryptoOverlaySizingSpec
from quanthack.cli import crypto_overlay_fold_diagnostic
from quanthack.core.config import load_config
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)


class CryptoOverlayFoldDiagnosticTest(TestCase):
    def test_builds_fold_diagnostic_and_writes_artifacts(self) -> None:
        config = load_config("configs/competition.toml")
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=36,
            interval_minutes=15,
            seed=111,
        )

        with TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "diag"
            diagnostic = build_crypto_overlay_fold_diagnostic(
                config=config,
                prices=data.prices,
                quotes=data.quotes,
                symbols=("EURUSD", "BTCUSD", "SOLUSD"),
                spec=CryptoOverlaySizingSpec(
                    label="test",
                    crypto_multiplier=0.75,
                    btc_multiplier=0.5,
                    sol_multiplier=1.0,
                    crypto_allowed_utc_hours=(7, 8, 9),
                ),
                train_size=12,
                test_size=8,
                step_size=8,
                output_prefix=prefix,
            )
            summary_path = Path(tmpdir) / "summary.csv"
            write_crypto_overlay_fold_diagnostic_summary_csv(
                diagnostic,
                summary_path,
            )
            symbol_summary_path = Path(tmpdir) / "symbol_summary.csv"
            write_crypto_overlay_fold_symbol_summary_csv(
                diagnostic,
                symbol_summary_path,
            )

            folds_text = (Path(tmpdir) / "diag_folds.csv").read_text(encoding="utf-8")
            fills_text = (Path(tmpdir) / "diag_fills.csv").read_text(encoding="utf-8")
            attribution_text = (Path(tmpdir) / "diag_attribution.csv").read_text(
                encoding="utf-8"
            )
            summary_text = summary_path.read_text(encoding="utf-8")
            symbol_summary_text = symbol_summary_path.read_text(encoding="utf-8")

        self.assertEqual(diagnostic.spec.crypto_allowed_utc_hours, (7, 8, 9))
        self.assertGreaterEqual(len(diagnostic.walk_forward.folds), 1)
        self.assertIn("fold,train_start,train_end,test_start", folds_text)
        self.assertIn("timestamp,symbol,side,fill_price", fills_text)
        self.assertIn("fold,fold_return_pct,symbol", attribution_text)
        self.assertIn("largest_positive_fold_contribution", summary_text)
        self.assertIn("fold,fold_return_pct,symbol,asset_class", symbol_summary_text)

    def test_crypto_overlay_fold_diagnostic_cli_writes_outputs(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD", "BTCUSD", "SOLUSD"),
            periods=32,
            interval_minutes=15,
            seed=112,
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            prefix = root / "diag"
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                crypto_overlay_fold_diagnostic.main(
                    [
                        "--config",
                        "configs/competition.toml",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--symbol",
                        "EURUSD",
                        "--symbol",
                        "BTCUSD",
                        "--symbol",
                        "SOLUSD",
                        "--candidate",
                        "label=test,crypto=0.75,btc=0.5,sol=1.0,crypto_hours=7|8|9",
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
            summary_text = (root / "diag_summary.csv").read_text(encoding="utf-8")
            symbol_summary_text = (root / "diag_symbol_summary.csv").read_text(
                encoding="utf-8"
            )

        self.assertIn("Crypto Overlay Fold Diagnostic", output)
        self.assertIn("Crypto hours: 7|8|9", output)
        self.assertIn("test", summary_text)
        self.assertIn("symbol,asset_class", symbol_summary_text)
