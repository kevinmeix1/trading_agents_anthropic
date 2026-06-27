from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.asset_class_attribution import (
    build_asset_class_attribution_report,
    write_asset_class_attribution_csv,
)


class AssetClassAttributionTest(TestCase):
    def test_groups_portfolio_pnl_by_asset_class(self) -> None:
        with TemporaryDirectory() as tmpdir:
            pnl_path = Path(tmpdir) / "pnl.csv"
            output_path = Path(tmpdir) / "asset.csv"
            pnl_path.write_text(
                "\n".join(
                    (
                        "symbol,fills,realized_pnl_usd,open_pnl_usd,total_pnl_usd,"
                        "final_position_units,average_entry_price,final_mark_price",
                        "EURUSD,2,100.0,0.0,100.0,0,,1.1",
                        "XAUUSD,1,-25.0,0.0,-25.0,0,,2400",
                        "BTCUSD,4,50.0,5.0,55.0,0,,100000",
                        "PORTFOLIO,7,125.0,5.0,130.0,,,",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_asset_class_attribution_report(pnl_path)
            write_asset_class_attribution_csv(report, output_path)
            text = output_path.read_text(encoding="utf-8")

        by_class = {row.asset_class.value: row for row in report.asset_class_rows}
        self.assertAlmostEqual(report.portfolio_total_pnl_usd, 130.0)
        self.assertAlmostEqual(by_class["FOREX"].total_pnl_usd, 100.0)
        self.assertAlmostEqual(by_class["METAL"].total_pnl_usd, -25.0)
        self.assertAlmostEqual(by_class["CRYPTO"].share_of_portfolio_pnl, 55.0 / 130.0)
        self.assertAlmostEqual(by_class["CRYPTO"].share_of_gross_abs_pnl, 55.0 / 180.0)
        self.assertIn("asset_class,symbols,fills", text)
        self.assertIn("share_of_gross_abs_pnl", text)
        self.assertIn("CRYPTO,BTCUSD,4", text)
