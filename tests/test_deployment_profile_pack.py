from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.cli import deployment_profile_pack
from quanthack.reporting.deployment_profile_pack import (
    build_deployment_profile_pack,
    write_deployment_profile_pack_csv,
    write_deployment_profile_pack_json,
)


class DeploymentProfilePackTest(TestCase):
    def test_build_deployment_profile_pack_selects_profile_slots(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.csv"
            stability = root / "stability.csv"
            _write_summary(summary, data_source="mixed_proxy")
            _write_stability(stability)

            pack = build_deployment_profile_pack(
                promotion_summary_csv=summary,
                asset_class_stability_csv=stability,
            )

        profiles = {profile.slot: profile for profile in pack.profiles}
        self.assertEqual(pack.recommended_slot, "paper_only")
        self.assertIn("mixed_proxy", pack.recommendation_reason)
        self.assertEqual(profiles["aggressive"].label, "current_london_full")
        self.assertEqual(profiles["conservative"].label, "current_asia_metal75")
        self.assertEqual(profiles["survival"].label, "current_asia_metal25")
        self.assertEqual(profiles["conservative"].evidence_status, "PAPER_ONLY")

    def test_official_stable_backup_can_be_recommended(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.csv"
            stability = root / "stability.csv"
            _write_summary(summary, data_source="official")
            _write_stability(stability)

            pack = build_deployment_profile_pack(
                promotion_summary_csv=summary,
                asset_class_stability_csv=stability,
            )

        self.assertEqual(pack.recommended_slot, "conservative")
        conservative = next(profile for profile in pack.profiles if profile.slot == "conservative")
        self.assertEqual(conservative.evidence_status, "LIVE_CANDIDATE")

    def test_write_deployment_profile_pack_csv_and_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.csv"
            stability = root / "stability.csv"
            output_csv = root / "pack.csv"
            output_json = root / "pack.json"
            _write_summary(summary, data_source="mixed_proxy")
            _write_stability(stability)
            pack = build_deployment_profile_pack(
                promotion_summary_csv=summary,
                asset_class_stability_csv=stability,
            )

            write_deployment_profile_pack_csv(pack, output_csv)
            write_deployment_profile_pack_json(pack, output_json)

            csv_text = output_csv.read_text(encoding="utf-8")
            json_text = output_json.read_text(encoding="utf-8")

        self.assertIn("slot,label,evidence_status", csv_text)
        self.assertIn("current_asia_metal75", csv_text)
        self.assertIn('"recommended_slot": "paper_only"', json_text)

    def test_deployment_profile_pack_cli_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.csv"
            stability = root / "stability.csv"
            output_csv = root / "pack.csv"
            output_json = root / "pack.json"
            _write_summary(summary, data_source="mixed_proxy")
            _write_stability(stability)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_pack.main(
                    [
                        "--promotion-summary-csv",
                        str(summary),
                        "--asset-class-stability-csv",
                        str(stability),
                        "--output-csv",
                        str(output_csv),
                        "--output-json",
                        str(output_json),
                    ]
                )

            output = stdout.getvalue()
            csv_text = output_csv.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Pack", output)
        self.assertIn("Recommended slot: paper_only", output)
        self.assertIn("conservative", csv_text)


def _write_summary(path: Path, *, data_source: str) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    "data_source,best_sizing_label,best_sizing_return_pct,"
                    "best_sizing_drawdown_pct,best_sizing_sharpe_15m,"
                    "best_sizing_risk_score,largest_positive_fold_contribution,"
                    "promotion_readiness,live_ready,promotion_reason"
                ),
                (
                    f"{data_source},primary,0.018,0.014,0.04,100,0.94,"
                    "PAPER_ONLY,False,primary fold concentration too high"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stability(path: Path) -> None:
    header = (
        "rank,label,stability_status,stability_score,return_retention,"
        "fx_multiplier,metal_multiplier,crypto_profile,crypto_allowed_utc_hours,"
        "strategy_map,multiplier_map,return_pct,max_drawdown_pct,sharpe_15m,"
        "risk_discipline_score,promotion_status,promotion_reason,"
        "wf_largest_positive_fold_contribution"
    )
    rows = (
        (
            "1,current_london_full,FRAGILE_PROFILE,76,1.0,1.0,1.0,current_london,"
            "7|8,s=m,m=one,0.018,0.014,0.04,100,PAPER_ONLY,fragile,0.94"
        ),
        (
            "2,current_asia_metal75,STABLE_PROFILE,69,0.41,1.0,0.75,current_asia,"
            "0|1,s=m,m=one,0.0076,0.0074,0.039,100,PROMOTE,stable,0.79"
        ),
        (
            "3,current_asia_metal25,STABLE_PROFILE,70,0.24,1.0,0.25,current_asia,"
            "0|1,s=m,m=one,0.0044,0.0046,0.037,100,PROMOTE,stable,0.65"
        ),
    )
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
