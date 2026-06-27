from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.research_candidate_gate import (
    ResearchCandidateSource,
    ResearchDataSource,
    ResearchReadiness,
    build_research_candidate_gate,
    write_research_candidate_gate_csv,
)
from quanthack.cli import research_candidate_gate


class ResearchCandidateGateTest(TestCase):
    def test_gates_official_proxy_and_rejected_evidence(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official = _write_comparison(root / "official.csv", "official_promote")
            proxy = _write_comparison(root / "proxy.csv", "proxy_promote")
            rejected = _write_comparison(
                root / "rejected.csv",
                "rejected_candidate",
                return_pct=-0.001,
                promotion_status="REJECT",
                promotion_reason="non-negative fold fraction below gate",
                wf_non_negative=0.5,
            )

            rows = build_research_candidate_gate(
                (
                    ResearchCandidateSource(
                        path=str(official),
                        data_source=ResearchDataSource.OFFICIAL,
                    ),
                    ResearchCandidateSource(
                        path=str(proxy),
                        data_source=ResearchDataSource.PROXY,
                    ),
                    ResearchCandidateSource(
                        path=str(rejected),
                        data_source=ResearchDataSource.OFFICIAL,
                    ),
                )
            )

        by_label = {row.label: row for row in rows}
        self.assertEqual(
            by_label["official_promote"].readiness,
            ResearchReadiness.LIVE_READY,
        )
        self.assertTrue(by_label["official_promote"].live_ready)
        self.assertEqual(
            by_label["proxy_promote"].readiness,
            ResearchReadiness.PAPER_ONLY,
        )
        self.assertIn("proxy data cannot be live-ready", by_label["proxy_promote"].reason)
        self.assertEqual(
            by_label["rejected_candidate"].readiness,
            ResearchReadiness.REJECT,
        )
        self.assertEqual(rows[0].label, "official_promote")

    def test_writes_gate_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            comparison = _write_comparison(root / "comparison.csv", "candidate")
            output = root / "gate.csv"

            rows = build_research_candidate_gate(
                (
                    ResearchCandidateSource(
                        path=str(comparison),
                        data_source=ResearchDataSource.MIXED_PROXY,
                    ),
                )
            )
            write_research_candidate_gate_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("rank,label,source_file,data_source,readiness", text)
        self.assertIn("crypto_allowed_utc_hours", text)
        self.assertIn("candidate", text)
        self.assertIn("mixed_proxy", text)

    def test_research_candidate_gate_cli_writes_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            comparison = _write_comparison(root / "comparison.csv", "candidate")
            output_path = root / "gate.csv"

            stdout = StringIO()
            with redirect_stdout(stdout):
                research_candidate_gate.main(
                    [
                        "--source",
                        f"path={comparison},data_source=mixed_proxy",
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("Research Candidate Gate", output)
        self.assertIn("PAPER_ONLY", output)
        self.assertIn("candidate", text)


def _write_comparison(
    path: Path,
    label: str,
    *,
    return_pct: float = 0.01,
    max_drawdown_pct: float = 0.01,
    sharpe_15m: float = 0.03,
    risk_score: float = 100.0,
    promotion_status: str = "PROMOTE",
    promotion_reason: str = "passed",
    wf_positive: float = 1.0,
    wf_active_positive: float = 1.0,
    wf_non_negative: float = 1.0,
    wf_median_active: float = 0.002,
    selection_score: float = 80.0,
    proxy_score: float = 75.0,
) -> Path:
    path.write_text(
        "label,return_pct,max_drawdown_pct,sharpe_15m,risk_discipline_score,"
        "promotion_status,promotion_reason,wf_positive_fold_fraction,"
        "wf_active_positive_fold_fraction,wf_non_negative_fold_fraction,"
        "wf_median_active_test_return_pct,selection_score,proxy_score,"
        "strategy_map,crypto_map\n"
        f"{label},{return_pct},{max_drawdown_pct},{sharpe_15m},{risk_score},"
        f"{promotion_status},{promotion_reason},{wf_positive},"
        f"{wf_active_positive},{wf_non_negative},{wf_median_active},"
        f"{selection_score},{proxy_score},EURUSD=macd,BTCUSD=macd\n",
        encoding="utf-8",
    )
    return path
