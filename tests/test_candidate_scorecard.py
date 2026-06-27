from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.candidate_scorecard import (
    CandidateBundle,
    build_candidate_scorecard,
    write_candidate_scorecard_csv,
)


class CandidateScorecardTest(TestCase):
    def test_ranks_portfolio_output_bundles(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strong = _write_candidate(
                root,
                "strong",
                equities=(1_000_000, 1_002_000, 1_004_000),
                fills=4,
                pnl=4_000,
            )
            weak = _write_candidate(
                root,
                "weak",
                equities=(1_000_000, 999_000, 1_000_500),
                fills=1,
                pnl=500,
            )
            output = root / "scorecard.csv"

            rows = build_candidate_scorecard((strong, weak))
            write_candidate_scorecard_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(rows[0].label, "strong")
        self.assertGreater(rows[0].composite_score, rows[1].composite_score)
        self.assertEqual(rows[0].trade_count, 4)
        self.assertIn("rank,label,return_pct", text)
        self.assertIn("strong", text)


def _write_candidate(
    root: Path,
    label: str,
    *,
    equities: tuple[int, ...],
    fills: int,
    pnl: float,
) -> CandidateBundle:
    equity_path = root / f"{label}_equity.csv"
    fills_path = root / f"{label}_fills.csv"
    pnl_path = root / f"{label}_pnl.csv"
    equity_rows = ["timestamp,equity,cash,gross_notional_usd,net_notional_usd,drawdown_pct,positions"]
    for index, equity in enumerate(equities):
        equity_rows.append(
            f"2026-06-01T00:{index * 15:02d}:00+00:00,{equity},{equity},0,0,0,"
        )
    equity_path.write_text("\n".join(equity_rows) + "\n", encoding="utf-8")
    fill_rows = [
        "timestamp,symbol,side,fill_price,trade_units,turnover_notional_usd,"
        "requested_notional_usd,adjusted_notional_usd,risk_reason,primary_signal,"
        "supporting_signals,conflicting_signals"
    ]
    for index in range(fills):
        fill_rows.append(
            f"2026-06-01T00:{index:02d}:00+00:00,EURUSD,BUY,1.1,1000,1100,1100,1100,"
            "approved,unit,,"
        )
    fills_path.write_text("\n".join(fill_rows) + "\n", encoding="utf-8")
    pnl_path.write_text(
        "symbol,fills,realized_pnl_usd,open_pnl_usd,total_pnl_usd,"
        "final_position_units,average_entry_price,final_mark_price\n"
        f"PORTFOLIO,{fills},{pnl},0.0,{pnl},,,\n",
        encoding="utf-8",
    )
    return CandidateBundle(
        label=label,
        equity_csv=str(equity_path),
        fills_csv=str(fills_path),
        pnl_csv=str(pnl_path),
    )
