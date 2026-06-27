from __future__ import annotations

import csv
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.deployment_profile_backtest import load_deployment_profile
from quanthack.cli import deployment_profile_symbol_universe_refine
from quanthack.reporting.symbol_universe_profile_pack import (
    build_symbol_universe_profile_pack,
    write_symbol_universe_profile_pack_csv,
    write_symbol_universe_profile_pack_json,
)


class SymbolUniverseProfilePackTest(TestCase):
    def test_builds_executable_profile_pack_from_ranked_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eligibility_path = root / "eligibility.csv"
            json_path = root / "pack.json"
            csv_path = root / "pack.csv"
            _write_symbol_eligibility_csv(eligibility_path)

            pack = build_symbol_universe_profile_pack(
                symbol_eligibility_csv=eligibility_path,
                candidate="rank:1",
                selected_slot="macd_defensive",
                selected_label="macd_top_two",
                data_source="official_research",
            )
            write_symbol_universe_profile_pack_json(pack, json_path)
            write_symbol_universe_profile_pack_csv(pack, csv_path)
            loaded = load_deployment_profile(
                profile_pack_json=json_path,
                slot="macd_defensive",
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertEqual(pack.recommended_slot, "macd_defensive")
        self.assertEqual(loaded.strategy_by_symbol, (("EURUSD", "macd_momentum"), ("XAUUSD", "macd_momentum")))
        self.assertEqual(loaded.multipliers_by_symbol, (("EURUSD", 1.0), ("XAUUSD", 1.0)))
        self.assertEqual(payload["profiles"][1]["source_candidate"], "top_2_pnl")
        self.assertIn("excluded_symbols", csv_text)
        self.assertIn("GBPUSD", csv_text)

    def test_cli_writes_symbol_universe_pack(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eligibility_path = root / "eligibility.csv"
            json_path = root / "pack.json"
            csv_path = root / "pack.csv"
            _write_symbol_eligibility_csv(eligibility_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                deployment_profile_symbol_universe_refine.main(
                    [
                        "--symbol-eligibility-csv",
                        str(eligibility_path),
                        "--candidate",
                        "top_2_pnl",
                        "--selected-slot",
                        "macd_defensive",
                        "--output-json",
                        str(json_path),
                        "--output-csv",
                        str(csv_path),
                    ]
            )
            output = stdout.getvalue()
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertIn("Deployment Profile Symbol Universe Refinement", output)
        self.assertEqual(payload["recommended_slot"], "macd_defensive")
        self.assertIn("macd_defensive", csv_text)


def _write_symbol_eligibility_csv(path: Path) -> None:
    fieldnames = [
        "rank",
        "candidate",
        "strategy",
        "symbols",
        "excluded_symbols",
        "reason",
        "risk_discipline_score",
        "official_return_pct",
        "official_max_drawdown_pct",
        "official_15m_sharpe",
        "wf_largest_positive_fold_contribution",
    ]
    rows = [
        {
            "rank": "1",
            "candidate": "top_2_pnl",
            "strategy": "macd_momentum",
            "symbols": "EURUSD XAUUSD",
            "excluded_symbols": "GBPUSD",
            "reason": "top symbols by attribution P&L",
            "risk_discipline_score": "100",
            "official_return_pct": "0.045",
            "official_max_drawdown_pct": "0.008",
            "official_15m_sharpe": "0.04",
            "wf_largest_positive_fold_contribution": "0.47",
        },
        {
            "rank": "2",
            "candidate": "all_symbols",
            "strategy": "macd_momentum",
            "symbols": "EURUSD GBPUSD XAUUSD",
            "excluded_symbols": "",
            "reason": "baseline",
            "risk_discipline_score": "100",
            "official_return_pct": "0.060",
            "official_max_drawdown_pct": "0.009",
            "official_15m_sharpe": "0.038",
            "wf_largest_positive_fold_contribution": "0.50",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
