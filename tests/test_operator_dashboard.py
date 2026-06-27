from __future__ import annotations

import csv
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.cli import operator_dashboard
from quanthack.reporting.operator_dashboard import build_operator_dashboard


class OperatorDashboardTest(TestCase):
    def test_build_operator_dashboard_from_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_artifacts(root)

            dashboard = build_operator_dashboard(
                profile_pack_json=paths["profile"],
                snapshot_csv=paths["snapshot"],
                allocation_csv=paths["allocation"],
                monitor_csv=paths["monitor"],
                ticket_csv=paths["tickets"],
            )

        self.assertIn("QuanHack Live Operator Dashboard", dashboard.html)
        self.assertIn("conservative: demo", dashboard.html)
        self.assertIn("NEEDS_CONTRACT_SPEC", dashboard.html)
        self.assertIn("Fill MT5 contract specs", dashboard.html)
        self.assertIn("Profile Snapshot", dashboard.html)
        self.assertIn("Live Monitor", dashboard.html)

    def test_operator_dashboard_marks_missing_sources(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_artifacts(root)

            dashboard = build_operator_dashboard(
                profile_pack_json=paths["profile"],
                snapshot_csv=paths["snapshot"],
                allocation_csv=paths["allocation"],
                monitor_csv=paths["monitor"],
                ticket_csv=root / "missing_tickets.csv",
            )

        self.assertIn("one or more dashboard sources are missing", dashboard.html)
        self.assertIn("file not found", dashboard.html)
        self.assertIn("MISSING", dashboard.html)

    def test_operator_dashboard_cli_writes_html(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_artifacts(root)
            output = root / "operator.html"

            stdout = StringIO()
            with redirect_stdout(stdout):
                operator_dashboard.main(
                    [
                        "--profile-pack-json",
                        str(paths["profile"]),
                        "--snapshot-csv",
                        str(paths["snapshot"]),
                        "--allocation-csv",
                        str(paths["allocation"]),
                        "--monitor-csv",
                        str(paths["monitor"]),
                        "--ticket-csv",
                        str(paths["tickets"]),
                        "--output",
                        str(output),
                    ]
                )
            text = output.read_text(encoding="utf-8")

        self.assertIn("Dashboard:", stdout.getvalue())
        self.assertIn("Operating View", text)
        self.assertIn("MT5 Ticket Sheet", text)


def _write_artifacts(root: Path) -> dict[str, Path]:
    profile = root / "pack.json"
    snapshot = root / "snapshot.csv"
    allocation = root / "allocation.csv"
    monitor = root / "monitor.csv"
    tickets = root / "tickets.csv"
    profile.write_text(
        json.dumps(
            {
                "recommended_slot": "paper_only",
                "recommendation_reason": "mixed data",
                "profiles": [
                    {
                        "slot": "conservative",
                        "label": "demo",
                        "evidence_status": "PAPER_ONLY",
                        "return_pct": 0.01,
                        "max_drawdown_pct": 0.004,
                        "fold_contribution": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        snapshot,
        rows=(
            {
                "profile_slot": "conservative",
                "profile_label": "demo",
                "timestamp": "2026-06-10T08:15:00+00:00",
                "symbol": "SOLUSD",
                "order_side": "SELL",
                "change_notional_usd": "-75000",
                "allocated_target_notional_usd": "-75000",
                "risk_approved": "True",
                "strategy_reason": "test signal",
            },
        ),
    )
    _write_csv(
        allocation,
        rows=(
            {
                "timestamp": "2026-06-10T08:15:00+00:00",
                "requested_gross_notional_usd": "75000",
                "adjusted_gross_notional_usd": "75000",
                "net_directional_exposure": "1.0",
                "largest_symbol_concentration": "1.0",
                "active_symbols": "1",
                "estimated_risk_status": "WARN",
                "trim_reasons": "single symbol",
            },
        ),
    )
    _write_csv(
        monitor,
        rows=(
            {
                "timestamp": "2026-06-10T08:15:00+00:00",
                "equity": "1000000",
                "daily_pnl_pct": "0",
                "drawdown_pct": "0",
                "margin_level_pct": "2000",
                "gross_notional_usd": "0",
                "net_notional_usd": "0",
                "leverage": "0",
                "single_symbol_concentration": "0",
                "accepted_trade_count": "0",
            },
        ),
    )
    _write_csv(
        tickets,
        rows=(
            {
                "profile_slot": "conservative",
                "profile_label": "demo",
                "timestamp": "2026-06-10T08:15:00+00:00",
                "symbol": "SOLUSD",
                "broker_symbol": "SOLUSD",
                "side": "SELL",
                "status": "NEEDS_CONTRACT_SPEC",
                "action_notional_usd": "75000",
                "rounded_lots": "0",
                "instruction": "add contract_size",
            },
        ),
    )
    return {
        "profile": profile,
        "snapshot": snapshot,
        "allocation": allocation,
        "monitor": monitor,
        "tickets": tickets,
    }


def _write_csv(path: Path, *, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
