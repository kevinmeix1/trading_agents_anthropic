from __future__ import annotations

import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.cli import mt5_ticket_sheet
from quanthack.trading.mt5_ticket_sheet import (
    Mt5ContractSpec,
    build_mt5_ticket_sheet_from_snapshot_csv,
    load_contract_specs,
    write_contract_spec_template_from_snapshot_csv,
    write_mt5_ticket_sheet_csv,
)


class Mt5TicketSheetTest(TestCase):
    def test_ticket_sheet_uses_fx_defaults_and_flags_missing_specs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot.csv"
            _write_snapshot(
                snapshot_path,
                rows=(
                    _snapshot_row("EURUSD", "BUY", 1.1000, 1.1002, 50_000),
                    _snapshot_row("SOLUSD", "SELL", 63.80, 63.86, -75_000),
                    _snapshot_row("EURGBP", "BUY", 0.8620, 0.8622, 100_000),
                ),
            )

            tickets = build_mt5_ticket_sheet_from_snapshot_csv(snapshot_path)

        by_symbol = {ticket.symbol: ticket for ticket in tickets}
        self.assertEqual(by_symbol["EURUSD"].status, "READY")
        self.assertEqual(by_symbol["EURUSD"].broker_symbol, "EURUSD")
        self.assertAlmostEqual(by_symbol["EURUSD"].rounded_lots, 0.45)
        self.assertEqual(by_symbol["SOLUSD"].status, "NEEDS_CONTRACT_SPEC")
        self.assertEqual(by_symbol["EURGBP"].status, "NEEDS_QUOTE_USD_RATE")

    def test_ticket_sheet_uses_contract_specs_for_crypto_and_broker_symbols(self) -> None:
        with TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot.csv"
            spec_path = Path(tmpdir) / "specs.csv"
            _write_snapshot(
                snapshot_path,
                rows=(_snapshot_row("SOLUSD", "SELL", 63.80, 63.86, -75_000),),
            )
            _write_specs(
                spec_path,
                rows=(
                    {
                        "symbol": "SOLUSD",
                        "broker_symbol": "SOLUSD.raw",
                        "contract_size": "1",
                        "volume_step": "0.01",
                        "min_volume": "0.01",
                        "quote_usd_rate": "",
                    },
                ),
            )

            specs = load_contract_specs(spec_path)
            tickets = build_mt5_ticket_sheet_from_snapshot_csv(
                snapshot_path,
                contract_specs=specs,
                broker_symbol_by_symbol={"SOLUSD": "SOLUSD.pro"},
            )

        (ticket,) = tickets
        self.assertEqual(ticket.status, "READY")
        self.assertEqual(ticket.broker_symbol, "SOLUSD.pro")
        self.assertAlmostEqual(ticket.rounded_lots, 1175.54)
        self.assertIn("Market Execution SELL", ticket.instruction)

    def test_writes_ticket_sheet_and_contract_spec_template(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot_path = root / "snapshot.csv"
            output_path = root / "tickets.csv"
            template_path = root / "template.csv"
            _write_snapshot(
                snapshot_path,
                rows=(
                    _snapshot_row("EURUSD", "BUY", 1.1000, 1.1002, 50_000),
                    _snapshot_row("SOLUSD", "SELL", 63.80, 63.86, -75_000),
                ),
            )

            tickets = build_mt5_ticket_sheet_from_snapshot_csv(snapshot_path)
            write_mt5_ticket_sheet_csv(tickets, output_path)
            write_contract_spec_template_from_snapshot_csv(
                snapshot_path,
                template_path,
            )
            ticket_text = output_path.read_text(encoding="utf-8")
            template_text = template_path.read_text(encoding="utf-8")

        self.assertIn("broker_symbol,side,status", ticket_text)
        self.assertIn("EURUSD,EURUSD,BUY,READY", ticket_text)
        self.assertIn("SOLUSD,SOLUSD,,0.01,0.01", template_text)

    def test_mt5_ticket_sheet_cli_writes_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot_path = root / "snapshot.csv"
            output_path = root / "tickets.csv"
            template_path = root / "template.csv"
            spec_path = root / "specs.csv"
            _write_snapshot(
                snapshot_path,
                rows=(_snapshot_row("SOLUSD", "BUY", 63.80, 63.86, 75_000),),
            )
            _write_specs(
                spec_path,
                rows=(
                    {
                        "symbol": "SOLUSD",
                        "broker_symbol": "SOLUSD.raw",
                        "contract_size": "1",
                        "volume_step": "0.01",
                        "min_volume": "0.01",
                        "quote_usd_rate": "",
                    },
                ),
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                mt5_ticket_sheet.main(
                    [
                        "--snapshot-csv",
                        str(snapshot_path),
                        "--contract-spec-csv",
                        str(spec_path),
                        "--broker-symbol-map",
                        "SOLUSD=SOLUSD.pro",
                        "--output",
                        str(output_path),
                        "--write-contract-spec-template",
                        str(template_path),
                    ]
                )

            output = stdout.getvalue()
            ticket_text = output_path.read_text(encoding="utf-8")
            template_text = template_path.read_text(encoding="utf-8")

            self.assertIn("MT5 Ticket Sheet", output)
            self.assertIn("READY: 1", output)
            self.assertIn("SOLUSD.pro", ticket_text)
            self.assertIn("SOLUSD,SOLUSD,,0.01,0.01", template_text)


def _snapshot_row(
    symbol: str,
    side: str,
    bid: float,
    ask: float,
    change_notional: float,
) -> dict[str, str | float]:
    target = change_notional
    return {
        "profile_slot": "conservative",
        "profile_label": "demo",
        "timestamp": "2026-06-10T08:15:00+00:00",
        "symbol": symbol,
        "strategy_name": "test",
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "current_notional_usd": 0,
        "raw_target_notional_usd": target,
        "scaled_target_notional_usd": target,
        "allocated_target_notional_usd": target,
        "change_notional_usd": change_notional,
        "order_side": side,
        "risk_approved": "True",
        "risk_adjusted_notional_usd": abs(target),
        "risk_reason": "approved",
        "primary_signal": "test",
        "strategy_reason": "unit test",
        "allocation_reasons": "",
        "supporting_signals": "",
        "conflicting_signals": "",
    }


def _write_snapshot(path: Path, *, rows: tuple[dict[str, str | float], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_specs(path: Path, *, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
