import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.clock import CompetitionMode
from quanthack.trading.execution import DryRunExecutor, portfolio_from_journal, read_journal
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskDecision,
    RiskState,
    Side,
    TradeRequest,
)


class DryRunExecutorTest(TestCase):
    def test_approved_decision_is_journaled_as_accepted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            executor = DryRunExecutor(journal)
            account = AccountSnapshot(equity=1_000_000, margin_level_pct=2_000)
            request = TradeRequest(
                symbol="EURUSD",
                side=Side.BUY,
                target_notional_usd=50_000,
                reason="test",
            )
            decision = RiskDecision(
                approved=True,
                reason="approved",
                adjusted_notional_usd=50_000,
                state=RiskState.NORMAL,
            )

            record = executor.submit(
                account=account,
                request=request,
                decision=decision,
                mode=CompetitionMode.QUALIFY,
            )

            self.assertEqual(record.status, "DRY_RUN_ACCEPTED")
            self.assertTrue(journal.exists())

            rows = journal.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            saved = json.loads(rows[0])
            self.assertEqual(saved["request"]["symbol"], "EURUSD")
            self.assertEqual(saved["decision"]["reason"], "approved")
            self.assertEqual(saved["portfolio_before"]["positions"], [])

    def test_blocked_decision_is_journaled_as_blocked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            executor = DryRunExecutor(journal)
            account = AccountSnapshot(equity=974_000, margin_level_pct=2_000)
            request = TradeRequest(
                symbol="EURUSD",
                side=Side.BUY,
                target_notional_usd=50_000,
                reason="test",
            )
            decision = RiskDecision(
                approved=False,
                reason="daily loss stop reached",
                adjusted_notional_usd=0,
                state=RiskState.FROZEN,
            )

            record = executor.submit(
                account=account,
                request=request,
                decision=decision,
                mode=CompetitionMode.QUALIFY,
            )

            self.assertEqual(record.status, "DRY_RUN_BLOCKED")

            saved_records = read_journal(journal)
            self.assertEqual(len(saved_records), 1)
            self.assertEqual(saved_records[0]["status"], "DRY_RUN_BLOCKED")

    def test_missing_journal_reads_as_empty_list(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.jsonl"

            self.assertEqual(read_journal(missing), [])

    def test_portfolio_from_journal_uses_latest_accepted_target(self) -> None:
        records = [
            _record("EURUSD", "BUY", 50_000, accepted=True),
            _record("XAUUSD", "BUY", 25_000, accepted=True),
            _record("EURUSD", "SELL", 75_000, accepted=True),
            _record("BTCUSD", "BUY", 10_000, accepted=False),
        ]

        portfolio = portfolio_from_journal(records)

        self.assertEqual(portfolio.notional_for_symbol("EURUSD"), -75_000)
        self.assertEqual(portfolio.notional_for_symbol("XAUUSD"), 25_000)
        self.assertEqual(portfolio.notional_for_symbol("BTCUSD"), 0)
        self.assertEqual(portfolio.gross_notional_usd, 100_000)

    def test_portfolio_from_journal_clears_symbol_on_accepted_zero_target(self) -> None:
        records = [
            _record("EURUSD", "BUY", 50_000, accepted=True),
            _record("EURUSD", "SELL", 0, accepted=True),
        ]

        portfolio = portfolio_from_journal(records)

        self.assertEqual(portfolio.notional_for_symbol("EURUSD"), 0)
        self.assertEqual(portfolio.gross_notional_usd, 0)

    def test_executor_current_portfolio_reads_its_journal(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            executor = DryRunExecutor(journal)
            account = AccountSnapshot(equity=1_000_000, margin_level_pct=2_000)
            request = TradeRequest(
                symbol="EURUSD",
                side=Side.BUY,
                target_notional_usd=50_000,
                reason="test",
            )
            decision = RiskDecision(
                approved=True,
                reason="approved",
                adjusted_notional_usd=50_000,
                state=RiskState.NORMAL,
            )

            executor.submit(
                account=account,
                request=request,
                decision=decision,
                mode=CompetitionMode.QUALIFY,
            )

            portfolio = executor.current_portfolio()

        self.assertEqual(portfolio.notional_for_symbol("EURUSD"), 50_000)

    def test_submit_records_portfolio_seen_by_risk(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            executor = DryRunExecutor(journal)
            account = AccountSnapshot(equity=1_000_000, margin_level_pct=2_000)
            portfolio = PortfolioSnapshot(
                positions=(Position(symbol="EURUSD", notional_usd=50_000),)
            )
            request = TradeRequest(
                symbol="XAUUSD",
                side=Side.BUY,
                target_notional_usd=25_000,
                reason="test",
            )
            decision = RiskDecision(
                approved=True,
                reason="approved",
                adjusted_notional_usd=25_000,
                state=RiskState.NORMAL,
            )

            executor.submit(
                account=account,
                request=request,
                decision=decision,
                mode=CompetitionMode.QUALIFY,
                portfolio_before=portfolio,
            )

            saved = read_journal(journal)[0]

        self.assertEqual(
            saved["portfolio_before"]["positions"][0]["notional_usd"],
            50_000,
        )


def _record(
    symbol: str,
    side: str,
    adjusted_notional_usd: float,
    *,
    accepted: bool,
) -> dict:
    return {
        "status": "DRY_RUN_ACCEPTED" if accepted else "DRY_RUN_BLOCKED",
        "request": {"symbol": symbol, "side": side},
        "decision": {
            "approved": accepted,
            "adjusted_notional_usd": adjusted_notional_usd,
        },
    }
