from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from quanthack.core.clock import CompetitionMode, UTC
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskDecision,
    Side,
    TradeRequest,
)


@dataclass(frozen=True)
class ExecutionRecord:
    record_id: str
    created_at_utc: datetime
    mode: CompetitionMode
    account: AccountSnapshot
    request: TradeRequest
    decision: RiskDecision
    status: str
    portfolio_before: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)
    platform: str = "dry_run"

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["created_at_utc"] = self.created_at_utc.isoformat(timespec="seconds")
        return data


@dataclass(frozen=True)
class DryRunExecutor:
    journal_path: Path = field(default_factory=lambda: Path("outputs/dry_run_journal.jsonl"))

    def submit(
        self,
        *,
        account: AccountSnapshot,
        request: TradeRequest,
        decision: RiskDecision,
        mode: CompetitionMode,
        portfolio_before: PortfolioSnapshot | None = None,
    ) -> ExecutionRecord:
        status = "DRY_RUN_ACCEPTED" if decision.approved else "DRY_RUN_BLOCKED"
        portfolio = portfolio_before or self.current_portfolio()
        record = ExecutionRecord(
            record_id=f"dryrun-{uuid4().hex}",
            created_at_utc=datetime.now(tz=UTC),
            mode=mode,
            account=account,
            request=request,
            decision=decision,
            status=status,
            portfolio_before=portfolio,
        )
        self.append(record)
        return record

    def current_portfolio(self) -> PortfolioSnapshot:
        return portfolio_from_journal(read_journal(self.journal_path))

    def append(self, record: ExecutionRecord) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")


def read_journal(path: str | Path = "outputs/dry_run_journal.jsonl") -> list[dict]:
    journal_path = Path(path)
    if not journal_path.exists():
        return []

    records: list[dict] = []
    with journal_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def portfolio_from_journal(records: list[dict]) -> PortfolioSnapshot:
    signed_notional_by_symbol: dict[str, float] = {}

    for record in records:
        if record.get("status") != "DRY_RUN_ACCEPTED":
            continue

        request = record.get("request") or {}
        decision = record.get("decision") or {}
        if not decision.get("approved", False):
            continue

        symbol = request.get("symbol")
        side = request.get("side")
        adjusted_notional = _adjusted_notional(decision)
        if not symbol:
            continue
        if adjusted_notional <= 0:
            signed_notional_by_symbol.pop(str(symbol), None)
            continue

        signed_notional_by_symbol[str(symbol)] = _signed_notional(
            side=str(side),
            notional_usd=adjusted_notional,
        )

    positions = tuple(
        Position(symbol=symbol, notional_usd=notional)
        for symbol, notional in sorted(signed_notional_by_symbol.items())
        if abs(notional) > 0
    )
    return PortfolioSnapshot(positions=positions)


def _signed_notional(*, side: str, notional_usd: float) -> float:
    if side == Side.SELL.value:
        return -notional_usd
    return notional_usd


def _adjusted_notional(decision: dict) -> float:
    try:
        return float(decision.get("adjusted_notional_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0
