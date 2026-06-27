from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    RiskDisciplineSample,
    build_competition_metrics,
    build_risk_discipline_report,
)
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot


@dataclass(frozen=True)
class CompetitionMonitorSnapshot:
    timestamp: datetime
    account: AccountSnapshot
    portfolio: PortfolioSnapshot
    accepted_trade_count: int = 0

    @property
    def gross_notional_usd(self) -> float:
        return self.portfolio.gross_notional_usd

    @property
    def net_notional_usd(self) -> float:
        return sum(position.notional_usd for position in self.portfolio.positions)

    @property
    def largest_symbol_notional_usd(self) -> float:
        if not self.portfolio.positions:
            return 0.0
        return max(abs(position.notional_usd) for position in self.portfolio.positions)

    @property
    def leverage(self) -> float:
        return self.gross_notional_usd / self.account.equity

    @property
    def margin_usage(self) -> float:
        return self.leverage / 30.0

    @property
    def single_symbol_concentration(self) -> float:
        if self.gross_notional_usd == 0:
            return 0.0
        return self.largest_symbol_notional_usd / self.gross_notional_usd

    @property
    def net_directional_exposure(self) -> float:
        if self.gross_notional_usd == 0:
            return 0.0
        return abs(self.net_notional_usd) / self.gross_notional_usd

    def to_risk_sample(self) -> RiskDisciplineSample:
        return RiskDisciplineSample(
            timestamp=self.timestamp,
            equity=self.account.equity,
            gross_notional_usd=self.gross_notional_usd,
            net_notional_usd=self.net_notional_usd,
            largest_symbol_notional_usd=self.largest_symbol_notional_usd,
        )


@dataclass(frozen=True)
class _MonitorEquityPoint:
    timestamp: str
    equity: float


@dataclass(frozen=True)
class CompetitionMonitorReport:
    snapshots: tuple[CompetitionMonitorSnapshot, ...]
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport

    @property
    def latest(self) -> CompetitionMonitorSnapshot:
        if not self.snapshots:
            raise ValueError("competition monitor report has no snapshots")
        return self.snapshots[-1]


class CompetitionMonitor:
    def __init__(self) -> None:
        self._snapshots: list[CompetitionMonitorSnapshot] = []

    @property
    def snapshots(self) -> tuple[CompetitionMonitorSnapshot, ...]:
        return tuple(self._snapshots)

    def record(
        self,
        *,
        timestamp: datetime,
        account: AccountSnapshot,
        portfolio: PortfolioSnapshot,
        accepted_trade_count: int,
    ) -> CompetitionMonitorSnapshot:
        snapshot = CompetitionMonitorSnapshot(
            timestamp=timestamp,
            account=account,
            portfolio=portfolio,
            accepted_trade_count=accepted_trade_count,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def report(self) -> CompetitionMonitorReport:
        if not self._snapshots:
            raise ValueError("competition monitor has no snapshots")
        equity_points = tuple(
            _MonitorEquityPoint(
                timestamp=snapshot.timestamp.isoformat(timespec="seconds"),
                equity=snapshot.account.equity,
            )
            for snapshot in self._snapshots
        )
        latest_count = self._snapshots[-1].accepted_trade_count
        metrics = build_competition_metrics(
            equity_points=equity_points,
            fills=tuple(range(latest_count)),
        )
        discipline = build_risk_discipline_report(
            tuple(snapshot.to_risk_sample() for snapshot in self._snapshots)
        )
        return CompetitionMonitorReport(
            snapshots=self.snapshots,
            competition_metrics=metrics,
            risk_discipline=discipline,
        )


def write_monitor_csv(
    snapshots: tuple[CompetitionMonitorSnapshot, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "equity",
                "daily_pnl_pct",
                "drawdown_pct",
                "margin_level_pct",
                "gross_notional_usd",
                "net_notional_usd",
                "leverage",
                "margin_usage",
                "single_symbol_concentration",
                "net_directional_exposure",
                "accepted_trade_count",
            ],
        )
        writer.writeheader()
        for snapshot in snapshots:
            writer.writerow(
                {
                    "timestamp": snapshot.timestamp.isoformat(timespec="seconds"),
                    "equity": snapshot.account.equity,
                    "daily_pnl_pct": snapshot.account.daily_pnl_pct,
                    "drawdown_pct": snapshot.account.drawdown_pct,
                    "margin_level_pct": snapshot.account.margin_level_pct,
                    "gross_notional_usd": snapshot.gross_notional_usd,
                    "net_notional_usd": snapshot.net_notional_usd,
                    "leverage": snapshot.leverage,
                    "margin_usage": snapshot.margin_usage,
                    "single_symbol_concentration": snapshot.single_symbol_concentration,
                    "net_directional_exposure": snapshot.net_directional_exposure,
                    "accepted_trade_count": snapshot.accepted_trade_count,
                }
            )
