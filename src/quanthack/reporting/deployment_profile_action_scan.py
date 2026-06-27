from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quanthack.core.clock import CompetitionMode
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.trading.deployment_profile_snapshot import (
    DeploymentProfileSignalSnapshot,
    build_deployment_profile_signal_snapshot,
)
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot, Position


@dataclass(frozen=True)
class DeploymentProfileActionScanEvent:
    timestamp: str
    profile_slot: str
    profile_label: str
    symbol: str
    strategy_name: str
    order_side: str
    change_notional_usd: float
    allocated_target_notional_usd: float
    risk_approved: bool
    risk_adjusted_notional_usd: float
    risk_reason: str
    primary_signal: str
    strategy_reason: str
    allocation_status: str
    requested_gross_notional_usd: float
    adjusted_gross_notional_usd: float
    net_directional_exposure: float
    largest_symbol_concentration: float


@dataclass(frozen=True)
class DeploymentProfileActionHour:
    hour_utc: int
    action_rows: int
    approved_actions: int
    buy_actions: int
    sell_actions: int
    unique_symbols: tuple[str, ...]


@dataclass(frozen=True)
class DeploymentProfileActionScanResult:
    profile_slot: str
    profile_label: str
    stateful: bool
    scanned_timestamps: int
    first_timestamp: str
    last_timestamp: str
    events: tuple[DeploymentProfileActionScanEvent, ...]
    hourly: tuple[DeploymentProfileActionHour, ...]

    @property
    def actionable_timestamps(self) -> int:
        return len({event.timestamp for event in self.events})

    @property
    def action_rows(self) -> int:
        return len(self.events)

    @property
    def approved_actions(self) -> int:
        return len([event for event in self.events if event.risk_approved])

    @property
    def blocked_actions(self) -> int:
        return self.action_rows - self.approved_actions

    @property
    def buy_actions(self) -> int:
        return len([event for event in self.events if event.order_side == "BUY"])

    @property
    def sell_actions(self) -> int:
        return len([event for event in self.events if event.order_side == "SELL"])

    @property
    def unique_action_symbols(self) -> tuple[str, ...]:
        return tuple(sorted({event.symbol for event in self.events}))

    @property
    def first_action_timestamp(self) -> str:
        if not self.events:
            return ""
        return self.events[0].timestamp

    @property
    def last_action_timestamp(self) -> str:
        if not self.events:
            return ""
        return self.events[-1].timestamp

    @property
    def most_active_symbol(self) -> str:
        if not self.events:
            return ""
        return Counter(event.symbol for event in self.events).most_common(1)[0][0]

    @property
    def approved_action_rate(self) -> float:
        if not self.events:
            return 0.0
        return self.approved_actions / len(self.events)


def scan_deployment_profile_actions(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    account: AccountSnapshot,
    mode: CompetitionMode = CompetitionMode.QUALIFY,
    start: datetime | None = None,
    end: datetime | None = None,
    stride: int = 1,
    max_timestamps: int | None = 500,
    stateful: bool = True,
) -> DeploymentProfileActionScanResult:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_timestamps is not None and max_timestamps < 1:
        raise ValueError("max_timestamps must be at least 1 when provided")
    if start is not None and start.tzinfo is None:
        raise ValueError("start timestamp must include a timezone")
    if end is not None and end.tzinfo is None:
        raise ValueError("end timestamp must include a timezone")

    first_snapshot = build_deployment_profile_signal_snapshot(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=profile_pack_json,
        slot=slot,
        account=account,
        mode=mode,
        as_of=end,
    )
    symbols = tuple(row.symbol for row in first_snapshot.rows)
    timestamps = _selected_timestamps(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
        start=start,
        end=end,
        stride=stride,
        max_timestamps=max_timestamps,
    )
    portfolio = PortfolioSnapshot()
    events: list[DeploymentProfileActionScanEvent] = []
    for timestamp in timestamps:
        snapshot = build_deployment_profile_signal_snapshot(
            config=config,
            prices=prices,
            quotes=quotes,
            profile_pack_json=profile_pack_json,
            slot=slot,
            account=account,
            portfolio=portfolio if stateful else PortfolioSnapshot(),
            mode=mode,
            as_of=timestamp,
        )
        events.extend(_events_from_snapshot(snapshot))
        if stateful:
            portfolio = _portfolio_after_snapshot(portfolio, snapshot)

    profile = first_snapshot.profile
    return DeploymentProfileActionScanResult(
        profile_slot=profile.slot,
        profile_label=profile.label,
        stateful=stateful,
        scanned_timestamps=len(timestamps),
        first_timestamp="" if not timestamps else timestamps[0].isoformat(timespec="seconds"),
        last_timestamp="" if not timestamps else timestamps[-1].isoformat(timespec="seconds"),
        events=tuple(events),
        hourly=_hourly(events),
    )


def write_deployment_profile_action_scan_summary_csv(
    result: DeploymentProfileActionScanResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_slot",
                "profile_label",
                "stateful",
                "scanned_timestamps",
                "first_timestamp",
                "last_timestamp",
                "actionable_timestamps",
                "action_rows",
                "approved_actions",
                "blocked_actions",
                "approved_action_rate",
                "buy_actions",
                "sell_actions",
                "unique_action_symbols",
                "most_active_symbol",
                "first_action_timestamp",
                "last_action_timestamp",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "profile_slot": result.profile_slot,
                "profile_label": result.profile_label,
                "stateful": result.stateful,
                "scanned_timestamps": result.scanned_timestamps,
                "first_timestamp": result.first_timestamp,
                "last_timestamp": result.last_timestamp,
                "actionable_timestamps": result.actionable_timestamps,
                "action_rows": result.action_rows,
                "approved_actions": result.approved_actions,
                "blocked_actions": result.blocked_actions,
                "approved_action_rate": result.approved_action_rate,
                "buy_actions": result.buy_actions,
                "sell_actions": result.sell_actions,
                "unique_action_symbols": " ".join(result.unique_action_symbols),
                "most_active_symbol": result.most_active_symbol,
                "first_action_timestamp": result.first_action_timestamp,
                "last_action_timestamp": result.last_action_timestamp,
            }
        )


def write_deployment_profile_action_events_csv(
    result: DeploymentProfileActionScanResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "profile_slot",
                "profile_label",
                "symbol",
                "strategy_name",
                "order_side",
                "change_notional_usd",
                "allocated_target_notional_usd",
                "risk_approved",
                "risk_adjusted_notional_usd",
                "risk_reason",
                "primary_signal",
                "strategy_reason",
                "allocation_status",
                "requested_gross_notional_usd",
                "adjusted_gross_notional_usd",
                "net_directional_exposure",
                "largest_symbol_concentration",
            ],
        )
        writer.writeheader()
        for event in result.events:
            writer.writerow(event.__dict__)


def write_deployment_profile_action_hours_csv(
    result: DeploymentProfileActionScanResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hour_utc",
                "action_rows",
                "approved_actions",
                "buy_actions",
                "sell_actions",
                "unique_symbols",
            ],
        )
        writer.writeheader()
        for row in result.hourly:
            writer.writerow(
                {
                    "hour_utc": row.hour_utc,
                    "action_rows": row.action_rows,
                    "approved_actions": row.approved_actions,
                    "buy_actions": row.buy_actions,
                    "sell_actions": row.sell_actions,
                    "unique_symbols": " ".join(row.unique_symbols),
                }
            )


def _selected_timestamps(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
    start: datetime | None,
    end: datetime | None,
    stride: int,
    max_timestamps: int | None,
) -> tuple[datetime, ...]:
    common: set[datetime] | None = None
    for symbol in symbols:
        price_times = {bar.timestamp for bar in prices.for_symbol(symbol).bars}
        quote_times = {quote.timestamp for quote in quotes.for_symbol(symbol).quotes}
        symbol_common = price_times & quote_times
        if not symbol_common:
            raise ValueError(f"no aligned price/quote timestamps for {symbol}")
        common = symbol_common if common is None else common & symbol_common
    if not common:
        raise ValueError("no common timestamp across profile symbols")
    ordered = tuple(
        timestamp
        for timestamp in sorted(common)
        if (start is None or timestamp >= start) and (end is None or timestamp <= end)
    )
    if not ordered:
        raise ValueError("no common timestamps in requested scan window")
    strided = ordered[::stride]
    if max_timestamps is not None and len(strided) > max_timestamps:
        strided = strided[-max_timestamps:]
    return strided


def _events_from_snapshot(
    snapshot: DeploymentProfileSignalSnapshot,
) -> tuple[DeploymentProfileActionScanEvent, ...]:
    events: list[DeploymentProfileActionScanEvent] = []
    allocation = snapshot.allocation
    for row in snapshot.actionable_rows:
        events.append(
            DeploymentProfileActionScanEvent(
                timestamp=row.timestamp,
                profile_slot=snapshot.profile.slot,
                profile_label=snapshot.profile.label,
                symbol=row.symbol,
                strategy_name=row.strategy_name,
                order_side=row.order_side,
                change_notional_usd=row.change_notional_usd,
                allocated_target_notional_usd=row.allocated_target_notional_usd,
                risk_approved=row.risk_approved,
                risk_adjusted_notional_usd=row.risk_adjusted_notional_usd,
                risk_reason=row.risk_reason,
                primary_signal=row.primary_signal,
                strategy_reason=row.strategy_reason,
                allocation_status=allocation.estimated_risk_status,
                requested_gross_notional_usd=allocation.requested_gross_notional_usd,
                adjusted_gross_notional_usd=allocation.adjusted_gross_notional_usd,
                net_directional_exposure=allocation.net_directional_exposure,
                largest_symbol_concentration=allocation.largest_symbol_concentration,
            )
        )
    return tuple(events)


def _portfolio_after_snapshot(
    portfolio: PortfolioSnapshot,
    snapshot: DeploymentProfileSignalSnapshot,
) -> PortfolioSnapshot:
    positions = {position.symbol: position.notional_usd for position in portfolio.positions}
    for row in snapshot.actionable_rows:
        if not row.risk_approved:
            continue
        signed_target = _signed_target(
            requested_signed=row.allocated_target_notional_usd,
            adjusted_abs=row.risk_adjusted_notional_usd,
        )
        if abs(signed_target) <= 1e-9:
            positions.pop(row.symbol, None)
        else:
            positions[row.symbol] = signed_target
    return PortfolioSnapshot(
        positions=tuple(
            sorted(
                (
                    Position(symbol=symbol, notional_usd=notional)
                    for symbol, notional in positions.items()
                    if abs(notional) > 1e-9
                ),
                key=lambda position: position.symbol,
            )
        )
    )


def _signed_target(*, requested_signed: float, adjusted_abs: float) -> float:
    if abs(adjusted_abs) <= 1e-9:
        return 0.0
    return adjusted_abs if requested_signed > 0 else -adjusted_abs


def _hourly(
    events: list[DeploymentProfileActionScanEvent],
) -> tuple[DeploymentProfileActionHour, ...]:
    by_hour: dict[int, list[DeploymentProfileActionScanEvent]] = {}
    for event in events:
        timestamp = datetime.fromisoformat(event.timestamp)
        by_hour.setdefault(timestamp.hour, []).append(event)
    rows: list[DeploymentProfileActionHour] = []
    for hour in sorted(by_hour):
        hour_events = by_hour[hour]
        rows.append(
            DeploymentProfileActionHour(
                hour_utc=hour,
                action_rows=len(hour_events),
                approved_actions=len(
                    [event for event in hour_events if event.risk_approved]
                ),
                buy_actions=len(
                    [event for event in hour_events if event.order_side == "BUY"]
                ),
                sell_actions=len(
                    [event for event in hour_events if event.order_side == "SELL"]
                ),
                unique_symbols=tuple(sorted({event.symbol for event in hour_events})),
            )
        )
    return tuple(rows)
