from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from quanthack.backtesting.backtest import BacktestFill
from quanthack.backtesting.portfolio_allocator import EPSILON_NOTIONAL, SymbolIntent


EPSILON_UNITS = 1e-12


@dataclass(frozen=True)
class SymbolEvidenceGatePolicy:
    lookback_closed_events: int = 1
    min_closed_events: int = 1
    min_realized_pnl_usd: float = 0.0
    min_win_rate: float = 0.0
    allow_without_history: bool = True
    stale_after_bars: int | None = None
    target_symbols: tuple[str, ...] = ()
    no_history_target_multiplier: float = 0.0
    failed_evidence_target_multiplier: float = 0.0

    def __post_init__(self) -> None:
        if self.lookback_closed_events < 1:
            raise ValueError("lookback_closed_events must be at least 1")
        if self.min_closed_events < 0:
            raise ValueError("min_closed_events cannot be negative")
        if not isfinite(self.min_realized_pnl_usd):
            raise ValueError("min_realized_pnl_usd must be finite")
        if not 0 <= self.min_win_rate <= 1:
            raise ValueError("min_win_rate must be between 0 and 1")
        if self.stale_after_bars is not None and self.stale_after_bars < 1:
            raise ValueError("stale_after_bars must be at least 1 when set")
        if not 0 <= self.no_history_target_multiplier <= 1:
            raise ValueError("no_history_target_multiplier must be between 0 and 1")
        if not 0 <= self.failed_evidence_target_multiplier <= 1:
            raise ValueError("failed_evidence_target_multiplier must be between 0 and 1")


@dataclass(frozen=True)
class SymbolEvidenceGateReport:
    timestamp: str
    symbol: str
    primary_signal: str
    requested_before_usd: float
    requested_after_usd: float
    current_notional_usd: float
    prior_closed_events: int
    prior_wins: int
    prior_losses: int
    prior_win_rate: float
    prior_realized_pnl_usd: float
    allowed: bool
    applied: bool
    reason: str


@dataclass
class _SymbolEvidenceState:
    position_units: float = 0.0
    average_entry_price: float | None = None
    closed_pnls: list[float] = field(default_factory=list)
    bars_since_closed_event: int | None = None


class PortfolioSymbolEvidenceGate:
    def __init__(self, policy: SymbolEvidenceGatePolicy | None = None) -> None:
        self.policy = policy or SymbolEvidenceGatePolicy()
        self._state_by_symbol: dict[str, _SymbolEvidenceState] = {}

    def apply(
        self,
        intents: Iterable[SymbolIntent],
        *,
        timestamp: str = "",
    ) -> tuple[tuple[SymbolIntent, ...], tuple[SymbolEvidenceGateReport, ...]]:
        adjusted: list[SymbolIntent] = []
        reports: list[SymbolEvidenceGateReport] = []
        for intent in tuple(intents):
            self._advance_bar(intent.symbol)
            gated_intent, report = self._apply_one(intent, timestamp=timestamp)
            adjusted.append(gated_intent)
            reports.append(report)
        return tuple(adjusted), tuple(reports)

    def observe_fill(self, fill: BacktestFill) -> float:
        state = self._state_by_symbol.setdefault(fill.symbol, _SymbolEvidenceState())
        realized_pnl = _apply_fill_to_state(
            state,
            trade_units=fill.trade_units,
            fill_price=fill.fill_price,
        )
        if abs(realized_pnl) > EPSILON_NOTIONAL:
            state.closed_pnls.append(realized_pnl)
            state.bars_since_closed_event = 0
        return realized_pnl

    def _apply_one(
        self,
        intent: SymbolIntent,
        *,
        timestamp: str,
    ) -> tuple[SymbolIntent, SymbolEvidenceGateReport]:
        if not self._is_targeted(intent.symbol):
            return intent, SymbolEvidenceGateReport(
                timestamp=timestamp,
                symbol=intent.symbol,
                primary_signal=intent.primary_signal,
                requested_before_usd=intent.target_notional_usd,
                requested_after_usd=intent.target_notional_usd,
                current_notional_usd=intent.current_notional_usd,
                prior_closed_events=0,
                prior_wins=0,
                prior_losses=0,
                prior_win_rate=0.0,
                prior_realized_pnl_usd=0.0,
                allowed=True,
                applied=False,
                reason="allowed: symbol not targeted by evidence gate",
            )
        prior_pnls = self._recent_pnls(intent.symbol)
        allowed, reason = self._allowed(prior_pnls)
        restricted_target = (
            None
            if allowed
            else _restricted_target(
                intent,
                entry_multiplier=(
                    self.policy.no_history_target_multiplier
                    if not prior_pnls
                    else self.policy.failed_evidence_target_multiplier
                ),
            )
        )
        if (
            restricted_target is not None
            and abs(restricted_target - intent.target_notional_usd) <= EPSILON_NOTIONAL
        ):
            restricted_target = None
        applied = restricted_target is not None
        if applied:
            adjusted = SymbolIntent(
                symbol=intent.symbol,
                target_notional_usd=restricted_target,
                current_notional_usd=intent.current_notional_usd,
                reason=_with_reason(intent.reason, f"symbol evidence gate: {reason}"),
                primary_signal="symbol_evidence_gate",
                supporting_signals=(
                    *intent.supporting_signals,
                    f"prior_symbol_pnl={sum(prior_pnls):.2f}",
                ),
                conflicting_signals=intent.conflicting_signals,
            )
        else:
            adjusted = intent

        wins = sum(1 for value in prior_pnls if value > EPSILON_NOTIONAL)
        losses = sum(1 for value in prior_pnls if value < -EPSILON_NOTIONAL)
        events = len(prior_pnls)
        win_rate = wins / events if events > 0 else 0.0
        return adjusted, SymbolEvidenceGateReport(
            timestamp=timestamp,
            symbol=intent.symbol,
            primary_signal=intent.primary_signal,
            requested_before_usd=intent.target_notional_usd,
            requested_after_usd=adjusted.target_notional_usd,
            current_notional_usd=intent.current_notional_usd,
            prior_closed_events=events,
            prior_wins=wins,
            prior_losses=losses,
            prior_win_rate=win_rate,
            prior_realized_pnl_usd=sum(prior_pnls),
            allowed=allowed,
            applied=applied,
            reason=reason,
        )

    def _is_targeted(self, symbol: str) -> bool:
        if not self.policy.target_symbols:
            return True
        return symbol in self.policy.target_symbols

    def _recent_pnls(self, symbol: str) -> tuple[float, ...]:
        state = self._state_by_symbol.setdefault(symbol, _SymbolEvidenceState())
        if (
            self.policy.stale_after_bars is not None
            and state.bars_since_closed_event is not None
            and state.bars_since_closed_event >= self.policy.stale_after_bars
        ):
            return ()
        return tuple(state.closed_pnls[-self.policy.lookback_closed_events :])

    def _advance_bar(self, symbol: str) -> None:
        state = self._state_by_symbol.setdefault(symbol, _SymbolEvidenceState())
        if state.bars_since_closed_event is not None:
            state.bars_since_closed_event += 1

    def _allowed(self, prior_pnls: tuple[float, ...]) -> tuple[bool, str]:
        if not prior_pnls:
            if self.policy.allow_without_history:
                return True, "allowed: no prior closed-trade evidence"
            return False, "blocked: no prior closed-trade evidence"
        wins = sum(1 for value in prior_pnls if value > EPSILON_NOTIONAL)
        events = len(prior_pnls)
        pnl = sum(prior_pnls)
        win_rate = wins / events if events > 0 else 0.0
        if events < self.policy.min_closed_events:
            return (
                False,
                f"blocked: prior closed events {events} < {self.policy.min_closed_events}",
            )
        if pnl < self.policy.min_realized_pnl_usd:
            return (
                False,
                (
                    f"blocked: prior pnl {pnl:.2f} < "
                    f"{self.policy.min_realized_pnl_usd:.2f}"
                ),
            )
        if win_rate < self.policy.min_win_rate:
            return (
                False,
                (
                    f"blocked: prior win rate {win_rate:.1%} < "
                    f"{self.policy.min_win_rate:.1%}"
                ),
            )
        return True, f"allowed: prior pnl {pnl:.2f}, win rate {win_rate:.1%}"


def write_symbol_evidence_gate_report_csv(
    reports: Iterable[SymbolEvidenceGateReport],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "symbol",
                "primary_signal",
                "requested_before_usd",
                "requested_after_usd",
                "current_notional_usd",
                "prior_closed_events",
                "prior_wins",
                "prior_losses",
                "prior_win_rate",
                "prior_realized_pnl_usd",
                "allowed",
                "applied",
                "reason",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "timestamp": report.timestamp,
                    "symbol": report.symbol,
                    "primary_signal": report.primary_signal,
                    "requested_before_usd": report.requested_before_usd,
                    "requested_after_usd": report.requested_after_usd,
                    "current_notional_usd": report.current_notional_usd,
                    "prior_closed_events": report.prior_closed_events,
                    "prior_wins": report.prior_wins,
                    "prior_losses": report.prior_losses,
                    "prior_win_rate": report.prior_win_rate,
                    "prior_realized_pnl_usd": report.prior_realized_pnl_usd,
                    "allowed": report.allowed,
                    "applied": report.applied,
                    "reason": report.reason,
                }
            )


def _restricted_target(
    intent: SymbolIntent,
    *,
    entry_multiplier: float = 0.0,
) -> float | None:
    current_abs = abs(intent.current_notional_usd)
    target_abs = abs(intent.target_notional_usd)
    if target_abs <= current_abs + EPSILON_NOTIONAL:
        return None
    if current_abs <= EPSILON_NOTIONAL:
        if target_abs <= EPSILON_NOTIONAL:
            return None
        return intent.target_notional_usd * entry_multiplier
    current_direction = 1 if intent.current_notional_usd > 0 else -1
    target_direction = 1 if intent.target_notional_usd > 0 else -1
    if current_direction == target_direction:
        extra_target = target_abs - current_abs
        adjusted_abs = current_abs + (extra_target * entry_multiplier)
        return adjusted_abs * current_direction
    return 0.0


def _apply_fill_to_state(
    state: _SymbolEvidenceState,
    *,
    trade_units: float,
    fill_price: float,
) -> float:
    if not isfinite(trade_units):
        raise ValueError("trade_units must be finite")
    if not isfinite(fill_price) or fill_price <= 0:
        raise ValueError("fill_price must be positive and finite")
    if abs(trade_units) <= EPSILON_UNITS:
        return 0.0

    realized_pnl = 0.0
    if abs(state.position_units) <= EPSILON_UNITS:
        state.position_units = trade_units
        state.average_entry_price = fill_price
        return realized_pnl

    if _same_direction(state.position_units, trade_units):
        state.average_entry_price = _weighted_average_price(
            current_units=state.position_units,
            current_average_price=state.average_entry_price,
            trade_units=trade_units,
            trade_price=fill_price,
        )
        state.position_units += trade_units
        return realized_pnl

    close_units = min(abs(state.position_units), abs(trade_units))
    position_direction = 1 if state.position_units > 0 else -1
    entry_price = state.average_entry_price or fill_price
    realized_pnl = close_units * position_direction * (fill_price - entry_price)
    state.position_units += trade_units
    if abs(state.position_units) <= EPSILON_UNITS:
        state.position_units = 0.0
        state.average_entry_price = None
    elif abs(trade_units) > close_units + EPSILON_UNITS:
        state.average_entry_price = fill_price
    return realized_pnl


def _same_direction(left_units: float, right_units: float) -> bool:
    return (left_units > 0 and right_units > 0) or (left_units < 0 and right_units < 0)


def _weighted_average_price(
    *,
    current_units: float,
    current_average_price: float | None,
    trade_units: float,
    trade_price: float,
) -> float:
    current_price = current_average_price or trade_price
    total_units = abs(current_units) + abs(trade_units)
    return ((current_price * abs(current_units)) + (trade_price * abs(trade_units))) / total_units


def _with_reason(reason: str, addition: str) -> str:
    return f"{reason}; {addition}" if reason else addition
