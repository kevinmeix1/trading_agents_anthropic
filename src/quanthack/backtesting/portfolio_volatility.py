from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import pstdev

from quanthack.backtesting.portfolio_allocator import EPSILON_NOTIONAL, SymbolIntent


@dataclass(frozen=True)
class VolatilityTargetingPolicy:
    lookback: int = 32
    min_observations: int = 12
    target_bar_volatility: float = 0.00075
    min_scale: float = 0.25
    max_scale: float = 1.15

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if self.min_observations > self.lookback:
            raise ValueError("min_observations cannot exceed lookback")
        _validate_positive("target_bar_volatility", self.target_bar_volatility)
        _validate_positive("min_scale", self.min_scale)
        _validate_positive("max_scale", self.max_scale)
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale cannot exceed max_scale")


@dataclass(frozen=True)
class VolatilityTargetingReport:
    timestamp: str
    scale: float
    target_bar_volatility: float
    realized_bar_volatility: float | None
    observations: int
    active_symbols: int
    reason: str
    applied: bool


class PortfolioVolatilityTargeter:
    def __init__(self, policy: VolatilityTargetingPolicy | None = None) -> None:
        self.policy = policy or VolatilityTargetingPolicy()

    def apply(
        self,
        intents: Iterable[SymbolIntent],
        *,
        closes_by_symbol: Mapping[str, tuple[float, ...]],
        equity: float,
        timestamp: str = "",
    ) -> tuple[tuple[SymbolIntent, ...], VolatilityTargetingReport]:
        intent_tuple = tuple(intents)
        report = self.measure(
            intent_tuple,
            closes_by_symbol=closes_by_symbol,
            equity=equity,
            timestamp=timestamp,
        )
        if not report.applied or abs(report.scale - 1.0) <= EPSILON_NOTIONAL:
            return intent_tuple, report
        return (
            tuple(_scale_intent(intent, report) for intent in intent_tuple),
            report,
        )

    def measure(
        self,
        intents: Iterable[SymbolIntent],
        *,
        closes_by_symbol: Mapping[str, tuple[float, ...]],
        equity: float,
        timestamp: str = "",
    ) -> VolatilityTargetingReport:
        _validate_positive("equity", equity)
        intent_tuple = tuple(intents)
        active_weights = {
            intent.symbol: intent.target_notional_usd / equity
            for intent in intent_tuple
            if abs(intent.target_notional_usd) > EPSILON_NOTIONAL
            and intent.primary_signal != "market_quality"
        }
        if not active_weights:
            return VolatilityTargetingReport(
                timestamp=timestamp,
                scale=1.0,
                target_bar_volatility=self.policy.target_bar_volatility,
                realized_bar_volatility=None,
                observations=0,
                active_symbols=0,
                reason="no active volatility-managed targets",
                applied=False,
            )

        returns_by_symbol = {
            symbol: _simple_returns(closes_by_symbol.get(symbol, ()))
            for symbol in active_weights
        }
        observations = min(
            [len(returns) for returns in returns_by_symbol.values()],
            default=0,
        )
        observations = min(observations, self.policy.lookback)
        if observations < self.policy.min_observations:
            return VolatilityTargetingReport(
                timestamp=timestamp,
                scale=1.0,
                target_bar_volatility=self.policy.target_bar_volatility,
                realized_bar_volatility=None,
                observations=observations,
                active_symbols=len(active_weights),
                reason="insufficient realized volatility history",
                applied=False,
            )

        portfolio_returns = []
        for offset in range(-observations, 0):
            portfolio_returns.append(
                sum(
                    active_weights[symbol] * returns_by_symbol[symbol][offset]
                    for symbol in active_weights
                )
            )
        realized_volatility = pstdev(portfolio_returns)
        if realized_volatility <= 0:
            raw_scale = self.policy.max_scale
        else:
            raw_scale = self.policy.target_bar_volatility / realized_volatility
        scale = min(self.policy.max_scale, max(self.policy.min_scale, raw_scale))
        reason = (
            "portfolio volatility below target"
            if scale > 1.0
            else "portfolio volatility above target"
            if scale < 1.0
            else "portfolio volatility on target"
        )
        return VolatilityTargetingReport(
            timestamp=timestamp,
            scale=scale,
            target_bar_volatility=self.policy.target_bar_volatility,
            realized_bar_volatility=realized_volatility,
            observations=observations,
            active_symbols=len(active_weights),
            reason=reason,
            applied=True,
        )


def write_volatility_targeting_report_csv(
    reports: Iterable[VolatilityTargetingReport],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "scale",
                "target_bar_volatility",
                "realized_bar_volatility",
                "observations",
                "active_symbols",
                "reason",
                "applied",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "timestamp": report.timestamp,
                    "scale": report.scale,
                    "target_bar_volatility": report.target_bar_volatility,
                    "realized_bar_volatility": report.realized_bar_volatility,
                    "observations": report.observations,
                    "active_symbols": report.active_symbols,
                    "reason": report.reason,
                    "applied": report.applied,
                }
            )


def _scale_intent(
    intent: SymbolIntent,
    report: VolatilityTargetingReport,
) -> SymbolIntent:
    if intent.primary_signal in {"market_quality", "position_stop"}:
        return intent
    if (
        abs(intent.target_notional_usd) <= EPSILON_NOTIONAL
        and abs(intent.current_notional_usd) <= EPSILON_NOTIONAL
    ):
        return intent
    reason = _with_reason(
        intent.reason,
        (
            f"portfolio volatility targeting scale={report.scale:.3f} "
            f"({report.reason})"
        ),
    )
    return SymbolIntent(
        symbol=intent.symbol,
        target_notional_usd=intent.target_notional_usd * report.scale,
        current_notional_usd=intent.current_notional_usd,
        reason=reason,
        primary_signal=intent.primary_signal,
        supporting_signals=(
            *intent.supporting_signals,
            f"portfolio_vol_target_scale={report.scale:.3f}",
        ),
        conflicting_signals=intent.conflicting_signals,
    )


def _simple_returns(closes: tuple[float, ...]) -> tuple[float, ...]:
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        if previous <= 0:
            continue
        returns.append((current / previous) - 1.0)
    return tuple(returns)


def _with_reason(existing: str, addition: str) -> str:
    if not existing:
        return addition
    return f"{existing}; {addition}"


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
