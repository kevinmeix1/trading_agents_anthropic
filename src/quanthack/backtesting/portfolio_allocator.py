from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from quanthack.core.instruments import AssetClass, instrument_for


EPSILON_NOTIONAL = 1e-9


@dataclass(frozen=True)
class SymbolIntent:
    symbol: str
    target_notional_usd: float
    current_notional_usd: float = 0.0
    reason: str = ""
    primary_signal: str = "strategy"
    supporting_signals: tuple[str, ...] = ()
    conflicting_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        instrument_for(self.symbol)
        _validate_finite("target_notional_usd", self.target_notional_usd)
        _validate_finite("current_notional_usd", self.current_notional_usd)

    @property
    def requested_change_notional_usd(self) -> float:
        return self.target_notional_usd - self.current_notional_usd


@dataclass(frozen=True)
class AllocationPolicy:
    max_gross_leverage: float = 2.0
    max_symbol_gross_pct: float = 0.25
    max_net_directional_pct: float = 0.80
    max_forex_gross_pct: float = 0.80
    max_metal_gross_pct: float = 0.25
    max_crypto_gross_pct: float = 0.40
    min_active_symbols: int = 3
    min_position_notional_usd: float = 1_000.0
    apply_diversification_scale: bool = True
    min_rebalance_notional_usd: float = 500.0
    min_rebalance_change_pct: float = 0.02

    def __post_init__(self) -> None:
        _validate_positive("max_gross_leverage", self.max_gross_leverage)
        _validate_pct("max_symbol_gross_pct", self.max_symbol_gross_pct)
        _validate_pct("max_net_directional_pct", self.max_net_directional_pct)
        _validate_pct("max_forex_gross_pct", self.max_forex_gross_pct)
        _validate_pct("max_metal_gross_pct", self.max_metal_gross_pct)
        _validate_pct("max_crypto_gross_pct", self.max_crypto_gross_pct)
        if self.min_active_symbols < 1:
            raise ValueError("min_active_symbols must be at least 1")
        _validate_non_negative("min_position_notional_usd", self.min_position_notional_usd)
        _validate_non_negative("min_rebalance_notional_usd", self.min_rebalance_notional_usd)
        _validate_non_negative("min_rebalance_change_pct", self.min_rebalance_change_pct)

    def asset_class_cap(self, asset_class: AssetClass) -> float:
        if asset_class == AssetClass.CRYPTO:
            return self.max_crypto_gross_pct
        if asset_class == AssetClass.METAL:
            return self.max_metal_gross_pct
        return self.max_forex_gross_pct


@dataclass(frozen=True)
class AllocatedTarget:
    symbol: str
    requested_notional_usd: float
    adjusted_notional_usd: float
    current_notional_usd: float
    intent_reason: str = ""
    reasons: tuple[str, ...] = ()
    primary_signal: str = "strategy"
    supporting_signals: tuple[str, ...] = ()
    conflicting_signals: tuple[str, ...] = ()

    @property
    def requested_abs_notional_usd(self) -> float:
        return abs(self.requested_notional_usd)

    @property
    def adjusted_abs_notional_usd(self) -> float:
        return abs(self.adjusted_notional_usd)

    @property
    def change_notional_usd(self) -> float:
        return self.adjusted_notional_usd - self.current_notional_usd

    @property
    def was_trimmed(self) -> bool:
        return abs(self.adjusted_notional_usd - self.requested_notional_usd) > EPSILON_NOTIONAL


@dataclass(frozen=True)
class PortfolioAllocation:
    targets: tuple[AllocatedTarget, ...]
    policy: AllocationPolicy
    equity: float
    timestamp: str = ""

    @property
    def requested_gross_notional_usd(self) -> float:
        return sum(target.requested_abs_notional_usd for target in self.targets)

    @property
    def adjusted_gross_notional_usd(self) -> float:
        return sum(target.adjusted_abs_notional_usd for target in self.targets)

    @property
    def requested_net_notional_usd(self) -> float:
        return sum(target.requested_notional_usd for target in self.targets)

    @property
    def adjusted_net_notional_usd(self) -> float:
        return sum(target.adjusted_notional_usd for target in self.targets)

    @property
    def active_symbols(self) -> int:
        return len(
            [
                target
                for target in self.targets
                if target.adjusted_abs_notional_usd >= self.policy.min_position_notional_usd
            ]
        )

    @property
    def leverage(self) -> float:
        return self.adjusted_gross_notional_usd / self.equity

    @property
    def largest_symbol_concentration(self) -> float:
        gross = self.adjusted_gross_notional_usd
        if gross <= EPSILON_NOTIONAL:
            return 0.0
        return max(target.adjusted_abs_notional_usd for target in self.targets) / gross

    @property
    def net_directional_exposure(self) -> float:
        gross = self.adjusted_gross_notional_usd
        if gross <= EPSILON_NOTIONAL:
            return 0.0
        return abs(self.adjusted_net_notional_usd) / gross

    @property
    def trimmed_targets(self) -> tuple[AllocatedTarget, ...]:
        return tuple(target for target in self.targets if target.was_trimmed)

    @property
    def trim_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        seen: set[str] = set()
        for target in self.targets:
            for reason in target.reasons:
                if reason in seen:
                    continue
                reasons.append(reason)
                seen.add(reason)
        return tuple(reasons)

    @property
    def estimated_risk_status(self) -> str:
        if self.adjusted_gross_notional_usd <= EPSILON_NOTIONAL:
            return "OK"
        if (
            self.largest_symbol_concentration > 0.90
            or self.net_directional_exposure > 0.95
            or self.leverage > 28.0
        ):
            return "PENALTY_RISK"
        if (
            self.active_symbols < self.policy.min_active_symbols
            or self.largest_symbol_concentration > self.policy.max_symbol_gross_pct
            or self.net_directional_exposure > self.policy.max_net_directional_pct
            or self.leverage > self.policy.max_gross_leverage * 0.90
        ):
            return "WARN"
        return "OK"


class PortfolioAllocator:
    def __init__(self, policy: AllocationPolicy | None = None) -> None:
        self.policy = policy or AllocationPolicy()

    def allocate(
        self,
        intents: Iterable[SymbolIntent],
        *,
        equity: float,
        timestamp: str = "",
    ) -> PortfolioAllocation:
        _validate_positive("equity", equity)
        intent_by_symbol = _unique_intents(tuple(intents))
        requested = {
            symbol: intent.target_notional_usd
            for symbol, intent in intent_by_symbol.items()
        }
        current = {
            symbol: intent.current_notional_usd
            for symbol, intent in intent_by_symbol.items()
        }
        adjusted = dict(requested)
        reasons = {symbol: [] for symbol in adjusted}
        max_gross_notional = equity * self.policy.max_gross_leverage

        _apply_minimum_position_filter(
            adjusted,
            reasons,
            self.policy.min_position_notional_usd,
        )
        _scale_all_if_needed(
            adjusted,
            reasons,
            max_gross_notional,
            "gross leverage budget",
        )
        _apply_asset_class_caps(
            adjusted,
            reasons,
            max_gross_notional,
            self.policy,
        )
        _apply_symbol_budget_cap(
            adjusted,
            reasons,
            max_gross_notional * self.policy.max_symbol_gross_pct,
        )
        _apply_net_directional_cap(
            adjusted,
            reasons,
            self.policy.max_net_directional_pct,
        )
        if self.policy.apply_diversification_scale:
            _apply_diversification_preference(
                adjusted,
                reasons,
                current,
                min_active_symbols=self.policy.min_active_symbols,
                min_position_notional_usd=self.policy.min_position_notional_usd,
            )
        _apply_minimum_position_filter(
            adjusted,
            reasons,
            self.policy.min_position_notional_usd,
        )
        _apply_single_symbol_concentration_guard(
            adjusted,
            reasons,
            self.policy.min_position_notional_usd,
            self.policy.min_active_symbols,
        )
        _apply_rebalance_deadband(
            adjusted,
            reasons,
            current,
            min_rebalance_notional_usd=self.policy.min_rebalance_notional_usd,
            min_rebalance_change_pct=self.policy.min_rebalance_change_pct,
        )

        targets = tuple(
            AllocatedTarget(
                symbol=symbol,
                requested_notional_usd=requested[symbol],
                adjusted_notional_usd=adjusted[symbol],
                current_notional_usd=intent_by_symbol[symbol].current_notional_usd,
                intent_reason=intent_by_symbol[symbol].reason,
                reasons=tuple(reasons[symbol]),
                primary_signal=intent_by_symbol[symbol].primary_signal,
                supporting_signals=intent_by_symbol[symbol].supporting_signals,
                conflicting_signals=intent_by_symbol[symbol].conflicting_signals,
            )
            for symbol in sorted(adjusted)
        )
        return PortfolioAllocation(
            targets=targets,
            policy=self.policy,
            equity=equity,
            timestamp=timestamp,
        )


def write_allocation_report_csv(
    allocations: Iterable[PortfolioAllocation],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "requested_gross_notional_usd",
                "adjusted_gross_notional_usd",
                "requested_net_notional_usd",
                "adjusted_net_notional_usd",
                "leverage",
                "largest_symbol_concentration",
                "net_directional_exposure",
                "active_symbols",
                "estimated_risk_status",
                "trim_reasons",
                "targets",
            ],
        )
        writer.writeheader()
        for allocation in allocations:
            writer.writerow(
                {
                    "timestamp": allocation.timestamp,
                    "requested_gross_notional_usd": allocation.requested_gross_notional_usd,
                    "adjusted_gross_notional_usd": allocation.adjusted_gross_notional_usd,
                    "requested_net_notional_usd": allocation.requested_net_notional_usd,
                    "adjusted_net_notional_usd": allocation.adjusted_net_notional_usd,
                    "leverage": allocation.leverage,
                    "largest_symbol_concentration": (
                        allocation.largest_symbol_concentration
                    ),
                    "net_directional_exposure": allocation.net_directional_exposure,
                    "active_symbols": allocation.active_symbols,
                    "estimated_risk_status": allocation.estimated_risk_status,
                    "trim_reasons": "; ".join(allocation.trim_reasons),
                    "targets": _allocation_targets_text(allocation.targets),
                }
            )


def _unique_intents(intents: tuple[SymbolIntent, ...]) -> dict[str, SymbolIntent]:
    intent_by_symbol: dict[str, SymbolIntent] = {}
    for intent in intents:
        symbol = instrument_for(intent.symbol).symbol
        if symbol in intent_by_symbol:
            raise ValueError(f"duplicate allocation intent for {symbol}")
        intent_by_symbol[symbol] = intent
    if not intent_by_symbol:
        raise ValueError("at least one allocation intent is required")
    return intent_by_symbol


def _apply_minimum_position_filter(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    min_position_notional_usd: float,
) -> None:
    if min_position_notional_usd <= 0:
        return
    for symbol, target in list(adjusted.items()):
        if 0 < abs(target) < min_position_notional_usd:
            adjusted[symbol] = 0.0
            reasons[symbol].append(
                f"below minimum position ${min_position_notional_usd:,.0f}"
            )


def _scale_all_if_needed(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    max_gross_notional: float,
    label: str,
) -> None:
    gross = _gross_notional(adjusted.values())
    if gross <= max_gross_notional or gross <= EPSILON_NOTIONAL:
        return
    factor = max_gross_notional / gross
    _scale_symbols(
        adjusted,
        reasons,
        symbols=tuple(adjusted),
        factor=factor,
        reason=f"{label} scaled to {factor:.2%}",
    )


def _apply_asset_class_caps(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    max_gross_notional: float,
    policy: AllocationPolicy,
) -> None:
    for asset_class in AssetClass:
        symbols = tuple(
            symbol
            for symbol in adjusted
            if instrument_for(symbol).asset_class == asset_class
        )
        class_gross = _gross_notional(adjusted[symbol] for symbol in symbols)
        class_cap = max_gross_notional * policy.asset_class_cap(asset_class)
        if class_gross <= class_cap or class_gross <= EPSILON_NOTIONAL:
            continue
        factor = class_cap / class_gross
        _scale_symbols(
            adjusted,
            reasons,
            symbols=symbols,
            factor=factor,
            reason=f"{asset_class.value.lower()} asset-class cap scaled to {factor:.2%}",
        )


def _apply_symbol_budget_cap(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    symbol_cap_notional: float,
) -> None:
    if symbol_cap_notional <= 0:
        return
    for symbol, target in list(adjusted.items()):
        if abs(target) <= symbol_cap_notional:
            continue
        adjusted[symbol] = _sign(target) * symbol_cap_notional
        reasons[symbol].append(
            f"symbol cap limited target to ${symbol_cap_notional:,.0f}"
        )


def _apply_net_directional_cap(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    max_net_directional_pct: float,
) -> None:
    gross = _gross_notional(adjusted.values())
    net = sum(adjusted.values())
    if gross <= EPSILON_NOTIONAL:
        return
    if abs(net) <= max_net_directional_pct * gross:
        return

    long_gross = sum(value for value in adjusted.values() if value > 0)
    short_gross = sum(abs(value) for value in adjusted.values() if value < 0)
    if net > 0:
        majority_symbols = tuple(symbol for symbol, value in adjusted.items() if value > 0)
        allowed_majority = (
            short_gross * (1 + max_net_directional_pct) / (1 - max_net_directional_pct)
        )
        factor = 0.0 if long_gross <= EPSILON_NOTIONAL else allowed_majority / long_gross
    else:
        majority_symbols = tuple(symbol for symbol, value in adjusted.items() if value < 0)
        allowed_majority = (
            long_gross * (1 + max_net_directional_pct) / (1 - max_net_directional_pct)
        )
        factor = 0.0 if short_gross <= EPSILON_NOTIONAL else allowed_majority / short_gross

    _scale_symbols(
        adjusted,
        reasons,
        symbols=majority_symbols,
        factor=min(max(factor, 0.0), 1.0),
        reason=(
            f"net directional exposure capped at "
            f"{max_net_directional_pct:.0%}"
        ),
    )


def _apply_diversification_preference(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    current: dict[str, float],
    *,
    min_active_symbols: int,
    min_position_notional_usd: float,
) -> None:
    active_symbols = tuple(
        symbol
        for symbol, target in adjusted.items()
        if abs(target) >= min_position_notional_usd
    )
    active_count = len(active_symbols)
    if active_count == 0 or active_count >= min_active_symbols:
        return
    changed_symbols = tuple(
        symbol
        for symbol in active_symbols
        if abs(adjusted[symbol] - current.get(symbol, 0.0)) >= min_position_notional_usd
    )
    if not changed_symbols:
        return
    factor = active_count / min_active_symbols
    _scale_symbols(
        adjusted,
        reasons,
        symbols=changed_symbols,
        factor=factor,
        reason=(
            f"diversification preference scaled {active_count} active symbols "
            f"toward {min_active_symbols}"
        ),
    )


def _apply_single_symbol_concentration_guard(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    min_position_notional_usd: float,
    min_active_symbols: int,
) -> None:
    if min_active_symbols <= 1:
        return
    active_symbols = tuple(
        symbol
        for symbol, target in adjusted.items()
        if abs(target) >= min_position_notional_usd
    )
    if len(active_symbols) != 1:
        return
    symbol = active_symbols[0]
    adjusted[symbol] = 0.0
    reasons[symbol].append("single-symbol concentration guard blocked exposure")


def _apply_rebalance_deadband(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    current: dict[str, float],
    *,
    min_rebalance_notional_usd: float,
    min_rebalance_change_pct: float,
) -> None:
    for symbol, target in list(adjusted.items()):
        current_notional = current.get(symbol, 0.0)
        if abs(current_notional) <= EPSILON_NOTIONAL:
            continue
        if abs(target) <= EPSILON_NOTIONAL:
            continue
        if _sign(current_notional) != _sign(target):
            continue
        change = abs(target - current_notional)
        threshold = max(
            min_rebalance_notional_usd,
            abs(current_notional) * min_rebalance_change_pct,
        )
        if change <= threshold:
            adjusted[symbol] = current_notional
            if change > EPSILON_NOTIONAL:
                reasons[symbol].append(
                    f"rebalance deadband kept current position within ${threshold:,.0f}"
                )


def _scale_symbols(
    adjusted: dict[str, float],
    reasons: dict[str, list[str]],
    *,
    symbols: tuple[str, ...],
    factor: float,
    reason: str,
) -> None:
    for symbol in symbols:
        if abs(adjusted[symbol]) <= EPSILON_NOTIONAL:
            continue
        adjusted[symbol] *= factor
        reasons[symbol].append(reason)


def _allocation_targets_text(targets: tuple[AllocatedTarget, ...]) -> str:
    parts = []
    for target in targets:
        reason = f" ({'; '.join(target.reasons)})" if target.reasons else ""
        parts.append(
            f"{target.symbol}:{target.requested_notional_usd:.2f}"
            f"->{target.adjusted_notional_usd:.2f}{reason}"
        )
    return " | ".join(parts)


def _gross_notional(values: Iterable[float]) -> float:
    return sum(abs(value) for value in values)


def _sign(value: float) -> int:
    return 1 if value >= 0 else -1


def _validate_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be positive and finite")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0 or not isfinite(value):
        raise ValueError(f"{name} must be non-negative and finite")


def _validate_pct(name: str, value: float) -> None:
    if not 0 < value <= 1 or not isfinite(value):
        raise ValueError(f"{name} must be finite and between 0 and 1")
