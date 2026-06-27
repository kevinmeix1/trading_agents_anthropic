from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from quanthack.backtesting.portfolio_allocator import EPSILON_NOTIONAL, SymbolIntent
from quanthack.core.clock import UTC
from quanthack.core.instruments import AssetClass, instrument_for


@dataclass(frozen=True)
class SessionGatePolicy:
    allowed_utc_hours: tuple[int, ...] | None = None
    forex_allowed_utc_hours: tuple[int, ...] | None = None
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    symbol_allowed_utc_hours: Mapping[str, tuple[int, ...]] | tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        _normalize_hours(self, "allowed_utc_hours")
        _normalize_hours(self, "forex_allowed_utc_hours")
        _normalize_hours(self, "metal_allowed_utc_hours")
        _normalize_hours(self, "crypto_allowed_utc_hours")
        _normalize_symbol_hours(self)

    def hours_for_symbol(self, symbol: str) -> tuple[int, ...] | None:
        normalized_symbol = instrument_for(symbol).symbol
        symbol_hours = dict(self.symbol_allowed_utc_hours)
        if normalized_symbol in symbol_hours:
            return symbol_hours[normalized_symbol]
        asset_class = instrument_for(symbol).asset_class
        if asset_class == AssetClass.METAL and self.metal_allowed_utc_hours is not None:
            return self.metal_allowed_utc_hours
        if asset_class == AssetClass.CRYPTO and self.crypto_allowed_utc_hours is not None:
            return self.crypto_allowed_utc_hours
        if asset_class == AssetClass.FOREX and self.forex_allowed_utc_hours is not None:
            return self.forex_allowed_utc_hours
        return self.allowed_utc_hours


class PortfolioSessionGate:
    def __init__(self, policy: SessionGatePolicy) -> None:
        self.policy = policy

    def apply(
        self,
        intents: Iterable[SymbolIntent],
        *,
        timestamp: datetime,
    ) -> tuple[SymbolIntent, ...]:
        hour = timestamp.astimezone(UTC).hour
        return tuple(self._apply_one(intent, hour=hour) for intent in intents)

    def _apply_one(self, intent: SymbolIntent, *, hour: int) -> SymbolIntent:
        allowed_hours = self.policy.hours_for_symbol(intent.symbol)
        if allowed_hours is None or hour in allowed_hours:
            return intent
        restricted_target = _restricted_target(intent)
        if restricted_target is None:
            return intent
        return SymbolIntent(
            symbol=intent.symbol,
            target_notional_usd=restricted_target,
            current_notional_usd=intent.current_notional_usd,
            reason=_with_reason(
                intent.reason,
                f"session gate held target outside UTC hours {allowed_hours}",
            ),
            primary_signal="session_gate",
            supporting_signals=(
                *intent.supporting_signals,
                f"blocked_hour={hour}",
            ),
            conflicting_signals=intent.conflicting_signals,
        )


def _restricted_target(intent: SymbolIntent) -> float | None:
    current_abs = abs(intent.current_notional_usd)
    target_abs = abs(intent.target_notional_usd)
    if target_abs <= current_abs + EPSILON_NOTIONAL:
        return None
    if current_abs <= EPSILON_NOTIONAL:
        return 0.0 if target_abs > EPSILON_NOTIONAL else None
    current_direction = 1 if intent.current_notional_usd > 0 else -1
    target_direction = 1 if intent.target_notional_usd > 0 else -1
    if current_direction == target_direction:
        return intent.current_notional_usd
    return 0.0


def _normalize_hours(policy: SessionGatePolicy, name: str) -> None:
    value = getattr(policy, name)
    if value is None:
        return
    normalized = tuple(sorted({int(hour) for hour in value}))
    if any(hour < 0 or hour > 23 for hour in normalized):
        raise ValueError(f"{name} must contain hours between 0 and 23")
    object.__setattr__(policy, name, normalized)


def _normalize_symbol_hours(policy: SessionGatePolicy) -> None:
    raw_value = policy.symbol_allowed_utc_hours
    if isinstance(raw_value, Mapping):
        items = raw_value.items()
    else:
        items = raw_value
    normalized: dict[str, tuple[int, ...]] = {}
    for raw_symbol, raw_hours in items:
        symbol = instrument_for(raw_symbol).symbol
        hours = tuple(sorted({int(hour) for hour in raw_hours}))
        if not hours:
            raise ValueError("symbol_allowed_utc_hours cannot contain empty hour sets")
        if any(hour < 0 or hour > 23 for hour in hours):
            raise ValueError("symbol_allowed_utc_hours must contain hours between 0 and 23")
        normalized[symbol] = hours
    object.__setattr__(
        policy,
        "symbol_allowed_utc_hours",
        tuple(sorted(normalized.items())),
    )


def _with_reason(reason: str, addition: str) -> str:
    return f"{reason}; {addition}" if reason else addition
