from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from quanthack.backtesting.portfolio_allocator import EPSILON_NOTIONAL, SymbolIntent
from quanthack.strategies.time_series import (
    KalmanTrendConfig,
    TimeSeriesRegime,
    read_kalman_regime,
)


TREND_SIGNALS = frozenset(
    {
        "request",
        "momentum",
        "multi_horizon_momentum",
        "autocorrelation_regime",
        "intraday_seasonality",
        "conditional_seasonality",
        "ma_crossover",
        "breakout",
        "session_breakout",
        "volatility_squeeze",
        "dual_squeeze",
        "asset_adaptive_dual_squeeze",
        "range_expansion_trend",
        "trend_pullback",
        "macd_momentum",
        "kalman_trend",
        "quality_trend",
        "champion_ensemble",
        "alpha_router",
        "relative_strength",
        "usd_pressure_router",
    }
)
REVERSION_SIGNALS = frozenset(
    {
        "mean_reversion",
        "cross_rate_reversion",
        "exhaustion_reversal",
        "fixing_reversal",
    }
)
PROTECTED_SIGNALS = frozenset({"market_quality", "position_stop"})


@dataclass(frozen=True)
class RegimeTiltPolicy:
    lookback: int = 80
    chop_trend_scale: float = 0.70
    chop_reversion_scale: float = 1.15
    trend_aligned_scale: float = 1.10
    trend_counter_scale: float = 0.60
    trend_reversion_scale: float = 0.80
    high_volatility_scale: float = 0.75
    min_abs_slope_bps: float = 0.75
    min_trend_efficiency: float = 0.25
    max_realized_volatility_bps: float = 120.0

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5")
        _validate_positive("chop_trend_scale", self.chop_trend_scale)
        _validate_positive("chop_reversion_scale", self.chop_reversion_scale)
        _validate_positive("trend_aligned_scale", self.trend_aligned_scale)
        _validate_positive("trend_counter_scale", self.trend_counter_scale)
        _validate_positive("trend_reversion_scale", self.trend_reversion_scale)
        _validate_positive("high_volatility_scale", self.high_volatility_scale)
        _validate_non_negative("min_abs_slope_bps", self.min_abs_slope_bps)
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_positive(
            "max_realized_volatility_bps",
            self.max_realized_volatility_bps,
        )

    def kalman_config(self) -> KalmanTrendConfig:
        return KalmanTrendConfig(
            lookback=self.lookback,
            min_abs_slope_bps=self.min_abs_slope_bps,
            min_trend_efficiency=self.min_trend_efficiency,
            max_realized_volatility_bps=self.max_realized_volatility_bps,
        )


@dataclass(frozen=True)
class RegimeTiltReport:
    timestamp: str
    symbol: str
    primary_signal: str
    regime: str
    scale: float
    requested_before_usd: float
    requested_after_usd: float
    reason: str
    applied: bool


class PortfolioRegimeTilter:
    def __init__(self, policy: RegimeTiltPolicy | None = None) -> None:
        self.policy = policy or RegimeTiltPolicy()

    def apply(
        self,
        intents: Iterable[SymbolIntent],
        *,
        closes_by_symbol: Mapping[str, tuple[float, ...]],
        timestamp: str = "",
    ) -> tuple[tuple[SymbolIntent, ...], tuple[RegimeTiltReport, ...]]:
        adjusted: list[SymbolIntent] = []
        reports: list[RegimeTiltReport] = []
        for intent in tuple(intents):
            scaled_intent, report = self._apply_one(
                intent,
                closes=closes_by_symbol.get(intent.symbol, ()),
                timestamp=timestamp,
            )
            adjusted.append(scaled_intent)
            reports.append(report)
        return tuple(adjusted), tuple(reports)

    def _apply_one(
        self,
        intent: SymbolIntent,
        *,
        closes: tuple[float, ...],
        timestamp: str,
    ) -> tuple[SymbolIntent, RegimeTiltReport]:
        if intent.primary_signal in PROTECTED_SIGNALS:
            return intent, _report(
                timestamp=timestamp,
                intent=intent,
                regime="PROTECTED",
                scale=1.0,
                reason="protected signal",
                applied=False,
            )
        if abs(intent.target_notional_usd) <= EPSILON_NOTIONAL:
            return intent, _report(
                timestamp=timestamp,
                intent=intent,
                regime="FLAT",
                scale=1.0,
                reason="flat target",
                applied=False,
            )
        if len(closes) < self.policy.lookback:
            return intent, _report(
                timestamp=timestamp,
                intent=intent,
                regime="UNKNOWN",
                scale=1.0,
                reason="insufficient regime history",
                applied=False,
            )

        try:
            reading = read_kalman_regime(
                closes,
                symbol=intent.symbol,
                config=self.policy.kalman_config(),
            )
        except ValueError as exc:
            return intent, _report(
                timestamp=timestamp,
                intent=intent,
                regime="UNKNOWN",
                scale=1.0,
                reason=str(exc),
                applied=False,
            )

        scale, reason = self._scale_for_intent(intent, reading.regime)
        if abs(scale - 1.0) <= EPSILON_NOTIONAL:
            return intent, _report(
                timestamp=timestamp,
                intent=intent,
                regime=reading.regime.value,
                scale=scale,
                reason=reason,
                applied=False,
            )
        scaled = SymbolIntent(
            symbol=intent.symbol,
            target_notional_usd=intent.target_notional_usd * scale,
            current_notional_usd=intent.current_notional_usd,
            reason=_with_reason(intent.reason, f"regime tilt scale={scale:.3f} ({reason})"),
            primary_signal=intent.primary_signal,
            supporting_signals=(
                *intent.supporting_signals,
                f"regime_tilt={reading.regime.value}:{scale:.3f}",
            ),
            conflicting_signals=intent.conflicting_signals,
        )
        return scaled, _report(
            timestamp=timestamp,
            intent=intent,
            regime=reading.regime.value,
            scale=scale,
            reason=reason,
            applied=True,
        )

    def _scale_for_intent(
        self,
        intent: SymbolIntent,
        regime: TimeSeriesRegime,
    ) -> tuple[float, str]:
        signal_type = _signal_type(intent.primary_signal)
        direction = _target_direction(intent.target_notional_usd)
        if regime == TimeSeriesRegime.HIGH_VOLATILITY:
            return self.policy.high_volatility_scale, "high-volatility regime"
        if regime == TimeSeriesRegime.CHOP:
            if signal_type == "trend":
                return self.policy.chop_trend_scale, "chop regime dampens trend sleeve"
            if signal_type == "reversion":
                return self.policy.chop_reversion_scale, "chop regime boosts reversion sleeve"
            return 1.0, "chop regime leaves neutral signal unchanged"

        if regime in {TimeSeriesRegime.TREND_UP, TimeSeriesRegime.TREND_DOWN}:
            trend_direction = 1 if regime == TimeSeriesRegime.TREND_UP else -1
            if signal_type == "trend":
                if direction == trend_direction:
                    return self.policy.trend_aligned_scale, "trend-aligned signal"
                return self.policy.trend_counter_scale, "counter-trend signal"
            if signal_type == "reversion":
                return self.policy.trend_reversion_scale, "trend regime dampens reversion sleeve"
        return 1.0, "neutral regime tilt"


def write_regime_tilt_report_csv(
    reports: Iterable[RegimeTiltReport],
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
                "regime",
                "scale",
                "requested_before_usd",
                "requested_after_usd",
                "reason",
                "applied",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "timestamp": report.timestamp,
                    "symbol": report.symbol,
                    "primary_signal": report.primary_signal,
                    "regime": report.regime,
                    "scale": report.scale,
                    "requested_before_usd": report.requested_before_usd,
                    "requested_after_usd": report.requested_after_usd,
                    "reason": report.reason,
                    "applied": report.applied,
                }
            )


def _report(
    *,
    timestamp: str,
    intent: SymbolIntent,
    regime: str,
    scale: float,
    reason: str,
    applied: bool,
) -> RegimeTiltReport:
    return RegimeTiltReport(
        timestamp=timestamp,
        symbol=intent.symbol,
        primary_signal=intent.primary_signal,
        regime=regime,
        scale=scale,
        requested_before_usd=intent.target_notional_usd,
        requested_after_usd=intent.target_notional_usd * scale,
        reason=reason,
        applied=applied,
    )


def _signal_type(primary_signal: str) -> str:
    if primary_signal in TREND_SIGNALS:
        return "trend"
    if primary_signal in REVERSION_SIGNALS:
        return "reversion"
    return "neutral"


def _target_direction(target_notional_usd: float) -> int:
    if target_notional_usd > EPSILON_NOTIONAL:
        return 1
    if target_notional_usd < -EPSILON_NOTIONAL:
        return -1
    return 0


def _with_reason(existing: str, addition: str) -> str:
    if not existing:
        return addition
    return f"{existing}; {addition}"


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")
