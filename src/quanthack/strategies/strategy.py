from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC
from enum import StrEnum
from math import exp, isfinite, log, sqrt
from typing import Protocol

from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import QuoteSnapshot
from quanthack.strategies.time_series import (
    KalmanTrendConfig,
    TimeSeriesRegime,
    TimeSeriesRegimeReading,
    read_kalman_regime,
)
from quanthack.trading.risk import Side, TradeRequest


STRATEGY_NAMES = (
    "simple_momentum",
    "session_momentum",
    "multi_horizon_momentum",
    "autocorrelation_regime",
    "intraday_seasonality",
    "conditional_seasonality",
    "ma_crossover",
    "macd_momentum",
    "asset_adaptive_macd",
    "macd_conditional_fallback",
    "macd_squeeze_complement",
    "breakout",
    "session_breakout",
    "volatility_squeeze",
    "dual_squeeze",
    "asset_adaptive_dual_squeeze",
    "range_expansion_trend",
    "trend_pullback",
    "exhaustion_reversal",
    "liquidity_sweep_reversal",
    "fixing_reversal",
    "kalman_trend",
    "quality_trend",
    "champion_ensemble",
    "mean_reversion",
    "crypto_mean_reversion",
    "regime_switch",
    "alpha_router",
    "crypto_trend_reversion",
    "usd_pressure_router",
    "relative_strength",
    "cross_rate_reversion",
)
EPSILON_NOTIONAL = 1e-9
CRYPTO_MACD_MIN_HISTOGRAM_BPS = 5.0
CRYPTO_MACD_MIN_MACD_BPS = 2.0
CRYPTO_MACD_MIN_TREND_EFFICIENCY = 0.25
CRYPTO_MACD_MAX_HOLDING_PERIOD = 10


class StrategyAction(StrEnum):
    ENTER = "ENTER"
    HOLD = "HOLD"
    EXIT = "EXIT"
    REVERSE = "REVERSE"
    NO_ACTION = "NO_ACTION"


class RegimeState(StrEnum):
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    FLAT = "FLAT"


class SignalDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SignalHorizon(StrEnum):
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"


@dataclass(frozen=True)
class SignalAttribution:
    primary_signal: str
    supporting_signals: tuple[str, ...] = ()
    conflicting_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyDecision:
    action: StrategyAction
    symbol: str
    target_notional_usd: float
    reason: str
    diagnostics: tuple[tuple[str, float | str], ...] = ()
    primary_signal: str = "strategy"
    supporting_signals: tuple[str, ...] = ()
    conflicting_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("strategy decision symbol is required")
        if not isfinite(self.target_notional_usd):
            raise ValueError("target_notional_usd must be finite")

    @property
    def is_trade_intent(self) -> bool:
        return self.action in {
            StrategyAction.ENTER,
            StrategyAction.EXIT,
            StrategyAction.REVERSE,
        }

    def to_trade_request(self) -> TradeRequest | None:
        if abs(self.target_notional_usd) <= EPSILON_NOTIONAL:
            return None
        side = Side.BUY if self.target_notional_usd > 0 else Side.SELL
        return TradeRequest(
            symbol=self.symbol,
            side=side,
            target_notional_usd=abs(self.target_notional_usd),
            reason=f"{self.action.value}: {self.reason}",
        )


@dataclass(frozen=True)
class StrategySignal:
    strategy_name: str
    symbol: str
    direction: SignalDirection
    confidence: float
    expected_edge_bps: float
    cost_bps: float
    weight: float
    horizon: SignalHorizon
    reason: str
    diagnostics: tuple[tuple[str, float | str], ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name is required")
        _validate_symbol(self.symbol)
        if not 0 <= self.confidence <= 1 or not isfinite(self.confidence):
            raise ValueError("signal confidence must be finite and between 0 and 1")
        _validate_non_negative_finite("expected_edge_bps", self.expected_edge_bps)
        _validate_non_negative_finite("cost_bps", self.cost_bps)
        _validate_non_negative_finite("weight", self.weight)

    @property
    def signed_score(self) -> float:
        return _signal_direction_sign(self.direction) * self.confidence * self.weight

    @property
    def edge_after_cost_bps(self) -> float:
        return self.expected_edge_bps - self.cost_bps


class Strategy(Protocol):
    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        """Return a trade request, or None when the strategy wants no new entry."""
        ...


@dataclass(frozen=True)
class MomentumConfig:
    symbol: str = "EURUSD"
    lookback: int = 5
    threshold_bps: float = 8.0
    exit_threshold_bps: float = 0.0
    min_trend_efficiency: float = 0.40
    min_normalized_momentum: float = 0.0
    forex_allowed_utc_hours: tuple[int, ...] | None = None
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "fixed"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = None
    min_trade_notional_usd: float = 0.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = None

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2 prices")
        _validate_non_negative_finite("threshold_bps", self.threshold_bps)
        _validate_non_negative_finite("exit_threshold_bps", self.exit_threshold_bps)
        if self.exit_threshold_bps > self.threshold_bps:
            raise ValueError("exit_threshold_bps cannot exceed threshold_bps")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite(
            "min_normalized_momentum",
            self.min_normalized_momentum,
        )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MomentumReading:
    first_price: float
    last_price: float
    cumulative_log_return: float
    move_bps: float
    realized_volatility: float
    normalized_momentum: float
    trend_efficiency: float
    path_log_return: float


@dataclass(frozen=True)
class MultiHorizonMomentumConfig:
    symbol: str = "EURUSD"
    fast_lookback: int = 6
    slow_lookback: int = 24
    volatility_lookback: int = 12
    baseline_volatility_lookback: int = 48
    min_fast_move_bps: float = 2.0
    min_slow_move_bps: float = 5.0
    exit_slow_move_bps: float = 1.0
    min_trend_efficiency: float = 0.25
    min_normalized_slow_momentum: float = 0.0
    min_volatility_ratio: float = 0.35
    max_volatility_ratio: float = 2.50
    min_realized_volatility_bps: float = 0.1
    max_realized_volatility_bps: float = 80.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 2
    max_holding_period: int = 24
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    @property
    def lookback(self) -> int:
        return max(self.slow_lookback, self.baseline_volatility_lookback) + 1

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.fast_lookback < 2:
            raise ValueError("fast_lookback must be at least 2")
        if self.slow_lookback <= self.fast_lookback:
            raise ValueError("slow_lookback must be greater than fast_lookback")
        if self.volatility_lookback < 2:
            raise ValueError("volatility_lookback must be at least 2")
        if self.baseline_volatility_lookback < self.volatility_lookback:
            raise ValueError(
                "baseline_volatility_lookback must be at least volatility_lookback"
            )
        _validate_positive_finite("min_fast_move_bps", self.min_fast_move_bps)
        _validate_positive_finite("min_slow_move_bps", self.min_slow_move_bps)
        _validate_non_negative_finite("exit_slow_move_bps", self.exit_slow_move_bps)
        if self.exit_slow_move_bps >= self.min_slow_move_bps:
            raise ValueError("exit_slow_move_bps must be below min_slow_move_bps")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite(
            "min_normalized_slow_momentum",
            self.min_normalized_slow_momentum,
        )
        _validate_non_negative_finite("min_volatility_ratio", self.min_volatility_ratio)
        _validate_positive_finite("max_volatility_ratio", self.max_volatility_ratio)
        if self.max_volatility_ratio <= self.min_volatility_ratio:
            raise ValueError("max_volatility_ratio must exceed min_volatility_ratio")
        _validate_non_negative_finite(
            "min_realized_volatility_bps",
            self.min_realized_volatility_bps,
        )
        _validate_positive_finite(
            "max_realized_volatility_bps",
            self.max_realized_volatility_bps,
        )
        if self.max_realized_volatility_bps <= self.min_realized_volatility_bps:
            raise ValueError(
                "max_realized_volatility_bps must exceed min_realized_volatility_bps"
            )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MultiHorizonMomentumReading:
    fast_move_bps: float
    slow_move_bps: float
    realized_volatility: float
    realized_volatility_bps: float
    baseline_volatility_bps: float
    volatility_ratio: float
    normalized_slow_momentum: float
    trend_efficiency: float
    expected_edge_bps: float
    signal_direction: SignalDirection
    session_allowed: bool
    utc_hour: int | None


@dataclass(frozen=True)
class AutocorrelationRegimeConfig:
    symbol: str = "EURUSD"
    lookback: int = 32
    signal_lookback: int = 6
    min_abs_autocorrelation: float = 0.18
    exit_abs_autocorrelation: float = 0.05
    min_momentum_bps: float = 4.0
    min_trend_efficiency: float = 0.20
    min_reversion_zscore: float = 0.80
    min_reversion_move_bps: float = 2.0
    min_expected_edge_bps: float = 3.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14, 15, 16, 17
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14, 15, 16, 17
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 16
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 6:
            raise ValueError("autocorrelation lookback must be at least 6 prices")
        if self.signal_lookback < 2:
            raise ValueError("signal_lookback must be at least 2 returns")
        if self.signal_lookback >= self.lookback:
            raise ValueError("signal_lookback must be below lookback")
        if not 0 <= self.min_abs_autocorrelation <= 1:
            raise ValueError("min_abs_autocorrelation must be between 0 and 1")
        if not 0 <= self.exit_abs_autocorrelation <= 1:
            raise ValueError("exit_abs_autocorrelation must be between 0 and 1")
        if self.exit_abs_autocorrelation > self.min_abs_autocorrelation:
            raise ValueError("exit_abs_autocorrelation cannot exceed entry threshold")
        _validate_non_negative_finite("min_momentum_bps", self.min_momentum_bps)
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite("min_reversion_zscore", self.min_reversion_zscore)
        _validate_non_negative_finite("min_reversion_move_bps", self.min_reversion_move_bps)
        _validate_non_negative_finite(
            "min_expected_edge_bps",
            self.min_expected_edge_bps,
        )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class AutocorrelationRegimeReading:
    latest_price: float
    lag1_autocorrelation: float
    signal_move_bps: float
    zscore: float
    realized_volatility: float
    realized_volatility_bps: float
    trend_efficiency: float
    mode: str
    expected_edge_bps: float
    signal_direction: SignalDirection


@dataclass(frozen=True)
class IntradaySeasonalityConfig:
    symbol: str = "EURUSD"
    period_bars: int = 96
    lookback_periods: int = 5
    signal_mode: str = "momentum"
    entry_threshold_bps: float = 0.5
    exit_threshold_bps: float = 0.1
    min_consistency: float = 0.60
    forex_allowed_utc_hours: tuple[int, ...] | None = None
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 50_000.0
    min_trade_notional_usd: float = 1_000.0
    max_holding_period: int = 8
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    @property
    def lookback(self) -> int:
        return (self.period_bars * self.lookback_periods) + 2

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.period_bars < 2:
            raise ValueError("period_bars must be at least 2")
        if self.lookback_periods < 1:
            raise ValueError("lookback_periods must be at least 1")
        normalized_mode = self.signal_mode.strip().lower()
        if normalized_mode not in {"momentum", "reversal"}:
            raise ValueError("signal_mode must be momentum or reversal")
        object.__setattr__(self, "signal_mode", normalized_mode)
        _validate_positive_finite("entry_threshold_bps", self.entry_threshold_bps)
        _validate_non_negative_finite("exit_threshold_bps", self.exit_threshold_bps)
        if self.exit_threshold_bps > self.entry_threshold_bps:
            raise ValueError("exit_threshold_bps cannot exceed entry_threshold_bps")
        if not 0 <= self.min_consistency <= 1:
            raise ValueError("min_consistency must be between 0 and 1")
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class IntradaySeasonalityReading:
    mean_return_bps: float
    realized_volatility: float
    consistency: float
    positive_samples: int
    negative_samples: int
    sample_count: int
    session_allowed: bool
    utc_hour: int | None


@dataclass(frozen=True)
class ConditionalSeasonalityConfig:
    symbol: str = "EURUSD"
    period_bars: int = 96
    lookback_periods: int = 4
    horizon_bars: int = 4
    momentum_lookback: int = 4
    momentum_threshold_bps: float = 2.0
    signal_mode: str = "reversal"
    min_samples: int = 3
    entry_threshold_bps: float = 10.0
    exit_threshold_bps: float = 1.0
    min_consistency: float = 0.67
    min_abs_tstat: float = 1.50
    forex_allowed_utc_hours: tuple[int, ...] | None = None
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 50_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 4
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    @property
    def lookback(self) -> int:
        return (self.period_bars * self.lookback_periods) + self.momentum_lookback + 1

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.period_bars < 2:
            raise ValueError("period_bars must be at least 2")
        if self.lookback_periods < 1:
            raise ValueError("lookback_periods must be at least 1")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be at least 1")
        if self.horizon_bars >= self.period_bars:
            raise ValueError("horizon_bars must be below period_bars")
        if self.momentum_lookback < 1:
            raise ValueError("momentum_lookback must be at least 1")
        _validate_non_negative_finite(
            "momentum_threshold_bps",
            self.momentum_threshold_bps,
        )
        normalized_mode = self.signal_mode.strip().lower()
        if normalized_mode not in {"momentum", "reversal"}:
            raise ValueError("signal_mode must be momentum or reversal")
        object.__setattr__(self, "signal_mode", normalized_mode)
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if self.min_samples > self.lookback_periods:
            raise ValueError("min_samples cannot exceed lookback_periods")
        _validate_positive_finite("entry_threshold_bps", self.entry_threshold_bps)
        _validate_non_negative_finite("exit_threshold_bps", self.exit_threshold_bps)
        if self.exit_threshold_bps > self.entry_threshold_bps:
            raise ValueError("exit_threshold_bps cannot exceed entry_threshold_bps")
        if not 0 <= self.min_consistency <= 1:
            raise ValueError("min_consistency must be between 0 and 1")
        _validate_non_negative_finite("min_abs_tstat", self.min_abs_tstat)
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class ConditionalSeasonalityReading:
    current_condition: str
    current_momentum_bps: float
    mean_forward_return_bps: float
    realized_volatility: float
    realized_volatility_bps: float
    tstat: float
    consistency: float
    positive_samples: int
    negative_samples: int
    sample_count: int
    expected_edge_bps: float
    signal_direction: SignalDirection
    session_allowed: bool
    utc_hour: int | None


@dataclass(frozen=True)
class MovingAverageCrossoverConfig:
    symbol: str = "EURUSD"
    fast_window: int = 3
    slow_window: int = 8
    min_separation_bps: float = 2.0
    exit_separation_bps: float = 0.5
    min_trend_efficiency: float = 0.20
    target_notional_usd: float = 50_000.0
    position_sizing: str = "fixed"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = None
    min_trade_notional_usd: float = 0.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = None

    @property
    def lookback(self) -> int:
        return self.slow_window

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.fast_window < 2:
            raise ValueError("fast_window must be at least 2 prices")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        _validate_positive_finite("min_separation_bps", self.min_separation_bps)
        _validate_non_negative_finite("exit_separation_bps", self.exit_separation_bps)
        if self.exit_separation_bps >= self.min_separation_bps:
            raise ValueError("exit_separation_bps must be below min_separation_bps")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MovingAverageCrossoverReading:
    fast_average: float
    slow_average: float
    previous_fast_average: float | None
    previous_slow_average: float | None
    last_price: float
    separation_bps: float
    previous_separation_bps: float | None
    crossed_direction: SignalDirection
    realized_volatility: float
    trend_efficiency: float


@dataclass(frozen=True)
class MacdMomentumConfig:
    symbol: str = "EURUSD"
    fast_window: int = 12
    slow_window: int = 26
    signal_window: int = 9
    min_histogram_bps: float = 1.5
    exit_histogram_bps: float = 0.25
    min_macd_bps: float = 0.5
    min_histogram_slope_bps: float = 0.0
    min_trend_efficiency: float = 0.10
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 24
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    @property
    def lookback(self) -> int:
        return self.slow_window + self.signal_window + 2

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.fast_window < 2:
            raise ValueError("MACD fast_window must be at least 2")
        if self.slow_window <= self.fast_window:
            raise ValueError("MACD slow_window must be greater than fast_window")
        if self.signal_window < 2:
            raise ValueError("MACD signal_window must be at least 2")
        _validate_positive_finite("min_histogram_bps", self.min_histogram_bps)
        _validate_non_negative_finite("exit_histogram_bps", self.exit_histogram_bps)
        if self.exit_histogram_bps >= self.min_histogram_bps:
            raise ValueError("exit_histogram_bps must be below min_histogram_bps")
        _validate_non_negative_finite("min_macd_bps", self.min_macd_bps)
        _validate_non_negative_finite(
            "min_histogram_slope_bps",
            self.min_histogram_slope_bps,
        )
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MacdMomentumReading:
    fast_ema: float
    slow_ema: float
    macd: float
    signal: float
    histogram: float
    previous_histogram: float
    macd_bps: float
    signal_bps: float
    histogram_bps: float
    previous_histogram_bps: float
    histogram_slope_bps: float
    crossed_direction: SignalDirection
    last_price: float
    realized_volatility: float
    realized_volatility_bps: float
    trend_efficiency: float
    session_allowed: bool
    utc_hour: int | None


@dataclass(frozen=True)
class BreakoutConfig:
    symbol: str = "EURUSD"
    lookback: int = 8
    breakout_buffer_bps: float = 2.0
    exit_buffer_bps: float = 1.0
    min_channel_width_bps: float = 2.0
    target_notional_usd: float = 50_000.0
    position_sizing: str = "fixed"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = None
    min_trade_notional_usd: float = 0.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = None

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 3:
            raise ValueError("breakout lookback must be at least 3 prices")
        _validate_non_negative_finite("breakout_buffer_bps", self.breakout_buffer_bps)
        _validate_non_negative_finite("exit_buffer_bps", self.exit_buffer_bps)
        _validate_non_negative_finite("min_channel_width_bps", self.min_channel_width_bps)
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class VolatilitySqueezeConfig:
    symbol: str = "EURUSD"
    lookback: int = 24
    squeeze_window: int = 8
    band_stdev_multiplier: float = 2.0
    breakout_buffer_bps: float = 2.5
    exit_buffer_bps: float = 1.0
    max_squeeze_ratio: float = 0.50
    min_prior_volatility_bps: float = 0.5
    min_band_width_bps: float = 1.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    max_holding_period: int = 24
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _validate_volatility_squeeze_shape(
            lookback=self.lookback,
            squeeze_window=self.squeeze_window,
            label="volatility squeeze",
        )
        _validate_positive_finite(
            "band_stdev_multiplier",
            self.band_stdev_multiplier,
        )
        _validate_non_negative_finite("breakout_buffer_bps", self.breakout_buffer_bps)
        _validate_non_negative_finite("exit_buffer_bps", self.exit_buffer_bps)
        _validate_positive_finite("max_squeeze_ratio", self.max_squeeze_ratio)
        _validate_non_negative_finite(
            "min_prior_volatility_bps",
            self.min_prior_volatility_bps,
        )
        _validate_non_negative_finite("min_band_width_bps", self.min_band_width_bps)
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class SessionBreakoutConfig:
    symbol: str = "EURUSD"
    lookback: int = 8
    breakout_buffer_bps: float = 2.0
    exit_buffer_bps: float = 1.0
    min_channel_width_bps: float = 2.0
    min_expected_edge_bps: float = 0.0
    min_holding_period: int = 0
    min_realized_volatility_bps: float = 1.5
    max_realized_volatility_bps: float = 80.0
    allowed_utc_hours: tuple[int, ...] = (12, 13, 14, 15)
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    require_regime_confirmation: bool = False
    regime_lookback: int = 80
    regime_min_abs_slope_bps: float = 0.75
    regime_min_trend_efficiency: float = 0.25
    regime_max_realized_volatility_bps: float = 120.0
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = None

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 3:
            raise ValueError("session breakout lookback must be at least 3 prices")
        _validate_non_negative_finite("breakout_buffer_bps", self.breakout_buffer_bps)
        _validate_non_negative_finite("exit_buffer_bps", self.exit_buffer_bps)
        _validate_non_negative_finite("min_channel_width_bps", self.min_channel_width_bps)
        _validate_non_negative_finite("min_expected_edge_bps", self.min_expected_edge_bps)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        _validate_non_negative_finite(
            "min_realized_volatility_bps",
            self.min_realized_volatility_bps,
        )
        _validate_positive_finite(
            "max_realized_volatility_bps",
            self.max_realized_volatility_bps,
        )
        if self.max_realized_volatility_bps < self.min_realized_volatility_bps:
            raise ValueError(
                "max_realized_volatility_bps cannot be below min_realized_volatility_bps"
            )
        normalized_hours = tuple(int(hour) for hour in self.allowed_utc_hours)
        if any(hour < 0 or hour > 23 for hour in normalized_hours):
            raise ValueError("allowed_utc_hours must contain hours between 0 and 23")
        object.__setattr__(self, "allowed_utc_hours", normalized_hours)
        if self.metal_allowed_utc_hours is not None:
            normalized_metal_hours = tuple(int(hour) for hour in self.metal_allowed_utc_hours)
            if any(hour < 0 or hour > 23 for hour in normalized_metal_hours):
                raise ValueError(
                    "metal_allowed_utc_hours must contain hours between 0 and 23"
                )
            object.__setattr__(
                self,
                "metal_allowed_utc_hours",
                normalized_metal_hours,
            )
        if self.regime_lookback < 5:
            raise ValueError("regime_lookback must be at least 5")
        _validate_non_negative_finite(
            "regime_min_abs_slope_bps",
            self.regime_min_abs_slope_bps,
        )
        if not 0 <= self.regime_min_trend_efficiency <= 1:
            raise ValueError("regime_min_trend_efficiency must be between 0 and 1")
        _validate_positive_finite(
            "regime_max_realized_volatility_bps",
            self.regime_max_realized_volatility_bps,
        )
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class BreakoutReading:
    upper_band: float
    lower_band: float
    last_price: float
    channel_width_bps: float
    breakout_bps: float
    position_in_channel: float
    realized_volatility: float


@dataclass(frozen=True)
class VolatilitySqueezeReading:
    mean_price: float
    upper_band: float
    lower_band: float
    last_price: float
    band_width_bps: float
    breakout_bps: float
    recent_volatility_bps: float
    prior_volatility_bps: float
    squeeze_ratio: float
    realized_volatility: float


@dataclass(frozen=True)
class DualSqueezeConfig:
    symbol: str = "EURUSD"
    lookback: int = 16
    squeeze_window: int = 5
    band_stdev_multiplier: float = 1.8
    breakout_buffer_bps: float = 2.5
    exit_buffer_bps: float = 1.0
    max_squeeze_ratio: float = 0.55
    min_prior_volatility_bps: float = 0.5
    min_band_width_bps: float = 1.0
    confirmation_lookback: int = 32
    confirmation_squeeze_window: int = 10
    confirmation_band_stdev_multiplier: float = 2.0
    confirmation_max_squeeze_ratio: float = 0.70
    confirmation_mode: str = "squeeze_bias"
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    max_holding_period: int = 12
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _validate_volatility_squeeze_shape(
            lookback=self.lookback,
            squeeze_window=self.squeeze_window,
            label="dual squeeze fast",
        )
        _validate_volatility_squeeze_shape(
            lookback=self.confirmation_lookback,
            squeeze_window=self.confirmation_squeeze_window,
            label="dual squeeze confirmation",
        )
        _validate_positive_finite(
            "band_stdev_multiplier",
            self.band_stdev_multiplier,
        )
        _validate_positive_finite(
            "confirmation_band_stdev_multiplier",
            self.confirmation_band_stdev_multiplier,
        )
        _validate_non_negative_finite("breakout_buffer_bps", self.breakout_buffer_bps)
        _validate_non_negative_finite("exit_buffer_bps", self.exit_buffer_bps)
        _validate_positive_finite("max_squeeze_ratio", self.max_squeeze_ratio)
        _validate_positive_finite(
            "confirmation_max_squeeze_ratio",
            self.confirmation_max_squeeze_ratio,
        )
        _validate_non_negative_finite(
            "min_prior_volatility_bps",
            self.min_prior_volatility_bps,
        )
        _validate_non_negative_finite("min_band_width_bps", self.min_band_width_bps)
        if self.confirmation_mode not in {
            "bias",
            "breakout",
            "not_opposite",
            "squeeze_bias",
        }:
            raise ValueError(
                "confirmation_mode must be bias, breakout, not_opposite, or squeeze_bias"
            )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class AssetAdaptiveDualSqueezeConfig:
    symbol: str = "EURUSD"
    base_lookback: int = 14
    base_squeeze_window: int = 4
    base_band_stdev_multiplier: float = 1.8
    base_breakout_buffer_bps: float = 2.5
    base_max_squeeze_ratio: float = 0.60
    base_confirmation_lookback: int = 24
    base_confirmation_squeeze_window: int = 8
    base_confirmation_band_stdev_multiplier: float = 2.0
    base_confirmation_max_squeeze_ratio: float = 0.70
    base_confirmation_mode: str = "squeeze_bias"
    metal_lookback: int = 12
    metal_squeeze_window: int = 4
    metal_band_stdev_multiplier: float = 1.7
    metal_breakout_buffer_bps: float = 2.0
    metal_max_squeeze_ratio: float = 0.70
    metal_confirmation_lookback: int = 20
    metal_confirmation_squeeze_window: int = 6
    metal_confirmation_band_stdev_multiplier: float = 2.0
    metal_confirmation_max_squeeze_ratio: float = 0.80
    metal_confirmation_mode: str = "squeeze_bias"
    exit_buffer_bps: float = 1.0
    min_prior_volatility_bps: float = 0.5
    min_band_width_bps: float = 1.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    max_holding_period: int = 12
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        self.dual_config_for_asset_class(AssetClass.FOREX)
        self.dual_config_for_asset_class(AssetClass.METAL)
        self.dual_config_for_asset_class(AssetClass.CRYPTO)

    @property
    def lookback(self) -> int:
        return max(
            self.base_lookback,
            self.base_confirmation_lookback,
            self.metal_lookback,
            self.metal_confirmation_lookback,
        )

    def dual_config_for_asset_class(self, asset_class: AssetClass) -> DualSqueezeConfig:
        if asset_class == AssetClass.METAL:
            return self._metal_dual_config()
        return self._base_dual_config()

    def _base_dual_config(self) -> DualSqueezeConfig:
        return self._build_dual_config(
            lookback=self.base_lookback,
            squeeze_window=self.base_squeeze_window,
            band_stdev_multiplier=self.base_band_stdev_multiplier,
            breakout_buffer_bps=self.base_breakout_buffer_bps,
            max_squeeze_ratio=self.base_max_squeeze_ratio,
            confirmation_lookback=self.base_confirmation_lookback,
            confirmation_squeeze_window=self.base_confirmation_squeeze_window,
            confirmation_band_stdev_multiplier=(
                self.base_confirmation_band_stdev_multiplier
            ),
            confirmation_max_squeeze_ratio=self.base_confirmation_max_squeeze_ratio,
            confirmation_mode=self.base_confirmation_mode,
        )

    def _metal_dual_config(self) -> DualSqueezeConfig:
        return self._build_dual_config(
            lookback=self.metal_lookback,
            squeeze_window=self.metal_squeeze_window,
            band_stdev_multiplier=self.metal_band_stdev_multiplier,
            breakout_buffer_bps=self.metal_breakout_buffer_bps,
            max_squeeze_ratio=self.metal_max_squeeze_ratio,
            confirmation_lookback=self.metal_confirmation_lookback,
            confirmation_squeeze_window=self.metal_confirmation_squeeze_window,
            confirmation_band_stdev_multiplier=(
                self.metal_confirmation_band_stdev_multiplier
            ),
            confirmation_max_squeeze_ratio=self.metal_confirmation_max_squeeze_ratio,
            confirmation_mode=self.metal_confirmation_mode,
        )

    def _build_dual_config(
        self,
        *,
        lookback: int,
        squeeze_window: int,
        band_stdev_multiplier: float,
        breakout_buffer_bps: float,
        max_squeeze_ratio: float,
        confirmation_lookback: int,
        confirmation_squeeze_window: int,
        confirmation_band_stdev_multiplier: float,
        confirmation_max_squeeze_ratio: float,
        confirmation_mode: str,
    ) -> DualSqueezeConfig:
        return DualSqueezeConfig(
            symbol=self.symbol,
            lookback=lookback,
            squeeze_window=squeeze_window,
            band_stdev_multiplier=band_stdev_multiplier,
            breakout_buffer_bps=breakout_buffer_bps,
            exit_buffer_bps=self.exit_buffer_bps,
            max_squeeze_ratio=max_squeeze_ratio,
            min_prior_volatility_bps=self.min_prior_volatility_bps,
            min_band_width_bps=self.min_band_width_bps,
            confirmation_lookback=confirmation_lookback,
            confirmation_squeeze_window=confirmation_squeeze_window,
            confirmation_band_stdev_multiplier=confirmation_band_stdev_multiplier,
            confirmation_max_squeeze_ratio=confirmation_max_squeeze_ratio,
            confirmation_mode=confirmation_mode,
            forex_allowed_utc_hours=self.forex_allowed_utc_hours,
            metal_allowed_utc_hours=self.metal_allowed_utc_hours,
            crypto_allowed_utc_hours=self.crypto_allowed_utc_hours,
            target_notional_usd=self.target_notional_usd,
            position_sizing=self.position_sizing,
            target_volatility=self.target_volatility,
            volatility_floor=self.volatility_floor,
            max_target_notional_usd=self.max_target_notional_usd,
            min_trade_notional_usd=self.min_trade_notional_usd,
            max_holding_period=self.max_holding_period,
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
            cost_buffer=self.cost_buffer,
            max_spread_bps=self.max_spread_bps,
        )


@dataclass(frozen=True)
class RangeExpansionTrendConfig:
    symbol: str = "EURUSD"
    lookback: int = 40
    trigger_window: int = 4
    min_trigger_move_bps: float = 10.0
    exit_trigger_move_bps: float = 1.0
    min_range_break_bps: float = 3.0
    min_expansion_zscore: float = 2.5
    max_expansion_zscore: float = 8.0
    min_trend_efficiency: float = 0.65
    min_baseline_volatility_bps: float = 0.2
    max_trigger_volatility_bps: float = 80.0
    min_expected_edge_bps: float = 6.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 6
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 8:
            raise ValueError("range expansion lookback must be at least 8 prices")
        if self.trigger_window < 2:
            raise ValueError("range expansion trigger_window must be at least 2 returns")
        if self.lookback < self.trigger_window + 5:
            raise ValueError(
                "range expansion lookback must leave baseline prices before trigger_window"
            )
        _validate_non_negative_finite("min_trigger_move_bps", self.min_trigger_move_bps)
        _validate_non_negative_finite("exit_trigger_move_bps", self.exit_trigger_move_bps)
        if self.exit_trigger_move_bps > self.min_trigger_move_bps:
            raise ValueError("exit_trigger_move_bps cannot exceed min_trigger_move_bps")
        _validate_non_negative_finite("min_range_break_bps", self.min_range_break_bps)
        _validate_non_negative_finite("min_expansion_zscore", self.min_expansion_zscore)
        _validate_positive_finite("max_expansion_zscore", self.max_expansion_zscore)
        if self.max_expansion_zscore < self.min_expansion_zscore:
            raise ValueError("max_expansion_zscore cannot be below min_expansion_zscore")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite(
            "min_baseline_volatility_bps",
            self.min_baseline_volatility_bps,
        )
        _validate_positive_finite(
            "max_trigger_volatility_bps",
            self.max_trigger_volatility_bps,
        )
        _validate_non_negative_finite("min_expected_edge_bps", self.min_expected_edge_bps)
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class RangeExpansionTrendReading:
    baseline_high: float
    baseline_low: float
    trigger_start_price: float
    last_price: float
    trigger_move_bps: float
    range_break_bps: float
    baseline_volatility_bps: float
    trigger_volatility_bps: float
    expansion_zscore: float
    trend_efficiency: float
    expected_edge_bps: float
    realized_volatility: float
    signal_direction: SignalDirection


@dataclass(frozen=True)
class DualSqueezeReading:
    fast: VolatilitySqueezeReading
    confirmation: VolatilitySqueezeReading
    confirmation_passed: bool
    confirmation_reason: str


@dataclass(frozen=True)
class SessionBreakoutReading:
    breakout: BreakoutReading
    realized_volatility_bps: float
    utc_hour: int | None
    session_allowed: bool


@dataclass(frozen=True)
class TrendPullbackConfig:
    symbol: str = "EURUSD"
    lookback: int = 32
    pullback_window: int = 4
    min_trend_bps: float = 8.0
    min_trend_efficiency: float = 0.35
    min_pullback_bps: float = 1.0
    max_pullback_bps: float = 12.0
    min_resume_bps: float = 1.0
    min_expected_edge_bps: float = 3.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 2
    max_holding_period: int = 24
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 8:
            raise ValueError("trend pullback lookback must be at least 8 prices")
        if self.pullback_window < 2:
            raise ValueError("pullback_window must be at least 2 prices")
        if self.lookback < self.pullback_window + 4:
            raise ValueError("lookback must leave enough pre-pullback trend history")
        _validate_non_negative_finite("min_trend_bps", self.min_trend_bps)
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite("min_pullback_bps", self.min_pullback_bps)
        _validate_positive_finite("max_pullback_bps", self.max_pullback_bps)
        if self.max_pullback_bps < self.min_pullback_bps:
            raise ValueError("max_pullback_bps cannot be below min_pullback_bps")
        _validate_non_negative_finite("min_resume_bps", self.min_resume_bps)
        _validate_non_negative_finite(
            "min_expected_edge_bps",
            self.min_expected_edge_bps,
        )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class TrendPullbackReading:
    anchor_price: float
    previous_price: float
    last_price: float
    trend_move_bps: float
    pullback_bps: float
    resume_bps: float
    expected_edge_bps: float
    trend_efficiency: float
    realized_volatility: float
    signal_direction: SignalDirection


@dataclass(frozen=True)
class ExhaustionReversalConfig:
    symbol: str = "EURUSD"
    lookback: int = 32
    shock_window: int = 4
    min_shock_bps: float = 12.0
    min_reversal_bps: float = 2.0
    min_shock_zscore: float = 1.5
    min_shock_efficiency: float = 0.65
    min_expected_edge_bps: float = 3.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 12
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 8:
            raise ValueError("exhaustion reversal lookback must be at least 8 prices")
        if self.shock_window < 2:
            raise ValueError("shock_window must be at least 2 prices")
        if self.lookback < self.shock_window + 5:
            raise ValueError("lookback must leave enough pre-shock baseline history")
        _validate_non_negative_finite("min_shock_bps", self.min_shock_bps)
        _validate_non_negative_finite("min_reversal_bps", self.min_reversal_bps)
        _validate_positive_finite("min_shock_zscore", self.min_shock_zscore)
        if not 0 <= self.min_shock_efficiency <= 1:
            raise ValueError("min_shock_efficiency must be between 0 and 1")
        _validate_non_negative_finite(
            "min_expected_edge_bps",
            self.min_expected_edge_bps,
        )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class ExhaustionReversalReading:
    shock_start_price: float
    previous_price: float
    last_price: float
    shock_move_bps: float
    reversal_bps: float
    shock_zscore: float
    shock_efficiency: float
    baseline_volatility_bps: float
    realized_volatility: float
    expected_edge_bps: float
    signal_direction: SignalDirection


@dataclass(frozen=True)
class FixingReversalConfig:
    symbol: str = "EURUSD"
    pre_fix_lookback: int = 4
    min_pre_fix_move_bps: float = 8.0
    max_pre_fix_move_bps: float = 80.0
    min_pre_fix_efficiency: float = 0.35
    min_reversal_confirmation_bps: float = 1.5
    min_expected_edge_bps: float = 3.0
    min_realized_volatility_bps: float = 0.25
    max_realized_volatility_bps: float = 120.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (14,)
    metal_allowed_utc_hours: tuple[int, ...] | None = (14,)
    crypto_allowed_utc_hours: tuple[int, ...] | None = ()
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 4
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    @property
    def lookback(self) -> int:
        return self.pre_fix_lookback + 2

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.pre_fix_lookback < 2:
            raise ValueError("pre_fix_lookback must be at least 2 returns")
        _validate_non_negative_finite(
            "min_pre_fix_move_bps",
            self.min_pre_fix_move_bps,
        )
        _validate_positive_finite(
            "max_pre_fix_move_bps",
            self.max_pre_fix_move_bps,
        )
        if self.max_pre_fix_move_bps <= self.min_pre_fix_move_bps:
            raise ValueError("max_pre_fix_move_bps must exceed min_pre_fix_move_bps")
        if not 0 <= self.min_pre_fix_efficiency <= 1:
            raise ValueError("min_pre_fix_efficiency must be between 0 and 1")
        _validate_non_negative_finite(
            "min_reversal_confirmation_bps",
            self.min_reversal_confirmation_bps,
        )
        _validate_non_negative_finite(
            "min_expected_edge_bps",
            self.min_expected_edge_bps,
        )
        _validate_non_negative_finite(
            "min_realized_volatility_bps",
            self.min_realized_volatility_bps,
        )
        _validate_positive_finite(
            "max_realized_volatility_bps",
            self.max_realized_volatility_bps,
        )
        if self.max_realized_volatility_bps < self.min_realized_volatility_bps:
            raise ValueError(
                "max_realized_volatility_bps cannot be below min_realized_volatility_bps"
            )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class FixingReversalReading:
    anchor_price: float
    previous_price: float
    last_price: float
    pre_fix_move_bps: float
    confirmation_bps: float
    pre_fix_efficiency: float
    realized_volatility_bps: float
    expected_edge_bps: float
    signal_direction: SignalDirection


@dataclass(frozen=True)
class MeanReversionConfig:
    symbol: str = "EURUSD"
    lookback: int = 5
    entry_zscore: float = 1.0
    exit_zscore: float = 0.25
    max_trend_bps: float = 15.0
    min_stdev_bps: float = 0.05
    target_notional_usd: float = 50_000.0
    position_sizing: str = "fixed"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = None
    min_trade_notional_usd: float = 0.0
    max_holding_period: int = 20
    stop_zscore: float = 4.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = None

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 3:
            raise ValueError("lookback must be at least 3 prices")
        _validate_positive_finite("entry_zscore", self.entry_zscore)
        _validate_non_negative_finite("exit_zscore", self.exit_zscore)
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError("entry_zscore must be greater than exit_zscore")
        _validate_non_negative_finite("max_trend_bps", self.max_trend_bps)
        _validate_non_negative_finite("min_stdev_bps", self.min_stdev_bps)
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.stop_zscore <= self.entry_zscore or not isfinite(self.stop_zscore):
            raise ValueError("stop_zscore must be finite and above entry_zscore")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MeanReversionReading:
    mean_price: float
    last_price: float
    stdev_price: float
    residual: float
    zscore: float
    deviation_bps: float
    trend_strength_bps: float
    trend_efficiency: float
    estimated_half_life: float | None = None


@dataclass(frozen=True)
class RegimeConfig:
    symbol: str = "EURUSD"
    lookback: int = 10
    momentum_min_move_bps: float = 8.0
    momentum_min_score: float = 1.0
    momentum_min_efficiency: float = 0.60
    mean_reversion_min_abs_zscore: float = 1.0
    mean_reversion_max_trend_bps: float = 8.0
    mean_reversion_max_efficiency: float = 0.50
    max_spread_bps: float | None = 10.0
    hysteresis_bars: int = 2

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 4:
            raise ValueError("regime lookback must be at least 4 prices")
        _validate_non_negative_finite("momentum_min_move_bps", self.momentum_min_move_bps)
        _validate_non_negative_finite("momentum_min_score", self.momentum_min_score)
        if not 0 <= self.momentum_min_efficiency <= 1:
            raise ValueError("momentum_min_efficiency must be between 0 and 1")
        _validate_non_negative_finite(
            "mean_reversion_min_abs_zscore",
            self.mean_reversion_min_abs_zscore,
        )
        _validate_non_negative_finite(
            "mean_reversion_max_trend_bps",
            self.mean_reversion_max_trend_bps,
        )
        if not 0 <= self.mean_reversion_max_efficiency <= 1:
            raise ValueError("mean_reversion_max_efficiency must be between 0 and 1")
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)
        if self.hysteresis_bars < 1:
            raise ValueError("hysteresis_bars must be at least 1")


@dataclass(frozen=True)
class RegimeReading:
    selected: RegimeState
    candidate: RegimeState
    confidence: float
    reason: str
    momentum_move_bps: float
    momentum_score: float
    momentum_efficiency: float
    reversion_zscore: float
    reversion_trend_bps: float
    reversion_efficiency: float
    spread_bps: float


@dataclass(frozen=True)
class KalmanTrendStrategyConfig:
    symbol: str = "EURUSD"
    lookback: int = 80
    process_noise: float = 1e-6
    observation_noise: float = 1e-4
    min_abs_slope_bps: float = 0.25
    min_trend_efficiency: float = 0.20
    max_realized_volatility_bps: float = 120.0
    expected_holding_bars: int = 6
    min_expected_edge_bps: float = 5.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 2
    max_holding_period: int = 32
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 5:
            raise ValueError("kalman trend lookback must be at least 5")
        _validate_positive_finite("process_noise", self.process_noise)
        _validate_positive_finite("observation_noise", self.observation_noise)
        _validate_non_negative_finite("min_abs_slope_bps", self.min_abs_slope_bps)
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_positive_finite(
            "max_realized_volatility_bps",
            self.max_realized_volatility_bps,
        )
        if self.expected_holding_bars < 1:
            raise ValueError("expected_holding_bars must be at least 1")
        _validate_non_negative_finite(
            "min_expected_edge_bps",
            self.min_expected_edge_bps,
        )
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class KalmanTrendStrategyReading:
    regime_reading: TimeSeriesRegimeReading
    expected_edge_bps: float
    signal_direction: SignalDirection
    session_allowed: bool
    utc_hour: int | None


@dataclass(frozen=True)
class ChampionEnsembleConfig:
    symbol: str = "EURUSD"
    lookback: int = 80
    target_notional_usd: float = 50_000.0
    max_target_notional_usd: float = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    entry_score: float = 0.50
    exit_score: float = 0.15
    strong_lead_score: float = 0.50
    min_signal_confidence: float = 0.50
    cost_buffer: float = 1.20
    max_spread_bps: float | None = 10.0
    kalman_trend_weight: float = 0.70
    asset_adaptive_dual_squeeze_weight: float = 0.30
    dual_squeeze_weight: float = 0.0
    trend_pullback_weight: float = 0.0
    fixing_reversal_weight: float = 0.0
    macd_momentum_weight: float = 0.0
    conflict_penalty: float = 0.70
    slippage_bps: float = 1.0
    fee_bps: float = 0.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        if self.max_target_notional_usd < self.target_notional_usd:
            raise ValueError("max_target_notional_usd must be at least target_notional_usd")
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_positive_finite("entry_score", self.entry_score)
        _validate_non_negative_finite("exit_score", self.exit_score)
        if self.exit_score >= self.entry_score:
            raise ValueError("entry_score must be greater than exit_score")
        _validate_non_negative_finite("strong_lead_score", self.strong_lead_score)
        if self.strong_lead_score > self.entry_score:
            raise ValueError("strong_lead_score cannot exceed entry_score")
        if not 0 <= self.min_signal_confidence <= 1:
            raise ValueError("min_signal_confidence must be between 0 and 1")
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)
        _validate_non_negative_finite("kalman_trend_weight", self.kalman_trend_weight)
        _validate_non_negative_finite(
            "asset_adaptive_dual_squeeze_weight",
            self.asset_adaptive_dual_squeeze_weight,
        )
        _validate_non_negative_finite("dual_squeeze_weight", self.dual_squeeze_weight)
        _validate_non_negative_finite("trend_pullback_weight", self.trend_pullback_weight)
        _validate_non_negative_finite(
            "fixing_reversal_weight",
            self.fixing_reversal_weight,
        )
        _validate_non_negative_finite(
            "macd_momentum_weight",
            self.macd_momentum_weight,
        )
        total_weight = (
            self.kalman_trend_weight
            + self.asset_adaptive_dual_squeeze_weight
            + self.dual_squeeze_weight
            + self.trend_pullback_weight
            + self.fixing_reversal_weight
            + self.macd_momentum_weight
        )
        if total_weight <= 0:
            raise ValueError("at least one champion ensemble weight must be positive")
        if not 0 <= self.conflict_penalty <= 1:
            raise ValueError("conflict_penalty must be between 0 and 1")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)


@dataclass(frozen=True)
class AlphaRouterConfig:
    symbol: str = "EURUSD"
    target_notional_usd: float = 50_000.0
    max_target_notional_usd: float = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    entry_score: float = 0.35
    exit_score: float = 0.15
    min_signal_confidence: float = 0.20
    cost_buffer: float = 1.20
    max_spread_bps: float | None = 10.0
    momentum_weight: float = 0.30
    moving_average_weight: float = 0.15
    breakout_weight: float = 0.15
    session_breakout_weight: float = 0.25
    macd_momentum_weight: float = 0.0
    kalman_trend_weight: float = 0.0
    volatility_squeeze_weight: float = 0.0
    dual_squeeze_weight: float = 0.0
    exhaustion_reversal_weight: float = 0.0
    mean_reversion_weight: float = 0.35
    relative_strength_weight: float = 0.0
    cross_rate_weight: float = 0.0
    conflict_penalty: float = 0.50
    primary_signal_override_enabled: bool = True
    primary_signal_min_confidence: float = 0.90
    primary_signal_min_edge_bps: float = 4.0
    adaptive_weighting_enabled: bool = True
    adaptive_regime_lookback: int = 80
    chop_mean_reversion_multiplier: float = 1.20
    chop_trend_signal_multiplier: float = 0.75
    trend_aligned_signal_multiplier: float = 1.20
    trend_counter_signal_multiplier: float = 0.65
    metal_mean_reversion_multiplier: float = 1.25
    metal_raw_breakout_multiplier: float = 0.60
    volatility_regime_enabled: bool = True
    volatility_regime_lookback: int = 24
    high_volatility_ratio: float = 1.50
    low_volatility_ratio: float = 0.60
    high_volatility_reversion_multiplier: float = 1.15
    high_volatility_trend_multiplier: float = 0.90
    low_volatility_reversion_multiplier: float = 0.95
    low_volatility_trend_multiplier: float = 1.05
    min_high_volatility_bps: float = 0.0
    relative_strength_min_score_dispersion: float = 0.75
    relative_strength_min_target_trend_efficiency: float = 0.20
    ml_enabled: bool = False
    ml_weight: float = 0.30
    ml_lookback: int = 8
    ml_train_window: int = 80
    ml_min_train_samples: int = 12
    ml_learning_rate: float = 0.15
    ml_epochs: int = 80
    ml_l2: float = 0.001
    ml_label_threshold_bps: float = 2.0
    ml_entry_probability: float = 0.58
    ml_min_edge_bps: float = 3.0
    ml_min_training_accuracy: float = 0.55
    ml_min_samples_for_trade: int = 20
    ml_min_expected_edge_bps: float = 3.0
    ml_disable_on_negative_signed_return: bool = True

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        if self.max_target_notional_usd < self.target_notional_usd:
            raise ValueError("max_target_notional_usd must be at least target_notional_usd")
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_positive_finite("entry_score", self.entry_score)
        _validate_non_negative_finite("exit_score", self.exit_score)
        if self.exit_score >= self.entry_score:
            raise ValueError("entry_score must be greater than exit_score")
        if not 0 <= self.min_signal_confidence <= 1:
            raise ValueError("min_signal_confidence must be between 0 and 1")
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)
        _validate_non_negative_finite("momentum_weight", self.momentum_weight)
        _validate_non_negative_finite("moving_average_weight", self.moving_average_weight)
        _validate_non_negative_finite("breakout_weight", self.breakout_weight)
        _validate_non_negative_finite("session_breakout_weight", self.session_breakout_weight)
        _validate_non_negative_finite("macd_momentum_weight", self.macd_momentum_weight)
        _validate_non_negative_finite("kalman_trend_weight", self.kalman_trend_weight)
        _validate_non_negative_finite(
            "volatility_squeeze_weight",
            self.volatility_squeeze_weight,
        )
        _validate_non_negative_finite(
            "dual_squeeze_weight",
            self.dual_squeeze_weight,
        )
        _validate_non_negative_finite(
            "exhaustion_reversal_weight",
            self.exhaustion_reversal_weight,
        )
        _validate_non_negative_finite("mean_reversion_weight", self.mean_reversion_weight)
        _validate_non_negative_finite(
            "relative_strength_weight",
            self.relative_strength_weight,
        )
        _validate_non_negative_finite("cross_rate_weight", self.cross_rate_weight)
        _validate_non_negative_finite("ml_weight", self.ml_weight)
        if not 0 <= self.primary_signal_min_confidence <= 1:
            raise ValueError("primary_signal_min_confidence must be between 0 and 1")
        _validate_non_negative_finite(
            "primary_signal_min_edge_bps",
            self.primary_signal_min_edge_bps,
        )
        if self.adaptive_regime_lookback < 5:
            raise ValueError("adaptive_regime_lookback must be at least 5")
        _validate_non_negative_finite(
            "chop_mean_reversion_multiplier",
            self.chop_mean_reversion_multiplier,
        )
        _validate_non_negative_finite(
            "chop_trend_signal_multiplier",
            self.chop_trend_signal_multiplier,
        )
        _validate_non_negative_finite(
            "trend_aligned_signal_multiplier",
            self.trend_aligned_signal_multiplier,
        )
        _validate_non_negative_finite(
            "trend_counter_signal_multiplier",
            self.trend_counter_signal_multiplier,
        )
        _validate_non_negative_finite(
            "metal_mean_reversion_multiplier",
            self.metal_mean_reversion_multiplier,
        )
        _validate_non_negative_finite(
            "metal_raw_breakout_multiplier",
            self.metal_raw_breakout_multiplier,
        )
        if self.volatility_regime_lookback < 3:
            raise ValueError("volatility_regime_lookback must be at least 3")
        _validate_positive_finite("high_volatility_ratio", self.high_volatility_ratio)
        _validate_positive_finite("low_volatility_ratio", self.low_volatility_ratio)
        if self.low_volatility_ratio >= self.high_volatility_ratio:
            raise ValueError("low_volatility_ratio must be below high_volatility_ratio")
        _validate_non_negative_finite(
            "high_volatility_reversion_multiplier",
            self.high_volatility_reversion_multiplier,
        )
        _validate_non_negative_finite(
            "high_volatility_trend_multiplier",
            self.high_volatility_trend_multiplier,
        )
        _validate_non_negative_finite(
            "low_volatility_reversion_multiplier",
            self.low_volatility_reversion_multiplier,
        )
        _validate_non_negative_finite(
            "low_volatility_trend_multiplier",
            self.low_volatility_trend_multiplier,
        )
        _validate_non_negative_finite(
            "min_high_volatility_bps",
            self.min_high_volatility_bps,
        )
        _validate_non_negative_finite(
            "relative_strength_min_score_dispersion",
            self.relative_strength_min_score_dispersion,
        )
        if not 0 <= self.relative_strength_min_target_trend_efficiency <= 1:
            raise ValueError(
                "relative_strength_min_target_trend_efficiency must be between 0 and 1"
            )
        total_weight = (
            self.momentum_weight
            + self.moving_average_weight
            + self.breakout_weight
            + self.session_breakout_weight
            + self.macd_momentum_weight
            + self.kalman_trend_weight
            + self.volatility_squeeze_weight
            + self.dual_squeeze_weight
            + self.exhaustion_reversal_weight
            + self.mean_reversion_weight
            + self.relative_strength_weight
            + self.cross_rate_weight
            + (self.ml_weight if self.ml_enabled else 0.0)
        )
        if total_weight <= 0:
            raise ValueError("at least one alpha weight must be positive")
        if not 0 <= self.conflict_penalty <= 1:
            raise ValueError("conflict_penalty must be between 0 and 1")
        if self.ml_lookback < 3:
            raise ValueError("ml_lookback must be at least 3 prices")
        if self.ml_train_window < 1:
            raise ValueError("ml_train_window must be at least 1")
        if self.ml_min_train_samples < 1:
            raise ValueError("ml_min_train_samples must be at least 1")
        _validate_positive_finite("ml_learning_rate", self.ml_learning_rate)
        if self.ml_epochs < 1:
            raise ValueError("ml_epochs must be at least 1")
        _validate_non_negative_finite("ml_l2", self.ml_l2)
        _validate_non_negative_finite("ml_label_threshold_bps", self.ml_label_threshold_bps)
        if not 0.5 < self.ml_entry_probability < 1.0:
            raise ValueError("ml_entry_probability must be between 0.5 and 1.0")
        _validate_non_negative_finite("ml_min_edge_bps", self.ml_min_edge_bps)
        if not 0 <= self.ml_min_training_accuracy <= 1:
            raise ValueError("ml_min_training_accuracy must be between 0 and 1")
        if self.ml_min_samples_for_trade < 1:
            raise ValueError("ml_min_samples_for_trade must be at least 1")
        _validate_non_negative_finite(
            "ml_min_expected_edge_bps",
            self.ml_min_expected_edge_bps,
        )


@dataclass(frozen=True)
class MLAlphaReading:
    probability_up: float
    probability_down: float
    score: float
    sample_count: int
    training_accuracy: float
    training_signed_return_bps: float
    expected_edge_bps: float
    features: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class VolatilityRegimeReading:
    state: str
    short_realized_volatility_bps: float
    long_realized_volatility_bps: float
    ratio: float


@dataclass(frozen=True)
class UsdPressureConfig:
    symbol: str = "EURUSD"
    lookback: int = 8
    pressure_threshold_bps: float = 2.0
    component_threshold_bps: float = 0.5
    min_target_volatility_bps: float = 0.0
    min_component_symbols: int = 3
    min_confirming_symbols: int = 2
    exit_on_conflict: bool = True
    max_spread_bps: float | None = None

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 3:
            raise ValueError("USD pressure lookback must be at least 3 prices")
        _validate_non_negative_finite("pressure_threshold_bps", self.pressure_threshold_bps)
        _validate_non_negative_finite("component_threshold_bps", self.component_threshold_bps)
        _validate_non_negative_finite(
            "min_target_volatility_bps",
            self.min_target_volatility_bps,
        )
        if self.min_component_symbols < 1:
            raise ValueError("min_component_symbols must be at least 1")
        if self.min_confirming_symbols < 1:
            raise ValueError("min_confirming_symbols must be at least 1")
        if self.min_confirming_symbols > self.min_component_symbols:
            raise ValueError("min_confirming_symbols cannot exceed min_component_symbols")
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class UsdPressureReading:
    pressure_bps: float
    component_count: int
    confirming_symbols: int
    conflicting_symbols: int
    components: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RelativeStrengthConfig:
    symbol: str = "EURUSD"
    lookback: int = 12
    entry_zscore: float = 0.75
    exit_zscore: float = 0.25
    min_component_symbols: int = 4
    require_asset_class_confirmation: bool = False
    asset_class_entry_zscore: float = 0.35
    asset_class_min_symbols: int = 2
    require_metal_trend_confirmation: bool = False
    metal_trend_min_move_bps: float = 2.0
    metal_trend_min_efficiency: float = 0.20
    min_score_dispersion: float = 0.0
    min_target_trend_efficiency: float = 0.0
    min_abs_move_bps: float = 0.5
    volatility_floor_bps: float = 0.25
    target_notional_usd: float = 50_000.0
    max_target_notional_usd: float = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 3:
            raise ValueError("relative strength lookback must be at least 3 prices")
        _validate_positive_finite("entry_zscore", self.entry_zscore)
        _validate_non_negative_finite("exit_zscore", self.exit_zscore)
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError("entry_zscore must be greater than exit_zscore")
        if self.min_component_symbols < 2:
            raise ValueError("min_component_symbols must be at least 2")
        _validate_non_negative_finite(
            "asset_class_entry_zscore",
            self.asset_class_entry_zscore,
        )
        if self.asset_class_min_symbols < 2:
            raise ValueError("asset_class_min_symbols must be at least 2")
        _validate_non_negative_finite(
            "metal_trend_min_move_bps",
            self.metal_trend_min_move_bps,
        )
        if not 0 <= self.metal_trend_min_efficiency <= 1:
            raise ValueError("metal_trend_min_efficiency must be between 0 and 1")
        _validate_non_negative_finite("min_score_dispersion", self.min_score_dispersion)
        if not 0 <= self.min_target_trend_efficiency <= 1:
            raise ValueError("min_target_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite("min_abs_move_bps", self.min_abs_move_bps)
        _validate_positive_finite("volatility_floor_bps", self.volatility_floor_bps)
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        if self.max_target_notional_usd < self.target_notional_usd:
            raise ValueError("max_target_notional_usd must be at least target_notional_usd")
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class RelativeStrengthReading:
    target_score: float
    target_rank: int
    component_count: int
    relative_zscore: float
    score_dispersion: float
    move_bps: float
    realized_volatility_bps: float
    trend_efficiency: float
    strongest_symbol: str
    strongest_score: float
    weakest_symbol: str
    weakest_score: float
    components: tuple[tuple[str, float], ...]
    asset_class_zscore: float | None = None
    asset_class_rank: int | None = None
    asset_class_component_count: int = 0
    asset_class_components: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CrossRateReversionConfig:
    symbol: str = "EURUSD"
    allowed_symbols: tuple[str, ...] = ()
    lookback: int = 12
    entry_zscore: float = 1.0
    exit_zscore: float = 0.25
    min_abs_deviation_bps: float = 1.0
    max_abs_deviation_bps: float = 80.0
    min_synthetic_components: int = 2
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    max_holding_period: int = 24
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        normalized_allowed: list[str] = []
        for allowed_symbol in self.allowed_symbols:
            instrument = instrument_for(allowed_symbol)
            if instrument.asset_class != AssetClass.FOREX:
                raise ValueError("cross-rate allowed_symbols must contain only FX symbols")
            normalized_allowed.append(instrument.symbol)
        object.__setattr__(self, "allowed_symbols", tuple(normalized_allowed))
        if self.lookback < 4:
            raise ValueError("cross-rate lookback must be at least 4 prices")
        _validate_positive_finite("entry_zscore", self.entry_zscore)
        _validate_non_negative_finite("exit_zscore", self.exit_zscore)
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError("entry_zscore must be greater than exit_zscore")
        _validate_non_negative_finite(
            "min_abs_deviation_bps",
            self.min_abs_deviation_bps,
        )
        _validate_positive_finite(
            "max_abs_deviation_bps",
            self.max_abs_deviation_bps,
        )
        if self.max_abs_deviation_bps <= self.min_abs_deviation_bps:
            raise ValueError(
                "max_abs_deviation_bps must be greater than min_abs_deviation_bps"
            )
        if self.min_synthetic_components < 1:
            raise ValueError("min_synthetic_components must be at least 1")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class CrossRateReversionReading:
    target_price: float
    synthetic_price: float
    deviation_bps: float
    mean_deviation_bps: float
    stdev_deviation_bps: float
    zscore: float
    realized_volatility: float
    component_symbols: tuple[str, ...]
    currency_path: tuple[str, ...]


@dataclass(frozen=True)
class LiquiditySweepReversalConfig:
    symbol: str = "EURUSD"
    lookback: int = 32
    min_sweep_bps: float = 2.0
    reentry_buffer_bps: float = 0.25
    min_range_width_bps: float = 4.0
    max_sweep_bps: float = 80.0
    max_trend_efficiency: float = 0.75
    min_expected_edge_bps: float = 2.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 60_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 8
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.lookback < 6:
            raise ValueError("liquidity sweep lookback must be at least 6 prices")
        _validate_positive_finite("min_sweep_bps", self.min_sweep_bps)
        _validate_non_negative_finite("reentry_buffer_bps", self.reentry_buffer_bps)
        _validate_positive_finite("min_range_width_bps", self.min_range_width_bps)
        _validate_positive_finite("max_sweep_bps", self.max_sweep_bps)
        if self.max_sweep_bps <= self.min_sweep_bps:
            raise ValueError("max_sweep_bps must exceed min_sweep_bps")
        if not 0 <= self.max_trend_efficiency <= 1:
            raise ValueError("max_trend_efficiency must be between 0 and 1")
        _validate_non_negative_finite("min_expected_edge_bps", self.min_expected_edge_bps)
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite("max_target_notional_usd", self.max_target_notional_usd)
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class LiquiditySweepReversalReading:
    prior_high: float
    prior_low: float
    midpoint_price: float
    previous_price: float
    last_price: float
    range_width_bps: float
    sweep_bps: float
    reentry_bps: float
    expected_edge_bps: float
    realized_volatility: float
    trend_efficiency: float
    signal_direction: SignalDirection


@dataclass(frozen=True)
class QualityTrendConfig:
    symbol: str = "EURUSD"
    kalman_lookback: int = 80
    kalman_process_noise: float = 1e-6
    kalman_observation_noise: float = 1e-4
    kalman_min_abs_slope_bps: float = 0.25
    kalman_min_trend_efficiency: float = 0.20
    kalman_max_realized_volatility_bps: float = 120.0
    kalman_expected_holding_bars: int = 6
    kalman_min_expected_edge_bps: float = 5.0
    macd_fast_window: int = 6
    macd_slow_window: int = 18
    macd_signal_window: int = 5
    macd_min_histogram_bps: float = 2.0
    macd_exit_histogram_bps: float = 0.25
    macd_min_macd_bps: float = 1.0
    macd_min_histogram_slope_bps: float = 0.0
    macd_min_trend_efficiency: float = 0.20
    min_combined_confidence: float = 0.30
    exit_combined_confidence: float = 0.10
    min_expected_edge_bps: float = 2.0
    forex_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14
    )
    metal_allowed_utc_hours: tuple[int, ...] | None = (
        10, 11, 12, 13, 14
    )
    crypto_allowed_utc_hours: tuple[int, ...] | None = None
    target_notional_usd: float = 50_000.0
    position_sizing: str = "volatility"
    target_volatility: float = 0.002
    volatility_floor: float = 0.00001
    max_target_notional_usd: float | None = 75_000.0
    min_trade_notional_usd: float = 1_000.0
    min_holding_period: int = 1
    max_holding_period: int = 16
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = 10.0

    @property
    def lookback(self) -> int:
        macd_lookback = self.macd_slow_window + self.macd_signal_window + 2
        return max(self.kalman_lookback, macd_lookback)

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if self.kalman_lookback < 5:
            raise ValueError("quality trend kalman_lookback must be at least 5")
        _validate_positive_finite("kalman_process_noise", self.kalman_process_noise)
        _validate_positive_finite(
            "kalman_observation_noise",
            self.kalman_observation_noise,
        )
        _validate_non_negative_finite(
            "kalman_min_abs_slope_bps",
            self.kalman_min_abs_slope_bps,
        )
        if not 0 <= self.kalman_min_trend_efficiency <= 1:
            raise ValueError("kalman_min_trend_efficiency must be between 0 and 1")
        _validate_positive_finite(
            "kalman_max_realized_volatility_bps",
            self.kalman_max_realized_volatility_bps,
        )
        if self.kalman_expected_holding_bars < 1:
            raise ValueError("kalman_expected_holding_bars must be at least 1")
        _validate_non_negative_finite(
            "kalman_min_expected_edge_bps",
            self.kalman_min_expected_edge_bps,
        )
        if self.macd_fast_window < 2:
            raise ValueError("quality trend macd_fast_window must be at least 2")
        if self.macd_slow_window <= self.macd_fast_window:
            raise ValueError("quality trend macd_slow_window must exceed fast window")
        if self.macd_signal_window < 2:
            raise ValueError("quality trend macd_signal_window must be at least 2")
        _validate_positive_finite(
            "macd_min_histogram_bps",
            self.macd_min_histogram_bps,
        )
        _validate_non_negative_finite(
            "macd_exit_histogram_bps",
            self.macd_exit_histogram_bps,
        )
        if self.macd_exit_histogram_bps >= self.macd_min_histogram_bps:
            raise ValueError("macd_exit_histogram_bps must be below histogram threshold")
        _validate_non_negative_finite("macd_min_macd_bps", self.macd_min_macd_bps)
        _validate_non_negative_finite(
            "macd_min_histogram_slope_bps",
            self.macd_min_histogram_slope_bps,
        )
        if not 0 <= self.macd_min_trend_efficiency <= 1:
            raise ValueError("macd_min_trend_efficiency must be between 0 and 1")
        if not 0 <= self.min_combined_confidence <= 1:
            raise ValueError("min_combined_confidence must be between 0 and 1")
        if not 0 <= self.exit_combined_confidence < self.min_combined_confidence:
            raise ValueError(
                "exit_combined_confidence must be below min_combined_confidence"
            )
        _validate_non_negative_finite("min_expected_edge_bps", self.min_expected_edge_bps)
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")
        _validate_positive_finite("target_notional_usd", self.target_notional_usd)
        _validate_position_sizing(self.position_sizing)
        _validate_positive_finite("target_volatility", self.target_volatility)
        _validate_positive_finite("volatility_floor", self.volatility_floor)
        if self.max_target_notional_usd is not None:
            _validate_positive_finite(
                "max_target_notional_usd",
                self.max_target_notional_usd,
            )
        _validate_non_negative_finite("min_trade_notional_usd", self.min_trade_notional_usd)
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.max_holding_period < self.min_holding_period:
            raise ValueError("max_holding_period cannot be below min_holding_period")
        _validate_non_negative_finite("slippage_bps", self.slippage_bps)
        _validate_non_negative_finite("fee_bps", self.fee_bps)
        _validate_positive_finite("cost_buffer", self.cost_buffer)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MacdConditionalFallbackConfig:
    symbol: str = "EURUSD"
    conditional_notional_multiplier: float = 0.25
    macd_inactive_reason_keywords: tuple[str, ...] = ("below",)
    max_spread_bps: float | None = None

    @property
    def lookback(self) -> int:
        return max(MacdMomentumConfig(symbol=self.symbol).lookback, ConditionalSeasonalityConfig(symbol=self.symbol).lookback)

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _validate_non_negative_finite(
            "conditional_notional_multiplier",
            self.conditional_notional_multiplier,
        )
        if self.conditional_notional_multiplier > 1:
            raise ValueError("conditional_notional_multiplier must be at most 1")
        keywords = tuple(str(value).strip().lower() for value in self.macd_inactive_reason_keywords)
        if not keywords or any(not value for value in keywords):
            raise ValueError("macd_inactive_reason_keywords cannot be empty")
        object.__setattr__(self, "macd_inactive_reason_keywords", keywords)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class MacdSqueezeComplementConfig:
    symbol: str = "EURUSD"
    squeeze_notional_multiplier: float = 1.0
    macd_inactive_reason_keywords: tuple[str, ...] = ("below",)
    max_spread_bps: float | None = None

    @property
    def lookback(self) -> int:
        return max(
            MacdMomentumConfig(symbol=self.symbol).lookback,
            VolatilitySqueezeConfig(symbol=self.symbol).lookback,
        )

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _validate_non_negative_finite(
            "squeeze_notional_multiplier",
            self.squeeze_notional_multiplier,
        )
        if self.squeeze_notional_multiplier > 1:
            raise ValueError("squeeze_notional_multiplier must be at most 1")
        keywords = tuple(
            str(value).strip().lower()
            for value in self.macd_inactive_reason_keywords
        )
        if any(not value for value in keywords):
            raise ValueError("macd_inactive_reason_keywords cannot contain empty values")
        object.__setattr__(self, "macd_inactive_reason_keywords", keywords)
        if self.max_spread_bps is not None:
            _validate_positive_finite("max_spread_bps", self.max_spread_bps)


@dataclass(frozen=True)
class QualityTrendReading:
    macd: MacdMomentumReading
    kalman: KalmanTrendStrategyReading
    macd_direction: SignalDirection
    kalman_direction: SignalDirection
    aligned_direction: SignalDirection
    macd_confidence: float
    kalman_confidence: float
    combined_confidence: float
    expected_edge_bps: float


StrategyConfig = (
    MomentumConfig
    | MultiHorizonMomentumConfig
    | AutocorrelationRegimeConfig
    | IntradaySeasonalityConfig
    | ConditionalSeasonalityConfig
    | MovingAverageCrossoverConfig
    | MacdMomentumConfig
    | MacdConditionalFallbackConfig
    | MacdSqueezeComplementConfig
    | BreakoutConfig
    | VolatilitySqueezeConfig
    | DualSqueezeConfig
    | AssetAdaptiveDualSqueezeConfig
    | RangeExpansionTrendConfig
    | SessionBreakoutConfig
    | TrendPullbackConfig
    | ExhaustionReversalConfig
    | LiquiditySweepReversalConfig
    | FixingReversalConfig
    | MeanReversionConfig
    | RegimeConfig
    | KalmanTrendStrategyConfig
    | QualityTrendConfig
    | ChampionEnsembleConfig
    | AlphaRouterConfig
    | UsdPressureConfig
    | RelativeStrengthConfig
    | CrossRateReversionConfig
)


class SimpleMomentumStrategy:
    def __init__(self, config: MomentumConfig | None = None) -> None:
        self.config = config or MomentumConfig()

    def read_momentum(self, prices: Sequence[float]) -> MomentumReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        log_returns = _log_returns(recent_prices)
        cumulative_log_return = sum(log_returns)
        path_log_return = sum(abs(value) for value in log_returns)
        trend_efficiency = (
            abs(cumulative_log_return) / path_log_return
            if path_log_return > 0
            else 0.0
        )
        realized_volatility = _population_stdev(log_returns)
        normalized_momentum = (
            cumulative_log_return / max(realized_volatility, self.config.volatility_floor)
            if cumulative_log_return != 0
            else 0.0
        )

        return MomentumReading(
            first_price=recent_prices[0],
            last_price=recent_prices[-1],
            cumulative_log_return=cumulative_log_return,
            move_bps=cumulative_log_return * 10_000,
            realized_volatility=realized_volatility,
            normalized_momentum=normalized_momentum,
            trend_efficiency=trend_efficiency,
            path_log_return=path_log_return,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_momentum(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for momentum reading",
            )

        diagnostics = _momentum_diagnostics(reading)
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.EXIT
            )
            return _decision(
                action,
                self.config.symbol,
                0.0,
                f"outside momentum UTC hours ({allowed}); current hour={utc_hour}",
                diagnostics
                + (
                    ("utc_hour", "n/a" if utc_hour is None else float(utc_hour)),
                    ("session_allowed", "no"),
                ),
            )
        if reading.move_bps == 0:
            return self._flat_or_exit(
                current_direction=current_direction,
                current_notional_usd=current_notional_usd,
                reason="zero momentum",
                diagnostics=diagnostics,
            )

        signal_direction = 1 if reading.move_bps > 0 else -1
        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)

        if current_direction == 0:
            if not passed:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
            )

        if signal_direction != current_direction:
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite momentum signal; {reason}",
                    diagnostics,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "momentum direction no longer supports current position",
                diagnostics,
            )

        if abs(reading.move_bps) <= self.config.exit_threshold_bps:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"momentum {abs(reading.move_bps):.1f} bps is at or below "
                    f"exit threshold {self.config.exit_threshold_bps:.1f} bps"
                ),
                diagnostics,
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"momentum still supports current position after {holding_period} bars",
            diagnostics,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: MomentumReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        abs_move = abs(reading.move_bps)
        abs_score = abs(reading.normalized_momentum)
        has_enough_move = abs_move >= self.config.threshold_bps
        has_enough_score = (
            self.config.min_normalized_momentum > 0
            and abs_score >= self.config.min_normalized_momentum
        )
        if not has_enough_move and not has_enough_score:
            return (
                False,
                (
                    f"momentum {abs_move:.1f} bps below entry threshold "
                    f"{self.config.threshold_bps:.1f} bps"
                ),
            )
        if reading.trend_efficiency < self.config.min_trend_efficiency:
            return (
                False,
                (
                    f"trend efficiency {reading.trend_efficiency:.2f} below "
                    f"{self.config.min_trend_efficiency:.2f}"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_move,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        return (
            True,
            (
                f"{self.config.lookback}-price momentum {reading.move_bps:.1f} bps, "
                f"efficiency={reading.trend_efficiency:.2f}, "
                f"score={reading.normalized_momentum:.2f}"
            ),
        )

    def _flat_or_exit(
        self,
        *,
        current_direction: int,
        current_notional_usd: float,
        reason: str,
        diagnostics: tuple[tuple[str, float | str], ...],
    ) -> StrategyDecision:
        if current_direction == 0:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                reason,
                diagnostics,
            )
        return _decision(
            StrategyAction.EXIT,
            self.config.symbol,
            0.0,
            reason,
            diagnostics,
        )

    def _sized_notional(self, reading: MomentumReading) -> float:
        confidence = _bounded_confidence(
            abs(reading.move_bps),
            max(self.config.threshold_bps, 1e-12),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        asset_class = instrument_for(self.config.symbol).asset_class
        if asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours


class MultiHorizonMomentumStrategy:
    def __init__(self, config: MultiHorizonMomentumConfig | None = None) -> None:
        self.config = config or MultiHorizonMomentumConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_multi_horizon_momentum(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> MultiHorizonMomentumReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        fast_prices = recent_prices[-(self.config.fast_lookback + 1):]
        slow_prices = recent_prices[-(self.config.slow_lookback + 1):]
        volatility_prices = recent_prices[-(self.config.volatility_lookback + 1):]
        baseline_prices = recent_prices[-(self.config.baseline_volatility_lookback + 1):]

        fast_log_return = log(fast_prices[-1] / fast_prices[0])
        slow_log_return = log(slow_prices[-1] / slow_prices[0])
        slow_log_returns = _log_returns(slow_prices)
        slow_path_log_return = sum(abs(value) for value in slow_log_returns)
        trend_efficiency = (
            abs(slow_log_return) / slow_path_log_return
            if slow_path_log_return > 0
            else 0.0
        )
        recent_volatility = _population_stdev(_log_returns(volatility_prices))
        baseline_volatility = _population_stdev(_log_returns(baseline_prices))
        volatility_ratio = recent_volatility / max(
            baseline_volatility,
            self.config.volatility_floor,
        )
        normalized_slow_momentum = (
            slow_log_return / max(recent_volatility, self.config.volatility_floor)
            if slow_log_return != 0
            else 0.0
        )
        fast_direction = _signed_threshold_direction(
            fast_log_return * 10_000,
            self.config.min_fast_move_bps,
        )
        slow_direction = _signed_threshold_direction(
            slow_log_return * 10_000,
            self.config.min_slow_move_bps,
        )
        direction = (
            SignalDirection.LONG
            if fast_direction == slow_direction == 1
            else SignalDirection.SHORT
            if fast_direction == slow_direction == -1
            else SignalDirection.FLAT
        )
        expected_edge_bps = (
            min(abs(fast_log_return), abs(slow_log_return)) * 10_000
            if direction != SignalDirection.FLAT
            else 0.0
        )
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        return MultiHorizonMomentumReading(
            fast_move_bps=fast_log_return * 10_000,
            slow_move_bps=slow_log_return * 10_000,
            realized_volatility=recent_volatility,
            realized_volatility_bps=recent_volatility * 10_000,
            baseline_volatility_bps=baseline_volatility * 10_000,
            volatility_ratio=volatility_ratio,
            normalized_slow_momentum=normalized_slow_momentum,
            trend_efficiency=trend_efficiency,
            expected_edge_bps=expected_edge_bps,
            signal_direction=direction,
            session_allowed=session_allowed,
            utc_hour=utc_hour,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_multi_horizon_momentum(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for multi-horizon momentum reading",
                primary_signal="multi_horizon_momentum",
            )

        diagnostics = _multi_horizon_momentum_diagnostics(reading)
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "multi-horizon momentum max holding period "
                    f"{self.config.max_holding_period} bars reached"
                ),
                diagnostics,
                primary_signal="multi_horizon_momentum",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)
        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="multi_horizon_momentum",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="multi_horizon_momentum",
            )

        if signal_direction == 0 or abs(reading.slow_move_bps) <= self.config.exit_slow_move_bps:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"multi-horizon momentum alignment faded: {reason}",
                diagnostics,
                primary_signal="multi_horizon_momentum",
            )

        if signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite multi-horizon momentum seen but minimum holding "
                        f"period {self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="multi_horizon_momentum",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite multi-horizon momentum; {reason}",
                    diagnostics,
                    primary_signal="multi_horizon_momentum",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"opposite multi-horizon momentum but entry blocked; {reason}",
                diagnostics,
                primary_signal="multi_horizon_momentum",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"multi-horizon momentum still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="multi_horizon_momentum",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: MultiHorizonMomentumReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        if not reading.session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                (
                    "outside multi-horizon momentum UTC hours "
                    f"({allowed}); current hour={reading.utc_hour}"
                ),
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return (
                False,
                (
                    "fast and slow momentum are not aligned "
                    f"(fast={reading.fast_move_bps:.1f} bps, "
                    f"slow={reading.slow_move_bps:.1f} bps)"
                ),
            )
        if reading.trend_efficiency < self.config.min_trend_efficiency:
            return (
                False,
                (
                    f"trend efficiency {reading.trend_efficiency:.2f} below "
                    f"{self.config.min_trend_efficiency:.2f}"
                ),
            )
        if (
            abs(reading.normalized_slow_momentum)
            < self.config.min_normalized_slow_momentum
        ):
            return (
                False,
                (
                    "normalized slow momentum "
                    f"{abs(reading.normalized_slow_momentum):.2f} below "
                    f"{self.config.min_normalized_slow_momentum:.2f}"
                ),
            )
        if reading.realized_volatility_bps < self.config.min_realized_volatility_bps:
            return (
                False,
                (
                    f"realized volatility {reading.realized_volatility_bps:.2f} bps "
                    f"below {self.config.min_realized_volatility_bps:.2f} bps"
                ),
            )
        if reading.realized_volatility_bps > self.config.max_realized_volatility_bps:
            return (
                False,
                (
                    f"realized volatility {reading.realized_volatility_bps:.2f} bps "
                    f"above {self.config.max_realized_volatility_bps:.2f} bps"
                ),
            )
        if reading.volatility_ratio < self.config.min_volatility_ratio:
            return (
                False,
                (
                    f"volatility ratio {reading.volatility_ratio:.2f} below "
                    f"{self.config.min_volatility_ratio:.2f}"
                ),
            )
        if reading.volatility_ratio > self.config.max_volatility_ratio:
            return (
                False,
                (
                    f"volatility ratio {reading.volatility_ratio:.2f} above "
                    f"{self.config.max_volatility_ratio:.2f}"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        return (
            True,
            (
                f"{direction} multi-horizon momentum: "
                f"fast={reading.fast_move_bps:.1f} bps, "
                f"slow={reading.slow_move_bps:.1f} bps, "
                f"vol_ratio={reading.volatility_ratio:.2f}, "
                f"edge={reading.expected_edge_bps:.1f} bps"
            ),
        )

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: MultiHorizonMomentumReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.fast_move_bps),
                max(self.config.min_fast_move_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(reading.slow_move_bps),
                max(self.config.min_slow_move_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.trend_efficiency,
                max(self.config.min_trend_efficiency, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class AutocorrelationRegimeStrategy:
    def __init__(self, config: AutocorrelationRegimeConfig | None = None) -> None:
        self.config = config or AutocorrelationRegimeConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_regime(self, prices: Sequence[float]) -> AutocorrelationRegimeReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        returns = _log_returns(recent_prices)
        if len(returns) < 2:
            return None
        lag1_autocorrelation = _lag1_autocorrelation(returns)
        signal_prices = recent_prices[-(self.config.signal_lookback + 1):]
        signal_move = log(signal_prices[-1] / signal_prices[0])
        signal_move_bps = signal_move * 10_000
        path_move = sum(abs(value) for value in returns)
        full_move = log(recent_prices[-1] / recent_prices[0])
        trend_efficiency = abs(full_move) / path_move if path_move > 0 else 0.0
        realized_volatility = _population_stdev(returns)
        baseline = recent_prices[:-1]
        baseline_mean = sum(baseline) / len(baseline)
        baseline_stdev = _population_stdev(baseline)
        zscore = (
            0.0
            if baseline_stdev == 0
            else (recent_prices[-1] - baseline_mean) / baseline_stdev
        )
        mode, direction, expected_edge_bps = self._classify(
            lag1_autocorrelation=lag1_autocorrelation,
            signal_move_bps=signal_move_bps,
            zscore=zscore,
            realized_volatility_bps=realized_volatility * 10_000,
            trend_efficiency=trend_efficiency,
        )
        return AutocorrelationRegimeReading(
            latest_price=recent_prices[-1],
            lag1_autocorrelation=lag1_autocorrelation,
            signal_move_bps=signal_move_bps,
            zscore=zscore,
            realized_volatility=realized_volatility,
            realized_volatility_bps=realized_volatility * 10_000,
            trend_efficiency=trend_efficiency,
            mode=mode,
            expected_edge_bps=expected_edge_bps,
            signal_direction=direction,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_regime(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for autocorrelation regime reading",
                primary_signal="autocorrelation_regime",
            )

        diagnostics = _autocorrelation_regime_diagnostics(reading)
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "autocorrelation regime max holding period "
                    f"{self.config.max_holding_period} bars reached"
                ),
                diagnostics,
                primary_signal="autocorrelation_regime",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)
        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="autocorrelation_regime",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="autocorrelation_regime",
            )

        if signal_direction == 0 or abs(
            reading.lag1_autocorrelation
        ) <= self.config.exit_abs_autocorrelation:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "autocorrelation regime faded but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="autocorrelation_regime",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"autocorrelation regime faded; {reason}",
                diagnostics,
                primary_signal="autocorrelation_regime",
            )

        if signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite autocorrelation regime seen but minimum holding "
                        f"period {self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="autocorrelation_regime",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite autocorrelation regime; {reason}",
                    diagnostics,
                    primary_signal="autocorrelation_regime",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"opposite autocorrelation regime but entry blocked; {reason}",
                diagnostics,
                primary_signal="autocorrelation_regime",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            (
                "autocorrelation regime still supports current position "
                f"after {holding_period} bars"
            ),
            diagnostics,
            primary_signal="autocorrelation_regime",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _classify(
        self,
        *,
        lag1_autocorrelation: float,
        signal_move_bps: float,
        zscore: float,
        realized_volatility_bps: float,
        trend_efficiency: float,
    ) -> tuple[str, SignalDirection, float]:
        abs_autocorr = abs(lag1_autocorrelation)
        abs_signal_move = abs(signal_move_bps)
        if (
            lag1_autocorrelation >= self.config.min_abs_autocorrelation
            and abs_signal_move >= self.config.min_momentum_bps
            and trend_efficiency >= self.config.min_trend_efficiency
        ):
            direction = (
                SignalDirection.LONG
                if signal_move_bps > 0
                else SignalDirection.SHORT
            )
            edge = abs_signal_move * min(
                abs_autocorr / max(self.config.min_abs_autocorrelation, 1e-12),
                2.0,
            )
            return "MOMENTUM", direction, edge

        if (
            lag1_autocorrelation <= -self.config.min_abs_autocorrelation
            and abs(zscore) >= self.config.min_reversion_zscore
            and abs_signal_move >= self.config.min_reversion_move_bps
        ):
            direction = SignalDirection.SHORT if zscore > 0 else SignalDirection.LONG
            edge = max(
                abs_signal_move,
                abs(zscore) * realized_volatility_bps,
            )
            return "MEAN_REVERSION", direction, edge

        return "FLAT", SignalDirection.FLAT, 0.0

    def _passes_entry_filters(
        self,
        *,
        reading: AutocorrelationRegimeReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                (
                    "outside autocorrelation regime UTC hours "
                    f"({allowed}); current hour={utc_hour}"
                ),
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return (
                False,
                (
                    "no autocorrelation regime entry "
                    f"(rho={reading.lag1_autocorrelation:.2f}, "
                    f"move={reading.signal_move_bps:.1f} bps, "
                    f"z={reading.zscore:.2f})"
                ),
            )
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        return (
            True,
            (
                f"{direction} autocorrelation {reading.mode.lower()}: "
                f"rho={reading.lag1_autocorrelation:.2f}, "
                f"move={reading.signal_move_bps:.1f} bps, "
                f"z={reading.zscore:.2f}, "
                f"edge={reading.expected_edge_bps:.1f} bps"
            ),
        )

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: AutocorrelationRegimeReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.lag1_autocorrelation),
                max(self.config.min_abs_autocorrelation, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class IntradaySeasonalityStrategy:
    def __init__(self, config: IntradaySeasonalityConfig | None = None) -> None:
        self.config = config or IntradaySeasonalityConfig()

    def read_seasonality(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> IntradaySeasonalityReading | None:
        valid_prices = _recent_valid_prices(prices, self.config.lookback)
        if valid_prices is None:
            return None
        returns: list[float] = []
        last_index = len(valid_prices) - 1
        for offset in range(1, self.config.lookback_periods + 1):
            end_index = last_index - (offset * self.config.period_bars)
            start_index = end_index - 1
            if start_index < 0:
                continue
            returns.append(log(valid_prices[end_index] / valid_prices[start_index]))
        if not returns:
            return None

        mean_return = sum(returns) / len(returns)
        positive_samples = sum(1 for value in returns if value > 0)
        negative_samples = sum(1 for value in returns if value < 0)
        aligned_samples = (
            positive_samples if mean_return >= 0 else negative_samples
        )
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        return IntradaySeasonalityReading(
            mean_return_bps=mean_return * 10_000,
            realized_volatility=_population_stdev(returns),
            consistency=aligned_samples / len(returns),
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            sample_count=len(returns),
            session_allowed=session_allowed,
            utc_hour=utc_hour,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_seasonality(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough same-slot history for intraday seasonality",
            )

        diagnostics = _intraday_seasonality_diagnostics(reading)
        if not reading.session_allowed:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.EXIT
            )
            return _decision(
                action,
                self.config.symbol,
                0.0,
                (
                    "outside intraday seasonality UTC hours "
                    f"({self._allowed_hours_text()}); current hour={reading.utc_hour}"
                ),
                diagnostics,
                primary_signal="intraday_seasonality",
            )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"intraday seasonality max holding period {self.config.max_holding_period} reached",
                diagnostics,
                primary_signal="intraday_seasonality",
            )

        abs_edge = abs(reading.mean_return_bps)
        if abs_edge <= self.config.exit_threshold_bps and current_direction != 0:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"same-slot edge {abs_edge:.2f} bps is at or below "
                    f"exit threshold {self.config.exit_threshold_bps:.2f} bps"
                ),
                diagnostics,
                primary_signal="intraday_seasonality",
            )
        if abs_edge < self.config.entry_threshold_bps:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                (
                    f"same-slot edge {abs_edge:.2f} bps below entry threshold "
                    f"{self.config.entry_threshold_bps:.2f} bps"
                ),
                diagnostics,
                primary_signal="intraday_seasonality",
            )
        if reading.consistency < self.config.min_consistency:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                (
                    f"same-slot consistency {reading.consistency:.2f} below "
                    f"{self.config.min_consistency:.2f}"
                ),
                diagnostics,
                primary_signal="intraday_seasonality",
            )

        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_edge,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                cost_reason,
                diagnostics,
                primary_signal="intraday_seasonality",
            )

        raw_direction = 1 if reading.mean_return_bps > 0 else -1
        signal_direction = (
            raw_direction
            if self.config.signal_mode == "momentum"
            else -raw_direction
        )
        target = signal_direction * self._sized_notional(reading)
        if current_direction == 0:
            action = StrategyAction.ENTER
            target_for_decision = target
        elif signal_direction != current_direction:
            action = StrategyAction.REVERSE
            target_for_decision = target
        else:
            action = StrategyAction.HOLD
            target_for_decision = current_notional_usd

        return _decision(
            action,
            self.config.symbol,
            target_for_decision,
            (
                f"same-slot mean={reading.mean_return_bps:.2f} bps, "
                f"consistency={reading.consistency:.2f}, "
                f"mode={self.config.signal_mode}; {cost_reason}"
            ),
            diagnostics,
            primary_signal="intraday_seasonality",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _sized_notional(self, reading: IntradaySeasonalityReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.mean_return_bps),
                max(self.config.entry_threshold_bps, 1e-12),
            ),
            reading.consistency,
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        asset_class = instrument_for(self.config.symbol).asset_class
        if asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _allowed_hours_text(self) -> str:
        allowed = self._allowed_entry_hours()
        if allowed is None:
            return "all"
        return ",".join(str(hour) for hour in allowed)


class ConditionalSeasonalityStrategy:
    def __init__(self, config: ConditionalSeasonalityConfig | None = None) -> None:
        self.config = config or ConditionalSeasonalityConfig()

    def read_conditional_seasonality(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> ConditionalSeasonalityReading | None:
        valid_prices = _recent_valid_prices(prices, self.config.lookback)
        if valid_prices is None:
            return None
        last_index = len(valid_prices) - 1
        current_momentum_bps = _log_move_bps(
            valid_prices,
            last_index - self.config.momentum_lookback,
            last_index,
        )
        current_condition = self._condition(current_momentum_bps)
        forward_returns: list[float] = []
        for offset in range(1, self.config.lookback_periods + 1):
            event_index = last_index - (offset * self.config.period_bars)
            momentum_start = event_index - self.config.momentum_lookback
            forward_end = event_index + self.config.horizon_bars
            if momentum_start < 0 or forward_end >= last_index:
                continue
            historical_momentum_bps = _log_move_bps(
                valid_prices,
                momentum_start,
                event_index,
            )
            if self._condition(historical_momentum_bps) != current_condition:
                continue
            forward_returns.append(log(valid_prices[forward_end] / valid_prices[event_index]))

        if len(forward_returns) < self.config.min_samples:
            return None

        mean_return = sum(forward_returns) / len(forward_returns)
        stdev = _population_stdev(forward_returns)
        positive_samples = sum(1 for value in forward_returns if value > 0)
        negative_samples = sum(1 for value in forward_returns if value < 0)
        aligned_samples = positive_samples if mean_return >= 0 else negative_samples
        consistency = aligned_samples / len(forward_returns)
        tstat = (
            mean_return / (stdev / sqrt(len(forward_returns)))
            if stdev > 0
            else (float("inf") if mean_return > 0 else -float("inf") if mean_return < 0 else 0.0)
        )
        raw_direction = (
            SignalDirection.LONG
            if mean_return > 0
            else SignalDirection.SHORT
            if mean_return < 0
            else SignalDirection.FLAT
        )
        signal_direction = (
            raw_direction
            if self.config.signal_mode == "momentum"
            else SignalDirection.SHORT
            if raw_direction == SignalDirection.LONG
            else SignalDirection.LONG
            if raw_direction == SignalDirection.SHORT
            else SignalDirection.FLAT
        )
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        return ConditionalSeasonalityReading(
            current_condition=current_condition,
            current_momentum_bps=current_momentum_bps,
            mean_forward_return_bps=mean_return * 10_000,
            realized_volatility=stdev,
            realized_volatility_bps=stdev * 10_000,
            tstat=tstat,
            consistency=consistency,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            sample_count=len(forward_returns),
            expected_edge_bps=abs(mean_return) * 10_000,
            signal_direction=signal_direction,
            session_allowed=session_allowed,
            utc_hour=utc_hour,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_conditional_seasonality(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                "not enough matching historical same-slot conditions",
                primary_signal="conditional_seasonality",
            )

        diagnostics = _conditional_seasonality_diagnostics(reading)
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "conditional seasonality horizon reached "
                    f"after {holding_period} bars"
                ),
                diagnostics,
                primary_signal="conditional_seasonality",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)
        if current_direction == 0:
            if not passed:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="conditional_seasonality",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="conditional_seasonality",
            )

        if reading.expected_edge_bps <= self.config.exit_threshold_bps:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "conditional seasonality edge faded but minimum holding period not reached",
                    diagnostics,
                    primary_signal="conditional_seasonality",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"conditional edge {reading.expected_edge_bps:.2f} bps at or below "
                    f"exit threshold {self.config.exit_threshold_bps:.2f} bps"
                ),
                diagnostics,
                primary_signal="conditional_seasonality",
            )

        if signal_direction == 0:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "conditional seasonality became flat",
                diagnostics,
                primary_signal="conditional_seasonality",
            )

        if signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "opposite conditional seasonality but minimum holding period not reached",
                    diagnostics,
                    primary_signal="conditional_seasonality",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite conditional seasonality; {reason}",
                    diagnostics,
                    primary_signal="conditional_seasonality",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"opposite conditional seasonality but entry blocked; {reason}",
                diagnostics,
                primary_signal="conditional_seasonality",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"conditional seasonality still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="conditional_seasonality",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: ConditionalSeasonalityReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        if not reading.session_allowed:
            return (
                False,
                (
                    "outside conditional seasonality UTC hours "
                    f"({self._allowed_hours_text()}); current hour={reading.utc_hour}"
                ),
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, "conditional seasonality signal is flat"
        if reading.expected_edge_bps < self.config.entry_threshold_bps:
            return (
                False,
                (
                    f"conditional edge {reading.expected_edge_bps:.2f} bps below "
                    f"{self.config.entry_threshold_bps:.2f} bps threshold"
                ),
            )
        if reading.consistency < self.config.min_consistency:
            return (
                False,
                (
                    f"conditional consistency {reading.consistency:.2f} below "
                    f"{self.config.min_consistency:.2f}"
                ),
            )
        if abs(reading.tstat) < self.config.min_abs_tstat:
            return (
                False,
                (
                    f"conditional t-stat {reading.tstat:.2f} below "
                    f"{self.config.min_abs_tstat:.2f}"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        return (
            True,
            (
                f"{direction} conditional seasonality: "
                f"condition={reading.current_condition}, "
                f"mean={reading.mean_forward_return_bps:.2f} bps, "
                f"mode={self.config.signal_mode}, "
                f"t={reading.tstat:.2f}, consistency={reading.consistency:.2f}; "
                f"{cost_reason}"
            ),
        )

    def _sized_notional(self, reading: ConditionalSeasonalityReading) -> float:
        confidence = min(
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.entry_threshold_bps, 1e-12),
            ),
            min(abs(reading.tstat) / max(self.config.min_abs_tstat, 1e-12), 1.0),
            reading.consistency,
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )

    def _condition(self, momentum_bps: float) -> str:
        if momentum_bps >= self.config.momentum_threshold_bps:
            return "UP"
        if momentum_bps <= -self.config.momentum_threshold_bps:
            return "DOWN"
        return "FLAT"

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        asset_class = instrument_for(self.config.symbol).asset_class
        if asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _allowed_hours_text(self) -> str:
        allowed = self._allowed_entry_hours()
        if allowed is None:
            return "all"
        return ",".join(str(hour) for hour in allowed)


class BreakoutStrategy:
    def __init__(self, config: BreakoutConfig | None = None) -> None:
        self.config = config or BreakoutConfig()

    def read_breakout(self, prices: Sequence[float]) -> BreakoutReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        channel_prices = recent_prices[:-1]
        last_price = recent_prices[-1]
        upper_band = max(channel_prices)
        lower_band = min(channel_prices)
        midpoint = (upper_band + lower_band) / 2
        channel_width_bps = ((upper_band - lower_band) / midpoint) * 10_000
        if last_price > upper_band:
            breakout_bps = ((last_price / upper_band) - 1.0) * 10_000
        elif last_price < lower_band:
            breakout_bps = -((lower_band / last_price) - 1.0) * 10_000
        else:
            breakout_bps = 0.0

        width = upper_band - lower_band
        position_in_channel = (
            (last_price - lower_band) / width
            if width > 0
            else 0.5
        )
        realized_volatility = _population_stdev(_log_returns(recent_prices))

        return BreakoutReading(
            upper_band=upper_band,
            lower_band=lower_band,
            last_price=last_price,
            channel_width_bps=channel_width_bps,
            breakout_bps=breakout_bps,
            position_in_channel=position_in_channel,
            realized_volatility=realized_volatility,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_breakout(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for breakout reading",
            )

        diagnostics = _breakout_diagnostics(reading)
        if reading.channel_width_bps < self.config.min_channel_width_bps:
            return self._flat_or_exit(
                current_direction=current_direction,
                current_notional_usd=current_notional_usd,
                reason=(
                    f"channel width {reading.channel_width_bps:.1f} bps below "
                    f"{self.config.min_channel_width_bps:.1f} bps minimum"
                ),
                diagnostics=diagnostics,
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        if reading.breakout_bps > 0:
            signal_direction = 1
        elif reading.breakout_bps < 0:
            signal_direction = -1
        else:
            signal_direction = 0

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite breakout; {reason}",
                    diagnostics,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "opposite range pressure without confirmed breakout",
                diagnostics,
            )

        if self._should_exit(reading=reading, current_direction=current_direction):
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"breakout faded back inside channel after {holding_period} bars"
                ),
                diagnostics,
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"breakout still supports current position after {holding_period} bars",
            diagnostics,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: BreakoutReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        abs_breakout = abs(reading.breakout_bps)
        if abs_breakout < self.config.breakout_buffer_bps:
            return (
                False,
                (
                    f"breakout {reading.breakout_bps:.1f} bps below "
                    f"{self.config.breakout_buffer_bps:.1f} bps buffer"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_breakout,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        side = "upper" if reading.breakout_bps > 0 else "lower"
        return (
            True,
            (
                f"{self.config.lookback}-price {side} breakout "
                f"{reading.breakout_bps:.1f} bps beyond channel "
                f"width={reading.channel_width_bps:.1f} bps"
            ),
        )

    def _should_exit(self, *, reading: BreakoutReading, current_direction: int) -> bool:
        if current_direction > 0:
            exit_price = reading.upper_band * (1.0 - self.config.exit_buffer_bps / 10_000)
            return reading.last_price <= exit_price
        if current_direction < 0:
            exit_price = reading.lower_band * (1.0 + self.config.exit_buffer_bps / 10_000)
            return reading.last_price >= exit_price
        return False

    def _flat_or_exit(
        self,
        *,
        current_direction: int,
        current_notional_usd: float,
        reason: str,
        diagnostics: tuple[tuple[str, float | str], ...],
    ) -> StrategyDecision:
        if current_direction == 0:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                reason,
                diagnostics,
            )
        return _decision(
            StrategyAction.EXIT,
            self.config.symbol,
            0.0,
            reason,
            diagnostics,
        )

    def _sized_notional(self, reading: BreakoutReading) -> float:
        confidence = _bounded_confidence(
            abs(reading.breakout_bps),
            max(self.config.breakout_buffer_bps, 1e-12),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class VolatilitySqueezeStrategy:
    def __init__(self, config: VolatilitySqueezeConfig | None = None) -> None:
        self.config = config or VolatilitySqueezeConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_squeeze(self, prices: Sequence[float]) -> VolatilitySqueezeReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        band_prices = recent_prices[:-1]
        last_price = recent_prices[-1]
        mean_price = _simple_average(band_prices)
        band_stdev = _population_stdev(band_prices)
        upper_band = mean_price + (self.config.band_stdev_multiplier * band_stdev)
        lower_band = mean_price - (self.config.band_stdev_multiplier * band_stdev)
        band_width_bps = ((upper_band - lower_band) / mean_price) * 10_000

        if last_price > upper_band:
            breakout_bps = ((last_price / upper_band) - 1.0) * 10_000
        elif last_price < lower_band:
            breakout_bps = -((lower_band / last_price) - 1.0) * 10_000
        else:
            breakout_bps = 0.0

        baseline_returns = _log_returns(band_prices)
        if len(baseline_returns) <= self.config.squeeze_window:
            return None
        recent_returns = baseline_returns[-self.config.squeeze_window :]
        prior_returns = baseline_returns[: -self.config.squeeze_window]
        recent_volatility = _population_stdev(recent_returns)
        prior_volatility = _population_stdev(prior_returns)
        squeeze_ratio = (
            recent_volatility / prior_volatility
            if prior_volatility > 0
            else float("inf")
        )
        realized_volatility = _population_stdev(_log_returns(recent_prices))

        return VolatilitySqueezeReading(
            mean_price=mean_price,
            upper_band=upper_band,
            lower_band=lower_band,
            last_price=last_price,
            band_width_bps=band_width_bps,
            breakout_bps=breakout_bps,
            recent_volatility_bps=recent_volatility * 10_000,
            prior_volatility_bps=prior_volatility * 10_000,
            squeeze_ratio=squeeze_ratio,
            realized_volatility=realized_volatility,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_squeeze(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for volatility squeeze reading",
                primary_signal="volatility_squeeze",
            )

        diagnostics = _volatility_squeeze_diagnostics(reading) + (
            _volatility_squeeze_session_diagnostics(self._entry_session_allowed(quote))
        )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"volatility squeeze max holding period "
                    f"{self.config.max_holding_period} bars reached"
                ),
                diagnostics,
                primary_signal="volatility_squeeze",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        if reading.breakout_bps > 0:
            signal_direction = 1
        elif reading.breakout_bps < 0:
            signal_direction = -1
        else:
            signal_direction = 0

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="volatility_squeeze",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="volatility_squeeze",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite squeeze breakout; {reason}",
                    diagnostics,
                    primary_signal="volatility_squeeze",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "opposite band pressure without confirmed squeeze breakout",
                diagnostics,
                primary_signal="volatility_squeeze",
            )

        if self._should_exit(reading=reading, current_direction=current_direction):
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"volatility squeeze breakout faded after {holding_period} bars",
                diagnostics,
                primary_signal="volatility_squeeze",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"volatility squeeze breakout still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="volatility_squeeze",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: VolatilitySqueezeReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        squeeze_ok, squeeze_reason = self._passes_squeeze_filters(reading)
        if not squeeze_ok:
            return False, squeeze_reason
        session_allowed, hour = self._entry_session_allowed(quote)
        if not session_allowed:
            return (
                False,
                f"outside volatility squeeze UTC session at hour {hour}",
            )

        abs_breakout = abs(reading.breakout_bps)
        if abs_breakout < self.config.breakout_buffer_bps:
            return (
                False,
                (
                    f"breakout {reading.breakout_bps:.1f} bps below "
                    f"{self.config.breakout_buffer_bps:.1f} bps squeeze buffer"
                ),
            )

        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_breakout,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason

        side = "upper" if reading.breakout_bps > 0 else "lower"
        return (
            True,
            (
                f"{self.config.lookback}-price volatility squeeze {side} breakout "
                f"{reading.breakout_bps:.1f} bps; "
                f"squeeze ratio={reading.squeeze_ratio:.2f}, "
                f"prior vol={reading.prior_volatility_bps:.1f} bps"
            ),
        )

    def _passes_squeeze_filters(
        self,
        reading: VolatilitySqueezeReading,
    ) -> tuple[bool, str]:
        if reading.band_width_bps < self.config.min_band_width_bps:
            return (
                False,
                (
                    f"band width {reading.band_width_bps:.1f} bps below "
                    f"{self.config.min_band_width_bps:.1f} bps minimum"
                ),
            )
        if reading.prior_volatility_bps < self.config.min_prior_volatility_bps:
            return (
                False,
                (
                    f"prior volatility {reading.prior_volatility_bps:.1f} bps below "
                    f"{self.config.min_prior_volatility_bps:.1f} bps minimum"
                ),
            )
        if reading.squeeze_ratio > self.config.max_squeeze_ratio:
            return (
                False,
                (
                    f"squeeze ratio {reading.squeeze_ratio:.2f} above "
                    f"{self.config.max_squeeze_ratio:.2f} maximum"
                ),
            )
        return True, "volatility squeeze filters passed"

    def _entry_session_allowed(self, quote: QuoteSnapshot | None) -> tuple[bool, int | None]:
        hours = self._allowed_utc_hours()
        if quote is None or hours is None:
            return True, None
        hour = quote.timestamp.astimezone(UTC).hour
        return hour in hours, hour

    def _allowed_utc_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _should_exit(
        self,
        *,
        reading: VolatilitySqueezeReading,
        current_direction: int,
    ) -> bool:
        if current_direction > 0:
            exit_price = reading.upper_band * (1.0 - self.config.exit_buffer_bps / 10_000)
            return reading.last_price <= exit_price
        if current_direction < 0:
            exit_price = reading.lower_band * (1.0 + self.config.exit_buffer_bps / 10_000)
            return reading.last_price >= exit_price
        return False

    def _sized_notional(self, reading: VolatilitySqueezeReading) -> float:
        confidence = _bounded_confidence(
            abs(reading.breakout_bps),
            max(self.config.breakout_buffer_bps, 1e-12),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class DualSqueezeStrategy:
    def __init__(self, config: DualSqueezeConfig | None = None) -> None:
        self.config = config or DualSqueezeConfig()
        self.fast = VolatilitySqueezeStrategy(self._fast_config())
        self.confirmation = VolatilitySqueezeStrategy(self._confirmation_config())

    def read_dual_squeeze(self, prices: Sequence[float]) -> DualSqueezeReading | None:
        fast_reading = self.fast.read_squeeze(prices)
        confirmation_reading = self.confirmation.read_squeeze(prices)
        if fast_reading is None or confirmation_reading is None:
            return None

        target_direction = _signed_threshold_direction(
            fast_reading.breakout_bps,
            self.config.breakout_buffer_bps,
        )
        passed, reason = self._confirmation_passed(
            confirmation_reading,
            target_direction=target_direction,
        )
        return DualSqueezeReading(
            fast=fast_reading,
            confirmation=confirmation_reading,
            confirmation_passed=passed,
            confirmation_reason=reason,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        fast_decision = self.fast.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        current_direction = _notional_direction(current_notional_usd)

        if fast_decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _decision(
                fast_decision.action,
                self.config.symbol,
                fast_decision.target_notional_usd,
                fast_decision.reason,
                fast_decision.diagnostics,
                primary_signal="dual_squeeze",
            )

        reading = self.read_dual_squeeze(prices)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "dual squeeze blocked: not enough confirmation context",
                primary_signal="dual_squeeze",
            )

        diagnostics = _dual_squeeze_diagnostics(reading)
        target_direction = _notional_direction(fast_decision.target_notional_usd)
        passed, reason = self._confirmation_passed(
            reading.confirmation,
            target_direction=target_direction,
        )
        if not passed:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                f"dual squeeze blocked: {reason}",
                diagnostics,
                primary_signal="dual_squeeze",
            )

        return _decision(
            fast_decision.action,
            self.config.symbol,
            fast_decision.target_notional_usd,
            f"dual squeeze confirmed: {fast_decision.reason}; {reason}",
            fast_decision.diagnostics + diagnostics,
            primary_signal="dual_squeeze",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _confirmation_passed(
        self,
        reading: VolatilitySqueezeReading,
        *,
        target_direction: int,
    ) -> tuple[bool, str]:
        if target_direction == 0:
            return False, "fast squeeze has no breakout direction"

        confirmation_direction = _signed_threshold_direction(
            reading.breakout_bps,
            self.config.breakout_buffer_bps,
        )
        confirmation_bias = 1 if reading.last_price >= reading.mean_price else -1
        squeeze_ok = reading.squeeze_ratio <= self.config.confirmation_max_squeeze_ratio
        mode = self.config.confirmation_mode

        if mode == "bias":
            passed = confirmation_bias == target_direction
        elif mode == "breakout":
            passed = confirmation_direction == target_direction
        elif mode == "not_opposite":
            passed = (
                confirmation_direction in {0, target_direction}
                and confirmation_bias == target_direction
            )
        else:
            passed = squeeze_ok and confirmation_bias == target_direction

        reason = (
            f"confirmation mode={mode}, bias={confirmation_bias}, "
            f"breakout={confirmation_direction}, squeeze_ratio={reading.squeeze_ratio:.2f}"
        )
        return passed, reason

    def _fast_config(self) -> VolatilitySqueezeConfig:
        return VolatilitySqueezeConfig(
            symbol=self.config.symbol,
            lookback=self.config.lookback,
            squeeze_window=self.config.squeeze_window,
            band_stdev_multiplier=self.config.band_stdev_multiplier,
            breakout_buffer_bps=self.config.breakout_buffer_bps,
            exit_buffer_bps=self.config.exit_buffer_bps,
            max_squeeze_ratio=self.config.max_squeeze_ratio,
            min_prior_volatility_bps=self.config.min_prior_volatility_bps,
            min_band_width_bps=self.config.min_band_width_bps,
            forex_allowed_utc_hours=self.config.forex_allowed_utc_hours,
            metal_allowed_utc_hours=self.config.metal_allowed_utc_hours,
            crypto_allowed_utc_hours=self.config.crypto_allowed_utc_hours,
            target_notional_usd=self.config.target_notional_usd,
            position_sizing=self.config.position_sizing,
            target_volatility=self.config.target_volatility,
            volatility_floor=self.config.volatility_floor,
            max_target_notional_usd=self.config.max_target_notional_usd,
            min_trade_notional_usd=self.config.min_trade_notional_usd,
            max_holding_period=self.config.max_holding_period,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )

    def _confirmation_config(self) -> VolatilitySqueezeConfig:
        return VolatilitySqueezeConfig(
            symbol=self.config.symbol,
            lookback=self.config.confirmation_lookback,
            squeeze_window=self.config.confirmation_squeeze_window,
            band_stdev_multiplier=self.config.confirmation_band_stdev_multiplier,
            breakout_buffer_bps=0.0,
            exit_buffer_bps=self.config.exit_buffer_bps,
            max_squeeze_ratio=self.config.confirmation_max_squeeze_ratio,
            min_prior_volatility_bps=self.config.min_prior_volatility_bps,
            min_band_width_bps=self.config.min_band_width_bps,
            forex_allowed_utc_hours=self.config.forex_allowed_utc_hours,
            metal_allowed_utc_hours=self.config.metal_allowed_utc_hours,
            crypto_allowed_utc_hours=self.config.crypto_allowed_utc_hours,
            target_notional_usd=self.config.target_notional_usd,
            position_sizing=self.config.position_sizing,
            target_volatility=self.config.target_volatility,
            volatility_floor=self.config.volatility_floor,
            max_target_notional_usd=self.config.max_target_notional_usd,
            min_trade_notional_usd=self.config.min_trade_notional_usd,
            max_holding_period=self.config.max_holding_period,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )


class AssetAdaptiveDualSqueezeStrategy:
    def __init__(self, config: AssetAdaptiveDualSqueezeConfig | None = None) -> None:
        self.config = config or AssetAdaptiveDualSqueezeConfig()
        self._instrument = instrument_for(self.config.symbol)
        self.inner_config = self.config.dual_config_for_asset_class(
            self._instrument.asset_class
        )
        self.inner = DualSqueezeStrategy(self.inner_config)

    @property
    def selected_profile(self) -> str:
        if self._instrument.asset_class == AssetClass.METAL:
            return "metal_fast"
        return "base"

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        decision = self.inner.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        supporting_signals = (decision.primary_signal,) + decision.supporting_signals
        diagnostics = decision.diagnostics + (
            ("asset_adaptive_profile", self.selected_profile),
        )
        return _decision(
            decision.action,
            self.config.symbol,
            decision.target_notional_usd,
            (
                f"{self.selected_profile} asset-adaptive dual squeeze: "
                f"{decision.reason}"
            ),
            diagnostics,
            primary_signal="asset_adaptive_dual_squeeze",
            supporting_signals=supporting_signals,
            conflicting_signals=decision.conflicting_signals,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()


class RangeExpansionTrendStrategy:
    def __init__(self, config: RangeExpansionTrendConfig | None = None) -> None:
        self.config = config or RangeExpansionTrendConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_range_expansion(
        self,
        prices: Sequence[float],
    ) -> RangeExpansionTrendReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        trigger_start_index = len(recent_prices) - self.config.trigger_window - 1
        baseline_prices = recent_prices[: trigger_start_index + 1]
        trigger_prices = recent_prices[trigger_start_index:]
        trigger_start_price = trigger_prices[0]
        last_price = trigger_prices[-1]
        baseline_high = max(baseline_prices)
        baseline_low = min(baseline_prices)

        trigger_returns = _log_returns(trigger_prices)
        trigger_log_return = sum(trigger_returns)
        trigger_move_bps = trigger_log_return * 10_000
        trigger_path = sum(abs(value) for value in trigger_returns)
        trend_efficiency = (
            abs(trigger_log_return) / trigger_path
            if trigger_path > 0
            else 0.0
        )

        if last_price > baseline_high:
            range_break_bps = ((last_price / baseline_high) - 1.0) * 10_000
        elif last_price < baseline_low:
            range_break_bps = -((baseline_low / last_price) - 1.0) * 10_000
        else:
            range_break_bps = 0.0

        baseline_returns = _log_returns(baseline_prices)
        baseline_volatility_bps = _population_stdev(baseline_returns) * 10_000
        trigger_volatility_bps = _population_stdev(trigger_returns) * 10_000
        expected_trigger_move_bps = baseline_volatility_bps * sqrt(
            max(self.config.trigger_window, 1)
        )
        expansion_zscore = (
            abs(trigger_move_bps) / max(expected_trigger_move_bps, 1e-12)
        )
        realized_volatility = _population_stdev(_log_returns(recent_prices))
        signal_direction = self._signal_direction(
            trigger_move_bps=trigger_move_bps,
            range_break_bps=range_break_bps,
            trend_efficiency=trend_efficiency,
        )
        expected_edge_bps = (
            abs(range_break_bps) + (0.25 * abs(trigger_move_bps))
            if signal_direction != SignalDirection.FLAT
            else 0.0
        )

        return RangeExpansionTrendReading(
            baseline_high=baseline_high,
            baseline_low=baseline_low,
            trigger_start_price=trigger_start_price,
            last_price=last_price,
            trigger_move_bps=trigger_move_bps,
            range_break_bps=range_break_bps,
            baseline_volatility_bps=baseline_volatility_bps,
            trigger_volatility_bps=trigger_volatility_bps,
            expansion_zscore=expansion_zscore,
            trend_efficiency=trend_efficiency,
            expected_edge_bps=expected_edge_bps,
            realized_volatility=realized_volatility,
            signal_direction=signal_direction,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_range_expansion(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for range expansion trend reading",
                primary_signal="range_expansion_trend",
            )

        diagnostics = _range_expansion_trend_diagnostics(reading) + (
            _range_expansion_session_diagnostics(self._entry_session_allowed(quote))
        )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "range expansion trend max holding period "
                    f"{self.config.max_holding_period} bars reached"
                ),
                diagnostics,
                primary_signal="range_expansion_trend",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="range_expansion_trend",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="range_expansion_trend",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite range expansion seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="range_expansion_trend",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite range expansion trend; {reason}",
                    diagnostics,
                    primary_signal="range_expansion_trend",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "opposite range pressure without confirmed expansion trend",
                diagnostics,
                primary_signal="range_expansion_trend",
            )

        if self._should_exit(reading=reading, current_direction=current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "range expansion exit signal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="range_expansion_trend",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "range expansion impulse faded back toward the prior range",
                diagnostics,
                primary_signal="range_expansion_trend",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"range expansion trend still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="range_expansion_trend",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: RangeExpansionTrendReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                f"outside range expansion UTC hours ({allowed}); current hour={utc_hour}",
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, "range break, trigger move, and trend efficiency not aligned"
        if reading.baseline_volatility_bps < self.config.min_baseline_volatility_bps:
            return (
                False,
                (
                    f"baseline volatility {reading.baseline_volatility_bps:.2f} bps below "
                    f"{self.config.min_baseline_volatility_bps:.2f} bps minimum"
                ),
            )
        if reading.trigger_volatility_bps > self.config.max_trigger_volatility_bps:
            return (
                False,
                (
                    f"trigger volatility {reading.trigger_volatility_bps:.1f} bps above "
                    f"{self.config.max_trigger_volatility_bps:.1f} bps maximum"
                ),
            )
        if reading.expansion_zscore < self.config.min_expansion_zscore:
            return (
                False,
                (
                    f"expansion z-score {reading.expansion_zscore:.2f} below "
                    f"{self.config.min_expansion_zscore:.2f}"
                ),
            )
        if reading.expansion_zscore > self.config.max_expansion_zscore:
            return (
                False,
                (
                    f"expansion z-score {reading.expansion_zscore:.2f} above "
                    f"{self.config.max_expansion_zscore:.2f}"
                ),
            )
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        session = "quote-less" if utc_hour is None else f"{utc_hour:02d}:00 UTC"
        return (
            True,
            (
                f"{direction} range expansion at {session}: "
                f"trigger={reading.trigger_move_bps:.1f} bps, "
                f"range_break={reading.range_break_bps:.1f} bps, "
                f"z={reading.expansion_zscore:.2f}; {cost_reason}"
            ),
        )

    def _signal_direction(
        self,
        *,
        trigger_move_bps: float,
        range_break_bps: float,
        trend_efficiency: float,
    ) -> SignalDirection:
        if trend_efficiency < self.config.min_trend_efficiency:
            return SignalDirection.FLAT
        if (
            trigger_move_bps >= self.config.min_trigger_move_bps
            and range_break_bps >= self.config.min_range_break_bps
        ):
            return SignalDirection.LONG
        if (
            trigger_move_bps <= -self.config.min_trigger_move_bps
            and range_break_bps <= -self.config.min_range_break_bps
        ):
            return SignalDirection.SHORT
        return SignalDirection.FLAT

    def _should_exit(
        self,
        *,
        reading: RangeExpansionTrendReading,
        current_direction: int,
    ) -> bool:
        if current_direction > 0:
            return (
                reading.trigger_move_bps <= self.config.exit_trigger_move_bps
                or reading.last_price <= reading.baseline_high
            )
        if current_direction < 0:
            return (
                reading.trigger_move_bps >= -self.config.exit_trigger_move_bps
                or reading.last_price >= reading.baseline_low
            )
        return False

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: RangeExpansionTrendReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.trigger_move_bps),
                max(self.config.min_trigger_move_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(reading.range_break_bps),
                max(self.config.min_range_break_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.expansion_zscore,
                max(self.config.min_expansion_zscore, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class SessionBreakoutStrategy:
    def __init__(self, config: SessionBreakoutConfig | None = None) -> None:
        self.config = config or SessionBreakoutConfig()
        self._instrument = instrument_for(self.config.symbol)
        self._breakout = BreakoutStrategy(
            BreakoutConfig(
                symbol=self.config.symbol,
                lookback=self.config.lookback,
                breakout_buffer_bps=self.config.breakout_buffer_bps,
                exit_buffer_bps=self.config.exit_buffer_bps,
                min_channel_width_bps=self.config.min_channel_width_bps,
                target_notional_usd=self.config.target_notional_usd,
                position_sizing=self.config.position_sizing,
                target_volatility=self.config.target_volatility,
                volatility_floor=self.config.volatility_floor,
                max_target_notional_usd=self.config.max_target_notional_usd,
                min_trade_notional_usd=self.config.min_trade_notional_usd,
                slippage_bps=self.config.slippage_bps,
                fee_bps=self.config.fee_bps,
                cost_buffer=self.config.cost_buffer,
                max_spread_bps=self.config.max_spread_bps,
            )
        )

    def read_session_breakout(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> SessionBreakoutReading | None:
        breakout = self._breakout.read_breakout(prices)
        if breakout is None:
            return None
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        session_allowed = utc_hour is None or utc_hour in self._allowed_utc_hours()
        return SessionBreakoutReading(
            breakout=breakout,
            realized_volatility_bps=breakout.realized_volatility * 10_000,
            utc_hour=utc_hour,
            session_allowed=session_allowed,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_session_breakout(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for session breakout reading",
                primary_signal="session_breakout",
            )

        diagnostics = _session_breakout_diagnostics(reading)
        regime_reading = self._read_regime(prices)
        diagnostics += _session_regime_diagnostics(regime_reading)
        breakout = reading.breakout
        if breakout.channel_width_bps < self.config.min_channel_width_bps:
            return self._flat_or_exit(
                current_direction=current_direction,
                current_notional_usd=current_notional_usd,
                reason=(
                    f"channel width {breakout.channel_width_bps:.1f} bps below "
                    f"{self.config.min_channel_width_bps:.1f} bps minimum"
                ),
                diagnostics=diagnostics,
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        if breakout.breakout_bps > 0:
            signal_direction = 1
        elif breakout.breakout_bps < 0:
            signal_direction = -1
        else:
            signal_direction = 0

        if (
            signal_direction != 0
            and passed
            and not self._regime_confirms_signal(regime_reading, signal_direction)
        ):
            passed = False
            reason = self._regime_block_reason(regime_reading, signal_direction)

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="session_breakout",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="session_breakout",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        f"opposite session breakout seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="session_breakout",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite session breakout; {reason}",
                    diagnostics,
                    primary_signal="session_breakout",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "opposite range pressure without confirmed session breakout",
                diagnostics,
                primary_signal="session_breakout",
            )

        if self._should_exit(reading=reading, current_direction=current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        f"session breakout exit signal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="session_breakout",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"session breakout faded back inside channel after {holding_period} bars",
                diagnostics,
                primary_signal="session_breakout",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"session breakout still supports current position after {holding_period} bars",
            diagnostics,
            primary_signal="session_breakout",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: SessionBreakoutReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        if not reading.session_allowed:
            allowed_hours = self._allowed_utc_hours()
            allowed = ",".join(str(hour) for hour in allowed_hours)
            return (
                False,
                f"outside allowed UTC session hours ({allowed}); current hour={reading.utc_hour}",
            )
        if reading.realized_volatility_bps < self.config.min_realized_volatility_bps:
            return (
                False,
                (
                    f"realized volatility {reading.realized_volatility_bps:.1f} bps below "
                    f"{self.config.min_realized_volatility_bps:.1f} bps minimum"
                ),
            )
        if reading.realized_volatility_bps > self.config.max_realized_volatility_bps:
            return (
                False,
                (
                    f"realized volatility {reading.realized_volatility_bps:.1f} bps above "
                    f"{self.config.max_realized_volatility_bps:.1f} bps maximum"
                ),
            )
        breakout = reading.breakout
        abs_breakout = abs(breakout.breakout_bps)
        required_breakout_bps = max(
            self.config.breakout_buffer_bps,
            self.config.min_expected_edge_bps,
        )
        if abs_breakout < required_breakout_bps:
            return (
                False,
                (
                    f"breakout {breakout.breakout_bps:.1f} bps below "
                    f"{required_breakout_bps:.1f} bps required edge"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_breakout,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        side = "upper" if breakout.breakout_bps > 0 else "lower"
        session = "quote-less" if reading.utc_hour is None else f"{reading.utc_hour:02d}:00 UTC"
        return (
            True,
            (
                f"session {session} {self.config.lookback}-price {side} breakout "
                f"{breakout.breakout_bps:.1f} bps with "
                f"volatility={reading.realized_volatility_bps:.1f} bps"
            ),
        )

    def _allowed_utc_hours(self) -> tuple[int, ...]:
        if (
            self._instrument.asset_class == AssetClass.METAL
            and self.config.metal_allowed_utc_hours is not None
        ):
            return self.config.metal_allowed_utc_hours
        return self.config.allowed_utc_hours

    def _read_regime(
        self,
        prices: Sequence[float],
    ) -> TimeSeriesRegimeReading | None:
        if not self.config.require_regime_confirmation:
            return None
        if len(prices) < self.config.regime_lookback:
            return None
        return read_kalman_regime(
            prices,
            symbol=self.config.symbol,
            config=KalmanTrendConfig(
                lookback=self.config.regime_lookback,
                min_abs_slope_bps=self.config.regime_min_abs_slope_bps,
                min_trend_efficiency=self.config.regime_min_trend_efficiency,
                max_realized_volatility_bps=self.config.regime_max_realized_volatility_bps,
            ),
        )

    def _regime_confirms_signal(
        self,
        regime_reading: TimeSeriesRegimeReading | None,
        signal_direction: int,
    ) -> bool:
        if not self.config.require_regime_confirmation:
            return True
        if regime_reading is None:
            return False
        if signal_direction > 0:
            return regime_reading.regime == TimeSeriesRegime.TREND_UP
        if signal_direction < 0:
            return regime_reading.regime == TimeSeriesRegime.TREND_DOWN
        return False

    def _regime_block_reason(
        self,
        regime_reading: TimeSeriesRegimeReading | None,
        signal_direction: int,
    ) -> str:
        wanted = "TREND_UP" if signal_direction > 0 else "TREND_DOWN"
        if regime_reading is None:
            return f"regime confirmation requires {wanted} but no regime reading is available"
        return (
            f"regime confirmation requires {wanted} but current regime is "
            f"{regime_reading.regime.value}"
        )

    def _should_exit(
        self,
        *,
        reading: SessionBreakoutReading,
        current_direction: int,
    ) -> bool:
        breakout = reading.breakout
        if current_direction > 0:
            exit_price = breakout.upper_band * (1.0 - self.config.exit_buffer_bps / 10_000)
            return breakout.last_price <= exit_price
        if current_direction < 0:
            exit_price = breakout.lower_band * (1.0 + self.config.exit_buffer_bps / 10_000)
            return breakout.last_price >= exit_price
        return False

    def _flat_or_exit(
        self,
        *,
        current_direction: int,
        current_notional_usd: float,
        reason: str,
        diagnostics: tuple[tuple[str, float | str], ...],
    ) -> StrategyDecision:
        if current_direction == 0:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                reason,
                diagnostics,
                primary_signal="session_breakout",
            )
        return _decision(
            StrategyAction.EXIT,
            self.config.symbol,
            0.0,
            reason,
            diagnostics,
            primary_signal="session_breakout",
        )

    def _sized_notional(self, reading: SessionBreakoutReading) -> float:
        confidence = _bounded_confidence(
            abs(reading.breakout.breakout_bps),
            max(self.config.breakout_buffer_bps, 1e-12),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.breakout.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class TrendPullbackStrategy:
    def __init__(self, config: TrendPullbackConfig | None = None) -> None:
        self.config = config or TrendPullbackConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_pullback(self, prices: Sequence[float]) -> TrendPullbackReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        anchor_index = len(recent_prices) - self.config.pullback_window - 1
        anchor_price = recent_prices[anchor_index]
        previous_price = recent_prices[-2]
        last_price = recent_prices[-1]
        trend_prices = recent_prices[: anchor_index + 1]
        trend_returns = _log_returns(trend_prices)
        trend_log_return = log(anchor_price / recent_prices[0])
        trend_path = sum(abs(value) for value in trend_returns)
        trend_efficiency = (
            abs(trend_log_return) / trend_path
            if trend_path > 0
            else 0.0
        )
        trend_move_bps = trend_log_return * 10_000
        pullback_bps = log(previous_price / anchor_price) * 10_000
        resume_bps = log(last_price / previous_price) * 10_000
        realized_volatility = _population_stdev(_log_returns(recent_prices))
        signal_direction = self._signal_direction(
            trend_move_bps=trend_move_bps,
            pullback_bps=pullback_bps,
            resume_bps=resume_bps,
            trend_efficiency=trend_efficiency,
        )
        expected_edge_bps = (
            abs(resume_bps)
            + min(abs(pullback_bps), abs(trend_move_bps) * 0.25)
            if signal_direction != SignalDirection.FLAT
            else 0.0
        )

        return TrendPullbackReading(
            anchor_price=anchor_price,
            previous_price=previous_price,
            last_price=last_price,
            trend_move_bps=trend_move_bps,
            pullback_bps=pullback_bps,
            resume_bps=resume_bps,
            expected_edge_bps=expected_edge_bps,
            trend_efficiency=trend_efficiency,
            realized_volatility=realized_volatility,
            signal_direction=signal_direction,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_pullback(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for trend pullback reading",
                primary_signal="trend_pullback",
            )

        diagnostics = _trend_pullback_diagnostics(reading) + (
            _trend_pullback_session_diagnostics(self._entry_session_allowed(quote))
        )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"trend pullback max holding period {self.config.max_holding_period} bars reached",
                diagnostics,
                primary_signal="trend_pullback",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="trend_pullback",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="trend_pullback",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite trend pullback seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="trend_pullback",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite trend pullback; {reason}",
                    diagnostics,
                    primary_signal="trend_pullback",
                )

        if self._trend_no_longer_supports_position(reading, current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "trend pullback exit signal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="trend_pullback",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "trend no longer supports trend pullback position",
                diagnostics,
                primary_signal="trend_pullback",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"trend pullback still supports current position after {holding_period} bars",
            diagnostics,
            primary_signal="trend_pullback",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: TrendPullbackReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                f"outside trend pullback UTC hours ({allowed}); current hour={utc_hour}",
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, "trend, pullback, and resume pattern not aligned"
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        session = "quote-less" if utc_hour is None else f"{utc_hour:02d}:00 UTC"
        return (
            True,
            (
                f"{direction} trend pullback at {session}: "
                f"trend={reading.trend_move_bps:.1f} bps, "
                f"pullback={reading.pullback_bps:.1f} bps, "
                f"resume={reading.resume_bps:.1f} bps"
            ),
        )

    def _signal_direction(
        self,
        *,
        trend_move_bps: float,
        pullback_bps: float,
        resume_bps: float,
        trend_efficiency: float,
    ) -> SignalDirection:
        if trend_efficiency < self.config.min_trend_efficiency:
            return SignalDirection.FLAT
        pullback_size = abs(pullback_bps)
        if not self.config.min_pullback_bps <= pullback_size <= self.config.max_pullback_bps:
            return SignalDirection.FLAT
        if (
            trend_move_bps >= self.config.min_trend_bps
            and pullback_bps <= -self.config.min_pullback_bps
            and resume_bps >= self.config.min_resume_bps
        ):
            return SignalDirection.LONG
        if (
            trend_move_bps <= -self.config.min_trend_bps
            and pullback_bps >= self.config.min_pullback_bps
            and resume_bps <= -self.config.min_resume_bps
        ):
            return SignalDirection.SHORT
        return SignalDirection.FLAT

    def _trend_no_longer_supports_position(
        self,
        reading: TrendPullbackReading,
        current_direction: int,
    ) -> bool:
        if current_direction > 0:
            return reading.trend_move_bps < self.config.min_trend_bps * 0.5
        if current_direction < 0:
            return reading.trend_move_bps > -self.config.min_trend_bps * 0.5
        return False

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: TrendPullbackReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.trend_move_bps),
                max(self.config.min_trend_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class ExhaustionReversalStrategy:
    def __init__(self, config: ExhaustionReversalConfig | None = None) -> None:
        self.config = config or ExhaustionReversalConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_exhaustion(
        self,
        prices: Sequence[float],
    ) -> ExhaustionReversalReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        shock_start_index = len(recent_prices) - self.config.shock_window - 1
        shock_start_price = recent_prices[shock_start_index]
        previous_price = recent_prices[-2]
        last_price = recent_prices[-1]
        shock_prices = recent_prices[shock_start_index:-1]
        shock_returns = _log_returns(shock_prices)
        shock_log_return = log(previous_price / shock_start_price)
        shock_path = sum(abs(value) for value in shock_returns)
        shock_efficiency = (
            abs(shock_log_return) / shock_path
            if shock_path > 0
            else 0.0
        )
        shock_move_bps = shock_log_return * 10_000
        reversal_bps = log(last_price / previous_price) * 10_000
        baseline_prices = recent_prices[: shock_start_index + 1]
        baseline_returns = _log_returns(baseline_prices)
        baseline_volatility_bps = _population_stdev(baseline_returns) * 10_000
        expected_shock_vol_bps = baseline_volatility_bps * sqrt(
            max(len(shock_returns), 1)
        )
        shock_zscore = (
            abs(shock_move_bps) / max(expected_shock_vol_bps, 1e-12)
        )
        realized_volatility = _population_stdev(_log_returns(recent_prices))
        signal_direction = self._signal_direction(
            shock_move_bps=shock_move_bps,
            reversal_bps=reversal_bps,
            shock_zscore=shock_zscore,
            shock_efficiency=shock_efficiency,
        )
        expected_edge_bps = (
            abs(reversal_bps) + (abs(shock_move_bps) * 0.25)
            if signal_direction != SignalDirection.FLAT
            else 0.0
        )

        return ExhaustionReversalReading(
            shock_start_price=shock_start_price,
            previous_price=previous_price,
            last_price=last_price,
            shock_move_bps=shock_move_bps,
            reversal_bps=reversal_bps,
            shock_zscore=shock_zscore,
            shock_efficiency=shock_efficiency,
            baseline_volatility_bps=baseline_volatility_bps,
            realized_volatility=realized_volatility,
            expected_edge_bps=expected_edge_bps,
            signal_direction=signal_direction,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_exhaustion(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for exhaustion reversal reading",
                primary_signal="exhaustion_reversal",
            )

        diagnostics = _exhaustion_reversal_diagnostics(reading) + (
            _exhaustion_reversal_session_diagnostics(
                self._entry_session_allowed(quote)
            )
        )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "exhaustion reversal max holding period "
                    f"{self.config.max_holding_period} bars reached"
                ),
                diagnostics,
                primary_signal="exhaustion_reversal",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="exhaustion_reversal",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="exhaustion_reversal",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite exhaustion reversal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="exhaustion_reversal",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite exhaustion reversal; {reason}",
                    diagnostics,
                    primary_signal="exhaustion_reversal",
                )

        if self._reversal_no_longer_supports_position(reading, current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "exhaustion exit signal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="exhaustion_reversal",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "reversal impulse no longer supports exhaustion position",
                diagnostics,
                primary_signal="exhaustion_reversal",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"exhaustion reversal still supports current position after {holding_period} bars",
            diagnostics,
            primary_signal="exhaustion_reversal",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: ExhaustionReversalReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                f"outside exhaustion reversal UTC hours ({allowed}); current hour={utc_hour}",
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, "shock and reversal pattern not aligned"
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        session = "quote-less" if utc_hour is None else f"{utc_hour:02d}:00 UTC"
        return (
            True,
            (
                f"{direction} exhaustion reversal at {session}: "
                f"shock={reading.shock_move_bps:.1f} bps, "
                f"reversal={reading.reversal_bps:.1f} bps, "
                f"z={reading.shock_zscore:.2f}"
            ),
        )

    def _signal_direction(
        self,
        *,
        shock_move_bps: float,
        reversal_bps: float,
        shock_zscore: float,
        shock_efficiency: float,
    ) -> SignalDirection:
        if shock_zscore < self.config.min_shock_zscore:
            return SignalDirection.FLAT
        if shock_efficiency < self.config.min_shock_efficiency:
            return SignalDirection.FLAT
        if (
            shock_move_bps >= self.config.min_shock_bps
            and reversal_bps <= -self.config.min_reversal_bps
        ):
            return SignalDirection.SHORT
        if (
            shock_move_bps <= -self.config.min_shock_bps
            and reversal_bps >= self.config.min_reversal_bps
        ):
            return SignalDirection.LONG
        return SignalDirection.FLAT

    def _reversal_no_longer_supports_position(
        self,
        reading: ExhaustionReversalReading,
        current_direction: int,
    ) -> bool:
        if current_direction > 0:
            return reading.reversal_bps <= -self.config.min_reversal_bps
        if current_direction < 0:
            return reading.reversal_bps >= self.config.min_reversal_bps
        return False

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: ExhaustionReversalReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.shock_move_bps),
                max(self.config.min_shock_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.shock_zscore,
                max(self.config.min_shock_zscore, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class LiquiditySweepReversalStrategy:
    def __init__(self, config: LiquiditySweepReversalConfig | None = None) -> None:
        self.config = config or LiquiditySweepReversalConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_liquidity_sweep(
        self,
        prices: Sequence[float],
    ) -> LiquiditySweepReversalReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        baseline = recent_prices[:-2]
        previous_price = recent_prices[-2]
        last_price = recent_prices[-1]
        prior_high = max(baseline)
        prior_low = min(baseline)
        midpoint_price = sqrt(prior_high * prior_low)
        range_width_bps = log(prior_high / prior_low) * 10_000 if prior_high > prior_low else 0.0
        baseline_returns = _log_returns(baseline)
        baseline_move = sum(baseline_returns)
        baseline_path = sum(abs(value) for value in baseline_returns)
        trend_efficiency = (
            abs(baseline_move) / baseline_path
            if baseline_path > 0
            else 0.0
        )
        realized_volatility = _population_stdev(_log_returns(recent_prices))

        high_sweep_bps = log(previous_price / prior_high) * 10_000
        low_sweep_bps = log(prior_low / previous_price) * 10_000
        high_reentry_bps = log(prior_high / last_price) * 10_000
        low_reentry_bps = log(last_price / prior_low) * 10_000

        signal_direction = SignalDirection.FLAT
        sweep_bps = max(high_sweep_bps, low_sweep_bps, 0.0)
        reentry_bps = 0.0
        expected_edge_bps = 0.0
        if (
            high_sweep_bps >= self.config.min_sweep_bps
            and high_reentry_bps >= self.config.reentry_buffer_bps
        ):
            signal_direction = SignalDirection.SHORT
            sweep_bps = high_sweep_bps
            reentry_bps = high_reentry_bps
            expected_edge_bps = max(log(last_price / midpoint_price) * 10_000, 0.0)
        elif (
            low_sweep_bps >= self.config.min_sweep_bps
            and low_reentry_bps >= self.config.reentry_buffer_bps
        ):
            signal_direction = SignalDirection.LONG
            sweep_bps = low_sweep_bps
            reentry_bps = low_reentry_bps
            expected_edge_bps = max(log(midpoint_price / last_price) * 10_000, 0.0)

        return LiquiditySweepReversalReading(
            prior_high=prior_high,
            prior_low=prior_low,
            midpoint_price=midpoint_price,
            previous_price=previous_price,
            last_price=last_price,
            range_width_bps=range_width_bps,
            sweep_bps=sweep_bps,
            reentry_bps=reentry_bps,
            expected_edge_bps=expected_edge_bps,
            realized_volatility=realized_volatility,
            trend_efficiency=trend_efficiency,
            signal_direction=signal_direction,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_liquidity_sweep(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for liquidity sweep reversal reading",
                primary_signal="liquidity_sweep_reversal",
            )

        session_allowed, utc_hour = self._entry_session_allowed(quote)
        diagnostics = _liquidity_sweep_reversal_diagnostics(reading) + (
            _liquidity_sweep_reversal_session_diagnostics((session_allowed, utc_hour))
        )
        if current_direction != 0 and not session_allowed:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"liquidity sweep session ended at hour {utc_hour}",
                diagnostics,
                primary_signal="liquidity_sweep_reversal",
            )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"liquidity sweep max holding period {self.config.max_holding_period} bars reached",
                diagnostics,
                primary_signal="liquidity_sweep_reversal",
            )
        if current_direction != 0 and self._midpoint_target_reached(reading, current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "liquidity sweep midpoint reached but minimum holding period not reached",
                    diagnostics,
                    primary_signal="liquidity_sweep_reversal",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "liquidity sweep mean target reached near range midpoint",
                diagnostics,
                primary_signal="liquidity_sweep_reversal",
            )
        if current_direction != 0 and self._sweep_invalidated(reading, current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "liquidity sweep invalidation seen but minimum holding period not reached",
                    diagnostics,
                    primary_signal="liquidity_sweep_reversal",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "liquidity sweep invalidated by renewed range break",
                diagnostics,
                primary_signal="liquidity_sweep_reversal",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="liquidity_sweep_reversal",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="liquidity_sweep_reversal",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "opposite liquidity sweep seen but minimum holding period not reached",
                    diagnostics,
                    primary_signal="liquidity_sweep_reversal",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite liquidity sweep reversal; {reason}",
                    diagnostics,
                    primary_signal="liquidity_sweep_reversal",
                )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"liquidity sweep reversal still managing position after {holding_period} bars",
            diagnostics,
            primary_signal="liquidity_sweep_reversal",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: LiquiditySweepReversalReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                f"outside liquidity sweep UTC hours ({allowed}); current hour={utc_hour}",
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, "no confirmed liquidity sweep back inside prior range"
        if reading.range_width_bps < self.config.min_range_width_bps:
            return (
                False,
                (
                    f"range width {reading.range_width_bps:.1f} bps below "
                    f"{self.config.min_range_width_bps:.1f} bps minimum"
                ),
            )
        if reading.sweep_bps > self.config.max_sweep_bps:
            return (
                False,
                (
                    f"sweep {reading.sweep_bps:.1f} bps above "
                    f"{self.config.max_sweep_bps:.1f} bps safety cap"
                ),
            )
        if reading.trend_efficiency > self.config.max_trend_efficiency:
            return (
                False,
                (
                    f"trend efficiency {reading.trend_efficiency:.2f} above "
                    f"{self.config.max_trend_efficiency:.2f} fade limit"
                ),
            )
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        session = "quote-less" if utc_hour is None else f"{utc_hour:02d}:00 UTC"
        return (
            True,
            (
                f"{direction} liquidity sweep reversal at {session}: "
                f"sweep={reading.sweep_bps:.1f} bps, "
                f"reentry={reading.reentry_bps:.1f} bps, "
                f"edge={reading.expected_edge_bps:.1f} bps"
            ),
        )

    def _midpoint_target_reached(
        self,
        reading: LiquiditySweepReversalReading,
        current_direction: int,
    ) -> bool:
        if current_direction > 0:
            return reading.last_price >= reading.midpoint_price
        if current_direction < 0:
            return reading.last_price <= reading.midpoint_price
        return False

    def _sweep_invalidated(
        self,
        reading: LiquiditySweepReversalReading,
        current_direction: int,
    ) -> bool:
        high_stop = reading.prior_high * exp(self.config.min_sweep_bps / 10_000)
        low_stop = reading.prior_low * exp(-self.config.min_sweep_bps / 10_000)
        if current_direction > 0:
            return reading.last_price <= low_stop
        if current_direction < 0:
            return reading.last_price >= high_stop
        return False

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: LiquiditySweepReversalReading) -> float:
        confidence = min(
            _bounded_confidence(
                reading.sweep_bps,
                max(self.config.min_sweep_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.range_width_bps,
                max(self.config.min_range_width_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class FixingReversalStrategy:
    def __init__(self, config: FixingReversalConfig | None = None) -> None:
        self.config = config or FixingReversalConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_fixing_reversal(
        self,
        prices: Sequence[float],
    ) -> FixingReversalReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        pre_fix_prices = recent_prices[:-1]
        anchor_price = pre_fix_prices[0]
        previous_price = pre_fix_prices[-1]
        last_price = recent_prices[-1]
        pre_fix_returns = _log_returns(pre_fix_prices)
        pre_fix_log_return = log(previous_price / anchor_price)
        pre_fix_path = sum(abs(value) for value in pre_fix_returns)
        pre_fix_efficiency = (
            abs(pre_fix_log_return) / pre_fix_path
            if pre_fix_path > 0
            else 0.0
        )
        pre_fix_move_bps = pre_fix_log_return * 10_000
        confirmation_bps = log(last_price / previous_price) * 10_000
        realized_volatility_bps = _population_stdev(_log_returns(recent_prices)) * 10_000
        signal_direction = self._signal_direction(
            pre_fix_move_bps=pre_fix_move_bps,
            confirmation_bps=confirmation_bps,
            pre_fix_efficiency=pre_fix_efficiency,
            realized_volatility_bps=realized_volatility_bps,
        )
        expected_edge_bps = (
            min(
                abs(pre_fix_move_bps),
                (abs(pre_fix_move_bps) * 0.5) + abs(confirmation_bps),
            )
            if signal_direction != SignalDirection.FLAT
            else 0.0
        )

        return FixingReversalReading(
            anchor_price=anchor_price,
            previous_price=previous_price,
            last_price=last_price,
            pre_fix_move_bps=pre_fix_move_bps,
            confirmation_bps=confirmation_bps,
            pre_fix_efficiency=pre_fix_efficiency,
            realized_volatility_bps=realized_volatility_bps,
            expected_edge_bps=expected_edge_bps,
            signal_direction=signal_direction,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_fixing_reversal(prices)
        current_direction = _notional_direction(current_notional_usd)
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for fixing reversal reading",
                primary_signal="fixing_reversal",
            )

        diagnostics = _fixing_reversal_diagnostics(reading) + (
            _fixing_reversal_session_diagnostics((session_allowed, utc_hour))
        )
        if current_direction != 0 and not session_allowed:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"fixing reversal session ended at hour {utc_hour}",
                diagnostics,
                primary_signal="fixing_reversal",
            )
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"fixing reversal max holding period "
                    f"{self.config.max_holding_period} bars reached"
                ),
                diagnostics,
                primary_signal="fixing_reversal",
            )

        passed, reason = self._passes_entry_filters(
            reading=reading,
            session_allowed=session_allowed,
            utc_hour=utc_hour,
            quote=quote,
        )
        signal_direction = _signal_direction_sign(reading.signal_direction)

        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="fixing_reversal",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="fixing_reversal",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite fixing reversal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="fixing_reversal",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite fixing reversal; {reason}",
                    diagnostics,
                    primary_signal="fixing_reversal",
                )

        if self._reversal_no_longer_supports_position(reading, current_direction):
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "fixing reversal exit signal seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="fixing_reversal",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "post-fix confirmation no longer supports reversal position",
                diagnostics,
                primary_signal="fixing_reversal",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"fixing reversal still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="fixing_reversal",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: FixingReversalReading,
        session_allowed: bool,
        utc_hour: int | None,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        if not session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "none"
                if allowed_hours == ()
                else "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                f"outside fixing reversal UTC hours ({allowed}); current hour={utc_hour}",
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, "pre-fix move and reversal confirmation not aligned"
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        session = "quote-less" if utc_hour is None else f"{utc_hour:02d}:00 UTC"
        return (
            True,
            (
                f"{direction} fixing reversal at {session}: "
                f"pre-fix={reading.pre_fix_move_bps:.1f} bps, "
                f"confirmation={reading.confirmation_bps:.1f} bps, "
                f"efficiency={reading.pre_fix_efficiency:.2f}"
            ),
        )

    def _signal_direction(
        self,
        *,
        pre_fix_move_bps: float,
        confirmation_bps: float,
        pre_fix_efficiency: float,
        realized_volatility_bps: float,
    ) -> SignalDirection:
        abs_pre_fix_move = abs(pre_fix_move_bps)
        if abs_pre_fix_move < self.config.min_pre_fix_move_bps:
            return SignalDirection.FLAT
        if abs_pre_fix_move > self.config.max_pre_fix_move_bps:
            return SignalDirection.FLAT
        if pre_fix_efficiency < self.config.min_pre_fix_efficiency:
            return SignalDirection.FLAT
        if realized_volatility_bps < self.config.min_realized_volatility_bps:
            return SignalDirection.FLAT
        if realized_volatility_bps > self.config.max_realized_volatility_bps:
            return SignalDirection.FLAT
        if (
            pre_fix_move_bps >= self.config.min_pre_fix_move_bps
            and confirmation_bps <= -self.config.min_reversal_confirmation_bps
        ):
            return SignalDirection.SHORT
        if (
            pre_fix_move_bps <= -self.config.min_pre_fix_move_bps
            and confirmation_bps >= self.config.min_reversal_confirmation_bps
        ):
            return SignalDirection.LONG
        return SignalDirection.FLAT

    def _reversal_no_longer_supports_position(
        self,
        reading: FixingReversalReading,
        current_direction: int,
    ) -> bool:
        if current_direction > 0:
            return reading.confirmation_bps <= -self.config.min_reversal_confirmation_bps
        if current_direction < 0:
            return reading.confirmation_bps >= self.config.min_reversal_confirmation_bps
        return False

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: FixingReversalReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.pre_fix_move_bps),
                max(self.config.min_pre_fix_move_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(reading.confirmation_bps),
                max(self.config.min_reversal_confirmation_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility_bps / 10_000,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class MovingAverageCrossoverStrategy:
    def __init__(self, config: MovingAverageCrossoverConfig | None = None) -> None:
        self.config = config or MovingAverageCrossoverConfig()

    def read_crossover(
        self,
        prices: Sequence[float],
    ) -> MovingAverageCrossoverReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.slow_window)
        if recent_prices is None:
            return None

        fast_average = _simple_average(recent_prices[-self.config.fast_window :])
        slow_average = _simple_average(recent_prices)
        separation_bps = ((fast_average / slow_average) - 1.0) * 10_000
        previous_fast_average: float | None = None
        previous_slow_average: float | None = None
        previous_separation_bps: float | None = None
        crossed_direction = SignalDirection.FLAT

        previous_prices = _recent_valid_prices(prices[:-1], self.config.slow_window)
        if previous_prices is not None:
            previous_fast_average = _simple_average(previous_prices[-self.config.fast_window :])
            previous_slow_average = _simple_average(previous_prices)
            previous_separation_bps = (
                (previous_fast_average / previous_slow_average) - 1.0
            ) * 10_000
            if previous_separation_bps <= 0 < separation_bps:
                crossed_direction = SignalDirection.LONG
            elif previous_separation_bps >= 0 > separation_bps:
                crossed_direction = SignalDirection.SHORT

        log_returns = _log_returns(recent_prices)
        cumulative_log_return = sum(log_returns)
        path_log_return = sum(abs(value) for value in log_returns)
        trend_efficiency = (
            abs(cumulative_log_return) / path_log_return
            if path_log_return > 0
            else 0.0
        )

        return MovingAverageCrossoverReading(
            fast_average=fast_average,
            slow_average=slow_average,
            previous_fast_average=previous_fast_average,
            previous_slow_average=previous_slow_average,
            last_price=recent_prices[-1],
            separation_bps=separation_bps,
            previous_separation_bps=previous_separation_bps,
            crossed_direction=crossed_direction,
            realized_volatility=_population_stdev(log_returns),
            trend_efficiency=trend_efficiency,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_crossover(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for moving-average crossover reading",
            )

        diagnostics = _ma_crossover_diagnostics(reading)
        signal_direction = _separation_direction(reading.separation_bps)
        if abs(reading.separation_bps) <= self.config.exit_separation_bps:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    (
                        f"average separation {reading.separation_bps:.1f} bps is inside "
                        f"exit band {self.config.exit_separation_bps:.1f} bps"
                    ),
                    diagnostics,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"average separation faded to {reading.separation_bps:.1f} bps "
                    f"after {holding_period} bars"
                ),
                diagnostics,
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite moving-average signal; {reason}",
                    diagnostics,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "moving-average relationship flipped but did not clear entry filters",
                diagnostics,
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"moving-average trend still supports current position after {holding_period} bars",
            diagnostics,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: MovingAverageCrossoverReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        abs_separation = abs(reading.separation_bps)
        if abs_separation < self.config.min_separation_bps:
            return (
                False,
                (
                    f"average separation {reading.separation_bps:.1f} bps below "
                    f"{self.config.min_separation_bps:.1f} bps entry threshold"
                ),
            )
        if reading.trend_efficiency < self.config.min_trend_efficiency:
            return (
                False,
                (
                    f"trend efficiency {reading.trend_efficiency:.2f} below "
                    f"{self.config.min_trend_efficiency:.2f}"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_separation,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason

        side = "above" if reading.separation_bps > 0 else "below"
        cross_text = (
            f", fresh {reading.crossed_direction.value.lower()} cross"
            if reading.crossed_direction != SignalDirection.FLAT
            else ""
        )
        return (
            True,
            (
                f"{self.config.fast_window}/{self.config.slow_window} moving average "
                f"fast average {side} slow by {reading.separation_bps:.1f} bps"
                f"{cross_text}"
            ),
        )

    def _sized_notional(self, reading: MovingAverageCrossoverReading) -> float:
        confidence = _bounded_confidence(
            abs(reading.separation_bps),
            self.config.min_separation_bps,
        )
        if self.config.min_trend_efficiency > 0:
            confidence = min(
                confidence,
                _bounded_confidence(
                    reading.trend_efficiency,
                    self.config.min_trend_efficiency,
                ),
            )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class MacdMomentumStrategy:
    def __init__(self, config: MacdMomentumConfig | None = None) -> None:
        self.config = config or MacdMomentumConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_macd(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> MacdMomentumReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        fast_series = _ema_series(recent_prices, self.config.fast_window)
        slow_series = _ema_series(recent_prices, self.config.slow_window)
        macd_series = tuple(
            fast - slow for fast, slow in zip(fast_series, slow_series, strict=True)
        )
        signal_series = _ema_series(macd_series, self.config.signal_window)
        histogram_series = tuple(
            macd - signal
            for macd, signal in zip(macd_series, signal_series, strict=True)
        )

        last_price = recent_prices[-1]
        histogram = histogram_series[-1]
        previous_histogram = histogram_series[-2]
        macd = macd_series[-1]
        signal = signal_series[-1]
        macd_bps = (macd / last_price) * 10_000
        signal_bps = (signal / last_price) * 10_000
        histogram_bps = (histogram / last_price) * 10_000
        previous_histogram_bps = (previous_histogram / recent_prices[-2]) * 10_000
        histogram_slope_bps = histogram_bps - previous_histogram_bps

        crossed_direction = SignalDirection.FLAT
        if previous_histogram_bps <= 0 < histogram_bps:
            crossed_direction = SignalDirection.LONG
        elif previous_histogram_bps >= 0 > histogram_bps:
            crossed_direction = SignalDirection.SHORT

        log_returns = _log_returns(recent_prices)
        cumulative_log_return = sum(log_returns)
        path_log_return = sum(abs(value) for value in log_returns)
        trend_efficiency = (
            abs(cumulative_log_return) / path_log_return
            if path_log_return > 0
            else 0.0
        )
        session_allowed, utc_hour = self._entry_session_allowed(quote)

        return MacdMomentumReading(
            fast_ema=fast_series[-1],
            slow_ema=slow_series[-1],
            macd=macd,
            signal=signal,
            histogram=histogram,
            previous_histogram=previous_histogram,
            macd_bps=macd_bps,
            signal_bps=signal_bps,
            histogram_bps=histogram_bps,
            previous_histogram_bps=previous_histogram_bps,
            histogram_slope_bps=histogram_slope_bps,
            crossed_direction=crossed_direction,
            last_price=last_price,
            realized_volatility=_population_stdev(log_returns),
            realized_volatility_bps=_population_stdev(log_returns) * 10_000,
            trend_efficiency=trend_efficiency,
            session_allowed=session_allowed,
            utc_hour=utc_hour,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_macd(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for MACD momentum reading",
            )

        diagnostics = _macd_momentum_diagnostics(reading)
        signal_direction = _separation_direction(reading.histogram_bps)

        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"MACD momentum max holding period {self.config.max_holding_period} bars reached",
                diagnostics,
                primary_signal="macd_momentum",
            )

        if abs(reading.histogram_bps) <= self.config.exit_histogram_bps:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    (
                        f"MACD histogram {reading.histogram_bps:.2f} bps is inside "
                        f"exit band {self.config.exit_histogram_bps:.2f} bps"
                    ),
                    diagnostics,
                    primary_signal="macd_momentum",
                )
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "MACD histogram softened but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="macd_momentum",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"MACD histogram faded after {holding_period} bars",
                diagnostics,
                primary_signal="macd_momentum",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="macd_momentum",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="macd_momentum",
            )

        if signal_direction != 0 and signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite MACD histogram but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="macd_momentum",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite MACD momentum signal; {reason}",
                    diagnostics,
                    primary_signal="macd_momentum",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "MACD histogram flipped but did not clear entry filters",
                diagnostics,
                primary_signal="macd_momentum",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"MACD momentum still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="macd_momentum",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: MacdMomentumReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        if not reading.session_allowed:
            allowed = self._allowed_entry_hours()
            return (
                False,
                (
                    f"outside MACD momentum UTC hours ({allowed}); "
                    f"current hour={reading.utc_hour}"
                ),
            )
        abs_histogram = abs(reading.histogram_bps)
        abs_macd = abs(reading.macd_bps)
        if abs_histogram < self.config.min_histogram_bps:
            return (
                False,
                (
                    f"MACD histogram {reading.histogram_bps:.2f} bps below "
                    f"{self.config.min_histogram_bps:.2f} bps threshold"
                ),
            )
        if abs_macd < self.config.min_macd_bps:
            return (
                False,
                (
                    f"MACD line {reading.macd_bps:.2f} bps below "
                    f"{self.config.min_macd_bps:.2f} bps threshold"
                ),
            )
        if reading.histogram_bps * reading.macd_bps <= 0:
            return False, "MACD line and histogram disagree on direction"
        if abs(reading.histogram_slope_bps) < self.config.min_histogram_slope_bps:
            return (
                False,
                (
                    f"MACD histogram slope {reading.histogram_slope_bps:.2f} bps below "
                    f"{self.config.min_histogram_slope_bps:.2f} bps threshold"
                ),
            )
        if reading.trend_efficiency < self.config.min_trend_efficiency:
            return (
                False,
                (
                    f"trend efficiency {reading.trend_efficiency:.2f} below "
                    f"{self.config.min_trend_efficiency:.2f}"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs_histogram,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason

        direction = "long" if reading.histogram_bps > 0 else "short"
        cross_text = (
            f", fresh {reading.crossed_direction.value.lower()} histogram cross"
            if reading.crossed_direction != SignalDirection.FLAT
            else ""
        )
        return (
            True,
            (
                f"MACD {self.config.fast_window}/{self.config.slow_window}/"
                f"{self.config.signal_window} {direction} histogram "
                f"{reading.histogram_bps:.2f} bps, macd={reading.macd_bps:.2f} bps"
                f"{cross_text}; {cost_reason}"
            ),
        )

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: MacdMomentumReading) -> float:
        confidence = min(
            _bounded_confidence(
                abs(reading.histogram_bps),
                max(self.config.min_histogram_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(reading.macd_bps),
                max(self.config.min_macd_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class MacdConditionalFallbackStrategy:
    def __init__(
        self,
        config: MacdConditionalFallbackConfig | None = None,
        *,
        macd_momentum: MacdMomentumConfig | None = None,
        conditional_seasonality: ConditionalSeasonalityConfig | None = None,
    ) -> None:
        self.config = config or MacdConditionalFallbackConfig()
        self.macd = MacdMomentumStrategy(
            replace(macd_momentum or MacdMomentumConfig(), symbol=self.config.symbol)
        )
        self.conditional = ConditionalSeasonalityStrategy(
            replace(
                conditional_seasonality or ConditionalSeasonalityConfig(),
                symbol=self.config.symbol,
            )
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        macd_decision = self.macd.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        if macd_decision.is_trade_intent or macd_decision.action == StrategyAction.HOLD:
            return macd_decision
        if not self._fallback_allowed(macd_decision.reason):
            return macd_decision

        conditional_decision = self.conditional.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        if conditional_decision.is_trade_intent:
            return self._wrap_conditional_decision(
                conditional_decision,
                macd_reason=macd_decision.reason,
            )

        return _decision(
            StrategyAction.NO_ACTION,
            self.config.symbol,
            0.0,
            (
                "MACD inactive and conditional fallback inactive; "
                f"macd={macd_decision.reason}; "
                f"conditional={conditional_decision.reason}"
            ),
            macd_decision.diagnostics + conditional_decision.diagnostics,
            primary_signal="macd_conditional_fallback",
            supporting_signals=(
                macd_decision.primary_signal,
                conditional_decision.primary_signal,
            ),
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _fallback_allowed(self, macd_reason: str) -> bool:
        normalized = macd_reason.lower()
        return any(
            keyword in normalized
            for keyword in self.config.macd_inactive_reason_keywords
        )

    def _wrap_conditional_decision(
        self,
        decision: StrategyDecision,
        *,
        macd_reason: str,
    ) -> StrategyDecision:
        target = decision.target_notional_usd
        if decision.action in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            target *= self.config.conditional_notional_multiplier
        return _decision(
            decision.action,
            self.config.symbol,
            target,
            (
                "conditional fallback after inactive MACD; "
                f"macd={macd_reason}; conditional={decision.reason}; "
                f"fallback_size={self.config.conditional_notional_multiplier:.2f}"
            ),
            decision.diagnostics,
            primary_signal="macd_conditional_fallback",
            supporting_signals=("macd_momentum", decision.primary_signal)
            + decision.supporting_signals,
            conflicting_signals=decision.conflicting_signals,
        )


class MacdSqueezeComplementStrategy:
    def __init__(
        self,
        config: MacdSqueezeComplementConfig | None = None,
        *,
        macd_momentum: MacdMomentumConfig | None = None,
        volatility_squeeze: VolatilitySqueezeConfig | None = None,
    ) -> None:
        self.config = config or MacdSqueezeComplementConfig()
        self.macd = MacdMomentumStrategy(
            replace(macd_momentum or MacdMomentumConfig(), symbol=self.config.symbol)
        )
        self.squeeze = VolatilitySqueezeStrategy(
            replace(
                volatility_squeeze or VolatilitySqueezeConfig(),
                symbol=self.config.symbol,
            )
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        macd_decision = self.macd.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        if macd_decision.is_trade_intent or macd_decision.action == StrategyAction.HOLD:
            return macd_decision
        if not self._squeeze_allowed(macd_decision.reason):
            return macd_decision

        squeeze_decision = self.squeeze.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        if squeeze_decision.is_trade_intent or squeeze_decision.action == StrategyAction.HOLD:
            return self._wrap_squeeze_decision(
                squeeze_decision,
                macd_reason=macd_decision.reason,
            )

        return _decision(
            StrategyAction.NO_ACTION,
            self.config.symbol,
            0.0,
            (
                "MACD inactive and squeeze complement inactive; "
                f"macd={macd_decision.reason}; "
                f"squeeze={squeeze_decision.reason}"
            ),
            macd_decision.diagnostics + squeeze_decision.diagnostics,
            primary_signal="macd_squeeze_complement",
            supporting_signals=(
                macd_decision.primary_signal,
                squeeze_decision.primary_signal,
            ),
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _squeeze_allowed(self, macd_reason: str) -> bool:
        if not self.config.macd_inactive_reason_keywords:
            return True
        normalized = macd_reason.lower()
        return any(
            keyword in normalized
            for keyword in self.config.macd_inactive_reason_keywords
        )

    def _wrap_squeeze_decision(
        self,
        decision: StrategyDecision,
        *,
        macd_reason: str,
    ) -> StrategyDecision:
        target = decision.target_notional_usd
        if decision.action in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            target *= self.config.squeeze_notional_multiplier
        return _decision(
            decision.action,
            self.config.symbol,
            target,
            (
                "squeeze complement after inactive MACD; "
                f"macd={macd_reason}; squeeze={decision.reason}; "
                f"squeeze_size={self.config.squeeze_notional_multiplier:.2f}"
            ),
            decision.diagnostics,
            primary_signal="macd_squeeze_complement",
            supporting_signals=("macd_momentum", decision.primary_signal)
            + decision.supporting_signals,
            conflicting_signals=decision.conflicting_signals,
        )


class MeanReversionStrategy:
    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    def read_reversion(self, prices: Sequence[float]) -> MeanReversionReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        baseline = recent_prices[:-1]
        latest = recent_prices[-1]
        mean_price = sum(baseline) / len(baseline)
        stdev_price = _population_stdev(baseline)
        residual = latest - mean_price
        zscore = 0.0 if stdev_price == 0 else residual / stdev_price
        deviation_bps = (residual / mean_price) * 10_000
        trend_log_returns = _log_returns(baseline)
        trend_strength_bps = sum(trend_log_returns) * 10_000
        path_log_return = sum(abs(value) for value in trend_log_returns)
        trend_efficiency = (
            abs(sum(trend_log_returns)) / path_log_return
            if path_log_return > 0
            else 0.0
        )

        return MeanReversionReading(
            mean_price=mean_price,
            last_price=latest,
            stdev_price=stdev_price,
            residual=residual,
            zscore=zscore,
            deviation_bps=deviation_bps,
            trend_strength_bps=trend_strength_bps,
            trend_efficiency=trend_efficiency,
            estimated_half_life=None,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_reversion(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for mean-reversion reading",
            )

        diagnostics = _reversion_diagnostics(reading)
        stdev_bps = (reading.stdev_price / reading.mean_price) * 10_000
        if stdev_bps <= self.config.min_stdev_bps:
            return self._flat_or_exit(
                current_direction=current_direction,
                current_notional_usd=current_notional_usd,
                reason=(
                    f"baseline volatility {stdev_bps:.2f} bps too small for "
                    "meaningful z-score"
                ),
                diagnostics=diagnostics,
            )

        trend_too_strong = abs(reading.trend_strength_bps) > self.config.max_trend_bps
        if trend_too_strong:
            return self._flat_or_exit(
                current_direction=current_direction,
                current_notional_usd=current_notional_usd,
                reason=(
                    f"trend filter blocked mean reversion: "
                    f"{reading.trend_strength_bps:.1f} bps"
                ),
                diagnostics=diagnostics,
            )

        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"max holding period reached: {holding_period} bars",
                diagnostics,
            )

        if current_direction != 0 and abs(reading.zscore) >= self.config.stop_zscore:
            signal_direction = _reversion_direction(reading.zscore)
            if signal_direction != current_direction:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite reversion stop signal z={reading.zscore:.2f}",
                    diagnostics,
                )
            return _decision(
                StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                f"large z-score still supports current position z={reading.zscore:.2f}",
                diagnostics,
            )

        if current_direction != 0 and abs(reading.zscore) <= self.config.exit_zscore:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"z-score {reading.zscore:.2f} reached exit threshold "
                    f"{self.config.exit_zscore:.2f}"
                ),
                diagnostics,
            )

        if abs(reading.zscore) < self.config.entry_zscore:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    (
                        f"z-score {reading.zscore:.2f} below entry threshold "
                        f"{self.config.entry_zscore:.2f}"
                    ),
                    diagnostics,
                )
            return _decision(
                StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                f"z-score {reading.zscore:.2f} remains between entry and exit",
                diagnostics,
            )

        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=abs(reading.deviation_bps),
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return self._flat_or_exit(
                current_direction=current_direction,
                current_notional_usd=current_notional_usd,
                reason=cost_reason,
                diagnostics=diagnostics,
            )

        signal_direction = _reversion_direction(reading.zscore)
        target = signal_direction * self._sized_notional(reading)
        if current_direction == 0:
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                target,
                (
                    f"{self.config.lookback}-price mean reversion z={reading.zscore:.2f}, "
                    f"deviation={reading.deviation_bps:.1f} bps"
                ),
                diagnostics,
            )
        if signal_direction != current_direction:
            return _decision(
                StrategyAction.REVERSE,
                self.config.symbol,
                target,
                f"opposite mean-reversion signal z={reading.zscore:.2f}",
                diagnostics,
            )
        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"mean-reversion signal still supports current position z={reading.zscore:.2f}",
            diagnostics,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _flat_or_exit(
        self,
        *,
        current_direction: int,
        current_notional_usd: float,
        reason: str,
        diagnostics: tuple[tuple[str, float | str], ...],
    ) -> StrategyDecision:
        if current_direction == 0:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                reason,
                diagnostics,
            )
        return _decision(
            StrategyAction.EXIT,
            self.config.symbol,
            0.0,
            reason,
            diagnostics,
        )

    def _sized_notional(self, reading: MeanReversionReading) -> float:
        confidence = _bounded_confidence(abs(reading.zscore), self.config.entry_zscore)
        realized_volatility = reading.stdev_price / reading.mean_price
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class RegimeSwitchingStrategy:
    def __init__(
        self,
        *,
        config: RegimeConfig | None = None,
        momentum: MomentumConfig | None = None,
        mean_reversion: MeanReversionConfig | None = None,
    ) -> None:
        self.config = config or RegimeConfig()
        self.momentum = SimpleMomentumStrategy(
            replace(momentum or MomentumConfig(), symbol=self.config.symbol)
        )
        self.mean_reversion = MeanReversionStrategy(
            replace(mean_reversion or MeanReversionConfig(), symbol=self.config.symbol)
        )
        self._active_regime = RegimeState.FLAT
        self._pending_regime = RegimeState.FLAT
        self._pending_count = 0

    @property
    def active_regime(self) -> RegimeState:
        return self._active_regime

    def read_regime(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> RegimeReading | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None

        log_returns = _log_returns(recent_prices)
        cumulative_log_return = sum(log_returns)
        path_log_return = sum(abs(value) for value in log_returns)
        momentum_efficiency = (
            abs(cumulative_log_return) / path_log_return
            if path_log_return > 0
            else 0.0
        )
        realized_volatility = _population_stdev(log_returns)
        momentum_score = (
            cumulative_log_return / max(realized_volatility, self.momentum.config.volatility_floor)
            if cumulative_log_return != 0
            else 0.0
        )
        momentum_move_bps = cumulative_log_return * 10_000

        baseline = recent_prices[:-1]
        latest = recent_prices[-1]
        baseline_mean = sum(baseline) / len(baseline)
        baseline_stdev = _population_stdev(baseline)
        reversion_zscore = (
            0.0 if baseline_stdev == 0 else (latest - baseline_mean) / baseline_stdev
        )
        baseline_returns = _log_returns(baseline)
        reversion_trend_bps = sum(baseline_returns) * 10_000
        reversion_path = sum(abs(value) for value in baseline_returns)
        reversion_efficiency = (
            abs(sum(baseline_returns)) / reversion_path
            if reversion_path > 0
            else 0.0
        )
        spread_bps = quote.spread_bps if quote is not None else 0.0

        candidate, confidence, reason = self._candidate_regime(
            momentum_move_bps=momentum_move_bps,
            momentum_score=momentum_score,
            momentum_efficiency=momentum_efficiency,
            reversion_zscore=reversion_zscore,
            reversion_trend_bps=reversion_trend_bps,
            reversion_efficiency=reversion_efficiency,
            spread_bps=spread_bps,
        )
        return RegimeReading(
            selected=candidate,
            candidate=candidate,
            confidence=confidence,
            reason=reason,
            momentum_move_bps=momentum_move_bps,
            momentum_score=momentum_score,
            momentum_efficiency=momentum_efficiency,
            reversion_zscore=reversion_zscore,
            reversion_trend_bps=reversion_trend_bps,
            reversion_efficiency=reversion_efficiency,
            spread_bps=spread_bps,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_regime(prices, quote=quote)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for regime selection",
            )

        selected = self._apply_hysteresis(reading.candidate)
        reading = replace(reading, selected=selected)
        diagnostics = _regime_diagnostics(reading)

        if selected == RegimeState.FLAT:
            if _notional_direction(current_notional_usd) == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    f"regime FLAT: {reading.reason}",
                    diagnostics,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"regime FLAT: exit because {reading.reason}",
                diagnostics,
            )

        if selected == RegimeState.MOMENTUM:
            decision = self.momentum.generate_decision(
                prices,
                current_notional_usd=current_notional_usd,
                holding_period=holding_period,
                quote=quote,
            )
            return self._regime_decision(decision, reading)

        decision = self.mean_reversion.generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        if (
            decision.action == StrategyAction.NO_ACTION
            and _notional_direction(current_notional_usd) != 0
        ):
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"regime MEAN_REVERSION has no active signal; exit stale position",
                _regime_diagnostics(reading) + decision.diagnostics,
            )
        return self._regime_decision(decision, reading)

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _candidate_regime(
        self,
        *,
        momentum_move_bps: float,
        momentum_score: float,
        momentum_efficiency: float,
        reversion_zscore: float,
        reversion_trend_bps: float,
        reversion_efficiency: float,
        spread_bps: float,
    ) -> tuple[RegimeState, float, str]:
        if self.config.max_spread_bps is not None and spread_bps > self.config.max_spread_bps:
            return (
                RegimeState.FLAT,
                1.0,
                (
                    f"spread {spread_bps:.2f} bps above regime limit "
                    f"{self.config.max_spread_bps:.2f} bps"
                ),
            )

        momentum_confidence = min(
            _bounded_confidence(
                abs(momentum_move_bps),
                max(self.config.momentum_min_move_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(momentum_score),
                max(self.config.momentum_min_score, 1e-12),
            ),
            _bounded_confidence(
                momentum_efficiency,
                max(self.config.momentum_min_efficiency, 1e-12),
            ),
        )
        if (
            abs(momentum_move_bps) >= self.config.momentum_min_move_bps
            and abs(momentum_score) >= self.config.momentum_min_score
            and momentum_efficiency >= self.config.momentum_min_efficiency
        ):
            return (
                RegimeState.MOMENTUM,
                momentum_confidence,
                (
                    f"smooth directional move: {momentum_move_bps:.1f} bps, "
                    f"score={momentum_score:.2f}, efficiency={momentum_efficiency:.2f}"
                ),
            )

        reversion_confidence = min(
            _bounded_confidence(
                abs(reversion_zscore),
                max(self.config.mean_reversion_min_abs_zscore, 1e-12),
            ),
            1.0 - min(
                _bounded_confidence(
                    abs(reversion_trend_bps),
                    max(self.config.mean_reversion_max_trend_bps, 1e-12),
                ),
                1.0,
            ),
            1.0 - min(
                _bounded_confidence(
                    reversion_efficiency,
                    max(self.config.mean_reversion_max_efficiency, 1e-12),
                ),
                1.0,
            ),
        )
        if (
            abs(reversion_zscore) >= self.config.mean_reversion_min_abs_zscore
            and abs(reversion_trend_bps) <= self.config.mean_reversion_max_trend_bps
            and reversion_efficiency <= self.config.mean_reversion_max_efficiency
        ):
            return (
                RegimeState.MEAN_REVERSION,
                reversion_confidence,
                (
                    f"stable residual: z={reversion_zscore:.2f}, "
                    f"trend={reversion_trend_bps:.1f} bps, "
                    f"efficiency={reversion_efficiency:.2f}"
                ),
            )

        return (
            RegimeState.FLAT,
            0.0,
            (
                "ambiguous regime: no smooth trend and no stable reversion "
                f"(momentum={momentum_move_bps:.1f} bps, z={reversion_zscore:.2f})"
            ),
        )

    def _apply_hysteresis(self, candidate: RegimeState) -> RegimeState:
        if candidate == self._active_regime:
            self._pending_regime = candidate
            self._pending_count = 0
            return self._active_regime

        if candidate == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = candidate
            self._pending_count = 1

        if self._pending_count >= self.config.hysteresis_bars:
            self._active_regime = candidate
            self._pending_count = 0
        return self._active_regime

    def _regime_decision(
        self,
        decision: StrategyDecision,
        reading: RegimeReading,
    ) -> StrategyDecision:
        return _decision(
            decision.action,
            decision.symbol,
            decision.target_notional_usd,
            f"regime {reading.selected.value}: {reading.reason}; {decision.reason}",
            _regime_diagnostics(reading) + decision.diagnostics,
        )


class KalmanTrendStrategy:
    def __init__(self, config: KalmanTrendStrategyConfig | None = None) -> None:
        self.config = config or KalmanTrendStrategyConfig()
        self._instrument = instrument_for(self.config.symbol)

    def read_kalman_trend(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> KalmanTrendStrategyReading | None:
        if len(prices) < self.config.lookback:
            return None
        regime_reading = read_kalman_regime(
            prices,
            symbol=self.config.symbol,
            config=KalmanTrendConfig(
                lookback=self.config.lookback,
                process_noise=self.config.process_noise,
                observation_noise=self.config.observation_noise,
                min_abs_slope_bps=self.config.min_abs_slope_bps,
                min_trend_efficiency=self.config.min_trend_efficiency,
                max_realized_volatility_bps=self.config.max_realized_volatility_bps,
            ),
        )
        signal_direction = self._signal_direction(regime_reading)
        expected_edge_bps = (
            abs(regime_reading.kalman_slope_bps) * self.config.expected_holding_bars
            if signal_direction != SignalDirection.FLAT
            else 0.0
        )
        session_allowed, utc_hour = self._entry_session_allowed(quote)
        return KalmanTrendStrategyReading(
            regime_reading=regime_reading,
            expected_edge_bps=expected_edge_bps,
            signal_direction=signal_direction,
            session_allowed=session_allowed,
            utc_hour=utc_hour,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_kalman_trend(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for Kalman trend reading",
                primary_signal="kalman_trend",
            )

        diagnostics = _kalman_trend_diagnostics(reading)
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"Kalman trend max holding period {self.config.max_holding_period} bars reached",
                diagnostics,
                primary_signal="kalman_trend",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        signal_direction = _signal_direction_sign(reading.signal_direction)
        if current_direction == 0:
            if not passed or signal_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="kalman_trend",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                signal_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="kalman_trend",
            )

        if signal_direction == 0:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"Kalman regime no longer trends: {reason}",
                diagnostics,
                primary_signal="kalman_trend",
            )

        if signal_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    (
                        "opposite Kalman trend seen but minimum holding period "
                        f"{self.config.min_holding_period} bars not reached"
                    ),
                    diagnostics,
                    primary_signal="kalman_trend",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    signal_direction * self._sized_notional(reading),
                    f"opposite Kalman trend; {reason}",
                    diagnostics,
                    primary_signal="kalman_trend",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"opposite Kalman trend but entry blocked; {reason}",
                diagnostics,
                primary_signal="kalman_trend",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"Kalman trend still supports position after {holding_period} bars",
            diagnostics,
            primary_signal="kalman_trend",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: KalmanTrendStrategyReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        regime = reading.regime_reading
        if not reading.session_allowed:
            allowed_hours = self._allowed_entry_hours()
            allowed = (
                "all"
                if allowed_hours is None
                else ",".join(str(hour) for hour in allowed_hours)
            )
            return (
                False,
                f"outside Kalman trend UTC hours ({allowed}); current hour={reading.utc_hour}",
            )
        if reading.signal_direction == SignalDirection.FLAT:
            return False, f"Kalman regime is {regime.regime.value}"
        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"expected edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )
        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason
        direction = "long" if reading.signal_direction == SignalDirection.LONG else "short"
        return (
            True,
            (
                f"{direction} Kalman trend: regime={regime.regime.value}, "
                f"slope={regime.kalman_slope_bps:.2f} bps, "
                f"efficiency={regime.trend_efficiency:.2f}, "
                f"edge={reading.expected_edge_bps:.1f} bps"
            ),
        )

    def _signal_direction(
        self,
        reading: TimeSeriesRegimeReading,
    ) -> SignalDirection:
        if reading.regime == TimeSeriesRegime.TREND_UP:
            return SignalDirection.LONG
        if reading.regime == TimeSeriesRegime.TREND_DOWN:
            return SignalDirection.SHORT
        return SignalDirection.FLAT

    def _entry_session_allowed(
        self,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, int | None]:
        utc_hour = quote.timestamp.astimezone(UTC).hour if quote is not None else None
        allowed_hours = self._allowed_entry_hours()
        if utc_hour is None or allowed_hours is None:
            return True, utc_hour
        return utc_hour in allowed_hours, utc_hour

    def _allowed_entry_hours(self) -> tuple[int, ...] | None:
        if self._instrument.asset_class == AssetClass.CRYPTO:
            return self.config.crypto_allowed_utc_hours
        if self._instrument.asset_class == AssetClass.METAL:
            return self.config.metal_allowed_utc_hours
        return self.config.forex_allowed_utc_hours

    def _sized_notional(self, reading: KalmanTrendStrategyReading) -> float:
        regime = reading.regime_reading
        confidence = min(
            _bounded_confidence(
                abs(regime.kalman_slope_bps),
                max(self.config.min_abs_slope_bps, 1e-12),
            ),
            _bounded_confidence(
                regime.trend_efficiency,
                max(self.config.min_trend_efficiency, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=regime.realized_volatility_bps / 10_000,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )


class QualityTrendStrategy:
    def __init__(self, config: QualityTrendConfig | None = None) -> None:
        self.config = config or QualityTrendConfig()
        self.macd = MacdMomentumStrategy(self._macd_config())
        self.kalman = KalmanTrendStrategy(self._kalman_config())

    def read_quality_trend(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> QualityTrendReading | None:
        macd = self.macd.read_macd(prices, quote=quote)
        kalman = self.kalman.read_kalman_trend(prices, quote=quote)
        if macd is None or kalman is None:
            return None

        macd_direction = _signed_direction_to_signal(
            _separation_direction(macd.histogram_bps)
        )
        kalman_direction = kalman.signal_direction
        aligned_direction = (
            macd_direction
            if macd_direction != SignalDirection.FLAT
            and macd_direction == kalman_direction
            else SignalDirection.FLAT
        )
        macd_confidence = min(
            _bounded_confidence(
                abs(macd.histogram_bps),
                max(self.config.macd_min_histogram_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(macd.macd_bps),
                max(self.config.macd_min_macd_bps, 1e-12),
            ),
            _bounded_confidence(
                macd.trend_efficiency,
                max(self.config.macd_min_trend_efficiency, 1e-12),
            ),
        )
        kalman_regime = kalman.regime_reading
        kalman_confidence = min(
            _bounded_confidence(
                abs(kalman_regime.kalman_slope_bps),
                max(self.config.kalman_min_abs_slope_bps, 1e-12),
            ),
            _bounded_confidence(
                kalman_regime.trend_efficiency,
                max(self.config.kalman_min_trend_efficiency, 1e-12),
            ),
            _bounded_confidence(
                kalman.expected_edge_bps,
                max(self.config.kalman_min_expected_edge_bps, 1e-12),
            ),
        )
        return QualityTrendReading(
            macd=macd,
            kalman=kalman,
            macd_direction=macd_direction,
            kalman_direction=kalman_direction,
            aligned_direction=aligned_direction,
            macd_confidence=macd_confidence,
            kalman_confidence=kalman_confidence,
            combined_confidence=min(macd_confidence, kalman_confidence),
            expected_edge_bps=min(abs(macd.histogram_bps), kalman.expected_edge_bps),
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_quality_trend(prices, quote=quote)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            action = (
                StrategyAction.NO_ACTION
                if current_direction == 0
                else StrategyAction.HOLD
            )
            return _decision(
                action,
                self.config.symbol,
                current_notional_usd,
                "not enough prices for quality trend reading",
                primary_signal="quality_trend",
            )

        diagnostics = _quality_trend_diagnostics(reading)
        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"quality trend max holding period {self.config.max_holding_period} bars reached",
                diagnostics,
                primary_signal="quality_trend",
            )

        passed, reason = self._passes_entry_filters(reading=reading, quote=quote)
        target_direction = _signal_direction_sign(reading.aligned_direction)

        if current_direction == 0:
            if not passed:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    reason,
                    diagnostics,
                    primary_signal="quality_trend",
                )
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                target_direction * self._sized_notional(reading),
                reason,
                diagnostics,
                primary_signal="quality_trend",
            )

        if target_direction == 0:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "quality trend agreement faded but minimum holding period not reached",
                    diagnostics,
                    primary_signal="quality_trend",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "quality trend agreement faded or components disagreed",
                diagnostics,
                primary_signal="quality_trend",
            )

        if target_direction != current_direction:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "opposite quality trend seen but minimum holding period not reached",
                    diagnostics,
                    primary_signal="quality_trend",
                )
            if passed:
                return _decision(
                    StrategyAction.REVERSE,
                    self.config.symbol,
                    target_direction * self._sized_notional(reading),
                    f"opposite quality trend; {reason}",
                    diagnostics,
                    primary_signal="quality_trend",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                f"opposite quality trend but entry blocked; {reason}",
                diagnostics,
                primary_signal="quality_trend",
            )

        if reading.combined_confidence <= self.config.exit_combined_confidence:
            if holding_period < self.config.min_holding_period:
                return _decision(
                    StrategyAction.HOLD,
                    self.config.symbol,
                    current_notional_usd,
                    "quality trend confidence faded but minimum holding period not reached",
                    diagnostics,
                    primary_signal="quality_trend",
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    f"quality trend confidence {reading.combined_confidence:.2f} "
                    f"fell below exit threshold {self.config.exit_combined_confidence:.2f}"
                ),
                diagnostics,
                primary_signal="quality_trend",
            )

        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            f"quality trend still aligned after {holding_period} bars",
            diagnostics,
            primary_signal="quality_trend",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _passes_entry_filters(
        self,
        *,
        reading: QualityTrendReading,
        quote: QuoteSnapshot | None,
    ) -> tuple[bool, str]:
        macd_passed, macd_reason = self.macd._passes_entry_filters(
            reading=reading.macd,
            quote=quote,
        )
        if not macd_passed:
            return False, f"MACD gate failed: {macd_reason}"

        kalman_passed, kalman_reason = self.kalman._passes_entry_filters(
            reading=reading.kalman,
            quote=quote,
        )
        if not kalman_passed:
            return False, f"Kalman gate failed: {kalman_reason}"

        if reading.aligned_direction == SignalDirection.FLAT:
            return (
                False,
                (
                    "quality trend components disagree: "
                    f"macd={reading.macd_direction.value}, "
                    f"kalman={reading.kalman_direction.value}"
                ),
            )

        if reading.combined_confidence < self.config.min_combined_confidence:
            return (
                False,
                (
                    f"combined confidence {reading.combined_confidence:.2f} below "
                    f"{self.config.min_combined_confidence:.2f}"
                ),
            )

        if reading.expected_edge_bps < self.config.min_expected_edge_bps:
            return (
                False,
                (
                    f"quality trend edge {reading.expected_edge_bps:.1f} bps below "
                    f"{self.config.min_expected_edge_bps:.1f} bps minimum"
                ),
            )

        cost_ok, cost_reason = _passes_cost_filter(
            edge_bps=reading.expected_edge_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not cost_ok:
            return False, cost_reason

        direction = "long" if reading.aligned_direction == SignalDirection.LONG else "short"
        return (
            True,
            (
                f"{direction} quality trend: confidence={reading.combined_confidence:.2f}, "
                f"edge={reading.expected_edge_bps:.1f} bps; {cost_reason}"
            ),
        )

    def _sized_notional(self, reading: QualityTrendReading) -> float:
        realized_volatility = max(
            reading.macd.realized_volatility,
            reading.kalman.regime_reading.realized_volatility_bps / 10_000,
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=reading.combined_confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )

    def _macd_config(self) -> MacdMomentumConfig:
        return MacdMomentumConfig(
            symbol=self.config.symbol,
            fast_window=self.config.macd_fast_window,
            slow_window=self.config.macd_slow_window,
            signal_window=self.config.macd_signal_window,
            min_histogram_bps=self.config.macd_min_histogram_bps,
            exit_histogram_bps=self.config.macd_exit_histogram_bps,
            min_macd_bps=self.config.macd_min_macd_bps,
            min_histogram_slope_bps=self.config.macd_min_histogram_slope_bps,
            min_trend_efficiency=self.config.macd_min_trend_efficiency,
            forex_allowed_utc_hours=self.config.forex_allowed_utc_hours,
            metal_allowed_utc_hours=self.config.metal_allowed_utc_hours,
            crypto_allowed_utc_hours=self.config.crypto_allowed_utc_hours,
            target_notional_usd=self.config.target_notional_usd,
            position_sizing=self.config.position_sizing,
            target_volatility=self.config.target_volatility,
            volatility_floor=self.config.volatility_floor,
            max_target_notional_usd=self.config.max_target_notional_usd,
            min_trade_notional_usd=self.config.min_trade_notional_usd,
            min_holding_period=self.config.min_holding_period,
            max_holding_period=self.config.max_holding_period,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )

    def _kalman_config(self) -> KalmanTrendStrategyConfig:
        return KalmanTrendStrategyConfig(
            symbol=self.config.symbol,
            lookback=self.config.kalman_lookback,
            process_noise=self.config.kalman_process_noise,
            observation_noise=self.config.kalman_observation_noise,
            min_abs_slope_bps=self.config.kalman_min_abs_slope_bps,
            min_trend_efficiency=self.config.kalman_min_trend_efficiency,
            max_realized_volatility_bps=self.config.kalman_max_realized_volatility_bps,
            expected_holding_bars=self.config.kalman_expected_holding_bars,
            min_expected_edge_bps=self.config.kalman_min_expected_edge_bps,
            forex_allowed_utc_hours=self.config.forex_allowed_utc_hours,
            metal_allowed_utc_hours=self.config.metal_allowed_utc_hours,
            crypto_allowed_utc_hours=self.config.crypto_allowed_utc_hours,
            target_notional_usd=self.config.target_notional_usd,
            position_sizing=self.config.position_sizing,
            target_volatility=self.config.target_volatility,
            volatility_floor=self.config.volatility_floor,
            max_target_notional_usd=self.config.max_target_notional_usd,
            min_trade_notional_usd=self.config.min_trade_notional_usd,
            min_holding_period=self.config.min_holding_period,
            max_holding_period=self.config.max_holding_period,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )


class AlphaRouterStrategy:
    def __init__(
        self,
        *,
        config: AlphaRouterConfig | None = None,
        momentum: MomentumConfig | None = None,
        moving_average: MovingAverageCrossoverConfig | None = None,
        breakout: BreakoutConfig | None = None,
        volatility_squeeze: VolatilitySqueezeConfig | None = None,
        dual_squeeze: DualSqueezeConfig | None = None,
        exhaustion_reversal: ExhaustionReversalConfig | None = None,
        session_breakout: SessionBreakoutConfig | None = None,
        macd_momentum: MacdMomentumConfig | None = None,
        kalman_trend: KalmanTrendStrategyConfig | None = None,
        mean_reversion: MeanReversionConfig | None = None,
        relative_strength: RelativeStrengthConfig | None = None,
        cross_rate_reversion: CrossRateReversionConfig | None = None,
    ) -> None:
        self.config = config or AlphaRouterConfig()
        self.momentum = SimpleMomentumStrategy(
            replace(momentum or MomentumConfig(), symbol=self.config.symbol)
        )
        self.moving_average = MovingAverageCrossoverStrategy(
            replace(moving_average or MovingAverageCrossoverConfig(), symbol=self.config.symbol)
        )
        self.breakout = BreakoutStrategy(
            replace(breakout or BreakoutConfig(), symbol=self.config.symbol)
        )
        self.volatility_squeeze = VolatilitySqueezeStrategy(
            replace(
                volatility_squeeze or VolatilitySqueezeConfig(),
                symbol=self.config.symbol,
            )
        )
        self.dual_squeeze = DualSqueezeStrategy(
            replace(
                dual_squeeze or DualSqueezeConfig(),
                symbol=self.config.symbol,
            )
        )
        self.exhaustion_reversal = ExhaustionReversalStrategy(
            replace(
                exhaustion_reversal or ExhaustionReversalConfig(),
                symbol=self.config.symbol,
            )
        )
        self.session_breakout = SessionBreakoutStrategy(
            replace(session_breakout or SessionBreakoutConfig(), symbol=self.config.symbol)
        )
        self.macd_momentum = MacdMomentumStrategy(
            replace(macd_momentum or MacdMomentumConfig(), symbol=self.config.symbol)
        )
        self.kalman_trend = KalmanTrendStrategy(
            replace(kalman_trend or KalmanTrendStrategyConfig(), symbol=self.config.symbol)
        )
        self.mean_reversion = MeanReversionStrategy(
            replace(mean_reversion or MeanReversionConfig(), symbol=self.config.symbol)
        )
        self.relative_strength = RelativeStrengthStrategy(
            replace(
                relative_strength or RelativeStrengthConfig(),
                symbol=self.config.symbol,
            )
        )
        self.cross_rate_reversion = CrossRateReversionStrategy(
            replace(
                cross_rate_reversion or CrossRateReversionConfig(),
                symbol=self.config.symbol,
            )
        )

    def update_portfolio_context(
        self,
        *,
        closes_by_symbol: Mapping[str, Sequence[float]],
        quotes_by_symbol: Mapping[str, QuoteSnapshot] | None = None,
    ) -> None:
        self.relative_strength.update_portfolio_context(
            closes_by_symbol=closes_by_symbol,
            quotes_by_symbol=quotes_by_symbol,
        )
        self.cross_rate_reversion.update_portfolio_context(
            closes_by_symbol=closes_by_symbol,
            quotes_by_symbol=quotes_by_symbol,
        )

    def generate_signals(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> tuple[StrategySignal, ...]:
        signals = [
            self._momentum_signal(prices, quote=quote),
        ]
        if self.config.moving_average_weight > 0:
            signals.append(self._moving_average_signal(prices, quote=quote))
        signals.extend(
            [
                self._breakout_signal(prices, quote=quote),
            ]
        )
        if self.config.session_breakout_weight > 0:
            signals.append(self._session_breakout_signal(prices, quote=quote))
        if self.config.macd_momentum_weight > 0:
            signals.append(self._macd_momentum_signal(prices, quote=quote))
        if self.config.kalman_trend_weight > 0:
            signals.append(self._kalman_trend_signal(prices, quote=quote))
        if self.config.volatility_squeeze_weight > 0:
            signals.append(self._volatility_squeeze_signal(prices, quote=quote))
        if self.config.dual_squeeze_weight > 0:
            signals.append(self._dual_squeeze_signal(prices, quote=quote))
        if self.config.exhaustion_reversal_weight > 0:
            signals.append(self._exhaustion_reversal_signal(prices, quote=quote))
        signals.append(self._mean_reversion_signal(prices, quote=quote))
        if self.config.relative_strength_weight > 0:
            signals.append(self._relative_strength_signal(prices, quote=quote))
        if self.config.cross_rate_weight > 0:
            signals.append(self._cross_rate_reversion_signal(prices, quote=quote))
        if self.config.ml_enabled:
            signals.append(self._ml_signal(prices, quote=quote))
        return self._apply_adaptive_weights(tuple(signals), prices)

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        signals = self.generate_signals(prices, quote=quote)
        active = tuple(
            signal
            for signal in signals
            if signal.direction != SignalDirection.FLAT
            and signal.confidence >= self.config.min_signal_confidence
            and signal.expected_edge_bps > signal.cost_bps * self.config.cost_buffer
        )
        total_weight = sum(signal.weight for signal in signals)
        raw_score = (
            sum(signal.signed_score for signal in active) / total_weight
            if total_weight > 0
            else 0.0
        )
        has_conflict = _has_signal_conflict(active)
        combined_score = (
            raw_score * (1.0 - self.config.conflict_penalty)
            if has_conflict
            else raw_score
        )
        attribution = _router_attribution(active, combined_score)
        attribution_kwargs = _attribution_kwargs(attribution)
        current_direction = _notional_direction(current_notional_usd)
        diagnostics = _router_diagnostics(
            signals=signals,
            raw_score=raw_score,
            combined_score=combined_score,
            has_conflict=has_conflict,
            holding_period=holding_period,
        )
        override_signal = self._primary_signal_override(
            active,
            has_conflict=has_conflict,
        )

        if (
            override_signal is not None
            and current_direction == 0
            and abs(combined_score) < self.config.entry_score
        ):
            target_direction = _signal_direction_sign(override_signal.direction)
            target_notional = target_direction * self._target_notional(
                max(override_signal.confidence, self.config.entry_score)
            )
            if abs(target_notional) >= self.config.min_trade_notional_usd:
                override_attribution = SignalAttribution(
                    primary_signal=override_signal.strategy_name,
                    supporting_signals=(override_signal.strategy_name,),
                    conflicting_signals=(),
                )
                override_diagnostics = diagnostics + (
                    ("router_override_signal", override_signal.strategy_name),
                    ("router_override_confidence", override_signal.confidence),
                    (
                        "router_override_edge_after_cost_bps",
                        override_signal.edge_after_cost_bps,
                    ),
                )
                return _decision(
                    StrategyAction.ENTER,
                    self.config.symbol,
                    target_notional,
                    (
                        "primary signal override; "
                        f"{override_signal.strategy_name}: {override_signal.reason}; "
                        f"combined_score={combined_score:.2f}"
                    ),
                    override_diagnostics,
                    **_attribution_kwargs(override_attribution),
                )

        if abs(combined_score) <= self.config.exit_score:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    _router_reason("flat score", signals, combined_score, has_conflict),
                    diagnostics,
                    **attribution_kwargs,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                _router_reason("score faded; exit", signals, combined_score, has_conflict),
                diagnostics,
                **attribution_kwargs,
            )

        target_direction = 1 if combined_score > 0 else -1
        if abs(combined_score) < self.config.entry_score:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    _router_reason(
                        "score below entry threshold",
                        signals,
                        combined_score,
                        has_conflict,
                    ),
                    diagnostics,
                    **attribution_kwargs,
                )
            if target_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    _router_reason(
                        "weak opposite score; exit",
                        signals,
                        combined_score,
                        has_conflict,
                    ),
                    diagnostics,
                    **attribution_kwargs,
                )
            return _decision(
                StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                _router_reason(
                    "score supports existing position but not resizing",
                    signals,
                        combined_score,
                        has_conflict,
                    ),
                    diagnostics,
                    **attribution_kwargs,
                )

        target_notional = target_direction * self._target_notional(abs(combined_score))
        if abs(target_notional) < self.config.min_trade_notional_usd:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    "router target below minimum trade size",
                    diagnostics,
                    **attribution_kwargs,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "router target below minimum trade size; exit current position",
                diagnostics,
                **attribution_kwargs,
            )

        if current_direction == 0:
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                target_notional,
                _router_reason("router entry", signals, combined_score, has_conflict),
                diagnostics,
                **attribution_kwargs,
            )
        if target_direction != current_direction:
            return _decision(
                StrategyAction.REVERSE,
                self.config.symbol,
                target_notional,
                _router_reason("router reversal", signals, combined_score, has_conflict),
                diagnostics,
                **attribution_kwargs,
            )
        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            _router_reason("router hold", signals, combined_score, has_conflict),
            diagnostics,
            **attribution_kwargs,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _apply_adaptive_weights(
        self,
        signals: tuple[StrategySignal, ...],
        prices: Sequence[float],
    ) -> tuple[StrategySignal, ...]:
        if not self.config.adaptive_weighting_enabled:
            return signals

        regime = self._adaptive_regime(prices)
        volatility_regime = self._adaptive_volatility_regime(prices)
        asset_class = instrument_for(self.config.symbol).asset_class
        weighted: list[StrategySignal] = []
        for signal in signals:
            multiplier = self._adaptive_weight_multiplier(
                signal=signal,
                regime=regime,
                volatility_regime=volatility_regime,
                asset_class=asset_class,
            )
            diagnostics = signal.diagnostics + (
                ("adaptive_router_asset_class", asset_class.value),
                (
                    "adaptive_router_regime",
                    regime.value if regime is not None else "UNKNOWN",
                ),
                (
                    "adaptive_router_volatility_regime",
                    volatility_regime.state
                    if volatility_regime is not None
                    else "UNKNOWN",
                ),
                (
                    "adaptive_router_volatility_ratio",
                    volatility_regime.ratio if volatility_regime is not None else 0.0,
                ),
                ("adaptive_router_multiplier", multiplier),
            )
            weighted.append(
                replace(
                    signal,
                    weight=signal.weight * multiplier,
                    diagnostics=diagnostics,
                )
            )
        return tuple(weighted)

    def _adaptive_regime(self, prices: Sequence[float]) -> TimeSeriesRegime | None:
        if len(prices) < self.config.adaptive_regime_lookback:
            return None
        try:
            reading = read_kalman_regime(
                prices,
                symbol=self.config.symbol,
                config=KalmanTrendConfig(lookback=self.config.adaptive_regime_lookback),
            )
        except ValueError:
            return None
        return reading.regime

    def _adaptive_volatility_regime(
        self,
        prices: Sequence[float],
    ) -> VolatilityRegimeReading | None:
        if not self.config.volatility_regime_enabled:
            return None
        short_window = self.config.volatility_regime_lookback
        long_window = max(self.config.adaptive_regime_lookback, short_window + 1)
        if len(prices) < long_window + 1:
            return None

        try:
            short_returns = _log_returns(prices[-(short_window + 1) :])
            long_returns = _log_returns(prices[-(long_window + 1) :])
        except ValueError:
            return None

        short_volatility_bps = _population_stdev(short_returns) * 10_000
        long_volatility_bps = _population_stdev(long_returns) * 10_000
        if long_volatility_bps <= 0:
            return None
        ratio = short_volatility_bps / long_volatility_bps
        if (
            ratio >= self.config.high_volatility_ratio
            and short_volatility_bps >= self.config.min_high_volatility_bps
        ):
            state = "HIGH_VOL"
        elif ratio <= self.config.low_volatility_ratio:
            state = "LOW_VOL"
        else:
            state = "NORMAL_VOL"
        return VolatilityRegimeReading(
            state=state,
            short_realized_volatility_bps=short_volatility_bps,
            long_realized_volatility_bps=long_volatility_bps,
            ratio=ratio,
        )

    def _adaptive_weight_multiplier(
        self,
        *,
        signal: StrategySignal,
        regime: TimeSeriesRegime | None,
        volatility_regime: VolatilityRegimeReading | None,
        asset_class: AssetClass,
    ) -> float:
        multiplier = 1.0
        trend_signals = {
            "momentum",
            "ma_crossover",
            "breakout",
            "session_breakout",
            "macd_momentum",
            "kalman_trend",
            "volatility_squeeze",
            "dual_squeeze",
            "relative_strength",
        }
        reversion_signals = {
            "mean_reversion",
            "cross_rate_reversion",
            "exhaustion_reversal",
        }
        if asset_class == AssetClass.METAL:
            if signal.strategy_name == "mean_reversion":
                multiplier *= self.config.metal_mean_reversion_multiplier
            elif signal.strategy_name == "breakout":
                multiplier *= self.config.metal_raw_breakout_multiplier

        if regime == TimeSeriesRegime.CHOP:
            if signal.strategy_name in reversion_signals:
                multiplier *= self.config.chop_mean_reversion_multiplier
            elif signal.strategy_name in trend_signals:
                multiplier *= self.config.chop_trend_signal_multiplier
        elif regime in {TimeSeriesRegime.TREND_UP, TimeSeriesRegime.TREND_DOWN}:
            trend_direction = (
                SignalDirection.LONG
                if regime == TimeSeriesRegime.TREND_UP
                else SignalDirection.SHORT
            )
            if signal.strategy_name in trend_signals:
                if signal.direction == trend_direction:
                    multiplier *= self.config.trend_aligned_signal_multiplier
                elif signal.direction != SignalDirection.FLAT:
                    multiplier *= self.config.trend_counter_signal_multiplier
            elif (
                signal.strategy_name in reversion_signals
                and signal.direction not in {SignalDirection.FLAT, trend_direction}
            ):
                multiplier *= self.config.trend_counter_signal_multiplier

        if volatility_regime is not None:
            if volatility_regime.state == "HIGH_VOL":
                if signal.strategy_name in reversion_signals:
                    multiplier *= self.config.high_volatility_reversion_multiplier
                elif signal.strategy_name in trend_signals:
                    multiplier *= self.config.high_volatility_trend_multiplier
            elif volatility_regime.state == "LOW_VOL":
                if signal.strategy_name in trend_signals:
                    multiplier *= self.config.low_volatility_trend_multiplier
                elif signal.strategy_name in reversion_signals:
                    multiplier *= self.config.low_volatility_reversion_multiplier

        return multiplier

    def _primary_signal_override(
        self,
        active: tuple[StrategySignal, ...],
        *,
        has_conflict: bool,
    ) -> StrategySignal | None:
        if not self.config.primary_signal_override_enabled or has_conflict:
            return None
        candidates = tuple(
            signal
            for signal in active
            if signal.weight > 0
            and signal.confidence >= self.config.primary_signal_min_confidence
            and signal.edge_after_cost_bps >= self.config.primary_signal_min_edge_bps
        )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda signal: (
                signal.edge_after_cost_bps,
                signal.confidence,
                abs(signal.signed_score),
            ),
        )

    def _momentum_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.momentum.read_momentum(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.momentum.config.slippage_bps,
            fee_bps=self.momentum.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "momentum",
                self.config.symbol,
                self.config.momentum_weight,
                SignalHorizon.SHORT,
                "not enough prices",
                cost_bps,
            )

        if not self._quote_is_acceptable(quote):
            return _flat_signal(
                "momentum",
                self.config.symbol,
                self.config.momentum_weight,
                SignalHorizon.SHORT,
                "spread above router limit",
                cost_bps,
                _momentum_diagnostics(reading),
            )

        abs_move = abs(reading.move_bps)
        abs_score = abs(reading.normalized_momentum)
        has_enough_score = (
            self.momentum.config.min_normalized_momentum <= 0
            or abs_score >= self.momentum.config.min_normalized_momentum
        )
        if (
            reading.move_bps == 0
            or abs_move < self.momentum.config.threshold_bps
            or not has_enough_score
            or reading.trend_efficiency < self.momentum.config.min_trend_efficiency
            or abs_move <= cost_bps * self.config.cost_buffer
        ):
            return _flat_signal(
                "momentum",
                self.config.symbol,
                self.config.momentum_weight,
                SignalHorizon.SHORT,
                (
                    "momentum signal blocked "
                    f"(move={reading.move_bps:.1f} bps, "
                    f"efficiency={reading.trend_efficiency:.2f})"
                ),
                cost_bps,
                _momentum_diagnostics(reading),
                expected_edge_bps=abs_move,
            )

        direction = SignalDirection.LONG if reading.move_bps > 0 else SignalDirection.SHORT
        confidence = min(
            _bounded_confidence(abs_move, max(self.momentum.config.threshold_bps, 1e-12)),
            _bounded_confidence(
                reading.trend_efficiency,
                max(self.momentum.config.min_trend_efficiency, 1e-12),
            ),
        )
        return StrategySignal(
            strategy_name="momentum",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs_move,
            cost_bps=cost_bps,
            weight=self.config.momentum_weight,
            horizon=SignalHorizon.SHORT,
            reason=(
                f"momentum {reading.move_bps:.1f} bps, "
                f"efficiency={reading.trend_efficiency:.2f}"
            ),
            diagnostics=_momentum_diagnostics(reading),
        )

    def _moving_average_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.moving_average.read_crossover(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.moving_average.config.slippage_bps,
            fee_bps=self.moving_average.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "ma_crossover",
                self.config.symbol,
                self.config.moving_average_weight,
                SignalHorizon.MEDIUM,
                "not enough prices",
                cost_bps,
            )

        if not self._quote_is_acceptable(quote):
            return _flat_signal(
                "ma_crossover",
                self.config.symbol,
                self.config.moving_average_weight,
                SignalHorizon.MEDIUM,
                "spread above router limit",
                cost_bps,
                _ma_crossover_diagnostics(reading),
            )

        abs_separation = abs(reading.separation_bps)
        if (
            abs_separation < self.moving_average.config.min_separation_bps
            or reading.trend_efficiency < self.moving_average.config.min_trend_efficiency
            or abs_separation <= cost_bps * self.config.cost_buffer
        ):
            return _flat_signal(
                "ma_crossover",
                self.config.symbol,
                self.config.moving_average_weight,
                SignalHorizon.MEDIUM,
                (
                    "moving-average signal blocked "
                    f"(separation={reading.separation_bps:.1f} bps, "
                    f"efficiency={reading.trend_efficiency:.2f})"
                ),
                cost_bps,
                _ma_crossover_diagnostics(reading),
                expected_edge_bps=abs_separation,
            )

        direction = (
            SignalDirection.LONG
            if reading.separation_bps > 0
            else SignalDirection.SHORT
        )
        confidence = _bounded_confidence(
            abs_separation,
            self.moving_average.config.min_separation_bps,
        )
        if self.moving_average.config.min_trend_efficiency > 0:
            confidence = min(
                confidence,
                _bounded_confidence(
                    reading.trend_efficiency,
                    self.moving_average.config.min_trend_efficiency,
                ),
            )
        return StrategySignal(
            strategy_name="ma_crossover",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs_separation,
            cost_bps=cost_bps,
            weight=self.config.moving_average_weight,
            horizon=SignalHorizon.MEDIUM,
            reason=(
                f"MA {self.moving_average.config.fast_window}/"
                f"{self.moving_average.config.slow_window} "
                f"separation={reading.separation_bps:.1f} bps, "
                f"efficiency={reading.trend_efficiency:.2f}"
            ),
            diagnostics=_ma_crossover_diagnostics(reading),
        )

    def _mean_reversion_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.mean_reversion.read_reversion(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.mean_reversion.config.slippage_bps,
            fee_bps=self.mean_reversion.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "mean_reversion",
                self.config.symbol,
                self.config.mean_reversion_weight,
                SignalHorizon.MEDIUM,
                "not enough prices",
                cost_bps,
            )

        stdev_bps = (reading.stdev_price / reading.mean_price) * 10_000
        if not self._quote_is_acceptable(quote):
            return _flat_signal(
                "mean_reversion",
                self.config.symbol,
                self.config.mean_reversion_weight,
                SignalHorizon.MEDIUM,
                "spread above router limit",
                cost_bps,
                _reversion_diagnostics(reading),
            )
        if (
            stdev_bps <= self.mean_reversion.config.min_stdev_bps
            or abs(reading.trend_strength_bps) > self.mean_reversion.config.max_trend_bps
            or abs(reading.zscore) < self.mean_reversion.config.entry_zscore
            or abs(reading.deviation_bps) <= cost_bps * self.config.cost_buffer
        ):
            return _flat_signal(
                "mean_reversion",
                self.config.symbol,
                self.config.mean_reversion_weight,
                SignalHorizon.MEDIUM,
                (
                    "mean-reversion signal blocked "
                    f"(z={reading.zscore:.2f}, "
                    f"trend={reading.trend_strength_bps:.1f} bps)"
                ),
                cost_bps,
                _reversion_diagnostics(reading),
                expected_edge_bps=abs(reading.deviation_bps),
            )

        direction = SignalDirection.SHORT if reading.zscore > 0 else SignalDirection.LONG
        trend_confidence = 1.0 - min(
            _bounded_confidence(
                abs(reading.trend_strength_bps),
                max(self.mean_reversion.config.max_trend_bps, 1e-12),
            ),
            1.0,
        )
        confidence = min(
            _bounded_confidence(
                abs(reading.zscore),
                self.mean_reversion.config.entry_zscore,
            ),
            max(trend_confidence, 0.0),
        )
        return StrategySignal(
            strategy_name="mean_reversion",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs(reading.deviation_bps),
            cost_bps=cost_bps,
            weight=self.config.mean_reversion_weight,
            horizon=SignalHorizon.MEDIUM,
            reason=(
                f"mean reversion z={reading.zscore:.2f}, "
                f"deviation={reading.deviation_bps:.1f} bps"
            ),
            diagnostics=_reversion_diagnostics(reading),
        )

    def _macd_momentum_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.macd_momentum.read_macd(prices, quote=quote)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.macd_momentum.config.slippage_bps,
            fee_bps=self.macd_momentum.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "macd_momentum",
                self.config.symbol,
                self.config.macd_momentum_weight,
                SignalHorizon.MEDIUM,
                "not enough prices",
                cost_bps,
            )

        decision = self.macd_momentum.generate_decision(prices, quote=quote)
        diagnostics = decision.diagnostics
        expected_edge_bps = abs(reading.histogram_bps)
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "macd_momentum",
                self.config.symbol,
                self.config.macd_momentum_weight,
                SignalHorizon.MEDIUM,
                f"MACD momentum blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=expected_edge_bps,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        confidence = min(
            _bounded_confidence(
                abs(reading.histogram_bps),
                max(self.macd_momentum.config.min_histogram_bps, 1e-12),
            ),
            _bounded_confidence(
                abs(reading.macd_bps),
                max(self.macd_momentum.config.min_macd_bps, 1e-12),
            )
            if self.macd_momentum.config.min_macd_bps > 0
            else 1.0,
            _bounded_confidence(
                reading.trend_efficiency,
                max(self.macd_momentum.config.min_trend_efficiency, 1e-12),
            )
            if self.macd_momentum.config.min_trend_efficiency > 0
            else 1.0,
        )
        return StrategySignal(
            strategy_name="macd_momentum",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=expected_edge_bps,
            cost_bps=cost_bps,
            weight=self.config.macd_momentum_weight,
            horizon=SignalHorizon.MEDIUM,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _kalman_trend_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.kalman_trend.read_kalman_trend(prices, quote=quote)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.kalman_trend.config.slippage_bps,
            fee_bps=self.kalman_trend.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "kalman_trend",
                self.config.symbol,
                self.config.kalman_trend_weight,
                SignalHorizon.MEDIUM,
                "not enough prices",
                cost_bps,
            )

        decision = self.kalman_trend.generate_decision(prices, quote=quote)
        diagnostics = decision.diagnostics
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "kalman_trend",
                self.config.symbol,
                self.config.kalman_trend_weight,
                SignalHorizon.MEDIUM,
                f"Kalman trend blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        regime = reading.regime_reading
        confidence = min(
            _bounded_confidence(
                abs(regime.kalman_slope_bps),
                max(self.kalman_trend.config.min_abs_slope_bps, 1e-12),
            ),
            _bounded_confidence(
                regime.trend_efficiency,
                max(self.kalman_trend.config.min_trend_efficiency, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.kalman_trend.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return StrategySignal(
            strategy_name="kalman_trend",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=reading.expected_edge_bps,
            cost_bps=cost_bps,
            weight=self.config.kalman_trend_weight,
            horizon=SignalHorizon.MEDIUM,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _breakout_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.breakout.read_breakout(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.breakout.config.slippage_bps,
            fee_bps=self.breakout.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "breakout",
                self.config.symbol,
                self.config.breakout_weight,
                SignalHorizon.SHORT,
                "not enough prices",
                cost_bps,
            )

        if not self._quote_is_acceptable(quote):
            return _flat_signal(
                "breakout",
                self.config.symbol,
                self.config.breakout_weight,
                SignalHorizon.SHORT,
                "spread above router limit",
                cost_bps,
                _breakout_diagnostics(reading),
            )

        abs_breakout = abs(reading.breakout_bps)
        if (
            reading.channel_width_bps < self.breakout.config.min_channel_width_bps
            or abs_breakout < self.breakout.config.breakout_buffer_bps
            or abs_breakout <= cost_bps * self.config.cost_buffer
        ):
            return _flat_signal(
                "breakout",
                self.config.symbol,
                self.config.breakout_weight,
                SignalHorizon.SHORT,
                (
                    "breakout signal blocked "
                    f"(breakout={reading.breakout_bps:.1f} bps, "
                    f"channel={reading.channel_width_bps:.1f} bps)"
                ),
                cost_bps,
                _breakout_diagnostics(reading),
                expected_edge_bps=abs_breakout,
            )

        direction = SignalDirection.LONG if reading.breakout_bps > 0 else SignalDirection.SHORT
        confidence = min(
            _bounded_confidence(
                abs_breakout,
                max(self.breakout.config.breakout_buffer_bps, 1e-12),
            ),
            1.0,
        )
        return StrategySignal(
            strategy_name="breakout",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs_breakout,
            cost_bps=cost_bps,
            weight=self.config.breakout_weight,
            horizon=SignalHorizon.SHORT,
            reason=(
                f"breakout {reading.breakout_bps:.1f} bps, "
                f"channel={reading.channel_width_bps:.1f} bps"
            ),
            diagnostics=_breakout_diagnostics(reading),
        )

    def _session_breakout_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.session_breakout.read_session_breakout(prices, quote=quote)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.session_breakout.config.slippage_bps,
            fee_bps=self.session_breakout.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "session_breakout",
                self.config.symbol,
                self.config.session_breakout_weight,
                SignalHorizon.SHORT,
                "not enough prices",
                cost_bps,
            )

        decision = self.session_breakout.generate_decision(prices, quote=quote)
        abs_breakout = abs(reading.breakout.breakout_bps)
        diagnostics = decision.diagnostics
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "session_breakout",
                self.config.symbol,
                self.config.session_breakout_weight,
                SignalHorizon.SHORT,
                f"session breakout blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=abs_breakout,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        required_edge_bps = max(
            self.session_breakout.config.breakout_buffer_bps,
            self.session_breakout.config.min_expected_edge_bps,
            1e-12,
        )
        confidence = min(
            _bounded_confidence(abs_breakout, required_edge_bps),
            _bounded_confidence(
                reading.realized_volatility_bps,
                max(self.session_breakout.config.min_realized_volatility_bps, 1e-12),
            ),
            1.0,
        )
        return StrategySignal(
            strategy_name="session_breakout",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs_breakout,
            cost_bps=cost_bps,
            weight=self.config.session_breakout_weight,
            horizon=SignalHorizon.SHORT,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _volatility_squeeze_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.volatility_squeeze.read_squeeze(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.volatility_squeeze.config.slippage_bps,
            fee_bps=self.volatility_squeeze.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "volatility_squeeze",
                self.config.symbol,
                self.config.volatility_squeeze_weight,
                SignalHorizon.SHORT,
                "not enough prices",
                cost_bps,
            )

        decision = self.volatility_squeeze.generate_decision(prices, quote=quote)
        abs_breakout = abs(reading.breakout_bps)
        diagnostics = decision.diagnostics
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "volatility_squeeze",
                self.config.symbol,
                self.config.volatility_squeeze_weight,
                SignalHorizon.SHORT,
                f"volatility squeeze blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=abs_breakout,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        confidence = min(
            _bounded_confidence(
                abs_breakout,
                max(self.volatility_squeeze.config.breakout_buffer_bps, 1e-12),
            ),
            1.0,
        )
        return StrategySignal(
            strategy_name="volatility_squeeze",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs_breakout,
            cost_bps=cost_bps,
            weight=self.config.volatility_squeeze_weight,
            horizon=SignalHorizon.SHORT,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _dual_squeeze_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.dual_squeeze.read_dual_squeeze(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.dual_squeeze.config.slippage_bps,
            fee_bps=self.dual_squeeze.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "dual_squeeze",
                self.config.symbol,
                self.config.dual_squeeze_weight,
                SignalHorizon.SHORT,
                "not enough prices",
                cost_bps,
            )

        decision = self.dual_squeeze.generate_decision(prices, quote=quote)
        abs_breakout = abs(reading.fast.breakout_bps)
        diagnostics = decision.diagnostics
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "dual_squeeze",
                self.config.symbol,
                self.config.dual_squeeze_weight,
                SignalHorizon.SHORT,
                f"dual squeeze blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=abs_breakout,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        confidence = min(
            _bounded_confidence(
                abs_breakout,
                max(self.dual_squeeze.config.breakout_buffer_bps, 1e-12),
            ),
            _bounded_confidence(
                self.dual_squeeze.config.confirmation_max_squeeze_ratio,
                max(reading.confirmation.squeeze_ratio, 1e-12),
            ),
            1.0,
        )
        return StrategySignal(
            strategy_name="dual_squeeze",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=abs_breakout,
            cost_bps=cost_bps,
            weight=self.config.dual_squeeze_weight,
            horizon=SignalHorizon.SHORT,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _exhaustion_reversal_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.exhaustion_reversal.read_exhaustion(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.exhaustion_reversal.config.slippage_bps,
            fee_bps=self.exhaustion_reversal.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "exhaustion_reversal",
                self.config.symbol,
                self.config.exhaustion_reversal_weight,
                SignalHorizon.SHORT,
                "not enough prices",
                cost_bps,
            )

        decision = self.exhaustion_reversal.generate_decision(prices, quote=quote)
        diagnostics = decision.diagnostics
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "exhaustion_reversal",
                self.config.symbol,
                self.config.exhaustion_reversal_weight,
                SignalHorizon.SHORT,
                f"exhaustion reversal blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        confidence = min(
            _bounded_confidence(
                abs(reading.shock_move_bps),
                max(self.exhaustion_reversal.config.min_shock_bps, 1e-12),
            ),
            _bounded_confidence(
                reading.shock_zscore,
                max(self.exhaustion_reversal.config.min_shock_zscore, 1e-12),
            ),
            _bounded_confidence(
                reading.expected_edge_bps,
                max(self.exhaustion_reversal.config.min_expected_edge_bps, 1e-12),
            ),
        )
        return StrategySignal(
            strategy_name="exhaustion_reversal",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=reading.expected_edge_bps,
            cost_bps=cost_bps,
            weight=self.config.exhaustion_reversal_weight,
            horizon=SignalHorizon.SHORT,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _relative_strength_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.relative_strength.read_relative_strength(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.relative_strength.config.slippage_bps,
            fee_bps=self.relative_strength.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "relative_strength",
                self.config.symbol,
                self.config.relative_strength_weight,
                SignalHorizon.MEDIUM,
                "not enough portfolio context",
                cost_bps,
            )

        diagnostics = _relative_strength_diagnostics(reading)
        if (
            reading.score_dispersion
            < self.config.relative_strength_min_score_dispersion
        ):
            return _flat_signal(
                "relative_strength",
                self.config.symbol,
                self.config.relative_strength_weight,
                SignalHorizon.MEDIUM,
                (
                    "relative-strength router gate blocked low dispersion "
                    f"({reading.score_dispersion:.2f} < "
                    f"{self.config.relative_strength_min_score_dispersion:.2f})"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=abs(reading.move_bps),
            )
        if (
            reading.trend_efficiency
            < self.config.relative_strength_min_target_trend_efficiency
        ):
            return _flat_signal(
                "relative_strength",
                self.config.symbol,
                self.config.relative_strength_weight,
                SignalHorizon.MEDIUM,
                (
                    "relative-strength router gate blocked inefficient target trend "
                    f"({reading.trend_efficiency:.2f} < "
                    f"{self.config.relative_strength_min_target_trend_efficiency:.2f})"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=abs(reading.move_bps),
            )

        decision = self.relative_strength.generate_decision(prices, quote=quote)
        expected_edge_bps = abs(reading.move_bps)
        diagnostics = decision.diagnostics
        if decision.action not in {StrategyAction.ENTER, StrategyAction.REVERSE}:
            return _flat_signal(
                "relative_strength",
                self.config.symbol,
                self.config.relative_strength_weight,
                SignalHorizon.MEDIUM,
                f"relative strength blocked ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=expected_edge_bps,
            )

        direction = (
            SignalDirection.LONG
            if decision.target_notional_usd > 0
            else SignalDirection.SHORT
        )
        confidence = _bounded_confidence(
            abs(reading.relative_zscore),
            self.relative_strength.config.entry_zscore,
        )
        if self.relative_strength.config.min_score_dispersion > 0:
            confidence = min(
                confidence,
                _bounded_confidence(
                    reading.score_dispersion,
                    self.relative_strength.config.min_score_dispersion,
                ),
            )
        if self.relative_strength.config.min_target_trend_efficiency > 0:
            confidence = min(
                confidence,
                _bounded_confidence(
                    reading.trend_efficiency,
                    self.relative_strength.config.min_target_trend_efficiency,
                ),
            )
        return StrategySignal(
            strategy_name="relative_strength",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=expected_edge_bps,
            cost_bps=cost_bps,
            weight=self.config.relative_strength_weight,
            horizon=SignalHorizon.MEDIUM,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _cross_rate_reversion_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        signals = self.cross_rate_reversion.generate_signals(prices, quote=quote)
        signal = signals[0]
        return replace(signal, weight=self.config.cross_rate_weight)

    def read_ml_alpha(self, prices: Sequence[float]) -> MLAlphaReading | None:
        return _train_and_score_ml_alpha(
            prices=prices,
            lookback=self.config.ml_lookback,
            train_window=self.config.ml_train_window,
            min_train_samples=self.config.ml_min_train_samples,
            learning_rate=self.config.ml_learning_rate,
            epochs=self.config.ml_epochs,
            l2=self.config.ml_l2,
            label_threshold_bps=self.config.ml_label_threshold_bps,
            min_edge_bps=self.config.ml_min_edge_bps,
        )

    def _ml_signal(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        reading = self.read_ml_alpha(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.momentum.config.slippage_bps,
            fee_bps=self.momentum.config.fee_bps,
        )
        if reading is None:
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                "not enough labeled history for ML alpha",
                cost_bps,
            )

        diagnostics = _ml_alpha_diagnostics(reading)
        if not self._quote_is_acceptable(quote):
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                "spread above router limit",
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        if reading.sample_count < self.config.ml_min_samples_for_trade:
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                (
                    f"ML sample count {reading.sample_count} below trade minimum "
                    f"{self.config.ml_min_samples_for_trade}"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        if reading.training_accuracy < self.config.ml_min_training_accuracy:
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                (
                    f"ML training accuracy {reading.training_accuracy:.1%} below "
                    f"{self.config.ml_min_training_accuracy:.1%}"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        if (
            self.config.ml_disable_on_negative_signed_return
            and reading.training_signed_return_bps <= 0
        ):
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                (
                    "ML training signed return is not positive "
                    f"({reading.training_signed_return_bps:.1f} bps)"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        if reading.expected_edge_bps < self.config.ml_min_expected_edge_bps:
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                (
                    f"ML expected edge {reading.expected_edge_bps:.2f} bps below "
                    f"{self.config.ml_min_expected_edge_bps:.2f} bps minimum"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        long_threshold = self.config.ml_entry_probability
        short_threshold = 1.0 - self.config.ml_entry_probability
        if reading.probability_up >= long_threshold:
            direction = SignalDirection.LONG
        elif reading.probability_up <= short_threshold:
            direction = SignalDirection.SHORT
        else:
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                (
                    f"ML probability {reading.probability_up:.2f} inside no-trade band "
                    f"{short_threshold:.2f}-{long_threshold:.2f}"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        if reading.expected_edge_bps <= cost_bps * self.config.cost_buffer:
            return _flat_signal(
                "ml_alpha",
                self.config.symbol,
                self.config.ml_weight,
                SignalHorizon.SHORT,
                (
                    f"ML edge {reading.expected_edge_bps:.1f} bps below "
                    f"estimated cost {cost_bps * self.config.cost_buffer:.1f} bps"
                ),
                cost_bps,
                diagnostics,
                expected_edge_bps=reading.expected_edge_bps,
            )

        confidence = _bounded_confidence(
            abs(reading.score),
            max((self.config.ml_entry_probability - 0.5) * 2.0, 1e-12),
        )
        return StrategySignal(
            strategy_name="ml_alpha",
            symbol=self.config.symbol,
            direction=direction,
            confidence=confidence,
            expected_edge_bps=reading.expected_edge_bps,
            cost_bps=cost_bps,
            weight=self.config.ml_weight,
            horizon=SignalHorizon.SHORT,
            reason=(
                f"ML p_up={reading.probability_up:.2f}, "
                f"score={reading.score:.2f}, "
                f"samples={reading.sample_count}, "
                f"train_acc={reading.training_accuracy:.1%}"
            ),
            diagnostics=diagnostics,
        )

    def _quote_is_acceptable(self, quote: QuoteSnapshot | None) -> bool:
        if quote is None or self.config.max_spread_bps is None:
            return True
        return quote.spread_bps <= self.config.max_spread_bps

    def _target_notional(self, abs_score: float) -> float:
        sized = self.config.target_notional_usd * min(abs_score, 1.0)
        return min(sized, self.config.max_target_notional_usd)


class ChampionEnsembleStrategy:
    def __init__(
        self,
        *,
        config: ChampionEnsembleConfig | None = None,
        kalman_trend: KalmanTrendStrategyConfig | None = None,
        asset_adaptive_dual_squeeze: AssetAdaptiveDualSqueezeConfig | None = None,
        dual_squeeze: DualSqueezeConfig | None = None,
        trend_pullback: TrendPullbackConfig | None = None,
        fixing_reversal: FixingReversalConfig | None = None,
        macd_momentum: MacdMomentumConfig | None = None,
        kalman_trend_strategy: object | None = None,
        asset_adaptive_dual_squeeze_strategy: object | None = None,
        dual_squeeze_strategy: object | None = None,
        trend_pullback_strategy: object | None = None,
        fixing_reversal_strategy: object | None = None,
        macd_momentum_strategy: object | None = None,
    ) -> None:
        self.config = config or ChampionEnsembleConfig()
        self.kalman_trend = kalman_trend_strategy or KalmanTrendStrategy(
            self._with_symbol(kalman_trend or KalmanTrendStrategyConfig())
        )
        self.asset_adaptive_dual_squeeze = (
            asset_adaptive_dual_squeeze_strategy
            or AssetAdaptiveDualSqueezeStrategy(
                self._with_symbol(
                    asset_adaptive_dual_squeeze
                    or AssetAdaptiveDualSqueezeConfig()
                )
            )
        )
        self.dual_squeeze = dual_squeeze_strategy or DualSqueezeStrategy(
            self._with_symbol(dual_squeeze or DualSqueezeConfig())
        )
        self.trend_pullback = trend_pullback_strategy or TrendPullbackStrategy(
            self._with_symbol(trend_pullback or TrendPullbackConfig())
        )
        self.fixing_reversal = fixing_reversal_strategy or FixingReversalStrategy(
            self._with_symbol(fixing_reversal or FixingReversalConfig())
        )
        self.macd_momentum = macd_momentum_strategy or MacdMomentumStrategy(
            self._with_symbol(macd_momentum or MacdMomentumConfig())
        )

    def update_portfolio_context(
        self,
        *,
        closes_by_symbol: Mapping[str, Sequence[float]],
        quotes_by_symbol: Mapping[str, QuoteSnapshot] | None = None,
    ) -> None:
        for strategy in (
            self.kalman_trend,
            self.asset_adaptive_dual_squeeze,
            self.dual_squeeze,
            self.trend_pullback,
            self.fixing_reversal,
            self.macd_momentum,
        ):
            update_context = getattr(strategy, "update_portfolio_context", None)
            if update_context is not None:
                update_context(
                    closes_by_symbol=closes_by_symbol,
                    quotes_by_symbol=quotes_by_symbol,
                )

    def generate_signals(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> tuple[StrategySignal, ...]:
        return tuple(
            self._decision_to_signal(
                strategy_name=strategy_name,
                strategy=strategy,
                weight=weight,
                horizon=horizon,
                prices=prices,
                current_notional_usd=current_notional_usd,
                holding_period=holding_period,
                quote=quote,
            )
            for strategy_name, strategy, weight, horizon in (
                (
                    "kalman_trend",
                    self.kalman_trend,
                    self.config.kalman_trend_weight,
                    SignalHorizon.MEDIUM,
                ),
                (
                    "asset_adaptive_dual_squeeze",
                    self.asset_adaptive_dual_squeeze,
                    self.config.asset_adaptive_dual_squeeze_weight,
                    SignalHorizon.SHORT,
                ),
                (
                    "dual_squeeze",
                    self.dual_squeeze,
                    self.config.dual_squeeze_weight,
                    SignalHorizon.SHORT,
                ),
                (
                    "trend_pullback",
                    self.trend_pullback,
                    self.config.trend_pullback_weight,
                    SignalHorizon.MEDIUM,
                ),
                (
                    "fixing_reversal",
                    self.fixing_reversal,
                    self.config.fixing_reversal_weight,
                    SignalHorizon.SHORT,
                ),
                (
                    "macd_momentum",
                    self.macd_momentum,
                    self.config.macd_momentum_weight,
                    SignalHorizon.SHORT,
                ),
            )
            if weight > 0
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        signals = self.generate_signals(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        active = tuple(
            signal
            for signal in signals
            if signal.direction != SignalDirection.FLAT
            and signal.confidence >= self.config.min_signal_confidence
        )
        total_weight = sum(signal.weight for signal in signals)
        raw_score = (
            sum(signal.signed_score for signal in active) / total_weight
            if total_weight > 0
            else 0.0
        )
        has_conflict = _has_signal_conflict(active)
        combined_score = (
            raw_score * (1.0 - self.config.conflict_penalty)
            if has_conflict
            else raw_score
        )
        attribution = _router_attribution(active, combined_score)
        attribution_kwargs = _attribution_kwargs(attribution)
        current_direction = _notional_direction(current_notional_usd)
        diagnostics = _champion_ensemble_diagnostics(
            signals=signals,
            raw_score=raw_score,
            combined_score=combined_score,
            has_conflict=has_conflict,
            holding_period=holding_period,
        )

        if abs(combined_score) <= self.config.exit_score:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    _champion_ensemble_reason(
                        "flat ensemble score",
                        signals,
                        combined_score,
                        has_conflict,
                    ),
                    diagnostics,
                    **attribution_kwargs,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                _champion_ensemble_reason(
                    "ensemble score faded; exit",
                    signals,
                    combined_score,
                    has_conflict,
                ),
                diagnostics,
                **attribution_kwargs,
            )

        target_direction = 1 if combined_score > 0 else -1
        entry_allowed = self._entry_allowed(
            active,
            combined_score=combined_score,
            has_conflict=has_conflict,
        )
        if not entry_allowed:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    _champion_ensemble_reason(
                        "ensemble score below entry threshold",
                        signals,
                        combined_score,
                        has_conflict,
                    ),
                    diagnostics,
                    **attribution_kwargs,
                )
            if target_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    _champion_ensemble_reason(
                        "weak opposite ensemble score; exit",
                        signals,
                        combined_score,
                        has_conflict,
                    ),
                    diagnostics,
                    **attribution_kwargs,
                )
            return _decision(
                StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                _champion_ensemble_reason(
                    "ensemble supports current position but not resizing",
                    signals,
                    combined_score,
                    has_conflict,
                ),
                diagnostics,
                **attribution_kwargs,
            )

        target_notional = target_direction * self._target_notional(
            abs(combined_score),
            active,
            target_direction=target_direction,
        )
        if abs(target_notional) < self.config.min_trade_notional_usd:
            if current_direction == 0:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    "champion ensemble target below minimum trade size",
                    diagnostics,
                    **attribution_kwargs,
                )
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                "champion ensemble target below minimum trade size; exit",
                diagnostics,
                **attribution_kwargs,
            )

        if current_direction == 0:
            return _decision(
                StrategyAction.ENTER,
                self.config.symbol,
                target_notional,
                _champion_ensemble_reason(
                    "champion ensemble entry",
                    signals,
                    combined_score,
                    has_conflict,
                ),
                diagnostics,
                **attribution_kwargs,
            )
        if target_direction != current_direction:
            return _decision(
                StrategyAction.REVERSE,
                self.config.symbol,
                target_notional,
                _champion_ensemble_reason(
                    "champion ensemble reversal",
                    signals,
                    combined_score,
                    has_conflict,
                ),
                diagnostics,
                **attribution_kwargs,
            )
        return _decision(
            StrategyAction.HOLD,
            self.config.symbol,
            current_notional_usd,
            _champion_ensemble_reason(
                "champion ensemble hold",
                signals,
                combined_score,
                has_conflict,
            ),
            diagnostics,
            **attribution_kwargs,
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        decision = self.generate_decision(prices)
        return decision.to_trade_request()

    def _with_symbol(self, config: StrategyConfig) -> StrategyConfig:
        return replace(
            config,
            symbol=self.config.symbol,
            max_spread_bps=self.config.max_spread_bps,
        )

    def _decision_to_signal(
        self,
        *,
        strategy_name: str,
        strategy: object,
        weight: float,
        horizon: SignalHorizon,
        prices: Sequence[float],
        current_notional_usd: float,
        holding_period: int,
        quote: QuoteSnapshot | None,
    ) -> StrategySignal:
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
        )
        if not self._quote_is_acceptable(quote):
            return _flat_signal(
                strategy_name,
                self.config.symbol,
                weight,
                horizon,
                "spread above champion ensemble limit",
                cost_bps,
            )

        decision = self._sub_decision(
            strategy,
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )
        expected_edge_bps = _decision_expected_edge_bps(decision)
        diagnostics = decision.diagnostics + (
            (f"{strategy_name}_source_action", decision.action.value),
            (
                f"{strategy_name}_source_target_notional_usd",
                abs(decision.target_notional_usd),
            ),
            (f"{strategy_name}_source_reason", decision.reason),
        )
        direction_sign = _notional_direction(decision.target_notional_usd)
        if (
            decision.action not in {StrategyAction.ENTER, StrategyAction.HOLD, StrategyAction.REVERSE}
            or direction_sign == 0
            or abs(decision.target_notional_usd) < self.config.min_trade_notional_usd
        ):
            return _flat_signal(
                strategy_name,
                self.config.symbol,
                weight,
                horizon,
                f"{strategy_name} inactive ({decision.reason})",
                cost_bps,
                diagnostics,
                expected_edge_bps=expected_edge_bps,
            )

        direction = SignalDirection.LONG if direction_sign > 0 else SignalDirection.SHORT
        return StrategySignal(
            strategy_name=strategy_name,
            symbol=self.config.symbol,
            direction=direction,
            confidence=1.0,
            expected_edge_bps=expected_edge_bps,
            cost_bps=cost_bps,
            weight=weight,
            horizon=horizon,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _sub_decision(
        self,
        strategy: object,
        prices: Sequence[float],
        *,
        current_notional_usd: float,
        holding_period: int,
        quote: QuoteSnapshot | None,
    ) -> StrategyDecision:
        generate_decision = getattr(strategy, "generate_decision", None)
        if generate_decision is None:
            request = strategy.generate_request(prices)
            if request is None:
                return _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    0.0,
                    "strategy produced no request",
                    (),
                )
            target = (
                request.target_notional_usd
                if request.side == Side.BUY
                else -request.target_notional_usd
            )
            return _decision(
                StrategyAction.ENTER,
                request.symbol,
                target,
                request.reason,
                (),
                primary_signal="strategy",
                supporting_signals=("strategy",),
            )
        return generate_decision(
            prices,
            current_notional_usd=current_notional_usd,
            holding_period=holding_period,
            quote=quote,
        )

    def _entry_allowed(
        self,
        active: tuple[StrategySignal, ...],
        *,
        combined_score: float,
        has_conflict: bool,
    ) -> bool:
        if abs(combined_score) >= self.config.entry_score:
            return True
        if has_conflict or not active:
            return False
        lead_signal = max(active, key=lambda signal: abs(signal.signed_score))
        return abs(lead_signal.signed_score) >= self.config.strong_lead_score

    def _target_notional(
        self,
        abs_score: float,
        active: tuple[StrategySignal, ...],
        *,
        target_direction: int,
    ) -> float:
        scaled = self.config.target_notional_usd * min(
            abs_score / max(self.config.entry_score, 1e-12),
            1.0,
        )
        supporting_targets = tuple(
            _signal_source_target_notional(signal)
            for signal in active
            if _signal_direction_sign(signal.direction) == target_direction
        )
        source_target = max(supporting_targets) if supporting_targets else 0.0
        return min(max(scaled, source_target), self.config.max_target_notional_usd)

    def _quote_is_acceptable(self, quote: QuoteSnapshot | None) -> bool:
        if quote is None or self.config.max_spread_bps is None:
            return True
        return quote.spread_bps <= self.config.max_spread_bps


class UsdPressureRouterStrategy:
    def __init__(
        self,
        *,
        config: UsdPressureConfig | None = None,
        base_strategy: Strategy | None = None,
    ) -> None:
        self.config = config or UsdPressureConfig()
        self.base_strategy = base_strategy or AlphaRouterStrategy(
            config=AlphaRouterConfig(symbol=self.config.symbol)
        )
        self._portfolio_closes: dict[str, tuple[float, ...]] = {}

    def update_portfolio_context(
        self,
        *,
        closes_by_symbol: Mapping[str, Sequence[float]],
        quotes_by_symbol: Mapping[str, QuoteSnapshot] | None = None,
    ) -> None:
        self._portfolio_closes = {
            instrument_for(symbol).symbol: tuple(prices)
            for symbol, prices in closes_by_symbol.items()
        }
        update_context = getattr(self.base_strategy, "update_portfolio_context", None)
        if update_context is not None:
            update_context(
                closes_by_symbol=closes_by_symbol,
                quotes_by_symbol=quotes_by_symbol,
            )

    def read_usd_pressure(self) -> UsdPressureReading | None:
        components: list[tuple[str, float]] = []
        target_symbol = instrument_for(self.config.symbol).symbol
        for symbol, prices in sorted(self._portfolio_closes.items()):
            instrument = instrument_for(symbol)
            if symbol == target_symbol:
                continue
            if instrument.asset_class != AssetClass.FOREX:
                continue
            if "USD" not in {instrument.base_currency, instrument.quote_currency}:
                continue
            recent_prices = _recent_valid_prices(prices, self.config.lookback)
            if recent_prices is None:
                continue
            move_bps = log(recent_prices[-1] / recent_prices[0]) * 10_000
            if instrument.base_currency == "USD":
                usd_move_bps = move_bps
            else:
                usd_move_bps = -move_bps
            components.append((instrument.symbol, usd_move_bps))

        if len(components) < self.config.min_component_symbols:
            return None

        pressure_bps = sum(move for _, move in components) / len(components)
        pressure_sign = _signed_threshold_direction(
            pressure_bps,
            self.config.pressure_threshold_bps,
        )
        confirming = 0
        conflicting = 0
        if pressure_sign != 0:
            for _, component_move in components:
                component_sign = _signed_threshold_direction(
                    component_move,
                    self.config.component_threshold_bps,
                )
                if component_sign == pressure_sign:
                    confirming += 1
                elif component_sign == -pressure_sign:
                    conflicting += 1

        return UsdPressureReading(
            pressure_bps=pressure_bps,
            component_count=len(components),
            confirming_symbols=confirming,
            conflicting_symbols=conflicting,
            components=tuple(components),
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        if hasattr(self.base_strategy, "generate_decision"):
            base_decision = self.base_strategy.generate_decision(
                prices,
                current_notional_usd=current_notional_usd,
                holding_period=holding_period,
                quote=quote,
            )
        else:
            request = self.base_strategy.generate_request(prices)
            if request is None:
                base_decision = _decision(
                    StrategyAction.NO_ACTION,
                    self.config.symbol,
                    current_notional_usd,
                    "base strategy produced no request",
                )
            else:
                direction = 1 if request.side == Side.BUY else -1
                base_decision = _decision(
                    StrategyAction.ENTER,
                    self.config.symbol,
                    direction * request.target_notional_usd,
                    request.reason,
                )

        reading = self.read_usd_pressure()
        target_volatility_bps = self._target_realized_volatility_bps(prices)
        diagnostics = (
            _usd_pressure_diagnostics(reading)
            if reading is not None
            else (("usd_pressure_status", "not enough USD basket context"),)
        ) + (
            (
                "target_realized_volatility_bps",
                target_volatility_bps
                if target_volatility_bps is not None
                else "not enough target history",
            ),
        ) + base_decision.diagnostics
        current_direction = _notional_direction(current_notional_usd)

        if base_decision.action == StrategyAction.EXIT:
            return self._with_pressure_context(base_decision, diagnostics)

        if (
            self.config.exit_on_conflict
            and current_direction != 0
            and reading is not None
            and self._pressure_conflicts(reading, current_direction)
        ):
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "USD pressure conflicts with held position; "
                    f"{self._pressure_summary(reading)}"
                ),
                diagnostics,
                primary_signal="usd_pressure_router",
                supporting_signals=(base_decision.primary_signal,),
                conflicting_signals=base_decision.conflicting_signals
                + ("usd_pressure",),
            )

        if not base_decision.is_trade_intent:
            return self._with_pressure_context(base_decision, diagnostics)

        target_direction = _notional_direction(base_decision.target_notional_usd)
        if target_direction == 0:
            return self._with_pressure_context(base_decision, diagnostics)

        volatility_ok, volatility_reason = self._target_volatility_allows_entry(
            target_volatility_bps,
        )
        if not volatility_ok:
            if current_direction != 0 and base_decision.action == StrategyAction.REVERSE:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    (
                        "base wanted reversal but target volatility gate blocked "
                        f"new exposure; {volatility_reason}"
                    ),
                    diagnostics,
                    primary_signal="usd_pressure_router",
                    supporting_signals=(base_decision.primary_signal,),
                    conflicting_signals=base_decision.conflicting_signals
                    + ("target_volatility",),
                )

            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                current_notional_usd if current_direction != 0 else 0.0,
                f"target volatility filter blocked entry; {volatility_reason}",
                diagnostics,
                primary_signal="usd_pressure_router",
                supporting_signals=(base_decision.primary_signal,),
                conflicting_signals=base_decision.conflicting_signals
                + ("target_volatility",),
            )

        accepted, reason = self._pressure_confirms(reading, target_direction)
        if accepted:
            return _decision(
                base_decision.action,
                self.config.symbol,
                base_decision.target_notional_usd,
                f"USD pressure confirmed; {base_decision.reason}; {reason}",
                diagnostics,
                primary_signal="usd_pressure_router",
                supporting_signals=base_decision.supporting_signals
                + (base_decision.primary_signal, "usd_pressure"),
                conflicting_signals=base_decision.conflicting_signals,
            )

        if current_direction != 0 and base_decision.action == StrategyAction.REVERSE:
            return _decision(
                StrategyAction.EXIT,
                self.config.symbol,
                0.0,
                (
                    "base wanted reversal but USD basket did not confirm; "
                    f"{reason}"
                ),
                diagnostics,
                primary_signal="usd_pressure_router",
                supporting_signals=(base_decision.primary_signal,),
                conflicting_signals=base_decision.conflicting_signals
                + ("usd_pressure",),
            )

        return _decision(
            StrategyAction.NO_ACTION,
            self.config.symbol,
            current_notional_usd if current_direction != 0 else 0.0,
            f"USD pressure filter blocked entry; {reason}",
            diagnostics,
            primary_signal="usd_pressure_router",
            supporting_signals=(base_decision.primary_signal,),
            conflicting_signals=base_decision.conflicting_signals + ("usd_pressure",),
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _target_realized_volatility_bps(
        self,
        prices: Sequence[float],
    ) -> float | None:
        recent_prices = _recent_valid_prices(prices, self.config.lookback)
        if recent_prices is None:
            return None
        return _population_stdev(_log_returns(recent_prices)) * 10_000

    def _target_volatility_allows_entry(
        self,
        target_volatility_bps: float | None,
    ) -> tuple[bool, str]:
        if self.config.min_target_volatility_bps <= 0:
            return True, "target volatility gate disabled"
        if target_volatility_bps is None:
            return False, "not enough target history for volatility gate"
        if target_volatility_bps < self.config.min_target_volatility_bps:
            return (
                False,
                (
                    f"target volatility {target_volatility_bps:.2f} bps below "
                    f"{self.config.min_target_volatility_bps:.2f} bps floor"
                ),
            )
        return (
            True,
            (
                f"target volatility {target_volatility_bps:.2f} bps cleared "
                f"{self.config.min_target_volatility_bps:.2f} bps floor"
            ),
        )

    def _pressure_confirms(
        self,
        reading: UsdPressureReading | None,
        target_direction: int,
    ) -> tuple[bool, str]:
        if reading is None:
            return False, "not enough USD basket symbols with history"
        required_usd_direction = _target_usd_pressure_direction(
            self.config.symbol,
            target_direction,
        )
        if required_usd_direction == 0:
            return True, "target symbol has no direct USD leg"
        alignment_bps = required_usd_direction * reading.pressure_bps
        if reading.confirming_symbols < self.config.min_confirming_symbols:
            return (
                False,
                (
                    f"only {reading.confirming_symbols} USD components confirm "
                    f"pressure; need {self.config.min_confirming_symbols}"
                ),
            )
        if alignment_bps < self.config.pressure_threshold_bps:
            return (
                False,
                (
                    f"USD pressure alignment {alignment_bps:.1f} bps below "
                    f"{self.config.pressure_threshold_bps:.1f} bps threshold"
                ),
            )
        return True, self._pressure_summary(reading)

    def _pressure_conflicts(
        self,
        reading: UsdPressureReading,
        current_direction: int,
    ) -> bool:
        required_usd_direction = _target_usd_pressure_direction(
            self.config.symbol,
            current_direction,
        )
        if required_usd_direction == 0:
            return False
        alignment_bps = required_usd_direction * reading.pressure_bps
        return (
            alignment_bps <= -self.config.pressure_threshold_bps
            and reading.confirming_symbols >= self.config.min_confirming_symbols
        )

    def _pressure_summary(self, reading: UsdPressureReading) -> str:
        return (
            f"USD pressure={reading.pressure_bps:.1f} bps, "
            f"confirming={reading.confirming_symbols}/{reading.component_count}"
        )

    def _with_pressure_context(
        self,
        decision: StrategyDecision,
        diagnostics: tuple[tuple[str, float | str], ...],
    ) -> StrategyDecision:
        return _decision(
            decision.action,
            self.config.symbol,
            decision.target_notional_usd,
            decision.reason,
            diagnostics,
            primary_signal=decision.primary_signal,
            supporting_signals=decision.supporting_signals,
            conflicting_signals=decision.conflicting_signals,
        )


class RelativeStrengthStrategy:
    def __init__(self, config: RelativeStrengthConfig | None = None) -> None:
        self.config = config or RelativeStrengthConfig()
        self._portfolio_closes: dict[str, tuple[float, ...]] = {}

    def update_portfolio_context(
        self,
        *,
        closes_by_symbol: Mapping[str, Sequence[float]],
        quotes_by_symbol: Mapping[str, QuoteSnapshot] | None = None,
    ) -> None:
        self._portfolio_closes = {
            instrument_for(symbol).symbol: tuple(prices)
            for symbol, prices in closes_by_symbol.items()
        }

    def read_relative_strength(
        self,
        prices: Sequence[float] | None = None,
    ) -> RelativeStrengthReading | None:
        target_symbol = instrument_for(self.config.symbol).symbol
        target_asset_class = instrument_for(target_symbol).asset_class
        closes_by_symbol = dict(self._portfolio_closes)
        if prices is not None:
            closes_by_symbol[target_symbol] = tuple(prices)

        scores: list[tuple[str, float, float, float, float]] = []
        for raw_symbol, symbol_prices in sorted(closes_by_symbol.items()):
            symbol = instrument_for(raw_symbol).symbol
            recent_prices = _recent_valid_prices(symbol_prices, self.config.lookback)
            if recent_prices is None:
                continue
            log_returns = _log_returns(recent_prices)
            cumulative_return = sum(log_returns)
            move_bps = cumulative_return * 10_000
            realized_volatility_bps = _population_stdev(log_returns) * 10_000
            path_return = sum(abs(value) for value in log_returns)
            trend_efficiency = (
                abs(cumulative_return) / path_return
                if path_return > 0
                else 0.0
            )
            score = (
                0.0
                if abs(move_bps) < self.config.min_abs_move_bps
                else move_bps
                / max(realized_volatility_bps, self.config.volatility_floor_bps)
            )
            scores.append(
                (symbol, score, move_bps, realized_volatility_bps, trend_efficiency)
            )

        if len(scores) < self.config.min_component_symbols:
            return None

        score_by_symbol = {symbol: score for symbol, score, _, _, _ in scores}
        if target_symbol not in score_by_symbol:
            return None

        score_values = [score for _, score, _, _, _ in scores]
        mean_score = sum(score_values) / len(score_values)
        score_stdev = _population_stdev(score_values)
        target_score = score_by_symbol[target_symbol]
        relative_zscore = (
            0.0
            if score_stdev <= 1e-12
            else (target_score - mean_score) / score_stdev
        )
        ranked = sorted(scores, key=lambda row: row[1], reverse=True)
        target_rank = next(
            index
            for index, (symbol, _, _, _, _) in enumerate(ranked, start=1)
            if symbol == target_symbol
        )
        target_row = next(row for row in scores if row[0] == target_symbol)
        strongest = ranked[0]
        weakest = ranked[-1]
        asset_class_rows = tuple(
            row
            for row in scores
            if instrument_for(row[0]).asset_class == target_asset_class
        )
        if len(asset_class_rows) >= self.config.asset_class_min_symbols:
            asset_class_values = [score for _, score, _, _, _ in asset_class_rows]
            asset_class_mean = sum(asset_class_values) / len(asset_class_values)
            asset_class_stdev = _population_stdev(asset_class_values)
            asset_class_zscore = (
                0.0
                if asset_class_stdev <= 1e-12
                else (target_score - asset_class_mean) / asset_class_stdev
            )
            asset_class_ranked = tuple(
                sorted(asset_class_rows, key=lambda row: row[1], reverse=True)
            )
            asset_class_rank = next(
                index
                for index, (symbol, _, _, _, _) in enumerate(asset_class_ranked, start=1)
                if symbol == target_symbol
            )
            asset_class_components = tuple(
                (symbol, score) for symbol, score, _, _, _ in asset_class_ranked
            )
        else:
            asset_class_zscore = None
            asset_class_rank = None
            asset_class_components = ()

        return RelativeStrengthReading(
            target_score=target_score,
            target_rank=target_rank,
            component_count=len(scores),
            relative_zscore=relative_zscore,
            score_dispersion=score_stdev,
            move_bps=target_row[2],
            realized_volatility_bps=target_row[3],
            trend_efficiency=target_row[4],
            strongest_symbol=strongest[0],
            strongest_score=strongest[1],
            weakest_symbol=weakest[0],
            weakest_score=weakest[1],
            components=tuple((symbol, score) for symbol, score, _, _, _ in ranked),
            asset_class_zscore=asset_class_zscore,
            asset_class_rank=asset_class_rank,
            asset_class_component_count=len(asset_class_rows),
            asset_class_components=asset_class_components,
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        reading = self.read_relative_strength(prices)
        current_direction = _notional_direction(current_notional_usd)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                "not enough portfolio context for relative strength ranking",
                primary_signal="relative_strength",
            )

        diagnostics = _relative_strength_diagnostics(reading) + (
            ("holding_period", float(holding_period)),
        )
        target_direction = _signed_threshold_direction(
            reading.relative_zscore,
            self.config.entry_zscore,
        )

        if target_direction == 0:
            if (
                current_direction != 0
                and abs(reading.relative_zscore) <= self.config.exit_zscore
            ):
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    (
                        f"relative z-score {reading.relative_zscore:.2f} faded below "
                        f"exit threshold {self.config.exit_zscore:.2f}"
                    ),
                    diagnostics,
                    primary_signal="relative_strength",
                )

            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                (
                    f"relative z-score {reading.relative_zscore:.2f} inside neutral band; "
                    f"rank {reading.target_rank}/{reading.component_count}"
                ),
                diagnostics,
                primary_signal="relative_strength",
            )

        asset_confirmed, asset_reason = self._asset_class_confirms(
            reading,
            target_direction,
        )
        if not asset_confirmed:
            if current_direction != 0 and target_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    (
                        "opposite relative-strength signal but asset-class "
                        f"confirmation blocked new exposure; {asset_reason}"
                    ),
                    diagnostics,
                    primary_signal="relative_strength",
                    conflicting_signals=("asset_class_confirmation",),
                )
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                f"asset-class confirmation blocked entry; {asset_reason}",
                diagnostics,
                primary_signal="relative_strength",
                conflicting_signals=("asset_class_confirmation",),
            )

        metal_confirmed, metal_reason = self._metal_trend_confirms(
            reading,
            target_direction,
        )
        if not metal_confirmed:
            if current_direction != 0 and target_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    (
                        "opposite relative-strength signal but metal trend "
                        f"confirmation blocked new exposure; {metal_reason}"
                    ),
                    diagnostics,
                    primary_signal="relative_strength",
                    conflicting_signals=("metal_trend_confirmation",),
                )
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                f"metal trend confirmation blocked entry; {metal_reason}",
                diagnostics,
                primary_signal="relative_strength",
                conflicting_signals=("metal_trend_confirmation",),
            )

        regime_confirmed, regime_reason = self._regime_confirms(reading)
        if not regime_confirmed:
            if current_direction != 0 and target_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    (
                        "opposite relative-strength signal but regime gate "
                        f"blocked new exposure; {regime_reason}"
                    ),
                    diagnostics,
                    primary_signal="relative_strength",
                    conflicting_signals=("relative_strength_regime",),
                )
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                f"relative-strength regime gate blocked entry; {regime_reason}",
                diagnostics,
                primary_signal="relative_strength",
                conflicting_signals=("relative_strength_regime",),
            )

        passed, reason = _passes_cost_filter(
            edge_bps=abs(reading.move_bps),
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        if not passed:
            if current_direction != 0 and target_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    f"opposite relative-strength signal but entry cost failed; {reason}",
                    diagnostics,
                    primary_signal="relative_strength",
                    conflicting_signals=("cost_filter",),
                )
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self.config.symbol,
                current_notional_usd,
                f"relative-strength entry blocked; {reason}",
                diagnostics,
                primary_signal="relative_strength",
                conflicting_signals=("cost_filter",),
            )

        target_notional = target_direction * self._target_notional(reading)
        if abs(target_notional) < self.config.min_trade_notional_usd:
            if current_direction != 0:
                return _decision(
                    StrategyAction.EXIT,
                    self.config.symbol,
                    0.0,
                    "relative-strength target below minimum trade size; exit",
                    diagnostics,
                    primary_signal="relative_strength",
                )
            return _decision(
                StrategyAction.NO_ACTION,
                self.config.symbol,
                0.0,
                "relative-strength target below minimum trade size",
                diagnostics,
                primary_signal="relative_strength",
            )

        action = (
            StrategyAction.ENTER
            if current_direction == 0
            else StrategyAction.REVERSE
            if current_direction != target_direction
            else StrategyAction.HOLD
        )
        target_for_decision = (
            current_notional_usd
            if action == StrategyAction.HOLD
            else target_notional
        )
        return _decision(
            action,
            self.config.symbol,
            target_for_decision,
            (
                f"relative z-score {reading.relative_zscore:.2f}, "
                f"rank {reading.target_rank}/{reading.component_count}; "
                f"{asset_reason}; {metal_reason}; {regime_reason}; {reason}"
            ),
            diagnostics,
            primary_signal="relative_strength",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _target_notional(self, reading: RelativeStrengthReading) -> float:
        confidence = _bounded_confidence(
            abs(reading.relative_zscore),
            self.config.entry_zscore,
        )
        sized = self.config.target_notional_usd * confidence
        return min(sized, self.config.max_target_notional_usd)

    def _asset_class_confirms(
        self,
        reading: RelativeStrengthReading,
        target_direction: int,
    ) -> tuple[bool, str]:
        if not self.config.require_asset_class_confirmation:
            return True, "asset-class confirmation disabled"
        if reading.asset_class_zscore is None or reading.asset_class_rank is None:
            return False, (
                "not enough same-asset symbols for confirmation "
                f"({reading.asset_class_component_count}/"
                f"{self.config.asset_class_min_symbols})"
            )
        aligned_score = target_direction * reading.asset_class_zscore
        if aligned_score < self.config.asset_class_entry_zscore:
            return False, (
                f"same-asset z-score {reading.asset_class_zscore:.2f} below "
                f"{self.config.asset_class_entry_zscore:.2f} confirmation threshold"
            )
        return True, (
            f"same-asset z-score {reading.asset_class_zscore:.2f}, "
            f"rank {reading.asset_class_rank}/"
            f"{reading.asset_class_component_count}"
        )

    def _metal_trend_confirms(
        self,
        reading: RelativeStrengthReading,
        target_direction: int,
    ) -> tuple[bool, str]:
        if not self.config.require_metal_trend_confirmation:
            return True, "metal trend confirmation disabled"
        if instrument_for(self.config.symbol).asset_class != AssetClass.METAL:
            return True, "metal trend confirmation not applicable"
        aligned_move_bps = target_direction * reading.move_bps
        if aligned_move_bps < self.config.metal_trend_min_move_bps:
            return False, (
                f"metal aligned move {aligned_move_bps:.1f} bps below "
                f"{self.config.metal_trend_min_move_bps:.1f} bps threshold"
            )
        if reading.trend_efficiency < self.config.metal_trend_min_efficiency:
            return False, (
                f"metal trend efficiency {reading.trend_efficiency:.2f} below "
                f"{self.config.metal_trend_min_efficiency:.2f} threshold"
            )
        return True, (
            f"metal trend move {aligned_move_bps:.1f} bps, "
            f"efficiency {reading.trend_efficiency:.2f}"
        )

    def _regime_confirms(
        self,
        reading: RelativeStrengthReading,
    ) -> tuple[bool, str]:
        if (
            self.config.min_score_dispersion <= 0
            and self.config.min_target_trend_efficiency <= 0
        ):
            return True, "relative-strength regime gate disabled"
        if reading.score_dispersion < self.config.min_score_dispersion:
            return False, (
                f"score dispersion {reading.score_dispersion:.2f} below "
                f"{self.config.min_score_dispersion:.2f} threshold"
            )
        if reading.trend_efficiency < self.config.min_target_trend_efficiency:
            return False, (
                f"target trend efficiency {reading.trend_efficiency:.2f} below "
                f"{self.config.min_target_trend_efficiency:.2f} threshold"
            )
        return True, (
            f"score dispersion {reading.score_dispersion:.2f}, "
            f"target trend efficiency {reading.trend_efficiency:.2f}"
        )


class CrossRateReversionStrategy:
    def __init__(self, config: CrossRateReversionConfig | None = None) -> None:
        self.config = config or CrossRateReversionConfig()
        self._target_instrument = instrument_for(self.config.symbol)
        self._portfolio_closes: dict[str, tuple[float, ...]] = {}

    def update_portfolio_context(
        self,
        *,
        closes_by_symbol: Mapping[str, Sequence[float]],
        quotes_by_symbol: Mapping[str, QuoteSnapshot] | None = None,
    ) -> None:
        self._portfolio_closes = {
            instrument_for(symbol).symbol: tuple(prices)
            for symbol, prices in closes_by_symbol.items()
        }

    def read_cross_rate_reversion(
        self,
        prices: Sequence[float] | None = None,
    ) -> CrossRateReversionReading | None:
        if self._target_instrument.asset_class != AssetClass.FOREX:
            return None
        if (
            self.config.allowed_symbols
            and self._target_instrument.symbol not in self.config.allowed_symbols
        ):
            return None

        target_symbol = self._target_instrument.symbol
        closes_by_symbol = dict(self._portfolio_closes)
        if prices is not None:
            closes_by_symbol[target_symbol] = tuple(prices)

        target_prices = closes_by_symbol.get(target_symbol)
        if _recent_valid_prices(target_prices or (), self.config.lookback) is None:
            return None

        latest_snapshot = _price_snapshot_at_offset(closes_by_symbol, 1)
        path = _find_fx_conversion_path(
            latest_snapshot,
            target_symbol=target_symbol,
        )
        if path is None or len(path) < self.config.min_synthetic_components:
            return None

        deviations_bps: list[float] = []
        synthetic_prices: list[float] = []
        target_window: list[float] = []
        for offset in range(self.config.lookback, 0, -1):
            snapshot = _price_snapshot_at_offset(closes_by_symbol, offset)
            target_price = snapshot.get(target_symbol)
            synthetic_price = _synthetic_price_from_path(path, snapshot)
            if target_price is None or synthetic_price is None:
                return None
            target_window.append(target_price)
            synthetic_prices.append(synthetic_price)
            deviations_bps.append(log(target_price / synthetic_price) * 10_000)

        baseline = deviations_bps[:-1]
        mean_deviation_bps = sum(baseline) / len(baseline)
        stdev_deviation_bps = _population_stdev(baseline)
        latest_deviation_bps = deviations_bps[-1]
        zscore = (
            0.0
            if stdev_deviation_bps <= 1e-12
            else (latest_deviation_bps - mean_deviation_bps) / stdev_deviation_bps
        )
        realized_volatility = _population_stdev(_log_returns(target_window))

        return CrossRateReversionReading(
            target_price=target_window[-1],
            synthetic_price=synthetic_prices[-1],
            deviation_bps=latest_deviation_bps,
            mean_deviation_bps=mean_deviation_bps,
            stdev_deviation_bps=stdev_deviation_bps,
            zscore=zscore,
            realized_volatility=realized_volatility,
            component_symbols=tuple(step.symbol for step in path),
            currency_path=(path[0].from_currency,)
            + tuple(step.to_currency for step in path),
        )

    def generate_signals(
        self,
        prices: Sequence[float],
        *,
        quote: QuoteSnapshot | None = None,
    ) -> tuple[StrategySignal, ...]:
        reading = self.read_cross_rate_reversion(prices)
        cost_bps = _estimated_round_trip_cost_bps(
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
        )
        if reading is None:
            return (
                _flat_signal(
                    "cross_rate_reversion",
                    self._target_instrument.symbol,
                    1.0,
                    SignalHorizon.MEDIUM,
                    self._missing_context_reason(),
                    cost_bps,
                ),
            )

        diagnostics = _cross_rate_reversion_diagnostics(reading)
        abs_deviation_bps = abs(reading.deviation_bps)
        if (
            abs(reading.zscore) < self.config.entry_zscore
            or abs_deviation_bps < self.config.min_abs_deviation_bps
            or abs_deviation_bps > self.config.max_abs_deviation_bps
            or abs_deviation_bps <= cost_bps * self.config.cost_buffer
        ):
            return (
                _flat_signal(
                    "cross_rate_reversion",
                    self._target_instrument.symbol,
                    1.0,
                    SignalHorizon.MEDIUM,
                    (
                        "cross-rate reversion blocked "
                        f"(z={reading.zscore:.2f}, "
                        f"deviation={reading.deviation_bps:.1f} bps)"
                    ),
                    cost_bps,
                    diagnostics,
                    expected_edge_bps=abs_deviation_bps,
                ),
            )

        direction = (
            SignalDirection.SHORT
            if reading.zscore > 0
            else SignalDirection.LONG
        )
        confidence = min(
            _bounded_confidence(abs(reading.zscore), self.config.entry_zscore),
            _bounded_confidence(
                abs_deviation_bps,
                max(self.config.min_abs_deviation_bps, 1e-12),
            ),
        )
        return (
            StrategySignal(
                strategy_name="cross_rate_reversion",
                symbol=self._target_instrument.symbol,
                direction=direction,
                confidence=confidence,
                expected_edge_bps=abs_deviation_bps,
                cost_bps=cost_bps,
                weight=1.0,
                horizon=SignalHorizon.MEDIUM,
                reason=self._entry_reason(reading),
                diagnostics=diagnostics,
            ),
        )

    def generate_decision(
        self,
        prices: Sequence[float],
        *,
        current_notional_usd: float = 0.0,
        holding_period: int = 0,
        quote: QuoteSnapshot | None = None,
    ) -> StrategyDecision:
        current_direction = _notional_direction(current_notional_usd)
        if self._target_instrument.asset_class != AssetClass.FOREX:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.EXIT,
                self._target_instrument.symbol,
                0.0 if current_direction != 0 else current_notional_usd,
                "cross-rate reversion only applies to FX instruments",
                primary_signal="cross_rate_reversion",
            )

        reading = self.read_cross_rate_reversion(prices)
        if reading is None:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.EXIT,
                self._target_instrument.symbol,
                0.0 if current_direction != 0 else current_notional_usd,
                self._missing_context_reason(),
                primary_signal="cross_rate_reversion",
            )

        diagnostics = _cross_rate_reversion_diagnostics(reading) + (
            ("holding_period", float(holding_period)),
        )
        abs_deviation_bps = abs(reading.deviation_bps)

        if current_direction != 0 and holding_period >= self.config.max_holding_period:
            return _decision(
                StrategyAction.EXIT,
                self._target_instrument.symbol,
                0.0,
                f"max cross-rate holding period reached: {holding_period} bars",
                diagnostics,
                primary_signal="cross_rate_reversion",
            )

        if current_direction != 0 and abs(reading.zscore) <= self.config.exit_zscore:
            return _decision(
                StrategyAction.EXIT,
                self._target_instrument.symbol,
                0.0,
                (
                    f"cross-rate z-score {reading.zscore:.2f} faded below "
                    f"exit threshold {self.config.exit_zscore:.2f}"
                ),
                diagnostics,
                primary_signal="cross_rate_reversion",
            )

        if abs_deviation_bps > self.config.max_abs_deviation_bps:
            action = StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.EXIT
            return _decision(
                action,
                self._target_instrument.symbol,
                0.0 if current_direction != 0 else current_notional_usd,
                (
                    f"cross-rate deviation {reading.deviation_bps:.1f} bps exceeds "
                    f"stability guard {self.config.max_abs_deviation_bps:.1f} bps"
                ),
                diagnostics,
                primary_signal="cross_rate_reversion",
            )

        if abs_deviation_bps < self.config.min_abs_deviation_bps:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self._target_instrument.symbol,
                current_notional_usd if current_direction != 0 else 0.0,
                (
                    f"cross-rate deviation {reading.deviation_bps:.1f} bps below "
                    f"{self.config.min_abs_deviation_bps:.1f} bps minimum"
                ),
                diagnostics,
                primary_signal="cross_rate_reversion",
            )

        if abs(reading.zscore) < self.config.entry_zscore:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self._target_instrument.symbol,
                current_notional_usd if current_direction != 0 else 0.0,
                (
                    f"cross-rate z-score {reading.zscore:.2f} below entry "
                    f"threshold {self.config.entry_zscore:.2f}"
                ),
                diagnostics,
                primary_signal="cross_rate_reversion",
            )

        passed, cost_reason = _passes_cost_filter(
            edge_bps=abs_deviation_bps,
            quote=quote,
            slippage_bps=self.config.slippage_bps,
            fee_bps=self.config.fee_bps,
            cost_buffer=self.config.cost_buffer,
            max_spread_bps=self.config.max_spread_bps,
        )
        signal_direction = _reversion_direction(reading.zscore)
        if not passed:
            if current_direction != 0 and signal_direction != current_direction:
                return _decision(
                    StrategyAction.EXIT,
                    self._target_instrument.symbol,
                    0.0,
                    f"opposite cross-rate signal but entry cost failed; {cost_reason}",
                    diagnostics,
                    primary_signal="cross_rate_reversion",
                    conflicting_signals=("cost_filter",),
                )
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.HOLD,
                self._target_instrument.symbol,
                current_notional_usd if current_direction != 0 else 0.0,
                f"cross-rate entry blocked; {cost_reason}",
                diagnostics,
                primary_signal="cross_rate_reversion",
                conflicting_signals=("cost_filter",),
            )

        target_notional = signal_direction * self._target_notional(reading)
        if abs(target_notional) < self.config.min_trade_notional_usd:
            return _decision(
                StrategyAction.NO_ACTION if current_direction == 0 else StrategyAction.EXIT,
                self._target_instrument.symbol,
                0.0,
                "cross-rate target below minimum trade size",
                diagnostics,
                primary_signal="cross_rate_reversion",
            )

        action = (
            StrategyAction.ENTER
            if current_direction == 0
            else StrategyAction.REVERSE
            if current_direction != signal_direction
            else StrategyAction.HOLD
        )
        target_for_decision = (
            current_notional_usd
            if action == StrategyAction.HOLD
            else target_notional
        )
        return _decision(
            action,
            self._target_instrument.symbol,
            target_for_decision,
            f"{self._entry_reason(reading)}; {cost_reason}",
            diagnostics,
            primary_signal="cross_rate_reversion",
        )

    def generate_request(self, prices: Sequence[float]) -> TradeRequest | None:
        return self.generate_decision(prices).to_trade_request()

    def _target_notional(self, reading: CrossRateReversionReading) -> float:
        confidence = min(
            _bounded_confidence(abs(reading.zscore), self.config.entry_zscore),
            _bounded_confidence(
                abs(reading.deviation_bps),
                max(self.config.min_abs_deviation_bps, 1e-12),
            ),
        )
        return _sized_notional(
            position_sizing=self.config.position_sizing,
            base_notional=self.config.target_notional_usd,
            target_volatility=self.config.target_volatility,
            realized_volatility=reading.realized_volatility,
            volatility_floor=self.config.volatility_floor,
            signal_confidence=confidence,
            max_notional=self.config.max_target_notional_usd,
            min_trade_notional=self.config.min_trade_notional_usd,
        )

    def _entry_reason(self, reading: CrossRateReversionReading) -> str:
        valuation = "rich" if reading.zscore > 0 else "cheap"
        components = "/".join(reading.component_symbols)
        path = "->".join(reading.currency_path)
        side = "short" if reading.zscore > 0 else "long"
        return (
            f"{self._target_instrument.symbol} is {valuation} vs synthetic "
            f"{components} ({path}); z={reading.zscore:.2f}, "
            f"deviation={reading.deviation_bps:.1f} bps, target={side}"
        )

    def _missing_context_reason(self) -> str:
        if self._target_instrument.asset_class != AssetClass.FOREX:
            return "cross-rate reversion only applies to FX instruments"
        if (
            self.config.allowed_symbols
            and self._target_instrument.symbol not in self.config.allowed_symbols
        ):
            allowed = ", ".join(self.config.allowed_symbols)
            return f"{self._target_instrument.symbol} is outside cross-rate allowlist ({allowed})"
        return (
            "not enough FX cross-rate context to synthesize "
            f"{self._target_instrument.symbol}"
        )


def normalize_strategy_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "momentum": "simple_momentum",
        "simple": "simple_momentum",
        "simple_momentum": "simple_momentum",
        "session_momentum": "session_momentum",
        "sessionmomentum": "session_momentum",
        "late_momentum": "session_momentum",
        "late_session_momentum": "session_momentum",
        "multi_horizon_momentum": "multi_horizon_momentum",
        "multihorizonmomentum": "multi_horizon_momentum",
        "multi_horizon": "multi_horizon_momentum",
        "multi_momentum": "multi_horizon_momentum",
        "volatility_managed_momentum": "multi_horizon_momentum",
        "vol_managed_momentum": "multi_horizon_momentum",
        "dual_horizon_momentum": "multi_horizon_momentum",
        "autocorrelation": "autocorrelation_regime",
        "autocorrelation_regime": "autocorrelation_regime",
        "autocorrelationregime": "autocorrelation_regime",
        "return_autocorrelation": "autocorrelation_regime",
        "rho_regime": "autocorrelation_regime",
        "serial_correlation": "autocorrelation_regime",
        "intraday": "intraday_seasonality",
        "intraday_seasonality": "intraday_seasonality",
        "intradayseasonality": "intraday_seasonality",
        "same_slot": "intraday_seasonality",
        "same_time": "intraday_seasonality",
        "time_of_day": "intraday_seasonality",
        "conditional": "conditional_seasonality",
        "conditional_seasonality": "conditional_seasonality",
        "conditionalseasonality": "conditional_seasonality",
        "conditioned_seasonality": "conditional_seasonality",
        "hourly_condition": "conditional_seasonality",
        "hourly_drift": "conditional_seasonality",
        "ma": "ma_crossover",
        "moving_average": "ma_crossover",
        "moving_average_crossover": "ma_crossover",
        "macrossover": "ma_crossover",
        "ma_crossover": "ma_crossover",
        "crossover": "ma_crossover",
        "macd": "macd_momentum",
        "macd_momentum": "macd_momentum",
        "macdmomentum": "macd_momentum",
        "macd_histogram": "macd_momentum",
        "momentum_acceleration": "macd_momentum",
        "asset_adaptive_macd": "asset_adaptive_macd",
        "adaptive_macd": "asset_adaptive_macd",
        "crypto_strict_macd": "asset_adaptive_macd",
        "crypto_macd": "asset_adaptive_macd",
        "macd_conditional": "macd_conditional_fallback",
        "macd_conditional_fallback": "macd_conditional_fallback",
        "macdconditionalfallback": "macd_conditional_fallback",
        "macd_fallback": "macd_conditional_fallback",
        "macd_seasonality_fallback": "macd_conditional_fallback",
        "conditional_fallback": "macd_conditional_fallback",
        "macd_squeeze": "macd_squeeze_complement",
        "macd_squeeze_complement": "macd_squeeze_complement",
        "macdsqueezecomplement": "macd_squeeze_complement",
        "squeeze_complement": "macd_squeeze_complement",
        "macd_vol_squeeze": "macd_squeeze_complement",
        "donchian": "breakout",
        "breakout": "breakout",
        "session": "session_breakout",
        "session_breakout": "session_breakout",
        "sessionbreakout": "session_breakout",
        "session_breakout_strategy": "session_breakout",
        "volatility_breakout": "session_breakout",
        "vol_breakout": "session_breakout",
        "volatility_squeeze": "volatility_squeeze",
        "squeeze": "volatility_squeeze",
        "squeeze_breakout": "volatility_squeeze",
        "vol_squeeze": "volatility_squeeze",
        "volsqueeze": "volatility_squeeze",
        "dual_squeeze": "dual_squeeze",
        "dualsqueeze": "dual_squeeze",
        "dual_squeeze_breakout": "dual_squeeze",
        "confirmed_squeeze": "dual_squeeze",
        "squeeze_confirmation": "dual_squeeze",
        "asset_adaptive_dual_squeeze": "asset_adaptive_dual_squeeze",
        "asset_adaptive_squeeze": "asset_adaptive_dual_squeeze",
        "adaptive_dual_squeeze": "asset_adaptive_dual_squeeze",
        "adaptive_squeeze": "asset_adaptive_dual_squeeze",
        "metal_fast_squeeze": "asset_adaptive_dual_squeeze",
        "range_expansion": "range_expansion_trend",
        "range_expansion_trend": "range_expansion_trend",
        "rangeexpansiontrend": "range_expansion_trend",
        "expansion_trend": "range_expansion_trend",
        "volatility_expansion": "range_expansion_trend",
        "vol_expansion": "range_expansion_trend",
        "range_break_trend": "range_expansion_trend",
        "pullback": "trend_pullback",
        "trend_pullback": "trend_pullback",
        "trendpullback": "trend_pullback",
        "momentum_pullback": "trend_pullback",
        "continuation_pullback": "trend_pullback",
        "exhaustion": "exhaustion_reversal",
        "exhaustion_reversal": "exhaustion_reversal",
        "exhaustionreversal": "exhaustion_reversal",
        "shock_reversal": "exhaustion_reversal",
        "jump_reversal": "exhaustion_reversal",
        "liquidity": "liquidity_sweep_reversal",
        "liquidity_sweep": "liquidity_sweep_reversal",
        "liquidity_sweep_reversal": "liquidity_sweep_reversal",
        "liquiditysweepreversal": "liquidity_sweep_reversal",
        "sweep_reversal": "liquidity_sweep_reversal",
        "false_breakout": "liquidity_sweep_reversal",
        "stop_hunt": "liquidity_sweep_reversal",
        "fixing": "fixing_reversal",
        "fixing_reversal": "fixing_reversal",
        "fixingreversal": "fixing_reversal",
        "fix_reversal": "fixing_reversal",
        "london_fix": "fixing_reversal",
        "wm_fix": "fixing_reversal",
        "post_fix_reversal": "fixing_reversal",
        "kalman": "kalman_trend",
        "kalman_trend": "kalman_trend",
        "kalmantrend": "kalman_trend",
        "kalman_trend_following": "kalman_trend",
        "time_series_trend": "kalman_trend",
        "quality": "quality_trend",
        "quality_trend": "quality_trend",
        "qualitytrend": "quality_trend",
        "trend_quality": "quality_trend",
        "confirmed_trend": "quality_trend",
        "macd_kalman": "quality_trend",
        "kalman_macd": "quality_trend",
        "champion": "champion_ensemble",
        "champion_ensemble": "champion_ensemble",
        "championensemble": "champion_ensemble",
        "champion_router": "champion_ensemble",
        "research_champion": "champion_ensemble",
        "ensemble": "champion_ensemble",
        "meanreversion": "mean_reversion",
        "mean_reversion": "mean_reversion",
        "reversion": "mean_reversion",
        "crypto_mean_reversion": "crypto_mean_reversion",
        "cryptomeanreversion": "crypto_mean_reversion",
        "crypto_reversion": "crypto_mean_reversion",
        "low_turnover_reversion": "crypto_mean_reversion",
        "strict_reversion": "crypto_mean_reversion",
        "regime": "regime_switch",
        "regime_switch": "regime_switch",
        "regimeswitch": "regime_switch",
        "controller": "regime_switch",
        "router": "alpha_router",
        "alpha": "alpha_router",
        "alpha_router": "alpha_router",
        "alpharouter": "alpha_router",
        "crypto_blend": "crypto_trend_reversion",
        "crypto_router": "crypto_trend_reversion",
        "crypto_trend_reversion": "crypto_trend_reversion",
        "cryptotrendreversion": "crypto_trend_reversion",
        "trend_reversion": "crypto_trend_reversion",
        "trend_reversion_router": "crypto_trend_reversion",
        "macd_reversion": "crypto_trend_reversion",
        "usd": "usd_pressure_router",
        "usd_pressure": "usd_pressure_router",
        "usd_pressure_router": "usd_pressure_router",
        "usdpressure": "usd_pressure_router",
        "usd_router": "usd_pressure_router",
        "dollar_pressure": "usd_pressure_router",
        "relative": "relative_strength",
        "relative_strength": "relative_strength",
        "rel_strength": "relative_strength",
        "cross_sectional": "relative_strength",
        "cross_sectional_momentum": "relative_strength",
        "xs_momentum": "relative_strength",
        "cross_rate": "cross_rate_reversion",
        "cross_rate_reversion": "cross_rate_reversion",
        "crossratereversion": "cross_rate_reversion",
        "fx_cross": "cross_rate_reversion",
        "fx_cross_rate": "cross_rate_reversion",
        "triangular": "cross_rate_reversion",
        "triangular_reversion": "cross_rate_reversion",
    }
    if normalized not in aliases:
        valid = ", ".join(STRATEGY_NAMES)
        raise ValueError(f"unknown strategy {name!r}; expected one of: {valid}")
    return aliases[normalized]


def build_strategy(
    name: str,
    *,
    simple_momentum: MomentumConfig,
    mean_reversion: MeanReversionConfig,
    session_momentum: MomentumConfig | None = None,
    multi_horizon_momentum: MultiHorizonMomentumConfig | None = None,
    autocorrelation_regime: AutocorrelationRegimeConfig | None = None,
    intraday_seasonality: IntradaySeasonalityConfig | None = None,
    conditional_seasonality: ConditionalSeasonalityConfig | None = None,
    ma_crossover: MovingAverageCrossoverConfig | None = None,
    macd_momentum: MacdMomentumConfig | None = None,
    macd_conditional_fallback: MacdConditionalFallbackConfig | None = None,
    macd_squeeze_complement: MacdSqueezeComplementConfig | None = None,
    breakout: BreakoutConfig | None = None,
    volatility_squeeze: VolatilitySqueezeConfig | None = None,
    dual_squeeze: DualSqueezeConfig | None = None,
    asset_adaptive_dual_squeeze: AssetAdaptiveDualSqueezeConfig | None = None,
    range_expansion_trend: RangeExpansionTrendConfig | None = None,
    session_breakout: SessionBreakoutConfig | None = None,
    trend_pullback: TrendPullbackConfig | None = None,
    exhaustion_reversal: ExhaustionReversalConfig | None = None,
    liquidity_sweep_reversal: LiquiditySweepReversalConfig | None = None,
    fixing_reversal: FixingReversalConfig | None = None,
    kalman_trend: KalmanTrendStrategyConfig | None = None,
    quality_trend: QualityTrendConfig | None = None,
    champion_ensemble: ChampionEnsembleConfig | None = None,
    regime_switch: RegimeConfig | None = None,
    alpha_router: AlphaRouterConfig | None = None,
    usd_pressure: UsdPressureConfig | None = None,
    relative_strength: RelativeStrengthConfig | None = None,
    cross_rate_reversion: CrossRateReversionConfig | None = None,
    symbol: str | None = None,
) -> Strategy:
    strategy_name = normalize_strategy_name(name)
    if strategy_name == "simple_momentum":
        config = simple_momentum if symbol is None else replace(simple_momentum, symbol=symbol)
        return SimpleMomentumStrategy(config)
    if strategy_name == "session_momentum":
        config = session_momentum or MomentumConfig(
            forex_allowed_utc_hours=(17, 18, 19, 20, 21),
            metal_allowed_utc_hours=(17, 18, 19, 20, 21),
            crypto_allowed_utc_hours=None,
        )
        config = config if symbol is None else replace(config, symbol=symbol)
        return SimpleMomentumStrategy(config)
    if strategy_name == "multi_horizon_momentum":
        config = multi_horizon_momentum or MultiHorizonMomentumConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MultiHorizonMomentumStrategy(config)
    if strategy_name == "autocorrelation_regime":
        config = autocorrelation_regime or AutocorrelationRegimeConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return AutocorrelationRegimeStrategy(config)
    if strategy_name == "intraday_seasonality":
        config = intraday_seasonality or IntradaySeasonalityConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return IntradaySeasonalityStrategy(config)
    if strategy_name == "conditional_seasonality":
        config = conditional_seasonality or ConditionalSeasonalityConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return ConditionalSeasonalityStrategy(config)
    if strategy_name == "ma_crossover":
        config = ma_crossover or MovingAverageCrossoverConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MovingAverageCrossoverStrategy(config)
    if strategy_name == "macd_momentum":
        config = macd_momentum or MacdMomentumConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MacdMomentumStrategy(config)
    if strategy_name == "asset_adaptive_macd":
        config = macd_momentum or MacdMomentumConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MacdMomentumStrategy(_asset_adaptive_macd_config(config))
    if strategy_name == "macd_conditional_fallback":
        config = macd_conditional_fallback or MacdConditionalFallbackConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MacdConditionalFallbackStrategy(
            config,
            macd_momentum=macd_momentum,
            conditional_seasonality=conditional_seasonality,
        )
    if strategy_name == "macd_squeeze_complement":
        config = macd_squeeze_complement or MacdSqueezeComplementConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MacdSqueezeComplementStrategy(
            config,
            macd_momentum=macd_momentum,
            volatility_squeeze=volatility_squeeze,
        )
    if strategy_name == "breakout":
        config = breakout or BreakoutConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return BreakoutStrategy(config)
    if strategy_name == "volatility_squeeze":
        config = volatility_squeeze or VolatilitySqueezeConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return VolatilitySqueezeStrategy(config)
    if strategy_name == "dual_squeeze":
        config = dual_squeeze or DualSqueezeConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return DualSqueezeStrategy(config)
    if strategy_name == "asset_adaptive_dual_squeeze":
        config = asset_adaptive_dual_squeeze or AssetAdaptiveDualSqueezeConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return AssetAdaptiveDualSqueezeStrategy(config)
    if strategy_name == "range_expansion_trend":
        config = range_expansion_trend or RangeExpansionTrendConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return RangeExpansionTrendStrategy(config)
    if strategy_name == "session_breakout":
        config = session_breakout or SessionBreakoutConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return SessionBreakoutStrategy(config)
    if strategy_name == "trend_pullback":
        config = trend_pullback or TrendPullbackConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return TrendPullbackStrategy(config)
    if strategy_name == "exhaustion_reversal":
        config = exhaustion_reversal or ExhaustionReversalConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return ExhaustionReversalStrategy(config)
    if strategy_name == "liquidity_sweep_reversal":
        config = liquidity_sweep_reversal or LiquiditySweepReversalConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return LiquiditySweepReversalStrategy(config)
    if strategy_name == "fixing_reversal":
        config = fixing_reversal or FixingReversalConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return FixingReversalStrategy(config)
    if strategy_name == "kalman_trend":
        config = kalman_trend or KalmanTrendStrategyConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return KalmanTrendStrategy(config)
    if strategy_name == "quality_trend":
        config = quality_trend or QualityTrendConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return QualityTrendStrategy(config)
    if strategy_name == "champion_ensemble":
        config = champion_ensemble or ChampionEnsembleConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return ChampionEnsembleStrategy(
            config=config,
            kalman_trend=kalman_trend,
            asset_adaptive_dual_squeeze=asset_adaptive_dual_squeeze,
            dual_squeeze=dual_squeeze,
            trend_pullback=trend_pullback,
            fixing_reversal=fixing_reversal,
            macd_momentum=macd_momentum,
        )
    if strategy_name == "mean_reversion":
        config = mean_reversion if symbol is None else replace(mean_reversion, symbol=symbol)
        return MeanReversionStrategy(config)
    if strategy_name == "crypto_mean_reversion":
        config = mean_reversion or MeanReversionConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return MeanReversionStrategy(_crypto_mean_reversion_config(config))
    if strategy_name == "regime_switch":
        config = regime_switch or RegimeConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return RegimeSwitchingStrategy(
            config=config,
            momentum=simple_momentum,
            mean_reversion=mean_reversion,
        )
    if strategy_name == "alpha_router":
        config = alpha_router or AlphaRouterConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return AlphaRouterStrategy(
            config=config,
            momentum=simple_momentum,
            moving_average=ma_crossover,
            breakout=breakout,
            volatility_squeeze=volatility_squeeze,
            dual_squeeze=dual_squeeze,
            exhaustion_reversal=exhaustion_reversal,
            session_breakout=session_breakout,
            macd_momentum=macd_momentum,
            kalman_trend=kalman_trend,
            mean_reversion=mean_reversion,
            relative_strength=relative_strength,
            cross_rate_reversion=cross_rate_reversion,
        )
    if strategy_name == "crypto_trend_reversion":
        config = alpha_router or AlphaRouterConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return AlphaRouterStrategy(
            config=_crypto_trend_reversion_router_config(
                config,
                macd_momentum=macd_momentum,
            ),
            momentum=simple_momentum,
            moving_average=ma_crossover,
            breakout=breakout,
            volatility_squeeze=volatility_squeeze,
            dual_squeeze=dual_squeeze,
            exhaustion_reversal=exhaustion_reversal,
            session_breakout=session_breakout,
            macd_momentum=macd_momentum,
            kalman_trend=kalman_trend,
            mean_reversion=mean_reversion,
            relative_strength=relative_strength,
            cross_rate_reversion=cross_rate_reversion,
        )
    if strategy_name == "usd_pressure_router":
        config = usd_pressure or UsdPressureConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        router_config = alpha_router or AlphaRouterConfig()
        router_config = replace(router_config, symbol=config.symbol)
        return UsdPressureRouterStrategy(
            config=config,
            base_strategy=AlphaRouterStrategy(
                config=router_config,
                momentum=simple_momentum,
                moving_average=ma_crossover,
                breakout=breakout,
                volatility_squeeze=volatility_squeeze,
                dual_squeeze=dual_squeeze,
                exhaustion_reversal=exhaustion_reversal,
                session_breakout=session_breakout,
                macd_momentum=macd_momentum,
                kalman_trend=kalman_trend,
                mean_reversion=mean_reversion,
                relative_strength=relative_strength,
                cross_rate_reversion=cross_rate_reversion,
            ),
        )
    if strategy_name == "relative_strength":
        config = relative_strength or RelativeStrengthConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return RelativeStrengthStrategy(config)
    if strategy_name == "cross_rate_reversion":
        config = cross_rate_reversion or CrossRateReversionConfig()
        config = config if symbol is None else replace(config, symbol=symbol)
        return CrossRateReversionStrategy(config)
    raise ValueError(f"unsupported strategy {name!r}")


def _asset_adaptive_macd_config(config: MacdMomentumConfig) -> MacdMomentumConfig:
    instrument = instrument_for(config.symbol)
    if instrument.asset_class != AssetClass.CRYPTO:
        return config
    return replace(
        config,
        min_histogram_bps=max(config.min_histogram_bps, CRYPTO_MACD_MIN_HISTOGRAM_BPS),
        min_macd_bps=max(config.min_macd_bps, CRYPTO_MACD_MIN_MACD_BPS),
        min_trend_efficiency=max(
            config.min_trend_efficiency,
            CRYPTO_MACD_MIN_TREND_EFFICIENCY,
        ),
        max_holding_period=min(
            config.max_holding_period,
            CRYPTO_MACD_MAX_HOLDING_PERIOD,
        ),
    )


def _crypto_mean_reversion_config(config: MeanReversionConfig) -> MeanReversionConfig:
    instrument = instrument_for(config.symbol)
    if instrument.asset_class != AssetClass.CRYPTO:
        return config

    max_target_notional = config.max_target_notional_usd
    if max_target_notional is None:
        max_target_notional = 150_000.0

    return replace(
        config,
        lookback=max(config.lookback, 16),
        entry_zscore=max(config.entry_zscore, 1.0),
        exit_zscore=min(config.exit_zscore, 0.25),
        max_trend_bps=max(config.max_trend_bps, 50.0),
        min_stdev_bps=max(config.min_stdev_bps, 0.05),
        target_notional_usd=max(config.target_notional_usd, 500_000.0),
        position_sizing="volatility",
        target_volatility=max(config.target_volatility, 0.002),
        max_target_notional_usd=max(max_target_notional, 150_000.0),
        min_trade_notional_usd=max(config.min_trade_notional_usd, 1_000.0),
        max_holding_period=max(config.max_holding_period, 20),
        stop_zscore=max(config.stop_zscore, 4.0),
        cost_buffer=max(config.cost_buffer, 1.0),
    )


def _crypto_trend_reversion_router_config(
    config: AlphaRouterConfig,
    *,
    macd_momentum: MacdMomentumConfig | None,
) -> AlphaRouterConfig:
    target_notional = config.target_notional_usd
    max_target_notional = config.max_target_notional_usd
    if macd_momentum is not None:
        target_notional = max(target_notional, macd_momentum.target_notional_usd)
        if macd_momentum.max_target_notional_usd is not None:
            max_target_notional = max(
                max_target_notional,
                macd_momentum.max_target_notional_usd,
            )
    max_target_notional = max(max_target_notional, target_notional)

    return replace(
        config,
        target_notional_usd=target_notional,
        max_target_notional_usd=max_target_notional,
        entry_score=0.30,
        exit_score=0.10,
        min_signal_confidence=0.20,
        cost_buffer=1.05,
        momentum_weight=0.0,
        moving_average_weight=0.0,
        breakout_weight=0.0,
        session_breakout_weight=0.0,
        macd_momentum_weight=0.70,
        kalman_trend_weight=0.0,
        volatility_squeeze_weight=0.0,
        dual_squeeze_weight=0.0,
        exhaustion_reversal_weight=0.0,
        mean_reversion_weight=0.30,
        relative_strength_weight=0.0,
        cross_rate_weight=0.0,
        conflict_penalty=0.75,
        primary_signal_override_enabled=True,
        primary_signal_min_confidence=0.75,
        primary_signal_min_edge_bps=3.0,
        adaptive_weighting_enabled=True,
        chop_mean_reversion_multiplier=1.25,
        chop_trend_signal_multiplier=0.75,
        trend_aligned_signal_multiplier=1.25,
        trend_counter_signal_multiplier=0.50,
        high_volatility_reversion_multiplier=1.05,
        high_volatility_trend_multiplier=0.95,
        low_volatility_reversion_multiplier=0.90,
        low_volatility_trend_multiplier=1.05,
    )


def _decision(
    action: StrategyAction,
    symbol: str,
    target_notional_usd: float,
    reason: str,
    diagnostics: tuple[tuple[str, float | str], ...] = (),
    *,
    primary_signal: str = "strategy",
    supporting_signals: tuple[str, ...] = (),
    conflicting_signals: tuple[str, ...] = (),
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        symbol=symbol,
        target_notional_usd=target_notional_usd,
        reason=reason,
        diagnostics=diagnostics,
        primary_signal=primary_signal,
        supporting_signals=supporting_signals,
        conflicting_signals=conflicting_signals,
    )


def _recent_valid_prices(prices: Sequence[float], lookback: int) -> list[float] | None:
    if len(prices) < lookback:
        return None

    recent_prices = list(prices)[-lookback:]
    for price in recent_prices:
        if price <= 0 or not isfinite(price):
            raise ValueError("prices must be positive finite numbers")
    return recent_prices


def _log_returns(prices: Sequence[float]) -> tuple[float, ...]:
    return tuple(log(current / previous) for previous, current in zip(prices, prices[1:]))


def _log_move_bps(prices: Sequence[float], start_index: int, end_index: int) -> float:
    if start_index < 0 or end_index >= len(prices) or start_index >= end_index:
        raise ValueError("invalid price indexes for log move")
    return log(prices[end_index] / prices[start_index]) * 10_000


def _population_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    x_values = values[:-1]
    y_values = values[1:]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    x_variance = sum((value - x_mean) ** 2 for value in x_values)
    y_variance = sum((value - y_mean) ** 2 for value in y_values)
    denominator = sqrt(x_variance * y_variance)
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, covariance / denominator))


def _simple_average(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _ema_series(values: Sequence[float], window: int) -> tuple[float, ...]:
    if window < 1:
        raise ValueError("EMA window must be at least 1")
    alpha = 2.0 / (window + 1.0)
    series: list[float] = []
    current: float | None = None
    for value in values:
        current = value if current is None else (alpha * value) + ((1.0 - alpha) * current)
        series.append(current)
    return tuple(series)


def _momentum_diagnostics(reading: MomentumReading) -> tuple[tuple[str, float | str], ...]:
    return (
        ("first_price", reading.first_price),
        ("last_price", reading.last_price),
        ("cumulative_log_return", reading.cumulative_log_return),
        ("move_bps", reading.move_bps),
        ("realized_volatility", reading.realized_volatility),
        ("normalized_momentum", reading.normalized_momentum),
        ("trend_efficiency", reading.trend_efficiency),
    )


def _multi_horizon_momentum_diagnostics(
    reading: MultiHorizonMomentumReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("multi_horizon_fast_move_bps", reading.fast_move_bps),
        ("multi_horizon_slow_move_bps", reading.slow_move_bps),
        ("multi_horizon_realized_volatility_bps", reading.realized_volatility_bps),
        ("multi_horizon_baseline_volatility_bps", reading.baseline_volatility_bps),
        ("multi_horizon_volatility_ratio", reading.volatility_ratio),
        (
            "multi_horizon_normalized_slow_momentum",
            reading.normalized_slow_momentum,
        ),
        ("multi_horizon_trend_efficiency", reading.trend_efficiency),
        ("multi_horizon_expected_edge_bps", reading.expected_edge_bps),
        ("multi_horizon_signal_direction", reading.signal_direction.value),
        (
            "multi_horizon_utc_hour",
            "n/a" if reading.utc_hour is None else float(reading.utc_hour),
        ),
        (
            "multi_horizon_session_allowed",
            "yes" if reading.session_allowed else "no",
        ),
    )


def _autocorrelation_regime_diagnostics(
    reading: AutocorrelationRegimeReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("autocorrelation_latest_price", reading.latest_price),
        ("autocorrelation_lag1", reading.lag1_autocorrelation),
        ("autocorrelation_signal_move_bps", reading.signal_move_bps),
        ("autocorrelation_zscore", reading.zscore),
        ("autocorrelation_realized_volatility_bps", reading.realized_volatility_bps),
        ("autocorrelation_trend_efficiency", reading.trend_efficiency),
        ("autocorrelation_mode", reading.mode),
        ("autocorrelation_expected_edge_bps", reading.expected_edge_bps),
        ("autocorrelation_signal_direction", reading.signal_direction.value),
    )


def _intraday_seasonality_diagnostics(
    reading: IntradaySeasonalityReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("intraday_mean_return_bps", reading.mean_return_bps),
        ("intraday_realized_volatility", reading.realized_volatility),
        ("intraday_consistency", reading.consistency),
        ("intraday_positive_samples", float(reading.positive_samples)),
        ("intraday_negative_samples", float(reading.negative_samples)),
        ("intraday_sample_count", float(reading.sample_count)),
        (
            "intraday_utc_hour",
            "n/a" if reading.utc_hour is None else float(reading.utc_hour),
        ),
        ("intraday_session_allowed", "yes" if reading.session_allowed else "no"),
    )


def _conditional_seasonality_diagnostics(
    reading: ConditionalSeasonalityReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("conditional_condition", reading.current_condition),
        ("conditional_current_momentum_bps", reading.current_momentum_bps),
        ("conditional_mean_forward_return_bps", reading.mean_forward_return_bps),
        ("conditional_realized_volatility_bps", reading.realized_volatility_bps),
        ("conditional_tstat", reading.tstat),
        ("conditional_consistency", reading.consistency),
        ("conditional_positive_samples", float(reading.positive_samples)),
        ("conditional_negative_samples", float(reading.negative_samples)),
        ("conditional_sample_count", float(reading.sample_count)),
        ("conditional_expected_edge_bps", reading.expected_edge_bps),
        ("conditional_signal_direction", reading.signal_direction.value),
        (
            "conditional_utc_hour",
            "n/a" if reading.utc_hour is None else float(reading.utc_hour),
        ),
        (
            "conditional_session_allowed",
            "yes" if reading.session_allowed else "no",
        ),
    )


def _ma_crossover_diagnostics(
    reading: MovingAverageCrossoverReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("fast_average", reading.fast_average),
        ("slow_average", reading.slow_average),
        (
            "previous_fast_average",
            "n/a" if reading.previous_fast_average is None else reading.previous_fast_average,
        ),
        (
            "previous_slow_average",
            "n/a" if reading.previous_slow_average is None else reading.previous_slow_average,
        ),
        ("latest_price", reading.last_price),
        ("separation_bps", reading.separation_bps),
        (
            "previous_separation_bps",
            (
                "n/a"
                if reading.previous_separation_bps is None
                else reading.previous_separation_bps
            ),
        ),
        ("crossed_direction", reading.crossed_direction.value),
        ("realized_volatility", reading.realized_volatility),
        ("trend_efficiency", reading.trend_efficiency),
    )


def _macd_momentum_diagnostics(
    reading: MacdMomentumReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("macd_fast_ema", reading.fast_ema),
        ("macd_slow_ema", reading.slow_ema),
        ("macd_line", reading.macd),
        ("macd_signal", reading.signal),
        ("macd_histogram", reading.histogram),
        ("macd_previous_histogram", reading.previous_histogram),
        ("macd_line_bps", reading.macd_bps),
        ("macd_signal_bps", reading.signal_bps),
        ("macd_histogram_bps", reading.histogram_bps),
        ("macd_previous_histogram_bps", reading.previous_histogram_bps),
        ("macd_histogram_slope_bps", reading.histogram_slope_bps),
        ("macd_crossed_direction", reading.crossed_direction.value),
        ("macd_last_price", reading.last_price),
        ("macd_realized_volatility_bps", reading.realized_volatility_bps),
        ("macd_trend_efficiency", reading.trend_efficiency),
        (
            "macd_utc_hour",
            "n/a" if reading.utc_hour is None else float(reading.utc_hour),
        ),
        ("macd_session_allowed", "yes" if reading.session_allowed else "no"),
    )


def _breakout_diagnostics(reading: BreakoutReading) -> tuple[tuple[str, float | str], ...]:
    return (
        ("upper_band", reading.upper_band),
        ("lower_band", reading.lower_band),
        ("last_price", reading.last_price),
        ("channel_width_bps", reading.channel_width_bps),
        ("breakout_bps", reading.breakout_bps),
        ("position_in_channel", reading.position_in_channel),
        ("realized_volatility", reading.realized_volatility),
    )


def _volatility_squeeze_diagnostics(
    reading: VolatilitySqueezeReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("squeeze_mean_price", reading.mean_price),
        ("squeeze_upper_band", reading.upper_band),
        ("squeeze_lower_band", reading.lower_band),
        ("squeeze_last_price", reading.last_price),
        ("squeeze_band_width_bps", reading.band_width_bps),
        ("squeeze_breakout_bps", reading.breakout_bps),
        ("squeeze_recent_volatility_bps", reading.recent_volatility_bps),
        ("squeeze_prior_volatility_bps", reading.prior_volatility_bps),
        ("squeeze_ratio", reading.squeeze_ratio),
        ("squeeze_realized_volatility", reading.realized_volatility),
    )


def _volatility_squeeze_session_diagnostics(
    session: tuple[bool, int | None],
) -> tuple[tuple[str, float | str], ...]:
    session_allowed, hour = session
    return (
        (
            "squeeze_utc_hour",
            "n/a" if hour is None else float(hour),
        ),
        ("squeeze_session_allowed", "yes" if session_allowed else "no"),
    )


def _dual_squeeze_diagnostics(
    reading: DualSqueezeReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("dual_fast_breakout_bps", reading.fast.breakout_bps),
        ("dual_fast_squeeze_ratio", reading.fast.squeeze_ratio),
        ("dual_fast_band_width_bps", reading.fast.band_width_bps),
        ("dual_confirmation_breakout_bps", reading.confirmation.breakout_bps),
        ("dual_confirmation_squeeze_ratio", reading.confirmation.squeeze_ratio),
        ("dual_confirmation_bias", "long" if reading.confirmation.last_price >= reading.confirmation.mean_price else "short"),
        ("dual_confirmation_passed", "yes" if reading.confirmation_passed else "no"),
        ("dual_confirmation_reason", reading.confirmation_reason),
    )


def _range_expansion_trend_diagnostics(
    reading: RangeExpansionTrendReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("range_expansion_baseline_high", reading.baseline_high),
        ("range_expansion_baseline_low", reading.baseline_low),
        ("range_expansion_trigger_start_price", reading.trigger_start_price),
        ("range_expansion_last_price", reading.last_price),
        ("range_expansion_trigger_move_bps", reading.trigger_move_bps),
        ("range_expansion_break_bps", reading.range_break_bps),
        ("range_expansion_baseline_volatility_bps", reading.baseline_volatility_bps),
        ("range_expansion_trigger_volatility_bps", reading.trigger_volatility_bps),
        ("range_expansion_zscore", reading.expansion_zscore),
        ("range_expansion_trend_efficiency", reading.trend_efficiency),
        ("range_expansion_expected_edge_bps", reading.expected_edge_bps),
        ("range_expansion_realized_volatility", reading.realized_volatility),
        ("range_expansion_signal_direction", reading.signal_direction.value),
    )


def _range_expansion_session_diagnostics(
    session: tuple[bool, int | None],
) -> tuple[tuple[str, float | str], ...]:
    session_allowed, hour = session
    return (
        (
            "range_expansion_utc_hour",
            "n/a" if hour is None else float(hour),
        ),
        ("range_expansion_session_allowed", "yes" if session_allowed else "no"),
    )


def _session_breakout_diagnostics(
    reading: SessionBreakoutReading,
) -> tuple[tuple[str, float | str], ...]:
    return _breakout_diagnostics(reading.breakout) + (
        ("realized_volatility_bps", reading.realized_volatility_bps),
        ("utc_hour", "n/a" if reading.utc_hour is None else float(reading.utc_hour)),
        ("session_allowed", "yes" if reading.session_allowed else "no"),
    )


def _session_regime_diagnostics(
    reading: TimeSeriesRegimeReading | None,
) -> tuple[tuple[str, float | str], ...]:
    if reading is None:
        return (("session_regime", "n/a"),)
    return (
        ("session_regime", reading.regime.value),
        ("session_regime_slope_bps", reading.kalman_slope_bps),
        ("session_regime_trend_efficiency", reading.trend_efficiency),
        ("session_regime_volatility_bps", reading.realized_volatility_bps),
        ("session_regime_confidence", reading.trend_confidence),
    )


def _trend_pullback_diagnostics(
    reading: TrendPullbackReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("trend_pullback_anchor_price", reading.anchor_price),
        ("trend_pullback_previous_price", reading.previous_price),
        ("trend_pullback_last_price", reading.last_price),
        ("trend_pullback_trend_move_bps", reading.trend_move_bps),
        ("trend_pullback_pullback_bps", reading.pullback_bps),
        ("trend_pullback_resume_bps", reading.resume_bps),
        ("trend_pullback_expected_edge_bps", reading.expected_edge_bps),
        ("trend_pullback_trend_efficiency", reading.trend_efficiency),
        ("trend_pullback_realized_volatility", reading.realized_volatility),
        ("trend_pullback_signal_direction", reading.signal_direction.value),
    )


def _trend_pullback_session_diagnostics(
    session: tuple[bool, int | None],
) -> tuple[tuple[str, float | str], ...]:
    session_allowed, hour = session
    return (
        (
            "trend_pullback_utc_hour",
            "n/a" if hour is None else float(hour),
        ),
        ("trend_pullback_session_allowed", "yes" if session_allowed else "no"),
    )


def _exhaustion_reversal_diagnostics(
    reading: ExhaustionReversalReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("exhaustion_shock_start_price", reading.shock_start_price),
        ("exhaustion_previous_price", reading.previous_price),
        ("exhaustion_last_price", reading.last_price),
        ("exhaustion_shock_move_bps", reading.shock_move_bps),
        ("exhaustion_reversal_bps", reading.reversal_bps),
        ("exhaustion_shock_zscore", reading.shock_zscore),
        ("exhaustion_shock_efficiency", reading.shock_efficiency),
        ("exhaustion_baseline_volatility_bps", reading.baseline_volatility_bps),
        ("exhaustion_realized_volatility", reading.realized_volatility),
        ("exhaustion_expected_edge_bps", reading.expected_edge_bps),
        ("exhaustion_signal_direction", reading.signal_direction.value),
    )


def _exhaustion_reversal_session_diagnostics(
    session: tuple[bool, int | None],
) -> tuple[tuple[str, float | str], ...]:
    session_allowed, hour = session
    return (
        (
            "exhaustion_utc_hour",
            "n/a" if hour is None else float(hour),
        ),
        ("exhaustion_session_allowed", "yes" if session_allowed else "no"),
    )


def _liquidity_sweep_reversal_diagnostics(
    reading: LiquiditySweepReversalReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("liquidity_sweep_prior_high", reading.prior_high),
        ("liquidity_sweep_prior_low", reading.prior_low),
        ("liquidity_sweep_midpoint_price", reading.midpoint_price),
        ("liquidity_sweep_previous_price", reading.previous_price),
        ("liquidity_sweep_last_price", reading.last_price),
        ("liquidity_sweep_range_width_bps", reading.range_width_bps),
        ("liquidity_sweep_sweep_bps", reading.sweep_bps),
        ("liquidity_sweep_reentry_bps", reading.reentry_bps),
        ("liquidity_sweep_expected_edge_bps", reading.expected_edge_bps),
        ("liquidity_sweep_realized_volatility", reading.realized_volatility),
        ("liquidity_sweep_trend_efficiency", reading.trend_efficiency),
        ("liquidity_sweep_signal_direction", reading.signal_direction.value),
    )


def _liquidity_sweep_reversal_session_diagnostics(
    session: tuple[bool, int | None],
) -> tuple[tuple[str, float | str], ...]:
    session_allowed, hour = session
    return (
        (
            "liquidity_sweep_utc_hour",
            "n/a" if hour is None else float(hour),
        ),
        ("liquidity_sweep_session_allowed", "yes" if session_allowed else "no"),
    )


def _fixing_reversal_diagnostics(
    reading: FixingReversalReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("fixing_anchor_price", reading.anchor_price),
        ("fixing_previous_price", reading.previous_price),
        ("fixing_last_price", reading.last_price),
        ("fixing_pre_fix_move_bps", reading.pre_fix_move_bps),
        ("fixing_confirmation_bps", reading.confirmation_bps),
        ("fixing_pre_fix_efficiency", reading.pre_fix_efficiency),
        ("fixing_realized_volatility_bps", reading.realized_volatility_bps),
        ("fixing_expected_edge_bps", reading.expected_edge_bps),
        ("fixing_signal_direction", reading.signal_direction.value),
    )


def _fixing_reversal_session_diagnostics(
    session: tuple[bool, int | None],
) -> tuple[tuple[str, float | str], ...]:
    session_allowed, hour = session
    return (
        (
            "fixing_utc_hour",
            "n/a" if hour is None else float(hour),
        ),
        ("fixing_session_allowed", "yes" if session_allowed else "no"),
    )


def _reversion_diagnostics(
    reading: MeanReversionReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("baseline_mean", reading.mean_price),
        ("baseline_stdev", reading.stdev_price),
        ("latest_price", reading.last_price),
        ("residual", reading.residual),
        ("zscore", reading.zscore),
        ("deviation_bps", reading.deviation_bps),
        ("trend_strength_bps", reading.trend_strength_bps),
        ("trend_efficiency", reading.trend_efficiency),
        (
            "estimated_half_life",
            "n/a" if reading.estimated_half_life is None else reading.estimated_half_life,
        ),
    )


def _regime_diagnostics(reading: RegimeReading) -> tuple[tuple[str, float | str], ...]:
    return (
        ("selected_regime", reading.selected.value),
        ("candidate_regime", reading.candidate.value),
        ("regime_confidence", reading.confidence),
        ("regime_reason", reading.reason),
        ("regime_momentum_move_bps", reading.momentum_move_bps),
        ("regime_momentum_score", reading.momentum_score),
        ("regime_momentum_efficiency", reading.momentum_efficiency),
        ("regime_reversion_zscore", reading.reversion_zscore),
        ("regime_reversion_trend_bps", reading.reversion_trend_bps),
        ("regime_reversion_efficiency", reading.reversion_efficiency),
        ("regime_spread_bps", reading.spread_bps),
    )


def _kalman_trend_diagnostics(
    reading: KalmanTrendStrategyReading,
) -> tuple[tuple[str, float | str], ...]:
    regime = reading.regime_reading
    return (
        ("kalman_trend_regime", regime.regime.value),
        ("kalman_trend_observations", float(regime.observations)),
        ("kalman_trend_level", regime.kalman_level),
        ("kalman_trend_slope_bps", regime.kalman_slope_bps),
        ("kalman_trend_efficiency", regime.trend_efficiency),
        ("kalman_trend_volatility_bps", regime.realized_volatility_bps),
        ("kalman_trend_confidence", regime.trend_confidence),
        ("kalman_trend_expected_edge_bps", reading.expected_edge_bps),
        ("kalman_trend_direction", reading.signal_direction.value),
        (
            "kalman_trend_utc_hour",
            "n/a" if reading.utc_hour is None else float(reading.utc_hour),
        ),
        ("kalman_trend_session_allowed", "yes" if reading.session_allowed else "no"),
    )


def _quality_trend_diagnostics(
    reading: QualityTrendReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("quality_trend_macd_direction", reading.macd_direction.value),
        ("quality_trend_kalman_direction", reading.kalman_direction.value),
        ("quality_trend_aligned_direction", reading.aligned_direction.value),
        ("quality_trend_macd_confidence", reading.macd_confidence),
        ("quality_trend_kalman_confidence", reading.kalman_confidence),
        ("quality_trend_combined_confidence", reading.combined_confidence),
        ("quality_trend_expected_edge_bps", reading.expected_edge_bps),
    ) + _macd_momentum_diagnostics(reading.macd) + _kalman_trend_diagnostics(
        reading.kalman
    )


def _ml_alpha_diagnostics(reading: MLAlphaReading) -> tuple[tuple[str, float | str], ...]:
    rows: list[tuple[str, float | str]] = [
        ("ml_probability_up", reading.probability_up),
        ("ml_probability_down", reading.probability_down),
        ("ml_score", reading.score),
        ("ml_sample_count", float(reading.sample_count)),
        ("ml_training_accuracy", reading.training_accuracy),
        ("ml_training_signed_return_bps", reading.training_signed_return_bps),
        ("ml_expected_edge_bps", reading.expected_edge_bps),
    ]
    rows.extend((f"ml_feature_{name}", value) for name, value in reading.features)
    return tuple(rows)


def _usd_pressure_diagnostics(
    reading: UsdPressureReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("usd_pressure_bps", reading.pressure_bps),
        ("usd_pressure_component_count", float(reading.component_count)),
        ("usd_pressure_confirming_symbols", float(reading.confirming_symbols)),
        ("usd_pressure_conflicting_symbols", float(reading.conflicting_symbols)),
        (
            "usd_pressure_components",
            ", ".join(
                f"{symbol}:{move_bps:.1f}"
                for symbol, move_bps in reading.components
            ),
        ),
    )


def _relative_strength_diagnostics(
    reading: RelativeStrengthReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("relative_strength_zscore", reading.relative_zscore),
        ("relative_strength_rank", float(reading.target_rank)),
        ("relative_strength_component_count", float(reading.component_count)),
        ("relative_strength_score", reading.target_score),
        ("relative_strength_score_dispersion", reading.score_dispersion),
        ("relative_strength_move_bps", reading.move_bps),
        ("relative_strength_realized_volatility_bps", reading.realized_volatility_bps),
        ("relative_strength_trend_efficiency", reading.trend_efficiency),
        ("relative_strength_strongest_symbol", reading.strongest_symbol),
        ("relative_strength_strongest_score", reading.strongest_score),
        ("relative_strength_weakest_symbol", reading.weakest_symbol),
        ("relative_strength_weakest_score", reading.weakest_score),
        (
            "relative_strength_asset_class_zscore",
            "n/a" if reading.asset_class_zscore is None else reading.asset_class_zscore,
        ),
        (
            "relative_strength_asset_class_rank",
            "n/a" if reading.asset_class_rank is None else float(reading.asset_class_rank),
        ),
        (
            "relative_strength_asset_class_component_count",
            float(reading.asset_class_component_count),
        ),
        (
            "relative_strength_components",
            ", ".join(
                f"{symbol}:{score:.2f}"
                for symbol, score in reading.components
            ),
        ),
        (
            "relative_strength_asset_class_components",
            ", ".join(
                f"{symbol}:{score:.2f}"
                for symbol, score in reading.asset_class_components
            ),
        ),
    )


def _cross_rate_reversion_diagnostics(
    reading: CrossRateReversionReading,
) -> tuple[tuple[str, float | str], ...]:
    return (
        ("cross_rate_target_price", reading.target_price),
        ("cross_rate_synthetic_price", reading.synthetic_price),
        ("cross_rate_deviation_bps", reading.deviation_bps),
        ("cross_rate_mean_deviation_bps", reading.mean_deviation_bps),
        ("cross_rate_stdev_deviation_bps", reading.stdev_deviation_bps),
        ("cross_rate_zscore", reading.zscore),
        ("cross_rate_realized_volatility", reading.realized_volatility),
        ("cross_rate_components", "/".join(reading.component_symbols)),
        ("cross_rate_currency_path", "->".join(reading.currency_path)),
    )


@dataclass(frozen=True)
class _CurrencyConversionStep:
    symbol: str
    from_currency: str
    to_currency: str
    inverted: bool


def _price_snapshot_at_offset(
    closes_by_symbol: Mapping[str, Sequence[float]],
    offset_from_end: int,
) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for raw_symbol, prices in closes_by_symbol.items():
        if len(prices) < offset_from_end:
            continue
        price = prices[-offset_from_end]
        if price <= 0 or not isfinite(price):
            continue
        snapshot[instrument_for(raw_symbol).symbol] = price
    return snapshot


def _find_fx_conversion_path(
    prices_by_symbol: Mapping[str, float],
    *,
    target_symbol: str,
) -> tuple[_CurrencyConversionStep, ...] | None:
    target = instrument_for(target_symbol)
    if target.asset_class != AssetClass.FOREX:
        return None

    adjacency: dict[str, list[_CurrencyConversionStep]] = {}
    for raw_symbol, price in sorted(prices_by_symbol.items()):
        if price <= 0 or not isfinite(price):
            continue
        instrument = instrument_for(raw_symbol)
        if instrument.asset_class != AssetClass.FOREX:
            continue
        if instrument.symbol == target.symbol:
            continue
        forward = _CurrencyConversionStep(
            symbol=instrument.symbol,
            from_currency=instrument.base_currency,
            to_currency=instrument.quote_currency,
            inverted=False,
        )
        reverse = _CurrencyConversionStep(
            symbol=instrument.symbol,
            from_currency=instrument.quote_currency,
            to_currency=instrument.base_currency,
            inverted=True,
        )
        adjacency.setdefault(forward.from_currency, []).append(forward)
        adjacency.setdefault(reverse.from_currency, []).append(reverse)

    start = target.base_currency
    end = target.quote_currency
    queue: list[tuple[str, tuple[_CurrencyConversionStep, ...]]] = [(start, ())]
    visited = {start}
    while queue:
        currency, path = queue.pop(0)
        if currency == end:
            return path
        for step in adjacency.get(currency, []):
            if step.to_currency in visited:
                continue
            visited.add(step.to_currency)
            queue.append((step.to_currency, path + (step,)))
    return None


def _synthetic_price_from_path(
    path: tuple[_CurrencyConversionStep, ...],
    prices_by_symbol: Mapping[str, float],
) -> float | None:
    synthetic_price = 1.0
    for step in path:
        price = prices_by_symbol.get(step.symbol)
        if price is None or price <= 0 or not isfinite(price):
            return None
        synthetic_price *= (1.0 / price) if step.inverted else price
    return synthetic_price


def _train_and_score_ml_alpha(
    *,
    prices: Sequence[float],
    lookback: int,
    train_window: int,
    min_train_samples: int,
    learning_rate: float,
    epochs: int,
    l2: float,
    label_threshold_bps: float,
    min_edge_bps: float,
) -> MLAlphaReading | None:
    price_list = list(prices)
    if len(price_list) < lookback + 2:
        return None
    for price in price_list:
        if price <= 0 or not isfinite(price):
            raise ValueError("prices must be positive finite numbers")

    samples: list[tuple[tuple[float, ...], float, float]] = []
    for end_index in range(len(price_list) - 2, lookback - 2, -1):
        feature_window = price_list[end_index - lookback + 1 : end_index + 1]
        features = _ml_feature_values(feature_window, lookback)
        forward_return_bps = log(price_list[end_index + 1] / price_list[end_index]) * 10_000
        if abs(forward_return_bps) <= label_threshold_bps:
            continue
        label = 1.0 if forward_return_bps > 0 else 0.0
        samples.append((features, label, forward_return_bps))
        if len(samples) >= train_window:
            break

    samples.reverse()
    if len(samples) < min_train_samples:
        return None

    weights = _fit_logistic_classifier(
        samples=samples,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    latest_features = _ml_feature_values(price_list[-lookback:], lookback)
    probability_up = _predict_probability(weights, latest_features)
    probability_down = 1.0 - probability_up
    score = (probability_up - 0.5) * 2.0
    training_accuracy = _training_accuracy(weights, samples)
    training_signed_return_bps = _training_signed_return_bps(weights, samples)
    mean_abs_forward_bps = sum(abs(sample[2]) for sample in samples) / len(samples)
    expected_edge_bps = max(mean_abs_forward_bps, min_edge_bps) * abs(score)

    return MLAlphaReading(
        probability_up=probability_up,
        probability_down=probability_down,
        score=score,
        sample_count=len(samples),
        training_accuracy=training_accuracy,
        training_signed_return_bps=training_signed_return_bps,
        expected_edge_bps=expected_edge_bps,
        features=tuple(zip(_ML_FEATURE_NAMES, latest_features, strict=True)),
    )


_ML_FEATURE_NAMES = (
    "cumulative_return",
    "last_return",
    "realized_volatility",
    "trend_efficiency",
    "zscore",
    "channel_position",
)


def _ml_feature_values(prices: Sequence[float], lookback: int) -> tuple[float, ...]:
    recent_prices = _recent_valid_prices(prices, lookback)
    if recent_prices is None:
        raise ValueError("not enough prices for ML features")

    log_returns = _log_returns(recent_prices)
    cumulative_return_bps = sum(log_returns) * 10_000
    last_return_bps = log_returns[-1] * 10_000 if log_returns else 0.0
    path_return = sum(abs(value) for value in log_returns)
    trend_efficiency = (
        abs(sum(log_returns)) / path_return
        if path_return > 0
        else 0.0
    )
    realized_volatility_bps = _population_stdev(log_returns) * 10_000
    baseline = recent_prices[:-1]
    latest = recent_prices[-1]
    baseline_mean = sum(baseline) / len(baseline)
    baseline_stdev = _population_stdev(baseline)
    zscore = 0.0 if baseline_stdev == 0 else (latest - baseline_mean) / baseline_stdev
    high = max(baseline)
    low = min(baseline)
    channel_position = 0.0 if high == low else ((latest - low) / (high - low) * 2.0) - 1.0

    return (
        cumulative_return_bps / 100.0,
        last_return_bps / 100.0,
        realized_volatility_bps / 100.0,
        trend_efficiency,
        _clamp(zscore / 3.0, -2.0, 2.0),
        _clamp(channel_position, -2.0, 2.0),
    )


def _fit_logistic_classifier(
    *,
    samples: Sequence[tuple[tuple[float, ...], float, float]],
    learning_rate: float,
    epochs: int,
    l2: float,
) -> tuple[float, ...]:
    feature_count = len(samples[0][0])
    weights = [0.0] * (feature_count + 1)
    for _ in range(epochs):
        for features, label, _ in samples:
            probability = _predict_probability(tuple(weights), features)
            error = probability - label
            weights[0] -= learning_rate * error
            for index, value in enumerate(features, start=1):
                gradient = (error * value) + (l2 * weights[index])
                weights[index] -= learning_rate * gradient
    return tuple(weights)


def _predict_probability(weights: tuple[float, ...], features: tuple[float, ...]) -> float:
    score = weights[0] + sum(
        weight * value
        for weight, value in zip(weights[1:], features, strict=True)
    )
    return _sigmoid(score)


def _training_accuracy(
    weights: tuple[float, ...],
    samples: Sequence[tuple[tuple[float, ...], float, float]],
) -> float:
    correct = 0
    for features, label, _ in samples:
        prediction = 1.0 if _predict_probability(weights, features) >= 0.5 else 0.0
        if prediction == label:
            correct += 1
    return correct / len(samples)


def _training_signed_return_bps(
    weights: tuple[float, ...],
    samples: Sequence[tuple[tuple[float, ...], float, float]],
) -> float:
    signed_return_bps = 0.0
    for features, _, forward_return_bps in samples:
        direction = 1.0 if _predict_probability(weights, features) >= 0.5 else -1.0
        signed_return_bps += direction * forward_return_bps
    return signed_return_bps


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _flat_signal(
    strategy_name: str,
    symbol: str,
    weight: float,
    horizon: SignalHorizon,
    reason: str,
    cost_bps: float,
    diagnostics: tuple[tuple[str, float | str], ...] = (),
    *,
    expected_edge_bps: float = 0.0,
) -> StrategySignal:
    return StrategySignal(
        strategy_name=strategy_name,
        symbol=symbol,
        direction=SignalDirection.FLAT,
        confidence=0.0,
        expected_edge_bps=expected_edge_bps,
        cost_bps=cost_bps,
        weight=weight,
        horizon=horizon,
        reason=reason,
        diagnostics=diagnostics,
    )


def _estimated_round_trip_cost_bps(
    *,
    quote: QuoteSnapshot | None,
    slippage_bps: float,
    fee_bps: float,
) -> float:
    spread_bps = quote.spread_bps if quote is not None else 0.0
    return spread_bps + (2.0 * slippage_bps) + fee_bps


def _has_signal_conflict(signals: tuple[StrategySignal, ...]) -> bool:
    directions = {signal.direction for signal in signals}
    return SignalDirection.LONG in directions and SignalDirection.SHORT in directions


def _router_attribution(
    active_signals: tuple[StrategySignal, ...],
    combined_score: float,
) -> SignalAttribution:
    target_direction = (
        SignalDirection.LONG
        if combined_score > EPSILON_NOTIONAL
        else SignalDirection.SHORT
        if combined_score < -EPSILON_NOTIONAL
        else SignalDirection.FLAT
    )
    if target_direction == SignalDirection.FLAT:
        return SignalAttribution(
            primary_signal="none",
            supporting_signals=(),
            conflicting_signals=tuple(
                signal.strategy_name
                for signal in active_signals
                if signal.direction != SignalDirection.FLAT
            ),
        )

    supporting = tuple(
        signal for signal in active_signals if signal.direction == target_direction
    )
    conflicting = tuple(
        signal
        for signal in active_signals
        if signal.direction not in {target_direction, SignalDirection.FLAT}
    )
    if not supporting:
        return SignalAttribution(
            primary_signal="none",
            supporting_signals=(),
            conflicting_signals=tuple(signal.strategy_name for signal in conflicting),
        )

    primary = max(supporting, key=lambda signal: abs(signal.signed_score))
    return SignalAttribution(
        primary_signal=primary.strategy_name,
        supporting_signals=tuple(signal.strategy_name for signal in supporting),
        conflicting_signals=tuple(signal.strategy_name for signal in conflicting),
    )


def _attribution_kwargs(attribution: SignalAttribution) -> dict[str, str | tuple[str, ...]]:
    return {
        "primary_signal": attribution.primary_signal,
        "supporting_signals": attribution.supporting_signals,
        "conflicting_signals": attribution.conflicting_signals,
    }


def _router_diagnostics(
    *,
    signals: tuple[StrategySignal, ...],
    raw_score: float,
    combined_score: float,
    has_conflict: bool,
    holding_period: int,
) -> tuple[tuple[str, float | str], ...]:
    rows: list[tuple[str, float | str]] = [
        ("router_raw_score", raw_score),
        ("router_combined_score", combined_score),
        ("router_conflict", str(has_conflict)),
        ("router_holding_period", float(holding_period)),
    ]
    for signal in signals:
        prefix = f"signal_{signal.strategy_name}"
        rows.extend(
            [
                (f"{prefix}_direction", signal.direction.value),
                (f"{prefix}_confidence", signal.confidence),
                (f"{prefix}_weight", signal.weight),
                (f"{prefix}_expected_edge_bps", signal.expected_edge_bps),
                (f"{prefix}_cost_bps", signal.cost_bps),
                (f"{prefix}_signed_score", signal.signed_score),
                (f"{prefix}_reason", signal.reason),
            ]
        )
    return tuple(rows)


def _router_reason(
    prefix: str,
    signals: tuple[StrategySignal, ...],
    combined_score: float,
    has_conflict: bool,
) -> str:
    signal_text = "; ".join(
        (
            f"{signal.strategy_name}={signal.direction.value} "
            f"conf={signal.confidence:.2f} edge={signal.expected_edge_bps:.1f} "
            f"cost={signal.cost_bps:.1f} reason={signal.reason}"
        )
        for signal in signals
    )
    conflict_text = "; conflict penalty applied" if has_conflict else ""
    return f"{prefix}: score={combined_score:.2f}{conflict_text}; {signal_text}"


def _champion_ensemble_diagnostics(
    *,
    signals: tuple[StrategySignal, ...],
    raw_score: float,
    combined_score: float,
    has_conflict: bool,
    holding_period: int,
) -> tuple[tuple[str, float | str], ...]:
    rows: list[tuple[str, float | str]] = [
        ("champion_raw_score", raw_score),
        ("champion_combined_score", combined_score),
        ("champion_conflict", str(has_conflict)),
        ("champion_holding_period", float(holding_period)),
    ]
    for signal in signals:
        prefix = f"champion_signal_{signal.strategy_name}"
        rows.extend(
            [
                (f"{prefix}_direction", signal.direction.value),
                (f"{prefix}_confidence", signal.confidence),
                (f"{prefix}_weight", signal.weight),
                (f"{prefix}_expected_edge_bps", signal.expected_edge_bps),
                (f"{prefix}_cost_bps", signal.cost_bps),
                (f"{prefix}_signed_score", signal.signed_score),
                (f"{prefix}_source_target_notional_usd", _signal_source_target_notional(signal)),
                (f"{prefix}_reason", signal.reason),
            ]
        )
    return tuple(rows)


def _champion_ensemble_reason(
    prefix: str,
    signals: tuple[StrategySignal, ...],
    combined_score: float,
    has_conflict: bool,
) -> str:
    signal_text = "; ".join(
        (
            f"{signal.strategy_name}={signal.direction.value} "
            f"w={signal.weight:.2f} score={signal.signed_score:.2f} "
            f"reason={signal.reason}"
        )
        for signal in signals
    )
    conflict_text = "; conflict penalty applied" if has_conflict else ""
    return f"{prefix}: score={combined_score:.2f}{conflict_text}; {signal_text}"


def _decision_expected_edge_bps(decision: StrategyDecision) -> float:
    candidates: list[float] = []
    for key, value in decision.diagnostics:
        if not isinstance(value, float):
            continue
        lowered = key.lower()
        if lowered.endswith("expected_edge_bps") or lowered.endswith("edge_bps"):
            candidates.append(abs(value))
        elif lowered.endswith("breakout_bps") or lowered.endswith("move_bps"):
            candidates.append(abs(value))
    return max(candidates) if candidates else 0.0


def _signal_source_target_notional(signal: StrategySignal) -> float:
    suffix = "_source_target_notional_usd"
    for key, value in signal.diagnostics:
        if key.endswith(suffix) and isinstance(value, float):
            return value
    return 0.0


def _passes_cost_filter(
    *,
    edge_bps: float,
    quote: QuoteSnapshot | None,
    slippage_bps: float,
    fee_bps: float,
    cost_buffer: float,
    max_spread_bps: float | None,
) -> tuple[bool, str]:
    spread_bps = quote.spread_bps if quote is not None else 0.0
    if max_spread_bps is not None and spread_bps > max_spread_bps:
        return (
            False,
            f"spread {spread_bps:.2f} bps above strategy limit {max_spread_bps:.2f} bps",
        )
    round_trip_cost_bps = spread_bps + (2.0 * slippage_bps) + fee_bps
    required_edge_bps = round_trip_cost_bps * cost_buffer
    if edge_bps <= required_edge_bps:
        return (
            False,
            (
                f"edge {edge_bps:.1f} bps not above estimated cost "
                f"{required_edge_bps:.1f} bps"
            ),
        )
    return True, f"edge {edge_bps:.1f} bps cleared estimated costs"


def _sized_notional(
    *,
    position_sizing: str,
    base_notional: float,
    target_volatility: float,
    realized_volatility: float,
    volatility_floor: float,
    signal_confidence: float,
    max_notional: float | None,
    min_trade_notional: float,
) -> float:
    if position_sizing == "fixed":
        sized = base_notional
    else:
        sized = (
            base_notional
            * target_volatility
            / max(realized_volatility, volatility_floor)
            * signal_confidence
        )

    cap = max_notional if max_notional is not None else base_notional
    sized = min(sized, cap)
    if sized < min_trade_notional:
        return 0.0
    return sized


def _bounded_confidence(signal_strength: float, entry_threshold: float) -> float:
    if entry_threshold <= 0:
        return 1.0
    return min(max(signal_strength / entry_threshold, 0.0), 1.0)


def _reversion_direction(zscore: float) -> int:
    return -1 if zscore > 0 else 1


def _separation_direction(separation_bps: float) -> int:
    if separation_bps > EPSILON_NOTIONAL:
        return 1
    if separation_bps < -EPSILON_NOTIONAL:
        return -1
    return 0


def _signal_direction_sign(direction: SignalDirection) -> int:
    if direction == SignalDirection.LONG:
        return 1
    if direction == SignalDirection.SHORT:
        return -1
    return 0


def _signed_direction_to_signal(direction: int) -> SignalDirection:
    if direction > 0:
        return SignalDirection.LONG
    if direction < 0:
        return SignalDirection.SHORT
    return SignalDirection.FLAT


def _signed_threshold_direction(value: float, threshold: float) -> int:
    if value >= threshold:
        return 1
    if value <= -threshold:
        return -1
    return 0


def _target_usd_pressure_direction(symbol: str, target_direction: int) -> int:
    if target_direction == 0:
        return 0
    instrument = instrument_for(symbol)
    if instrument.base_currency == "USD":
        return target_direction
    if instrument.quote_currency == "USD":
        return -target_direction
    return 0


def _notional_direction(notional_usd: float) -> int:
    if notional_usd > EPSILON_NOTIONAL:
        return 1
    if notional_usd < -EPSILON_NOTIONAL:
        return -1
    return 0


def _validate_symbol(symbol: str) -> None:
    if not symbol or not symbol.strip():
        raise ValueError("symbol is required")


def _validate_position_sizing(value: str) -> None:
    if value not in {"fixed", "volatility"}:
        raise ValueError("position_sizing must be 'fixed' or 'volatility'")


def _validate_volatility_squeeze_shape(
    *,
    lookback: int,
    squeeze_window: int,
    label: str,
) -> None:
    if lookback < 6:
        raise ValueError(f"{label} lookback must be at least 6 prices")
    if squeeze_window < 2:
        raise ValueError(f"{label} squeeze_window must be at least 2 returns")
    if lookback < squeeze_window + 4:
        raise ValueError(
            f"{label} lookback must leave at least two prior returns before squeeze_window"
        )


def _normalize_optional_hours(instance: object, field_name: str) -> None:
    raw_hours = getattr(instance, field_name)
    if raw_hours is None:
        return
    normalized_hours = tuple(int(hour) for hour in raw_hours)
    if any(hour < 0 or hour > 23 for hour in normalized_hours):
        raise ValueError(f"{field_name} must contain hours between 0 and 23")
    object.__setattr__(instance, field_name, normalized_hours)


def _validate_positive_finite(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be a positive finite number")


def _validate_non_negative_finite(name: str, value: float) -> None:
    if value < 0 or not isfinite(value):
        raise ValueError(f"{name} must be a non-negative finite number")
