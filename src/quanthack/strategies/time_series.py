from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import exp, isfinite, log, sqrt
from pathlib import Path

from quanthack.market.market_data import PriceHistory


class TimeSeriesRegime(StrEnum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    CHOP = "CHOP"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass(frozen=True)
class KalmanTrendConfig:
    lookback: int = 80
    process_noise: float = 1e-6
    observation_noise: float = 1e-4
    min_abs_slope_bps: float = 0.75
    min_trend_efficiency: float = 0.25
    max_realized_volatility_bps: float = 120.0

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5")
        _validate_positive("process_noise", self.process_noise)
        _validate_positive("observation_noise", self.observation_noise)
        _validate_non_negative("min_abs_slope_bps", self.min_abs_slope_bps)
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        _validate_positive("max_realized_volatility_bps", self.max_realized_volatility_bps)


@dataclass(frozen=True)
class TimeSeriesRegimeReading:
    symbol: str
    observations: int
    latest_close: float
    kalman_level: float
    kalman_slope_bps: float
    trend_efficiency: float
    realized_volatility_bps: float
    trend_confidence: float
    regime: TimeSeriesRegime


def read_kalman_regime(
    prices: Sequence[float],
    *,
    symbol: str,
    config: KalmanTrendConfig | None = None,
) -> TimeSeriesRegimeReading:
    cfg = config or KalmanTrendConfig()
    if len(prices) < cfg.lookback:
        raise ValueError(f"not enough prices for Kalman regime reading on {symbol}")
    window = tuple(float(price) for price in prices[-cfg.lookback :])
    if any(price <= 0 or not isfinite(price) for price in window):
        raise ValueError("prices must be positive finite numbers")

    log_prices = tuple(log(price) for price in window)
    filtered = _kalman_filter_levels(
        log_prices,
        process_noise=cfg.process_noise,
        observation_noise=cfg.observation_noise,
    )
    slope_bps = (exp(filtered[-1] - filtered[-2]) - 1.0) * 10_000
    returns = [current - previous for previous, current in zip(log_prices, log_prices[1:])]
    realized_vol_bps = _root_mean_square(returns) * 10_000
    efficiency = _trend_efficiency(log_prices)
    confidence = min(
        1.0,
        abs(slope_bps) / max(realized_vol_bps, cfg.min_abs_slope_bps, 1e-12),
    )
    regime = _classify_regime(
        slope_bps=slope_bps,
        realized_volatility_bps=realized_vol_bps,
        trend_efficiency=efficiency,
        config=cfg,
    )
    return TimeSeriesRegimeReading(
        symbol=symbol,
        observations=len(window),
        latest_close=window[-1],
        kalman_level=exp(filtered[-1]),
        kalman_slope_bps=slope_bps,
        trend_efficiency=efficiency,
        realized_volatility_bps=realized_vol_bps,
        trend_confidence=confidence,
        regime=regime,
    )


def evaluate_time_series_regimes(
    *,
    prices: PriceHistory,
    symbols: tuple[str, ...] | list[str] | None = None,
    config: KalmanTrendConfig | None = None,
) -> tuple[TimeSeriesRegimeReading, ...]:
    cfg = config or KalmanTrendConfig()
    selected_symbols = tuple(symbols or prices.symbols())
    return tuple(
        read_kalman_regime(
            prices.close_prices(symbol=symbol),
            symbol=symbol,
            config=cfg,
        )
        for symbol in selected_symbols
    )


def write_time_series_regime_csv(
    readings: tuple[TimeSeriesRegimeReading, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "observations",
                "latest_close",
                "kalman_level",
                "kalman_slope_bps",
                "trend_efficiency",
                "realized_volatility_bps",
                "trend_confidence",
                "regime",
            ],
        )
        writer.writeheader()
        for reading in readings:
            writer.writerow(
                {
                    "symbol": reading.symbol,
                    "observations": reading.observations,
                    "latest_close": reading.latest_close,
                    "kalman_level": reading.kalman_level,
                    "kalman_slope_bps": reading.kalman_slope_bps,
                    "trend_efficiency": reading.trend_efficiency,
                    "realized_volatility_bps": reading.realized_volatility_bps,
                    "trend_confidence": reading.trend_confidence,
                    "regime": reading.regime.value,
                }
            )


def _kalman_filter_levels(
    values: Sequence[float],
    *,
    process_noise: float,
    observation_noise: float,
) -> tuple[float, ...]:
    level = values[0]
    variance = observation_noise
    filtered = [level]
    for observation in values[1:]:
        predicted_level = level
        predicted_variance = variance + process_noise
        gain = predicted_variance / (predicted_variance + observation_noise)
        level = predicted_level + gain * (observation - predicted_level)
        variance = (1.0 - gain) * predicted_variance
        filtered.append(level)
    return tuple(filtered)


def _trend_efficiency(log_prices: Sequence[float]) -> float:
    net_move = abs(log_prices[-1] - log_prices[0])
    path_move = sum(
        abs(current - previous)
        for previous, current in zip(log_prices, log_prices[1:])
    )
    if path_move == 0:
        return 0.0
    return min(1.0, net_move / path_move)


def _root_mean_square(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(value * value for value in values) / len(values))


def _classify_regime(
    *,
    slope_bps: float,
    realized_volatility_bps: float,
    trend_efficiency: float,
    config: KalmanTrendConfig,
) -> TimeSeriesRegime:
    if realized_volatility_bps > config.max_realized_volatility_bps:
        return TimeSeriesRegime.HIGH_VOLATILITY
    if (
        abs(slope_bps) >= config.min_abs_slope_bps
        and trend_efficiency >= config.min_trend_efficiency
    ):
        return TimeSeriesRegime.TREND_UP if slope_bps > 0 else TimeSeriesRegime.TREND_DOWN
    return TimeSeriesRegime.CHOP


def _validate_positive(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be positive and finite")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0 or not isfinite(value):
        raise ValueError(f"{name} must be non-negative and finite")

