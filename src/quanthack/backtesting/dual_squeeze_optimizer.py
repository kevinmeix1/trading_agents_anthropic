from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.portfolio_strategy_compare import (
    PortfolioStrategyComparisonRow,
    compare_portfolio_strategies,
)
from quanthack.backtesting.portfolio_walk_forward import (
    PortfolioWalkForwardSummary,
    run_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class DualSqueezeParameterSet:
    label: str
    lookback: int
    squeeze_window: int
    max_squeeze_ratio: float
    breakout_buffer_bps: float
    band_stdev_multiplier: float
    confirmation_lookback: int
    confirmation_squeeze_window: int
    confirmation_max_squeeze_ratio: float
    confirmation_band_stdev_multiplier: float = 2.0
    confirmation_mode: str = "squeeze_bias"
    min_prior_volatility_bps: float = 0.5
    min_band_width_bps: float = 1.0
    max_holding_period: int = 12
    forex_allowed_utc_hours: tuple[int, ...] | None = None
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("dual squeeze parameter label is required")
        _validate_squeeze_shape(
            lookback=self.lookback,
            squeeze_window=self.squeeze_window,
            label="dual squeeze fast",
        )
        _validate_squeeze_shape(
            lookback=self.confirmation_lookback,
            squeeze_window=self.confirmation_squeeze_window,
            label="dual squeeze confirmation",
        )
        for name, value in (
            ("max_squeeze_ratio", self.max_squeeze_ratio),
            ("band_stdev_multiplier", self.band_stdev_multiplier),
            ("confirmation_max_squeeze_ratio", self.confirmation_max_squeeze_ratio),
            (
                "confirmation_band_stdev_multiplier",
                self.confirmation_band_stdev_multiplier,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("breakout_buffer_bps", self.breakout_buffer_bps),
            ("min_prior_volatility_bps", self.min_prior_volatility_bps),
            ("min_band_width_bps", self.min_band_width_bps),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.confirmation_mode not in {
            "bias",
            "breakout",
            "not_opposite",
            "squeeze_bias",
        }:
            raise ValueError(
                "confirmation_mode must be bias, breakout, not_opposite, or squeeze_bias"
            )
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")


def _validate_squeeze_shape(*, lookback: int, squeeze_window: int, label: str) -> None:
    if lookback < 6:
        raise ValueError(f"{label} lookback must be at least 6")
    if squeeze_window < 2:
        raise ValueError(f"{label} squeeze_window must be at least 2")
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


def _hours_text(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return ""
    return " ".join(str(hour) for hour in hours)


DEFAULT_DUAL_SQUEEZE_PARAMETER_SETS: tuple[DualSqueezeParameterSet, ...] = (
    DualSqueezeParameterSet(
        label="current_l16_w5_r0_55_b2_5_c32_cw10_cr0_70",
        lookback=16,
        squeeze_window=5,
        max_squeeze_ratio=0.55,
        breakout_buffer_bps=2.5,
        band_stdev_multiplier=1.8,
        confirmation_lookback=32,
        confirmation_squeeze_window=10,
        confirmation_max_squeeze_ratio=0.70,
    ),
    DualSqueezeParameterSet(
        label="strict_l16_w5_r0_50_b3_c32_cw10_cr0_65",
        lookback=16,
        squeeze_window=5,
        max_squeeze_ratio=0.50,
        breakout_buffer_bps=3.0,
        band_stdev_multiplier=1.8,
        confirmation_lookback=32,
        confirmation_squeeze_window=10,
        confirmation_max_squeeze_ratio=0.65,
    ),
    DualSqueezeParameterSet(
        label="fast_confirm_l14_w4_r0_60_b2_5_c24_cw8_cr0_70",
        lookback=14,
        squeeze_window=4,
        max_squeeze_ratio=0.60,
        breakout_buffer_bps=2.5,
        band_stdev_multiplier=1.8,
        confirmation_lookback=24,
        confirmation_squeeze_window=8,
        confirmation_max_squeeze_ratio=0.70,
    ),
    DualSqueezeParameterSet(
        label="slow_confirm_l20_w6_r0_55_b2_5_c40_cw12_cr0_75",
        lookback=20,
        squeeze_window=6,
        max_squeeze_ratio=0.55,
        breakout_buffer_bps=2.5,
        band_stdev_multiplier=2.0,
        confirmation_lookback=40,
        confirmation_squeeze_window=12,
        confirmation_max_squeeze_ratio=0.75,
    ),
)


@dataclass(frozen=True)
class DualSqueezeOptimizationCandidate:
    parameters: DualSqueezeParameterSet
    comparison_row: PortfolioStrategyComparisonRow
    walk_forward_summary: PortfolioWalkForwardSummary | None = None

    @property
    def rank_key(self) -> tuple[float, int, float, float, float, float]:
        metrics = self.comparison_row.competition_metrics
        if self.walk_forward_summary is not None:
            summary = self.walk_forward_summary
            return (
                100.0 if summary.eligible else 0.0,
                int(self.comparison_row.risk_discipline.score),
                summary.stable_fold_fraction,
                summary.median_test_return_pct,
                summary.median_test_sharpe_15m,
                -summary.worst_test_drawdown_pct,
            )
        return (
            self.comparison_row.proxy_score,
            self.comparison_row.risk_discipline.score,
            metrics.return_pct,
            metrics.sharpe_15m,
            -metrics.max_drawdown_pct,
            -float(metrics.trade_count),
        )


@dataclass(frozen=True)
class DualSqueezeOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[DualSqueezeOptimizationCandidate, ...]

    @property
    def best(self) -> DualSqueezeOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_dual_squeeze_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        DualSqueezeParameterSet, ...
    ] = DEFAULT_DUAL_SQUEEZE_PARAMETER_SETS,
    include_walk_forward: bool = False,
    walk_forward_train_size: int = 480,
    walk_forward_test_size: int = 240,
    walk_forward_step_size: int = 240,
    walk_forward_max_baskets: int = 30,
) -> DualSqueezeOptimizationResult:
    if not parameter_sets:
        raise ValueError("dual squeeze optimizer needs at least one parameter set")

    candidates: list[DualSqueezeOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("dual_squeeze",),
            symbols=symbols,
        )
        if comparison.best is None:
            continue
        selected_symbols = comparison.symbols

        walk_forward_summary: PortfolioWalkForwardSummary | None = None
        if include_walk_forward:
            walk_forward = run_portfolio_walk_forward(
                config=candidate_config,
                prices=prices,
                quotes=quotes,
                strategy_names=("dual_squeeze",),
                max_baskets=walk_forward_max_baskets,
                train_size=walk_forward_train_size,
                test_size=walk_forward_test_size,
                step_size=walk_forward_step_size,
                min_test_fills=1,
                min_stable_fold_fraction=0.50,
            )
            walk_forward_summary = walk_forward.summary

        candidates.append(
            DualSqueezeOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward_summary=walk_forward_summary,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return DualSqueezeOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_dual_squeeze_optimization_csv(
    result: DualSqueezeOptimizationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "label",
                "symbols",
                "lookback",
                "squeeze_window",
                "max_squeeze_ratio",
                "breakout_buffer_bps",
                "band_stdev_multiplier",
                "confirmation_lookback",
                "confirmation_squeeze_window",
                "confirmation_max_squeeze_ratio",
                "confirmation_band_stdev_multiplier",
                "confirmation_mode",
                "min_prior_volatility_bps",
                "min_band_width_bps",
                "max_holding_period",
                "forex_allowed_utc_hours",
                "metal_allowed_utc_hours",
                "crypto_allowed_utc_hours",
                "proxy_score",
                "final_equity",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "trade_count",
                "fills",
                "turnover_notional",
                "total_pnl_usd",
                "walk_forward_eligible",
                "walk_forward_stable_fold_fraction",
                "walk_forward_median_test_return_pct",
                "walk_forward_median_test_sharpe_15m",
                "walk_forward_worst_test_drawdown_pct",
                "walk_forward_total_test_fills",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            parameters = candidate.parameters
            row = candidate.comparison_row
            metrics = row.competition_metrics
            summary = candidate.walk_forward_summary
            writer.writerow(
                {
                    "rank": rank,
                    "label": parameters.label,
                    "symbols": " ".join(result.symbols),
                    "lookback": parameters.lookback,
                    "squeeze_window": parameters.squeeze_window,
                    "max_squeeze_ratio": parameters.max_squeeze_ratio,
                    "breakout_buffer_bps": parameters.breakout_buffer_bps,
                    "band_stdev_multiplier": parameters.band_stdev_multiplier,
                    "confirmation_lookback": parameters.confirmation_lookback,
                    "confirmation_squeeze_window": (
                        parameters.confirmation_squeeze_window
                    ),
                    "confirmation_max_squeeze_ratio": (
                        parameters.confirmation_max_squeeze_ratio
                    ),
                    "confirmation_band_stdev_multiplier": (
                        parameters.confirmation_band_stdev_multiplier
                    ),
                    "confirmation_mode": parameters.confirmation_mode,
                    "min_prior_volatility_bps": parameters.min_prior_volatility_bps,
                    "min_band_width_bps": parameters.min_band_width_bps,
                    "max_holding_period": parameters.max_holding_period,
                    "forex_allowed_utc_hours": _hours_text(
                        parameters.forex_allowed_utc_hours
                    ),
                    "metal_allowed_utc_hours": _hours_text(
                        parameters.metal_allowed_utc_hours
                    ),
                    "crypto_allowed_utc_hours": _hours_text(
                        parameters.crypto_allowed_utc_hours
                    ),
                    "proxy_score": row.proxy_score,
                    "final_equity": metrics.final_equity,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": row.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "fills": len(row.result.fills),
                    "turnover_notional": row.result.metrics.turnover_notional,
                    "total_pnl_usd": row.result.total_pnl_usd,
                    "walk_forward_eligible": (
                        "" if summary is None else summary.eligible
                    ),
                    "walk_forward_stable_fold_fraction": (
                        "" if summary is None else summary.stable_fold_fraction
                    ),
                    "walk_forward_median_test_return_pct": (
                        "" if summary is None else summary.median_test_return_pct
                    ),
                    "walk_forward_median_test_sharpe_15m": (
                        "" if summary is None else summary.median_test_sharpe_15m
                    ),
                    "walk_forward_worst_test_drawdown_pct": (
                        "" if summary is None else summary.worst_test_drawdown_pct
                    ),
                    "walk_forward_total_test_fills": (
                        "" if summary is None else summary.total_test_fills
                    ),
                }
            )


def _config_with_parameters(
    config: AppConfig,
    parameters: DualSqueezeParameterSet,
) -> AppConfig:
    dual_squeeze = replace(
        config.dual_squeeze,
        lookback=parameters.lookback,
        squeeze_window=parameters.squeeze_window,
        max_squeeze_ratio=parameters.max_squeeze_ratio,
        breakout_buffer_bps=parameters.breakout_buffer_bps,
        band_stdev_multiplier=parameters.band_stdev_multiplier,
        confirmation_lookback=parameters.confirmation_lookback,
        confirmation_squeeze_window=parameters.confirmation_squeeze_window,
        confirmation_max_squeeze_ratio=parameters.confirmation_max_squeeze_ratio,
        confirmation_band_stdev_multiplier=(
            parameters.confirmation_band_stdev_multiplier
        ),
        confirmation_mode=parameters.confirmation_mode,
        min_prior_volatility_bps=parameters.min_prior_volatility_bps,
        min_band_width_bps=parameters.min_band_width_bps,
        max_holding_period=parameters.max_holding_period,
        forex_allowed_utc_hours=(
            parameters.forex_allowed_utc_hours
            if parameters.forex_allowed_utc_hours is not None
            else config.dual_squeeze.forex_allowed_utc_hours
        ),
        metal_allowed_utc_hours=(
            parameters.metal_allowed_utc_hours
            if parameters.metal_allowed_utc_hours is not None
            else config.dual_squeeze.metal_allowed_utc_hours
        ),
        crypto_allowed_utc_hours=(
            parameters.crypto_allowed_utc_hours
            if parameters.crypto_allowed_utc_hours is not None
            else config.dual_squeeze.crypto_allowed_utc_hours
        ),
    )
    return replace(config, dual_squeeze=dual_squeeze)
