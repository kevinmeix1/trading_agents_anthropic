from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.backtesting.portfolio_strategy_compare import (
    PortfolioStrategyComparisonRow,
    compare_portfolio_strategies,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class MacdMomentumParameterSet:
    label: str
    fast_window: int
    slow_window: int
    signal_window: int
    min_histogram_bps: float
    min_macd_bps: float
    min_trend_efficiency: float
    max_holding_period: int
    allowed_utc_hours: tuple[int, ...] | None = None
    min_histogram_slope_bps: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("MACD momentum parameter label is required")
        if self.fast_window < 2:
            raise ValueError("fast_window must be at least 2")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        if self.signal_window < 2:
            raise ValueError("signal_window must be at least 2")
        if self.min_histogram_bps <= 0:
            raise ValueError("min_histogram_bps must be positive")
        if self.min_macd_bps < 0:
            raise ValueError("min_macd_bps cannot be negative")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.min_histogram_slope_bps < 0:
            raise ValueError("min_histogram_slope_bps cannot be negative")
        if self.allowed_utc_hours is not None:
            if not self.allowed_utc_hours:
                raise ValueError("allowed_utc_hours cannot be empty")
            if any(hour < 0 or hour > 23 for hour in self.allowed_utc_hours):
                raise ValueError("allowed_utc_hours must be between 0 and 23")


DEFAULT_MACD_MOMENTUM_PARAMETER_SETS: tuple[MacdMomentumParameterSet, ...] = (
    MacdMomentumParameterSet(
        "competition_current_6_18_5_h2p5_m1_eff20_hold12",
        6,
        18,
        5,
        2.5,
        1.0,
        0.20,
        12,
        allowed_utc_hours=(10, 11, 12, 13, 14),
    ),
    MacdMomentumParameterSet("baseline_12_26_9_h1_5_m0_5_eff10_hold24", 12, 26, 9, 1.5, 0.5, 0.10, 24),
    MacdMomentumParameterSet("strict_12_26_9_h3_m1_eff20_hold16", 12, 26, 9, 3.0, 1.0, 0.20, 16),
    MacdMomentumParameterSet("fast_6_18_5_h1_m0_5_eff10_hold16", 6, 18, 5, 1.0, 0.5, 0.10, 16),
    MacdMomentumParameterSet("fast_strict_6_18_5_h2_m1_eff20_hold12", 6, 18, 5, 2.0, 1.0, 0.20, 12),
    MacdMomentumParameterSet(
        "fast_slope_6_18_5_h2_m1_eff20_hold12_slope0_25",
        6,
        18,
        5,
        2.0,
        1.0,
        0.20,
        12,
        min_histogram_slope_bps=0.25,
    ),
    MacdMomentumParameterSet("smooth_8_21_8_h2_m0_75_eff15_hold20", 8, 21, 8, 2.0, 0.75, 0.15, 20),
)


@dataclass(frozen=True)
class MacdMomentumOptimizationCandidate:
    parameters: MacdMomentumParameterSet
    comparison_row: PortfolioStrategyComparisonRow
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None

    @property
    def rank_key(self) -> tuple[float, ...]:
        metrics = self.comparison_row.competition_metrics
        if self.walk_forward is not None:
            return (
                _coverage_adjusted_active_score(self.walk_forward),
                self.walk_forward.active_positive_fold_fraction,
                self.walk_forward.non_negative_fold_fraction,
                self.walk_forward.median_active_test_return_pct,
                self.walk_forward.positive_fold_fraction,
                -self.walk_forward.losing_fold_fraction,
                self.comparison_row.proxy_score,
                metrics.return_pct,
                metrics.sharpe_15m,
                -metrics.max_drawdown_pct,
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
class MacdMomentumOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[MacdMomentumOptimizationCandidate, ...]

    @property
    def best(self) -> MacdMomentumOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_macd_momentum_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        MacdMomentumParameterSet, ...
    ] = DEFAULT_MACD_MOMENTUM_PARAMETER_SETS,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> MacdMomentumOptimizationResult:
    if not parameter_sets:
        raise ValueError("MACD momentum optimizer needs at least one parameter set")

    candidates: list[MacdMomentumOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("macd_momentum",),
            symbols=symbols,
        )
        if comparison.best is None:
            continue
        selected_symbols = comparison.symbols
        walk_forward = (
            run_fixed_warmup_portfolio_walk_forward(
                config=candidate_config,
                prices=prices,
                quotes=quotes,
                strategy_name="macd_momentum",
                symbols=comparison.symbols,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            MacdMomentumOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return MacdMomentumOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_macd_momentum_optimization_csv(
    result: MacdMomentumOptimizationResult,
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
                "fast_window",
                "slow_window",
                "signal_window",
                "min_histogram_bps",
                "min_macd_bps",
                "min_histogram_slope_bps",
                "min_trend_efficiency",
                "max_holding_period",
                "allowed_utc_hours",
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
                "wf_positive_fold_fraction",
                "wf_active_fold_fraction",
                "wf_active_positive_fold_fraction",
                "wf_non_negative_fold_fraction",
                "wf_losing_fold_fraction",
                "wf_median_test_return_pct",
                "wf_median_active_test_return_pct",
                "wf_worst_test_drawdown_pct",
                "wf_total_evaluation_fills",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            parameters = candidate.parameters
            row = candidate.comparison_row
            metrics = row.competition_metrics
            walk_forward = candidate.walk_forward
            writer.writerow(
                {
                    "rank": rank,
                    "label": parameters.label,
                    "symbols": " ".join(result.symbols),
                    "fast_window": parameters.fast_window,
                    "slow_window": parameters.slow_window,
                    "signal_window": parameters.signal_window,
                    "min_histogram_bps": parameters.min_histogram_bps,
                    "min_macd_bps": parameters.min_macd_bps,
                    "min_histogram_slope_bps": parameters.min_histogram_slope_bps,
                    "min_trend_efficiency": parameters.min_trend_efficiency,
                    "max_holding_period": parameters.max_holding_period,
                    "allowed_utc_hours": _hours_text(parameters.allowed_utc_hours),
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
                    "wf_positive_fold_fraction": (
                        "" if walk_forward is None else walk_forward.positive_fold_fraction
                    ),
                    "wf_active_fold_fraction": (
                        "" if walk_forward is None else walk_forward.active_fold_fraction
                    ),
                    "wf_active_positive_fold_fraction": (
                        ""
                        if walk_forward is None
                        else walk_forward.active_positive_fold_fraction
                    ),
                    "wf_non_negative_fold_fraction": (
                        ""
                        if walk_forward is None
                        else walk_forward.non_negative_fold_fraction
                    ),
                    "wf_losing_fold_fraction": (
                        "" if walk_forward is None else walk_forward.losing_fold_fraction
                    ),
                    "wf_median_test_return_pct": (
                        "" if walk_forward is None else walk_forward.median_test_return_pct
                    ),
                    "wf_median_active_test_return_pct": (
                        ""
                        if walk_forward is None
                        else walk_forward.median_active_test_return_pct
                    ),
                    "wf_worst_test_drawdown_pct": (
                        "" if walk_forward is None else walk_forward.worst_test_drawdown_pct
                    ),
                    "wf_total_evaluation_fills": (
                        "" if walk_forward is None else walk_forward.total_evaluation_fills
                    ),
                }
            )


def _config_with_parameters(
    config: AppConfig,
    parameters: MacdMomentumParameterSet,
) -> AppConfig:
    macd_momentum = replace(
        config.macd_momentum,
        fast_window=parameters.fast_window,
        slow_window=parameters.slow_window,
        signal_window=parameters.signal_window,
        min_histogram_bps=parameters.min_histogram_bps,
        exit_histogram_bps=min(
            config.macd_momentum.exit_histogram_bps,
            parameters.min_histogram_bps * 0.5,
        ),
        min_macd_bps=parameters.min_macd_bps,
        min_histogram_slope_bps=parameters.min_histogram_slope_bps,
        min_trend_efficiency=parameters.min_trend_efficiency,
        max_holding_period=parameters.max_holding_period,
        forex_allowed_utc_hours=(
            config.macd_momentum.forex_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
        metal_allowed_utc_hours=(
            config.macd_momentum.metal_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
        crypto_allowed_utc_hours=(
            config.macd_momentum.crypto_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
    )
    return replace(config, macd_momentum=macd_momentum)


def _hours_text(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return ""
    return "|".join(str(hour) for hour in hours)


def _coverage_adjusted_active_score(
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    *,
    target_active_fold_fraction: float = 0.35,
) -> float:
    if target_active_fold_fraction <= 0:
        return walk_forward.active_positive_fold_fraction
    coverage = min(walk_forward.active_fold_fraction / target_active_fold_fraction, 1.0)
    return coverage * walk_forward.active_positive_fold_fraction
