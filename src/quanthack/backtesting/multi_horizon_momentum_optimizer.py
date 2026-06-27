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
class MultiHorizonMomentumParameterSet:
    label: str
    fast_lookback: int
    slow_lookback: int
    volatility_lookback: int
    baseline_volatility_lookback: int
    min_fast_move_bps: float
    min_slow_move_bps: float
    min_trend_efficiency: float
    min_volatility_ratio: float
    max_volatility_ratio: float
    max_holding_period: int
    allowed_utc_hours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("multi-horizon momentum parameter label is required")
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
        if self.min_fast_move_bps <= 0:
            raise ValueError("min_fast_move_bps must be positive")
        if self.min_slow_move_bps <= 0:
            raise ValueError("min_slow_move_bps must be positive")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        if self.min_volatility_ratio < 0:
            raise ValueError("min_volatility_ratio cannot be negative")
        if self.max_volatility_ratio <= self.min_volatility_ratio:
            raise ValueError("max_volatility_ratio must exceed min_volatility_ratio")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.allowed_utc_hours is not None:
            if not self.allowed_utc_hours:
                raise ValueError("allowed_utc_hours cannot be empty")
            if any(hour < 0 or hour > 23 for hour in self.allowed_utc_hours):
                raise ValueError("allowed_utc_hours must be between 0 and 23")


DEFAULT_MULTI_HORIZON_MOMENTUM_PARAMETER_SETS: tuple[
    MultiHorizonMomentumParameterSet, ...
] = (
    MultiHorizonMomentumParameterSet(
        "default_6_24_v12_b48_f2_s5_eff25_v035_250_hold24",
        6,
        24,
        12,
        48,
        2.0,
        5.0,
        0.25,
        0.35,
        2.50,
        24,
    ),
    MultiHorizonMomentumParameterSet(
        "strict_8_32_v12_b64_f3_s8_eff30_v050_220_hold18",
        8,
        32,
        12,
        64,
        3.0,
        8.0,
        0.30,
        0.50,
        2.20,
        18,
    ),
    MultiHorizonMomentumParameterSet(
        "fast_4_16_v8_b48_f2_s4_eff25_v025_250_hold16",
        4,
        16,
        8,
        48,
        2.0,
        4.0,
        0.25,
        0.25,
        2.50,
        16,
    ),
    MultiHorizonMomentumParameterSet(
        "slow_8_40_v16_b64_f4_s10_eff35_v035_200_hold24",
        8,
        40,
        16,
        64,
        4.0,
        10.0,
        0.35,
        0.35,
        2.00,
        24,
    ),
    MultiHorizonMomentumParameterSet(
        "liquid_10_14_6_24_v12_b48_f3_s7_eff30_v050_200_hold16",
        6,
        24,
        12,
        48,
        3.0,
        7.0,
        0.30,
        0.50,
        2.00,
        16,
        allowed_utc_hours=(10, 11, 12, 13, 14),
    ),
    MultiHorizonMomentumParameterSet(
        "broad_10_16_6_24_v12_b48_f3_s7_eff30_v035_250_hold16",
        6,
        24,
        12,
        48,
        3.0,
        7.0,
        0.30,
        0.35,
        2.50,
        16,
        allowed_utc_hours=(10, 11, 12, 13, 14, 15, 16),
    ),
)


@dataclass(frozen=True)
class MultiHorizonMomentumOptimizationCandidate:
    parameters: MultiHorizonMomentumParameterSet
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
class MultiHorizonMomentumOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[MultiHorizonMomentumOptimizationCandidate, ...]

    @property
    def best(self) -> MultiHorizonMomentumOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_multi_horizon_momentum_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        MultiHorizonMomentumParameterSet, ...
    ] = DEFAULT_MULTI_HORIZON_MOMENTUM_PARAMETER_SETS,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> MultiHorizonMomentumOptimizationResult:
    if not parameter_sets:
        raise ValueError("multi-horizon momentum optimizer needs at least one parameter set")

    candidates: list[MultiHorizonMomentumOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("multi_horizon_momentum",),
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
                strategy_name="multi_horizon_momentum",
                symbols=comparison.symbols,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            MultiHorizonMomentumOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return MultiHorizonMomentumOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_multi_horizon_momentum_optimization_csv(
    result: MultiHorizonMomentumOptimizationResult,
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
                "fast_lookback",
                "slow_lookback",
                "volatility_lookback",
                "baseline_volatility_lookback",
                "min_fast_move_bps",
                "min_slow_move_bps",
                "min_trend_efficiency",
                "min_volatility_ratio",
                "max_volatility_ratio",
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
                    "fast_lookback": parameters.fast_lookback,
                    "slow_lookback": parameters.slow_lookback,
                    "volatility_lookback": parameters.volatility_lookback,
                    "baseline_volatility_lookback": (
                        parameters.baseline_volatility_lookback
                    ),
                    "min_fast_move_bps": parameters.min_fast_move_bps,
                    "min_slow_move_bps": parameters.min_slow_move_bps,
                    "min_trend_efficiency": parameters.min_trend_efficiency,
                    "min_volatility_ratio": parameters.min_volatility_ratio,
                    "max_volatility_ratio": parameters.max_volatility_ratio,
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
    parameters: MultiHorizonMomentumParameterSet,
) -> AppConfig:
    multi_horizon_momentum = replace(
        config.multi_horizon_momentum,
        fast_lookback=parameters.fast_lookback,
        slow_lookback=parameters.slow_lookback,
        volatility_lookback=parameters.volatility_lookback,
        baseline_volatility_lookback=parameters.baseline_volatility_lookback,
        min_fast_move_bps=parameters.min_fast_move_bps,
        min_slow_move_bps=parameters.min_slow_move_bps,
        exit_slow_move_bps=min(
            config.multi_horizon_momentum.exit_slow_move_bps,
            parameters.min_slow_move_bps * 0.5,
        ),
        min_trend_efficiency=parameters.min_trend_efficiency,
        min_volatility_ratio=parameters.min_volatility_ratio,
        max_volatility_ratio=parameters.max_volatility_ratio,
        max_holding_period=parameters.max_holding_period,
        forex_allowed_utc_hours=(
            config.multi_horizon_momentum.forex_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
        metal_allowed_utc_hours=(
            config.multi_horizon_momentum.metal_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
    )
    return replace(config, multi_horizon_momentum=multi_horizon_momentum)


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
