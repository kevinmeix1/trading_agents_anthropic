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
class LiquiditySweepReversalParameterSet:
    label: str
    lookback: int
    min_sweep_bps: float
    reentry_buffer_bps: float
    min_range_width_bps: float
    max_sweep_bps: float
    max_trend_efficiency: float
    min_expected_edge_bps: float
    max_holding_period: int
    allowed_utc_hours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("liquidity sweep parameter label is required")
        if self.lookback < 6:
            raise ValueError("lookback must be at least 6 prices")
        if self.min_sweep_bps <= 0:
            raise ValueError("min_sweep_bps must be positive")
        if self.reentry_buffer_bps < 0:
            raise ValueError("reentry_buffer_bps cannot be negative")
        if self.min_range_width_bps <= 0:
            raise ValueError("min_range_width_bps must be positive")
        if self.max_sweep_bps <= self.min_sweep_bps:
            raise ValueError("max_sweep_bps must exceed min_sweep_bps")
        if not 0 <= self.max_trend_efficiency <= 1:
            raise ValueError("max_trend_efficiency must be between 0 and 1")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.allowed_utc_hours is not None:
            if not self.allowed_utc_hours:
                raise ValueError("allowed_utc_hours cannot be empty")
            if any(hour < 0 or hour > 23 for hour in self.allowed_utc_hours):
                raise ValueError("allowed_utc_hours must be between 0 and 23")


DEFAULT_LIQUIDITY_SWEEP_REVERSAL_PARAMETER_SETS: tuple[
    LiquiditySweepReversalParameterSet, ...
] = (
    LiquiditySweepReversalParameterSet(
        "baseline_l32_s2_r0_25_w4_e2_hold8",
        32,
        2.0,
        0.25,
        4.0,
        80.0,
        0.75,
        2.0,
        8,
    ),
    LiquiditySweepReversalParameterSet(
        "selective_l32_s4_r0_5_w6_e3_hold6",
        32,
        4.0,
        0.50,
        6.0,
        60.0,
        0.65,
        3.0,
        6,
    ),
    LiquiditySweepReversalParameterSet(
        "fast_l20_s2_r0_25_w3_e2_hold6",
        20,
        2.0,
        0.25,
        3.0,
        60.0,
        0.80,
        2.0,
        6,
    ),
    LiquiditySweepReversalParameterSet(
        "wide_l48_s3_r0_25_w8_e4_hold12",
        48,
        3.0,
        0.25,
        8.0,
        100.0,
        0.70,
        4.0,
        12,
    ),
    LiquiditySweepReversalParameterSet(
        "permissive_l24_s1_5_r0_w3_e1_5_hold8",
        24,
        1.5,
        0.0,
        3.0,
        80.0,
        0.85,
        1.5,
        8,
    ),
)


@dataclass(frozen=True)
class LiquiditySweepReversalOptimizationCandidate:
    parameters: LiquiditySweepReversalParameterSet
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
class LiquiditySweepReversalOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[LiquiditySweepReversalOptimizationCandidate, ...]

    @property
    def best(self) -> LiquiditySweepReversalOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_liquidity_sweep_reversal_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        LiquiditySweepReversalParameterSet, ...
    ] = DEFAULT_LIQUIDITY_SWEEP_REVERSAL_PARAMETER_SETS,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> LiquiditySweepReversalOptimizationResult:
    if not parameter_sets:
        raise ValueError("liquidity sweep optimizer needs at least one parameter set")

    candidates: list[LiquiditySweepReversalOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("liquidity_sweep_reversal",),
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
                strategy_name="liquidity_sweep_reversal",
                symbols=comparison.symbols,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            LiquiditySweepReversalOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return LiquiditySweepReversalOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_liquidity_sweep_reversal_optimization_csv(
    result: LiquiditySweepReversalOptimizationResult,
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
                "min_sweep_bps",
                "reentry_buffer_bps",
                "min_range_width_bps",
                "max_sweep_bps",
                "max_trend_efficiency",
                "min_expected_edge_bps",
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
                    "lookback": parameters.lookback,
                    "min_sweep_bps": parameters.min_sweep_bps,
                    "reentry_buffer_bps": parameters.reentry_buffer_bps,
                    "min_range_width_bps": parameters.min_range_width_bps,
                    "max_sweep_bps": parameters.max_sweep_bps,
                    "max_trend_efficiency": parameters.max_trend_efficiency,
                    "min_expected_edge_bps": parameters.min_expected_edge_bps,
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
    parameters: LiquiditySweepReversalParameterSet,
) -> AppConfig:
    liquidity_sweep_reversal = replace(
        config.liquidity_sweep_reversal,
        lookback=parameters.lookback,
        min_sweep_bps=parameters.min_sweep_bps,
        reentry_buffer_bps=parameters.reentry_buffer_bps,
        min_range_width_bps=parameters.min_range_width_bps,
        max_sweep_bps=parameters.max_sweep_bps,
        max_trend_efficiency=parameters.max_trend_efficiency,
        min_expected_edge_bps=parameters.min_expected_edge_bps,
        max_holding_period=parameters.max_holding_period,
        forex_allowed_utc_hours=(
            config.liquidity_sweep_reversal.forex_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
        metal_allowed_utc_hours=(
            config.liquidity_sweep_reversal.metal_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
        crypto_allowed_utc_hours=(
            config.liquidity_sweep_reversal.crypto_allowed_utc_hours
            if parameters.allowed_utc_hours is None
            else parameters.allowed_utc_hours
        ),
    )
    return replace(config, liquidity_sweep_reversal=liquidity_sweep_reversal)


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
