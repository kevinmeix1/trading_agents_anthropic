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
class RelativeStrengthParameterSet:
    label: str
    lookback: int
    entry_zscore: float
    exit_zscore: float
    min_abs_move_bps: float = 0.5
    require_asset_class_confirmation: bool = False
    asset_class_entry_zscore: float = 0.35
    require_metal_trend_confirmation: bool = False
    metal_trend_min_move_bps: float = 2.0
    metal_trend_min_efficiency: float = 0.20
    min_score_dispersion: float = 0.0
    min_target_trend_efficiency: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("relative strength parameter label is required")
        if self.lookback < 3:
            raise ValueError("relative strength lookback must be at least 3")
        if self.entry_zscore <= 0:
            raise ValueError("entry_zscore must be positive")
        if self.exit_zscore < 0 or self.exit_zscore >= self.entry_zscore:
            raise ValueError("exit_zscore must be non-negative and below entry_zscore")
        if self.min_abs_move_bps < 0:
            raise ValueError("min_abs_move_bps cannot be negative")
        if self.asset_class_entry_zscore < 0:
            raise ValueError("asset_class_entry_zscore cannot be negative")
        if self.metal_trend_min_move_bps < 0:
            raise ValueError("metal_trend_min_move_bps cannot be negative")
        if not 0 <= self.metal_trend_min_efficiency <= 1:
            raise ValueError("metal_trend_min_efficiency must be between 0 and 1")
        if self.min_score_dispersion < 0:
            raise ValueError("min_score_dispersion cannot be negative")
        if not 0 <= self.min_target_trend_efficiency <= 1:
            raise ValueError("min_target_trend_efficiency must be between 0 and 1")


DEFAULT_RELATIVE_STRENGTH_PARAMETER_SETS: tuple[RelativeStrengthParameterSet, ...] = (
    RelativeStrengthParameterSet(
        label="baseline_l12_z0_75",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.25,
    ),
    RelativeStrengthParameterSet(
        label="fast_l8_z0_75",
        lookback=8,
        entry_zscore=0.75,
        exit_zscore=0.25,
    ),
    RelativeStrengthParameterSet(
        label="slow_l16_z0_75",
        lookback=16,
        entry_zscore=0.75,
        exit_zscore=0.25,
    ),
    RelativeStrengthParameterSet(
        label="permissive_l12_z0_50",
        lookback=12,
        entry_zscore=0.50,
        exit_zscore=0.15,
    ),
    RelativeStrengthParameterSet(
        label="selective_l12_z1_00",
        lookback=12,
        entry_zscore=1.00,
        exit_zscore=0.30,
    ),
    RelativeStrengthParameterSet(
        label="metal_trend_l12_z0_75",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.25,
        require_metal_trend_confirmation=True,
        metal_trend_min_move_bps=2.0,
        metal_trend_min_efficiency=0.20,
    ),
    RelativeStrengthParameterSet(
        label="asset_confirm_l12_z0_75",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.25,
        require_asset_class_confirmation=True,
        asset_class_entry_zscore=0.35,
    ),
    RelativeStrengthParameterSet(
        label="dispersion_l12_z0_75_d2",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.25,
        min_score_dispersion=2.0,
    ),
    RelativeStrengthParameterSet(
        label="trend_l12_z0_75_eff0_20",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.25,
        min_target_trend_efficiency=0.20,
    ),
    RelativeStrengthParameterSet(
        label="regime_l12_z0_75_d2_eff0_20",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.25,
        min_score_dispersion=2.0,
        min_target_trend_efficiency=0.20,
    ),
)


@dataclass(frozen=True)
class RelativeStrengthOptimizationCandidate:
    parameters: RelativeStrengthParameterSet
    comparison_row: PortfolioStrategyComparisonRow
    walk_forward_summary: PortfolioWalkForwardSummary | None = None

    @property
    def rank_key(self) -> tuple[float, float, float, float, float, float]:
        metrics = self.comparison_row.competition_metrics
        if self.walk_forward_summary is None:
            return (
                self.comparison_row.proxy_score,
                self.comparison_row.risk_discipline.score,
                metrics.return_pct,
                metrics.sharpe_15m,
                -metrics.max_drawdown_pct,
                -float(metrics.trade_count),
            )
        summary = self.walk_forward_summary
        return (
            100.0 if summary.eligible else 0.0,
            summary.stable_fold_fraction,
            summary.median_test_return_pct,
            summary.median_test_sharpe_15m,
            -summary.worst_test_drawdown_pct,
            metrics.return_pct,
        )


@dataclass(frozen=True)
class RelativeStrengthOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[RelativeStrengthOptimizationCandidate, ...]

    @property
    def best(self) -> RelativeStrengthOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_relative_strength_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        RelativeStrengthParameterSet, ...
    ] = DEFAULT_RELATIVE_STRENGTH_PARAMETER_SETS,
    include_walk_forward: bool = False,
    walk_forward_train_size: int = 40,
    walk_forward_test_size: int = 16,
    walk_forward_step_size: int = 8,
    walk_forward_max_baskets: int = 10,
) -> RelativeStrengthOptimizationResult:
    if not parameter_sets:
        raise ValueError("relative strength optimizer needs at least one parameter set")

    candidates: list[RelativeStrengthOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("relative_strength",),
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
                strategy_names=("relative_strength",),
                max_baskets=walk_forward_max_baskets,
                train_size=walk_forward_train_size,
                test_size=walk_forward_test_size,
                step_size=walk_forward_step_size,
                min_test_fills=1,
                min_stable_fold_fraction=0.50,
            )
            walk_forward_summary = walk_forward.summary

        candidates.append(
            RelativeStrengthOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward_summary=walk_forward_summary,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return RelativeStrengthOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_relative_strength_optimization_csv(
    result: RelativeStrengthOptimizationResult,
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
                "entry_zscore",
                "exit_zscore",
                "min_abs_move_bps",
                "asset_class_confirmation",
                "asset_class_entry_zscore",
                "metal_trend_confirmation",
                "metal_trend_min_move_bps",
                "metal_trend_min_efficiency",
                "min_score_dispersion",
                "min_target_trend_efficiency",
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
                    "entry_zscore": parameters.entry_zscore,
                    "exit_zscore": parameters.exit_zscore,
                    "min_abs_move_bps": parameters.min_abs_move_bps,
                    "asset_class_confirmation": (
                        parameters.require_asset_class_confirmation
                    ),
                    "asset_class_entry_zscore": parameters.asset_class_entry_zscore,
                    "metal_trend_confirmation": (
                        parameters.require_metal_trend_confirmation
                    ),
                    "metal_trend_min_move_bps": parameters.metal_trend_min_move_bps,
                    "metal_trend_min_efficiency": (
                        parameters.metal_trend_min_efficiency
                    ),
                    "min_score_dispersion": parameters.min_score_dispersion,
                    "min_target_trend_efficiency": (
                        parameters.min_target_trend_efficiency
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
                }
            )


def _config_with_parameters(
    config: AppConfig,
    parameters: RelativeStrengthParameterSet,
) -> AppConfig:
    relative_strength = replace(
        config.relative_strength,
        lookback=parameters.lookback,
        entry_zscore=parameters.entry_zscore,
        exit_zscore=parameters.exit_zscore,
        min_abs_move_bps=parameters.min_abs_move_bps,
        require_asset_class_confirmation=parameters.require_asset_class_confirmation,
        asset_class_entry_zscore=parameters.asset_class_entry_zscore,
        require_metal_trend_confirmation=parameters.require_metal_trend_confirmation,
        metal_trend_min_move_bps=parameters.metal_trend_min_move_bps,
        metal_trend_min_efficiency=parameters.metal_trend_min_efficiency,
        min_score_dispersion=parameters.min_score_dispersion,
        min_target_trend_efficiency=parameters.min_target_trend_efficiency,
    )
    return replace(config, relative_strength=relative_strength)
