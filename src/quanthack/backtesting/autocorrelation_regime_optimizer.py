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
class AutocorrelationRegimeParameterSet:
    label: str
    lookback: int
    signal_lookback: int
    min_abs_autocorrelation: float
    exit_abs_autocorrelation: float
    min_momentum_bps: float
    min_trend_efficiency: float
    min_reversion_zscore: float
    min_reversion_move_bps: float
    min_expected_edge_bps: float
    max_holding_period: int
    allowed_utc_hours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("autocorrelation regime parameter label is required")
        if self.lookback < 6:
            raise ValueError("lookback must be at least 6")
        if self.signal_lookback < 2:
            raise ValueError("signal_lookback must be at least 2")
        if self.signal_lookback >= self.lookback:
            raise ValueError("signal_lookback must be below lookback")
        if not 0 <= self.min_abs_autocorrelation <= 1:
            raise ValueError("min_abs_autocorrelation must be between 0 and 1")
        if not 0 <= self.exit_abs_autocorrelation <= 1:
            raise ValueError("exit_abs_autocorrelation must be between 0 and 1")
        if self.exit_abs_autocorrelation > self.min_abs_autocorrelation:
            raise ValueError("exit_abs_autocorrelation cannot exceed entry threshold")
        if self.min_momentum_bps < 0:
            raise ValueError("min_momentum_bps cannot be negative")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        if self.min_reversion_zscore < 0:
            raise ValueError("min_reversion_zscore cannot be negative")
        if self.min_reversion_move_bps < 0:
            raise ValueError("min_reversion_move_bps cannot be negative")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.allowed_utc_hours is not None:
            if not self.allowed_utc_hours:
                raise ValueError("allowed_utc_hours cannot be empty")
            if any(hour < 0 or hour > 23 for hour in self.allowed_utc_hours):
                raise ValueError("allowed_utc_hours must be between 0 and 23")


LIQUID_FX_METAL_HOURS = (10, 11, 12, 13, 14, 15, 16, 17)
CORE_OVERLAP_HOURS = (10, 11, 12, 13, 14)


DEFAULT_AUTOCORRELATION_REGIME_PARAMETER_SETS: tuple[
    AutocorrelationRegimeParameterSet, ...
] = (
    AutocorrelationRegimeParameterSet(
        "baseline_l32_s6_rho18_m4_eff20_z80_edge3_hold16",
        32,
        6,
        0.18,
        0.05,
        4.0,
        0.20,
        0.80,
        2.0,
        3.0,
        16,
        LIQUID_FX_METAL_HOURS,
    ),
    AutocorrelationRegimeParameterSet(
        "strict_momentum_l48_s8_rho28_m8_eff45_z120_edge6_hold10",
        48,
        8,
        0.28,
        0.08,
        8.0,
        0.45,
        1.20,
        4.0,
        6.0,
        10,
        CORE_OVERLAP_HOURS,
    ),
    AutocorrelationRegimeParameterSet(
        "fast_reversion_l24_s4_rho25_m6_eff25_z150_edge5_hold8",
        24,
        4,
        0.25,
        0.06,
        6.0,
        0.25,
        1.50,
        4.0,
        5.0,
        8,
        CORE_OVERLAP_HOURS,
    ),
    AutocorrelationRegimeParameterSet(
        "selective_l40_s5_rho35_m8_eff35_z180_edge7_hold6",
        40,
        5,
        0.35,
        0.08,
        8.0,
        0.35,
        1.80,
        5.0,
        7.0,
        6,
        CORE_OVERLAP_HOURS,
    ),
    AutocorrelationRegimeParameterSet(
        "very_selective_l64_s8_rho40_m10_eff50_z200_edge8_hold6",
        64,
        8,
        0.40,
        0.10,
        10.0,
        0.50,
        2.00,
        6.0,
        8.0,
        6,
        CORE_OVERLAP_HOURS,
    ),
)


@dataclass(frozen=True)
class AutocorrelationRegimeOptimizationCandidate:
    parameters: AutocorrelationRegimeParameterSet
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
class AutocorrelationRegimeOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[AutocorrelationRegimeOptimizationCandidate, ...]

    @property
    def best(self) -> AutocorrelationRegimeOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_autocorrelation_regime_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        AutocorrelationRegimeParameterSet, ...
    ] = DEFAULT_AUTOCORRELATION_REGIME_PARAMETER_SETS,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> AutocorrelationRegimeOptimizationResult:
    if not parameter_sets:
        raise ValueError("autocorrelation regime optimizer needs at least one parameter set")

    candidates: list[AutocorrelationRegimeOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("autocorrelation_regime",),
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
                strategy_name="autocorrelation_regime",
                symbols=comparison.symbols,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            AutocorrelationRegimeOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return AutocorrelationRegimeOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_autocorrelation_regime_optimization_csv(
    result: AutocorrelationRegimeOptimizationResult,
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
                "signal_lookback",
                "min_abs_autocorrelation",
                "exit_abs_autocorrelation",
                "min_momentum_bps",
                "min_trend_efficiency",
                "min_reversion_zscore",
                "min_reversion_move_bps",
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
                    "signal_lookback": parameters.signal_lookback,
                    "min_abs_autocorrelation": parameters.min_abs_autocorrelation,
                    "exit_abs_autocorrelation": parameters.exit_abs_autocorrelation,
                    "min_momentum_bps": parameters.min_momentum_bps,
                    "min_trend_efficiency": parameters.min_trend_efficiency,
                    "min_reversion_zscore": parameters.min_reversion_zscore,
                    "min_reversion_move_bps": parameters.min_reversion_move_bps,
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
    parameters: AutocorrelationRegimeParameterSet,
) -> AppConfig:
    autocorrelation_regime = replace(
        config.autocorrelation_regime,
        lookback=parameters.lookback,
        signal_lookback=parameters.signal_lookback,
        min_abs_autocorrelation=parameters.min_abs_autocorrelation,
        exit_abs_autocorrelation=parameters.exit_abs_autocorrelation,
        min_momentum_bps=parameters.min_momentum_bps,
        min_trend_efficiency=parameters.min_trend_efficiency,
        min_reversion_zscore=parameters.min_reversion_zscore,
        min_reversion_move_bps=parameters.min_reversion_move_bps,
        min_expected_edge_bps=parameters.min_expected_edge_bps,
        max_holding_period=parameters.max_holding_period,
        forex_allowed_utc_hours=parameters.allowed_utc_hours,
        metal_allowed_utc_hours=parameters.allowed_utc_hours,
    )
    return replace(config, autocorrelation_regime=autocorrelation_regime)


def _coverage_adjusted_active_score(
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
) -> float:
    if walk_forward.active_fold_fraction <= 0:
        return 0.0
    return (
        0.45 * walk_forward.active_positive_fold_fraction
        + 0.25 * walk_forward.non_negative_fold_fraction
        + 0.20 * walk_forward.active_fold_fraction
        + 0.10 * min(max(walk_forward.median_active_test_return_pct * 1000, -1), 1)
    )


def _hours_text(hours: tuple[int, ...] | None) -> str:
    if hours is None:
        return ""
    return " ".join(str(hour) for hour in hours)
