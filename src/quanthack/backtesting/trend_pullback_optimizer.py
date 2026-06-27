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
class TrendPullbackParameterSet:
    label: str
    lookback: int
    pullback_window: int
    min_trend_bps: float
    min_trend_efficiency: float
    min_pullback_bps: float
    max_pullback_bps: float
    min_resume_bps: float
    min_expected_edge_bps: float
    max_holding_period: int = 24
    forex_allowed_utc_hours: tuple[int, ...] | None = None
    metal_allowed_utc_hours: tuple[int, ...] | None = None
    crypto_allowed_utc_hours: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("trend pullback parameter label is required")
        if self.lookback < 8:
            raise ValueError("trend pullback lookback must be at least 8")
        if self.pullback_window < 2:
            raise ValueError("pullback_window must be at least 2")
        if self.lookback < self.pullback_window + 4:
            raise ValueError("lookback must leave enough pre-pullback history")
        if self.min_trend_bps < 0:
            raise ValueError("min_trend_bps cannot be negative")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        if self.min_pullback_bps < 0:
            raise ValueError("min_pullback_bps cannot be negative")
        if self.max_pullback_bps <= 0:
            raise ValueError("max_pullback_bps must be positive")
        if self.max_pullback_bps < self.min_pullback_bps:
            raise ValueError("max_pullback_bps cannot be below min_pullback_bps")
        if self.min_resume_bps < 0:
            raise ValueError("min_resume_bps cannot be negative")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        _normalize_optional_hours(self, "forex_allowed_utc_hours")
        _normalize_optional_hours(self, "metal_allowed_utc_hours")
        _normalize_optional_hours(self, "crypto_allowed_utc_hours")


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


LIQUID_FX_METAL_HOURS = (11, 12, 13, 14, 15, 16, 17, 18, 19)


DEFAULT_TREND_PULLBACK_PARAMETER_SETS: tuple[TrendPullbackParameterSet, ...] = (
    TrendPullbackParameterSet(
        label="baseline_l32_p4_t8_e0_35_pb1_12_r1_edge3",
        lookback=32,
        pullback_window=4,
        min_trend_bps=8.0,
        min_trend_efficiency=0.35,
        min_pullback_bps=1.0,
        max_pullback_bps=12.0,
        min_resume_bps=1.0,
        min_expected_edge_bps=3.0,
        forex_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
        metal_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
    ),
    TrendPullbackParameterSet(
        label="strict_l48_p4_t12_e0_45_pb2_10_r1_5_edge4",
        lookback=48,
        pullback_window=4,
        min_trend_bps=12.0,
        min_trend_efficiency=0.45,
        min_pullback_bps=2.0,
        max_pullback_bps=10.0,
        min_resume_bps=1.5,
        min_expected_edge_bps=4.0,
        forex_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
        metal_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
    ),
    TrendPullbackParameterSet(
        label="fast_l24_p3_t6_e0_30_pb1_15_r1_edge3",
        lookback=24,
        pullback_window=3,
        min_trend_bps=6.0,
        min_trend_efficiency=0.30,
        min_pullback_bps=1.0,
        max_pullback_bps=15.0,
        min_resume_bps=1.0,
        min_expected_edge_bps=3.0,
        forex_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
        metal_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
    ),
    TrendPullbackParameterSet(
        label="tight_l32_p3_t10_e0_50_pb1_8_r1_5_edge4",
        lookback=32,
        pullback_window=3,
        min_trend_bps=10.0,
        min_trend_efficiency=0.50,
        min_pullback_bps=1.0,
        max_pullback_bps=8.0,
        min_resume_bps=1.5,
        min_expected_edge_bps=4.0,
        forex_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
        metal_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
    ),
    TrendPullbackParameterSet(
        label="ny_l32_p4_t8_e0_40_pb1_12_r1_edge3",
        lookback=32,
        pullback_window=4,
        min_trend_bps=8.0,
        min_trend_efficiency=0.40,
        min_pullback_bps=1.0,
        max_pullback_bps=12.0,
        min_resume_bps=1.0,
        min_expected_edge_bps=3.0,
        forex_allowed_utc_hours=(13, 14, 15, 16, 17, 18, 19),
        metal_allowed_utc_hours=(13, 14, 15, 16, 17, 18, 19),
    ),
    TrendPullbackParameterSet(
        label="messy_cont_l64_p8_t10_e0_10_pb1_35_r0_5_edge3",
        lookback=64,
        pullback_window=8,
        min_trend_bps=10.0,
        min_trend_efficiency=0.10,
        min_pullback_bps=1.0,
        max_pullback_bps=35.0,
        min_resume_bps=0.5,
        min_expected_edge_bps=3.0,
        max_holding_period=32,
        forex_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
        metal_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
    ),
    TrendPullbackParameterSet(
        label="metal_cont_l96_p12_t15_e0_06_pb2_45_r0_5_edge4",
        lookback=96,
        pullback_window=12,
        min_trend_bps=15.0,
        min_trend_efficiency=0.06,
        min_pullback_bps=2.0,
        max_pullback_bps=45.0,
        min_resume_bps=0.5,
        min_expected_edge_bps=4.0,
        max_holding_period=40,
        forex_allowed_utc_hours=(),
        metal_allowed_utc_hours=LIQUID_FX_METAL_HOURS,
    ),
    TrendPullbackParameterSet(
        label="late_metal_cont_l96_p12_t15_e0_06_pb2_45_r0_5_edge4",
        lookback=96,
        pullback_window=12,
        min_trend_bps=15.0,
        min_trend_efficiency=0.06,
        min_pullback_bps=2.0,
        max_pullback_bps=45.0,
        min_resume_bps=0.5,
        min_expected_edge_bps=4.0,
        max_holding_period=40,
        forex_allowed_utc_hours=(),
        metal_allowed_utc_hours=(15, 16, 17, 18, 19, 20),
    ),
)


@dataclass(frozen=True)
class TrendPullbackOptimizationCandidate:
    parameters: TrendPullbackParameterSet
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
class TrendPullbackOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[TrendPullbackOptimizationCandidate, ...]

    @property
    def best(self) -> TrendPullbackOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_trend_pullback_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        TrendPullbackParameterSet, ...
    ] = DEFAULT_TREND_PULLBACK_PARAMETER_SETS,
    include_walk_forward: bool = False,
    walk_forward_train_size: int = 480,
    walk_forward_test_size: int = 240,
    walk_forward_step_size: int = 240,
    walk_forward_max_baskets: int = 30,
) -> TrendPullbackOptimizationResult:
    if not parameter_sets:
        raise ValueError("trend pullback optimizer needs at least one parameter set")

    candidates: list[TrendPullbackOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("trend_pullback",),
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
                strategy_names=("trend_pullback",),
                max_baskets=walk_forward_max_baskets,
                train_size=walk_forward_train_size,
                test_size=walk_forward_test_size,
                step_size=walk_forward_step_size,
                min_test_fills=1,
                min_stable_fold_fraction=0.50,
            )
            walk_forward_summary = walk_forward.summary
        candidates.append(
            TrendPullbackOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward_summary=walk_forward_summary,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return TrendPullbackOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_trend_pullback_optimization_csv(
    result: TrendPullbackOptimizationResult,
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
                "pullback_window",
                "min_trend_bps",
                "min_trend_efficiency",
                "min_pullback_bps",
                "max_pullback_bps",
                "min_resume_bps",
                "min_expected_edge_bps",
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
                    "pullback_window": parameters.pullback_window,
                    "min_trend_bps": parameters.min_trend_bps,
                    "min_trend_efficiency": parameters.min_trend_efficiency,
                    "min_pullback_bps": parameters.min_pullback_bps,
                    "max_pullback_bps": parameters.max_pullback_bps,
                    "min_resume_bps": parameters.min_resume_bps,
                    "min_expected_edge_bps": parameters.min_expected_edge_bps,
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
    parameters: TrendPullbackParameterSet,
) -> AppConfig:
    trend_pullback = replace(
        config.trend_pullback,
        lookback=parameters.lookback,
        pullback_window=parameters.pullback_window,
        min_trend_bps=parameters.min_trend_bps,
        min_trend_efficiency=parameters.min_trend_efficiency,
        min_pullback_bps=parameters.min_pullback_bps,
        max_pullback_bps=parameters.max_pullback_bps,
        min_resume_bps=parameters.min_resume_bps,
        min_expected_edge_bps=parameters.min_expected_edge_bps,
        max_holding_period=parameters.max_holding_period,
        forex_allowed_utc_hours=parameters.forex_allowed_utc_hours,
        metal_allowed_utc_hours=parameters.metal_allowed_utc_hours,
        crypto_allowed_utc_hours=parameters.crypto_allowed_utc_hours,
    )
    return replace(config, trend_pullback=trend_pullback)
