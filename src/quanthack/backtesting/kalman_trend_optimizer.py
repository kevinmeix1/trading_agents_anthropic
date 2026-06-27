from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.portfolio_strategy_compare import (
    PortfolioStrategyComparisonRow,
    compare_portfolio_strategies,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class KalmanTrendParameterSet:
    label: str
    lookback: int
    min_abs_slope_bps: float
    min_trend_efficiency: float
    min_expected_edge_bps: float
    expected_holding_bars: int
    max_holding_period: int

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Kalman trend parameter label is required")
        if self.lookback < 5:
            raise ValueError("lookback must be at least 5")
        if self.min_abs_slope_bps < 0:
            raise ValueError("min_abs_slope_bps cannot be negative")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        if self.expected_holding_bars < 1:
            raise ValueError("expected_holding_bars must be at least 1")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")


DEFAULT_KALMAN_TREND_PARAMETER_SETS: tuple[KalmanTrendParameterSet, ...] = (
    KalmanTrendParameterSet(
        label="best_l80_s0_50_e0_20_edge5_hold32",
        lookback=80,
        min_abs_slope_bps=0.50,
        min_trend_efficiency=0.20,
        min_expected_edge_bps=5.0,
        expected_holding_bars=6,
        max_holding_period=32,
    ),
    KalmanTrendParameterSet(
        label="strict_l80_s1_00_e0_20_edge5_hold32",
        lookback=80,
        min_abs_slope_bps=1.00,
        min_trend_efficiency=0.20,
        min_expected_edge_bps=5.0,
        expected_holding_bars=6,
        max_holding_period=32,
    ),
    KalmanTrendParameterSet(
        label="loose_l80_s0_25_e0_20_edge5_hold32",
        lookback=80,
        min_abs_slope_bps=0.25,
        min_trend_efficiency=0.20,
        min_expected_edge_bps=5.0,
        expected_holding_bars=6,
        max_holding_period=32,
    ),
    KalmanTrendParameterSet(
        label="short_hold_l80_s0_50_e0_20_edge5_hold16",
        lookback=80,
        min_abs_slope_bps=0.50,
        min_trend_efficiency=0.20,
        min_expected_edge_bps=5.0,
        expected_holding_bars=6,
        max_holding_period=16,
    ),
    KalmanTrendParameterSet(
        label="sensitive_l60_s0_50_e0_20_edge5_hold32",
        lookback=60,
        min_abs_slope_bps=0.50,
        min_trend_efficiency=0.20,
        min_expected_edge_bps=5.0,
        expected_holding_bars=6,
        max_holding_period=32,
    ),
)


@dataclass(frozen=True)
class KalmanTrendOptimizationCandidate:
    parameters: KalmanTrendParameterSet
    comparison_row: PortfolioStrategyComparisonRow

    @property
    def rank_key(self) -> tuple[float, int, float, float, float, float]:
        metrics = self.comparison_row.competition_metrics
        return (
            self.comparison_row.proxy_score,
            self.comparison_row.risk_discipline.score,
            metrics.return_pct,
            metrics.sharpe_15m,
            -metrics.max_drawdown_pct,
            -float(metrics.trade_count),
        )


@dataclass(frozen=True)
class KalmanTrendOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[KalmanTrendOptimizationCandidate, ...]

    @property
    def best(self) -> KalmanTrendOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_kalman_trend_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        KalmanTrendParameterSet, ...
    ] = DEFAULT_KALMAN_TREND_PARAMETER_SETS,
) -> KalmanTrendOptimizationResult:
    if not parameter_sets:
        raise ValueError("Kalman trend optimizer needs at least one parameter set")

    candidates: list[KalmanTrendOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("kalman_trend",),
            symbols=symbols,
        )
        if comparison.best is None:
            continue
        selected_symbols = comparison.symbols
        candidates.append(
            KalmanTrendOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return KalmanTrendOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_kalman_trend_optimization_csv(
    result: KalmanTrendOptimizationResult,
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
                "min_abs_slope_bps",
                "min_trend_efficiency",
                "min_expected_edge_bps",
                "expected_holding_bars",
                "max_holding_period",
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
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            parameters = candidate.parameters
            row = candidate.comparison_row
            metrics = row.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "label": parameters.label,
                    "symbols": " ".join(result.symbols),
                    "lookback": parameters.lookback,
                    "min_abs_slope_bps": parameters.min_abs_slope_bps,
                    "min_trend_efficiency": parameters.min_trend_efficiency,
                    "min_expected_edge_bps": parameters.min_expected_edge_bps,
                    "expected_holding_bars": parameters.expected_holding_bars,
                    "max_holding_period": parameters.max_holding_period,
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
                }
            )


def _config_with_parameters(
    config: AppConfig,
    parameters: KalmanTrendParameterSet,
) -> AppConfig:
    kalman_trend = replace(
        config.kalman_trend,
        lookback=parameters.lookback,
        min_abs_slope_bps=parameters.min_abs_slope_bps,
        min_trend_efficiency=parameters.min_trend_efficiency,
        min_expected_edge_bps=parameters.min_expected_edge_bps,
        expected_holding_bars=parameters.expected_holding_bars,
        max_holding_period=parameters.max_holding_period,
    )
    return replace(config, kalman_trend=kalman_trend)
