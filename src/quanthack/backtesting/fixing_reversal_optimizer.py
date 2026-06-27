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
class FixingReversalParameterSet:
    label: str
    pre_fix_lookback: int
    min_pre_fix_move_bps: float
    min_reversal_confirmation_bps: float
    min_pre_fix_efficiency: float
    max_holding_period: int
    allowed_utc_hours: tuple[int, ...]
    min_expected_edge_bps: float = 3.0
    max_pre_fix_move_bps: float = 80.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("fixing reversal parameter label is required")
        if self.pre_fix_lookback < 2:
            raise ValueError("pre_fix_lookback must be at least 2")
        if self.min_pre_fix_move_bps < 0:
            raise ValueError("min_pre_fix_move_bps cannot be negative")
        if self.max_pre_fix_move_bps <= self.min_pre_fix_move_bps:
            raise ValueError("max_pre_fix_move_bps must exceed min_pre_fix_move_bps")
        if self.min_reversal_confirmation_bps < 0:
            raise ValueError("min_reversal_confirmation_bps cannot be negative")
        if not 0 <= self.min_pre_fix_efficiency <= 1:
            raise ValueError("min_pre_fix_efficiency must be between 0 and 1")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be at least 1")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        normalized_hours = tuple(int(hour) for hour in self.allowed_utc_hours)
        if not normalized_hours:
            raise ValueError("allowed_utc_hours cannot be empty")
        if any(hour < 0 or hour > 23 for hour in normalized_hours):
            raise ValueError("allowed_utc_hours must contain hours between 0 and 23")
        object.__setattr__(self, "allowed_utc_hours", normalized_hours)


DEFAULT_FIXING_REVERSAL_PARAMETER_SETS: tuple[FixingReversalParameterSet, ...] = (
    FixingReversalParameterSet(
        label="best_grid_h14_l4_m8_c1_5_e0_35_hold4",
        pre_fix_lookback=4,
        min_pre_fix_move_bps=8.0,
        min_reversal_confirmation_bps=1.5,
        min_pre_fix_efficiency=0.35,
        max_holding_period=4,
        allowed_utc_hours=(14,),
    ),
    FixingReversalParameterSet(
        label="strict_h14_l4_m12_c1_5_e0_35_hold4",
        pre_fix_lookback=4,
        min_pre_fix_move_bps=12.0,
        min_reversal_confirmation_bps=1.5,
        min_pre_fix_efficiency=0.35,
        max_holding_period=4,
        allowed_utc_hours=(14,),
    ),
    FixingReversalParameterSet(
        label="high_bar_h14_l8_m20_c1_5_e0_60_hold4",
        pre_fix_lookback=8,
        min_pre_fix_move_bps=20.0,
        min_reversal_confirmation_bps=1.5,
        min_pre_fix_efficiency=0.60,
        max_holding_period=4,
        allowed_utc_hours=(14,),
    ),
    FixingReversalParameterSet(
        label="london_fix_h15_l4_m8_c1_5_e0_35_hold4",
        pre_fix_lookback=4,
        min_pre_fix_move_bps=8.0,
        min_reversal_confirmation_bps=1.5,
        min_pre_fix_efficiency=0.35,
        max_holding_period=4,
        allowed_utc_hours=(15,),
    ),
    FixingReversalParameterSet(
        label="late_h16_l4_m12_c1_5_e0_60_hold4",
        pre_fix_lookback=4,
        min_pre_fix_move_bps=12.0,
        min_reversal_confirmation_bps=1.5,
        min_pre_fix_efficiency=0.60,
        max_holding_period=4,
        allowed_utc_hours=(16,),
    ),
)


@dataclass(frozen=True)
class FixingReversalOptimizationCandidate:
    parameters: FixingReversalParameterSet
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
class FixingReversalOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[FixingReversalOptimizationCandidate, ...]

    @property
    def best(self) -> FixingReversalOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_fixing_reversal_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        FixingReversalParameterSet, ...
    ] = DEFAULT_FIXING_REVERSAL_PARAMETER_SETS,
) -> FixingReversalOptimizationResult:
    if not parameter_sets:
        raise ValueError("fixing reversal optimizer needs at least one parameter set")

    candidates: list[FixingReversalOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("fixing_reversal",),
            symbols=symbols,
        )
        if comparison.best is None:
            continue
        selected_symbols = comparison.symbols
        candidates.append(
            FixingReversalOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return FixingReversalOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_fixing_reversal_optimization_csv(
    result: FixingReversalOptimizationResult,
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
                "pre_fix_lookback",
                "min_pre_fix_move_bps",
                "min_reversal_confirmation_bps",
                "min_pre_fix_efficiency",
                "max_holding_period",
                "allowed_utc_hours",
                "min_expected_edge_bps",
                "max_pre_fix_move_bps",
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
                    "pre_fix_lookback": parameters.pre_fix_lookback,
                    "min_pre_fix_move_bps": parameters.min_pre_fix_move_bps,
                    "min_reversal_confirmation_bps": (
                        parameters.min_reversal_confirmation_bps
                    ),
                    "min_pre_fix_efficiency": parameters.min_pre_fix_efficiency,
                    "max_holding_period": parameters.max_holding_period,
                    "allowed_utc_hours": _hours_text(parameters.allowed_utc_hours),
                    "min_expected_edge_bps": parameters.min_expected_edge_bps,
                    "max_pre_fix_move_bps": parameters.max_pre_fix_move_bps,
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
    parameters: FixingReversalParameterSet,
) -> AppConfig:
    fixing_reversal = replace(
        config.fixing_reversal,
        pre_fix_lookback=parameters.pre_fix_lookback,
        min_pre_fix_move_bps=parameters.min_pre_fix_move_bps,
        min_reversal_confirmation_bps=parameters.min_reversal_confirmation_bps,
        min_pre_fix_efficiency=parameters.min_pre_fix_efficiency,
        max_holding_period=parameters.max_holding_period,
        min_expected_edge_bps=parameters.min_expected_edge_bps,
        max_pre_fix_move_bps=parameters.max_pre_fix_move_bps,
        forex_allowed_utc_hours=parameters.allowed_utc_hours,
        metal_allowed_utc_hours=parameters.allowed_utc_hours,
        crypto_allowed_utc_hours=(),
    )
    return replace(config, fixing_reversal=fixing_reversal)


def _hours_text(hours: tuple[int, ...]) -> str:
    return " ".join(str(hour) for hour in hours)
