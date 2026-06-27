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
class SessionMomentumParameterSet:
    label: str
    lookback: int
    threshold_bps: float
    exit_threshold_bps: float
    min_trend_efficiency: float
    allowed_utc_hours: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("session momentum parameter label is required")
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.threshold_bps <= 0:
            raise ValueError("threshold_bps must be positive")
        if self.exit_threshold_bps < 0:
            raise ValueError("exit_threshold_bps cannot be negative")
        if self.exit_threshold_bps > self.threshold_bps:
            raise ValueError("exit_threshold_bps cannot exceed threshold_bps")
        if not 0 <= self.min_trend_efficiency <= 1:
            raise ValueError("min_trend_efficiency must be between 0 and 1")
        if not self.allowed_utc_hours:
            raise ValueError("allowed_utc_hours cannot be empty")
        if any(hour < 0 or hour > 23 for hour in self.allowed_utc_hours):
            raise ValueError("allowed_utc_hours must be between 0 and 23")


DEFAULT_SESSION_MOMENTUM_PARAMETER_SETS: tuple[SessionMomentumParameterSet, ...] = (
    SessionMomentumParameterSet("late_17_21_l5_t8_eff40", 5, 8.0, 3.0, 0.40, (17, 18, 19, 20, 21)),
    SessionMomentumParameterSet("late_18_21_l5_t8_eff40", 5, 8.0, 3.0, 0.40, (18, 19, 20, 21)),
    SessionMomentumParameterSet("late_18_20_l5_t8_eff40", 5, 8.0, 3.0, 0.40, (18, 19, 20)),
    SessionMomentumParameterSet("late_17_21_l6_t10_eff50", 6, 10.0, 4.0, 0.50, (17, 18, 19, 20, 21)),
    SessionMomentumParameterSet("ny_13_16_l5_t8_eff40", 5, 8.0, 3.0, 0.40, (13, 14, 15, 16)),
    SessionMomentumParameterSet("london_8_11_l5_t8_eff40", 5, 8.0, 3.0, 0.40, (8, 9, 10, 11)),
)


@dataclass(frozen=True)
class SessionMomentumOptimizationCandidate:
    parameters: SessionMomentumParameterSet
    comparison_row: PortfolioStrategyComparisonRow
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None

    @property
    def rank_key(self) -> tuple[float, ...]:
        metrics = self.comparison_row.competition_metrics
        if self.walk_forward is not None:
            return (
                _coverage_adjusted_active_score(self.walk_forward),
                self.walk_forward.non_negative_fold_fraction,
                self.walk_forward.active_positive_fold_fraction,
                self.walk_forward.median_active_test_return_pct,
                self.walk_forward.positive_fold_fraction,
                -self.walk_forward.worst_test_drawdown_pct,
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
class SessionMomentumOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[SessionMomentumOptimizationCandidate, ...]

    @property
    def best(self) -> SessionMomentumOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_session_momentum_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        SessionMomentumParameterSet, ...
    ] = DEFAULT_SESSION_MOMENTUM_PARAMETER_SETS,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> SessionMomentumOptimizationResult:
    if not parameter_sets:
        raise ValueError("session momentum optimizer needs at least one parameter set")

    candidates: list[SessionMomentumOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("session_momentum",),
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
                strategy_name="session_momentum",
                symbols=comparison.symbols,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            SessionMomentumOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return SessionMomentumOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_session_momentum_optimization_csv(
    result: SessionMomentumOptimizationResult,
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
                "threshold_bps",
                "exit_threshold_bps",
                "min_trend_efficiency",
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
                    "threshold_bps": parameters.threshold_bps,
                    "exit_threshold_bps": parameters.exit_threshold_bps,
                    "min_trend_efficiency": parameters.min_trend_efficiency,
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
    parameters: SessionMomentumParameterSet,
) -> AppConfig:
    session_momentum = replace(
        config.session_momentum,
        lookback=parameters.lookback,
        threshold_bps=parameters.threshold_bps,
        exit_threshold_bps=parameters.exit_threshold_bps,
        min_trend_efficiency=parameters.min_trend_efficiency,
        forex_allowed_utc_hours=parameters.allowed_utc_hours,
        metal_allowed_utc_hours=parameters.allowed_utc_hours,
    )
    return replace(config, session_momentum=session_momentum)


def _coverage_adjusted_active_score(
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
) -> float:
    if walk_forward.active_fold_fraction <= 0:
        return 0.0
    return (
        walk_forward.active_positive_fold_fraction
        * walk_forward.non_negative_fold_fraction
        * max(walk_forward.median_active_test_return_pct, 0.0)
    )


def _hours_text(hours: tuple[int, ...]) -> str:
    return "|".join(str(hour) for hour in hours)
