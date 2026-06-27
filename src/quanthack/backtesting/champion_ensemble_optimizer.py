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
class ChampionEnsembleParameterSet:
    label: str
    kalman_trend_weight: float
    asset_adaptive_dual_squeeze_weight: float
    dual_squeeze_weight: float
    trend_pullback_weight: float
    entry_score: float
    strong_lead_score: float
    conflict_penalty: float
    fixing_reversal_weight: float = 0.0
    macd_momentum_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Champion ensemble parameter label is required")
        for name, value in (
            ("kalman_trend_weight", self.kalman_trend_weight),
            (
                "asset_adaptive_dual_squeeze_weight",
                self.asset_adaptive_dual_squeeze_weight,
            ),
            ("dual_squeeze_weight", self.dual_squeeze_weight),
            ("trend_pullback_weight", self.trend_pullback_weight),
            ("fixing_reversal_weight", self.fixing_reversal_weight),
            ("macd_momentum_weight", self.macd_momentum_weight),
            ("strong_lead_score", self.strong_lead_score),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        total_weight = (
            self.kalman_trend_weight
            + self.asset_adaptive_dual_squeeze_weight
            + self.dual_squeeze_weight
            + self.trend_pullback_weight
            + self.fixing_reversal_weight
            + self.macd_momentum_weight
        )
        if total_weight <= 0:
            raise ValueError("at least one champion ensemble weight must be positive")
        if self.entry_score <= 0:
            raise ValueError("entry_score must be positive")
        if self.strong_lead_score > self.entry_score:
            raise ValueError("strong_lead_score cannot exceed entry_score")
        if not 0 <= self.conflict_penalty <= 1:
            raise ValueError("conflict_penalty must be between 0 and 1")


DEFAULT_CHAMPION_ENSEMBLE_PARAMETER_SETS: tuple[ChampionEnsembleParameterSet, ...] = (
    ChampionEnsembleParameterSet(
        label="strict_k70_a30_e50_s50_c70",
        kalman_trend_weight=0.70,
        asset_adaptive_dual_squeeze_weight=0.30,
        dual_squeeze_weight=0.0,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.50,
        conflict_penalty=0.70,
    ),
    ChampionEnsembleParameterSet(
        label="loose_k60_a25_d10_t05_e50_s25_c65",
        kalman_trend_weight=0.60,
        asset_adaptive_dual_squeeze_weight=0.25,
        dual_squeeze_weight=0.10,
        trend_pullback_weight=0.05,
        entry_score=0.50,
        strong_lead_score=0.25,
        conflict_penalty=0.65,
    ),
    ChampionEnsembleParameterSet(
        label="kalman_only_e50_s50_c70",
        kalman_trend_weight=1.00,
        asset_adaptive_dual_squeeze_weight=0.0,
        dual_squeeze_weight=0.0,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.50,
        conflict_penalty=0.70,
    ),
    ChampionEnsembleParameterSet(
        label="confirm_with_dual_k70_a20_d10_e50_s50_c75",
        kalman_trend_weight=0.70,
        asset_adaptive_dual_squeeze_weight=0.20,
        dual_squeeze_weight=0.10,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.50,
        conflict_penalty=0.75,
    ),
    ChampionEnsembleParameterSet(
        label="confirm_with_pullback_k70_a20_t10_e50_s50_c75",
        kalman_trend_weight=0.70,
        asset_adaptive_dual_squeeze_weight=0.20,
        dual_squeeze_weight=0.0,
        trend_pullback_weight=0.10,
        entry_score=0.50,
        strong_lead_score=0.50,
        conflict_penalty=0.75,
    ),
    ChampionEnsembleParameterSet(
        label="fixing_diversifier_k65_a25_f10_e50_s50_c75",
        kalman_trend_weight=0.65,
        asset_adaptive_dual_squeeze_weight=0.25,
        dual_squeeze_weight=0.0,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.50,
        conflict_penalty=0.75,
        fixing_reversal_weight=0.10,
    ),
    ChampionEnsembleParameterSet(
        label="fixing_diversifier_k60_a25_d05_f10_e50_s35_c75",
        kalman_trend_weight=0.60,
        asset_adaptive_dual_squeeze_weight=0.25,
        dual_squeeze_weight=0.05,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.35,
        conflict_penalty=0.75,
        fixing_reversal_weight=0.10,
    ),
    ChampionEnsembleParameterSet(
        label="macd_diversifier_k65_a25_m10_e50_s50_c75",
        kalman_trend_weight=0.65,
        asset_adaptive_dual_squeeze_weight=0.25,
        dual_squeeze_weight=0.0,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.50,
        conflict_penalty=0.75,
        macd_momentum_weight=0.10,
    ),
    ChampionEnsembleParameterSet(
        label="macd_fixing_mix_k60_a20_f10_m10_e50_s35_c75",
        kalman_trend_weight=0.60,
        asset_adaptive_dual_squeeze_weight=0.20,
        dual_squeeze_weight=0.0,
        trend_pullback_weight=0.0,
        entry_score=0.50,
        strong_lead_score=0.35,
        conflict_penalty=0.75,
        fixing_reversal_weight=0.10,
        macd_momentum_weight=0.10,
    ),
)


@dataclass(frozen=True)
class ChampionEnsembleOptimizationCandidate:
    parameters: ChampionEnsembleParameterSet
    comparison_row: PortfolioStrategyComparisonRow
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None = None

    @property
    def rank_key(self) -> tuple[float, ...]:
        metrics = self.comparison_row.competition_metrics
        if self.walk_forward is not None:
            active_quality = _coverage_adjusted_active_score(self.walk_forward)
            return (
                active_quality,
                self.walk_forward.active_positive_fold_fraction,
                self.walk_forward.median_active_test_return_pct,
                self.walk_forward.non_negative_fold_fraction,
                self.walk_forward.active_fold_fraction,
                self.walk_forward.positive_fold_fraction,
                self.walk_forward.median_test_return_pct,
                self.walk_forward.median_test_sharpe_15m,
                -self.walk_forward.losing_fold_fraction,
                self.comparison_row.proxy_score,
                self.comparison_row.risk_discipline.score,
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
class ChampionEnsembleOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[ChampionEnsembleOptimizationCandidate, ...]

    @property
    def best(self) -> ChampionEnsembleOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_champion_ensemble_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        ChampionEnsembleParameterSet, ...
    ] = DEFAULT_CHAMPION_ENSEMBLE_PARAMETER_SETS,
    include_walk_forward: bool = False,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> ChampionEnsembleOptimizationResult:
    if not parameter_sets:
        raise ValueError("Champion ensemble optimizer needs at least one parameter set")

    candidates: list[ChampionEnsembleOptimizationCandidate] = []
    selected_symbols: tuple[str, ...] | None = None
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        comparison = compare_portfolio_strategies(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_names=("champion_ensemble",),
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
                strategy_name="champion_ensemble",
                symbols=comparison.symbols,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
            if include_walk_forward
            else None
        )
        candidates.append(
            ChampionEnsembleOptimizationCandidate(
                parameters=parameters,
                comparison_row=comparison.best,
                walk_forward=walk_forward,
            )
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return ChampionEnsembleOptimizationResult(
        symbols=selected_symbols or (),
        candidates=tuple(candidates),
    )


def write_champion_ensemble_optimization_csv(
    result: ChampionEnsembleOptimizationResult,
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
                "kalman_trend_weight",
                "asset_adaptive_dual_squeeze_weight",
                "dual_squeeze_weight",
                "trend_pullback_weight",
                "fixing_reversal_weight",
                "macd_momentum_weight",
                "entry_score",
                "strong_lead_score",
                "conflict_penalty",
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
                "wf_median_test_sharpe_15m",
                "wf_worst_test_drawdown_pct",
                "wf_total_evaluation_fills",
                "wf_largest_positive_fold_contribution",
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
                    "kalman_trend_weight": parameters.kalman_trend_weight,
                    "asset_adaptive_dual_squeeze_weight": (
                        parameters.asset_adaptive_dual_squeeze_weight
                    ),
                    "dual_squeeze_weight": parameters.dual_squeeze_weight,
                    "trend_pullback_weight": parameters.trend_pullback_weight,
                    "fixing_reversal_weight": parameters.fixing_reversal_weight,
                    "macd_momentum_weight": parameters.macd_momentum_weight,
                    "entry_score": parameters.entry_score,
                    "strong_lead_score": parameters.strong_lead_score,
                    "conflict_penalty": parameters.conflict_penalty,
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
                    "wf_median_test_sharpe_15m": (
                        "" if walk_forward is None else walk_forward.median_test_sharpe_15m
                    ),
                    "wf_worst_test_drawdown_pct": (
                        "" if walk_forward is None else walk_forward.worst_test_drawdown_pct
                    ),
                    "wf_total_evaluation_fills": (
                        "" if walk_forward is None else walk_forward.total_evaluation_fills
                    ),
                    "wf_largest_positive_fold_contribution": (
                        ""
                        if walk_forward is None
                        else walk_forward.largest_positive_fold_contribution
                    ),
                }
            )


def _config_with_parameters(
    config: AppConfig,
    parameters: ChampionEnsembleParameterSet,
) -> AppConfig:
    champion_ensemble = replace(
        config.champion_ensemble,
        kalman_trend_weight=parameters.kalman_trend_weight,
        asset_adaptive_dual_squeeze_weight=(
            parameters.asset_adaptive_dual_squeeze_weight
        ),
        dual_squeeze_weight=parameters.dual_squeeze_weight,
        trend_pullback_weight=parameters.trend_pullback_weight,
        fixing_reversal_weight=parameters.fixing_reversal_weight,
        macd_momentum_weight=parameters.macd_momentum_weight,
        entry_score=parameters.entry_score,
        strong_lead_score=parameters.strong_lead_score,
        conflict_penalty=parameters.conflict_penalty,
    )
    return replace(config, champion_ensemble=champion_ensemble)


def _coverage_adjusted_active_score(
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    *,
    target_active_fold_fraction: float = 0.35,
) -> float:
    if target_active_fold_fraction <= 0:
        return walk_forward.active_positive_fold_fraction
    coverage = min(
        walk_forward.active_fold_fraction / target_active_fold_fraction,
        1.0,
    )
    return walk_forward.active_positive_fold_fraction * coverage
