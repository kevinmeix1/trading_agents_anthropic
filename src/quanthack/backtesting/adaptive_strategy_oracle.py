from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.adaptive_strategy_policy_sweep import (
    _build_policy_cache,
    _run_cached_policy,
)
from quanthack.backtesting.adaptive_strategy_selector import (
    CASH_FALLBACK_LABEL,
    AdaptiveStrategySelectionResult,
    _cash_evaluation,
)
from quanthack.backtesting.competition_score import CompetitionMetrics
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class AdaptiveStrategyOracleCandidate:
    fold_index: int
    strategy_name: str
    train_rank: int
    selected_by_policy: bool
    oracle_best: bool
    train_return_pct: float
    train_drawdown_adjusted_return_pct: float
    train_max_drawdown_pct: float
    train_sharpe_15m: float
    train_fills: int
    test_return_pct: float
    test_max_drawdown_pct: float
    test_sharpe_15m: float
    test_fills: int


@dataclass(frozen=True)
class AdaptiveStrategyOracleFold:
    fold_index: int
    test_start: str
    test_end: str
    selected_strategy: str
    oracle_strategy: str
    selected_return_pct: float
    oracle_return_pct: float
    regret_pct: float
    selected_max_drawdown_pct: float
    oracle_max_drawdown_pct: float
    selected_fills: int
    oracle_fills: int
    selected_was_oracle: bool
    selected_was_negative: bool
    oracle_was_cash: bool


@dataclass(frozen=True)
class AdaptiveStrategyOracleDiagnostic:
    policy_result: AdaptiveStrategySelectionResult
    folds: tuple[AdaptiveStrategyOracleFold, ...]
    candidates: tuple[AdaptiveStrategyOracleCandidate, ...]
    include_cash_oracle: bool

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def selected_was_oracle_fraction(self) -> float:
        if not self.folds:
            return 0.0
        return len([fold for fold in self.folds if fold.selected_was_oracle]) / len(
            self.folds
        )

    @property
    def total_regret_pct(self) -> float:
        return sum(fold.regret_pct for fold in self.folds)

    @property
    def average_regret_pct(self) -> float:
        if not self.folds:
            return 0.0
        return self.total_regret_pct / len(self.folds)

    @property
    def regret_folds(self) -> tuple[AdaptiveStrategyOracleFold, ...]:
        return tuple(fold for fold in self.folds if fold.regret_pct > 1e-12)

    @property
    def negative_selected_folds(self) -> int:
        return len([fold for fold in self.folds if fold.selected_was_negative])

    @property
    def cash_oracle_folds(self) -> int:
        return len([fold for fold in self.folds if fold.oracle_was_cash])


def build_adaptive_strategy_oracle_diagnostic(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...],
    symbols: tuple[str, ...] | None = None,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    loss_cooldown_folds: int = 1,
    min_train_fills: int = 0,
    min_train_drawdown_adjusted_return_pct: float | None = None,
    train_fill_penalty_pct: float = 0.0,
    train_stability_splits: int = 0,
    prefer_train_stability: bool = False,
    transition_risk_multiplier: float = 1.0,
    allow_cash_fallback: bool = False,
    include_cash_oracle: bool = True,
) -> AdaptiveStrategyOracleDiagnostic:
    cache = _build_policy_cache(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=strategy_names,
        symbols=symbols,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        stability_splits_values=(train_stability_splits,),
        transition_risk_multiplier_values=(transition_risk_multiplier,),
    )
    policy_result = _run_cached_policy(
        cache=cache,
        loss_cooldown_folds=loss_cooldown_folds,
        min_train_fills=min_train_fills,
        min_train_drawdown_adjusted_return_pct=(
            min_train_drawdown_adjusted_return_pct
        ),
        train_fill_penalty_pct=train_fill_penalty_pct,
        train_stability_splits=train_stability_splits,
        prefer_train_stability=prefer_train_stability,
        transition_risk_multiplier=transition_risk_multiplier,
        allow_cash_fallback=allow_cash_fallback,
    )

    diagnostic_folds: list[AdaptiveStrategyOracleFold] = []
    diagnostic_candidates: list[AdaptiveStrategyOracleCandidate] = []
    for cached_fold, selected_fold in zip(cache.folds, policy_result.folds):
        train_scores = tuple(
            sorted(
                cached_fold.train_scores_by_stability_splits[train_stability_splits],
                key=lambda score: score.rank_key_with_preferences(
                    fill_penalty_pct=train_fill_penalty_pct,
                    prefer_train_stability=prefer_train_stability,
                ),
                reverse=True,
            )
        )
        candidate_metrics: dict[str, CompetitionMetrics] = {}
        candidate_fills: dict[str, int] = {}
        for candidate in cache.candidates:
            evaluation, _full_run_fills = (
                cached_fold.evaluation_by_candidate_and_multiplier[
                    (candidate.label, transition_risk_multiplier)
                    if (
                        selected_fold.evaluation_risk_multiplier
                        == transition_risk_multiplier
                    )
                    else (candidate.label, 1.0)
                ]
            )
            candidate_metrics[candidate.label] = evaluation.competition_metrics
            candidate_fills[candidate.label] = len(evaluation.fills)
        if include_cash_oracle:
            cash_evaluation = _cash_evaluation(
                cached_fold.test_timestamps,
                starting_equity=cache.starting_equity,
            )
            candidate_metrics[CASH_FALLBACK_LABEL] = cash_evaluation.competition_metrics
            candidate_fills[CASH_FALLBACK_LABEL] = 0

        oracle_strategy = max(
            candidate_metrics,
            key=lambda label: (
                candidate_metrics[label].return_pct,
                -candidate_metrics[label].max_drawdown_pct,
                -candidate_fills[label],
            ),
        )
        oracle_metrics = candidate_metrics[oracle_strategy]
        selected_metrics = selected_fold.metrics
        regret_pct = max(0.0, oracle_metrics.return_pct - selected_metrics.return_pct)
        diagnostic_folds.append(
            AdaptiveStrategyOracleFold(
                fold_index=selected_fold.fold_index,
                test_start=selected_fold.test_start,
                test_end=selected_fold.test_end,
                selected_strategy=selected_fold.selected_strategy,
                oracle_strategy=oracle_strategy,
                selected_return_pct=selected_metrics.return_pct,
                oracle_return_pct=oracle_metrics.return_pct,
                regret_pct=regret_pct,
                selected_max_drawdown_pct=selected_metrics.max_drawdown_pct,
                oracle_max_drawdown_pct=oracle_metrics.max_drawdown_pct,
                selected_fills=len(selected_fold.evaluation.fills),
                oracle_fills=candidate_fills[oracle_strategy],
                selected_was_oracle=selected_fold.selected_strategy == oracle_strategy,
                selected_was_negative=selected_metrics.return_pct < 0.0,
                oracle_was_cash=oracle_strategy == CASH_FALLBACK_LABEL,
            )
        )

        train_rank_by_strategy = {
            score.strategy_name: rank for rank, score in enumerate(train_scores, start=1)
        }
        train_score_by_strategy = {
            score.strategy_name: score for score in train_scores
        }
        for label, metrics in candidate_metrics.items():
            train_score = train_score_by_strategy.get(label)
            diagnostic_candidates.append(
                AdaptiveStrategyOracleCandidate(
                    fold_index=selected_fold.fold_index,
                    strategy_name=label,
                    train_rank=train_rank_by_strategy.get(label, 0),
                    selected_by_policy=selected_fold.selected_strategy == label,
                    oracle_best=oracle_strategy == label,
                    train_return_pct=train_score.return_pct if train_score else 0.0,
                    train_drawdown_adjusted_return_pct=(
                        train_score.drawdown_adjusted_return_pct if train_score else 0.0
                    ),
                    train_max_drawdown_pct=(
                        train_score.max_drawdown_pct if train_score else 0.0
                    ),
                    train_sharpe_15m=train_score.sharpe_15m if train_score else 0.0,
                    train_fills=train_score.fills if train_score else 0,
                    test_return_pct=metrics.return_pct,
                    test_max_drawdown_pct=metrics.max_drawdown_pct,
                    test_sharpe_15m=metrics.sharpe_15m,
                    test_fills=candidate_fills[label],
                )
            )

    return AdaptiveStrategyOracleDiagnostic(
        policy_result=policy_result,
        folds=tuple(diagnostic_folds),
        candidates=tuple(diagnostic_candidates),
        include_cash_oracle=include_cash_oracle,
    )


def write_adaptive_strategy_oracle_summary_csv(
    diagnostic: AdaptiveStrategyOracleDiagnostic,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "folds",
                "include_cash_oracle",
                "selected_was_oracle_fraction",
                "total_regret_pct",
                "average_regret_pct",
                "regret_folds",
                "negative_selected_folds",
                "cash_oracle_folds",
                "policy_compounded_test_return_pct",
                "policy_active_positive_fold_fraction",
                "policy_non_negative_fold_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "folds": diagnostic.fold_count,
                "include_cash_oracle": (
                    "yes" if diagnostic.include_cash_oracle else "no"
                ),
                "selected_was_oracle_fraction": (
                    diagnostic.selected_was_oracle_fraction
                ),
                "total_regret_pct": diagnostic.total_regret_pct,
                "average_regret_pct": diagnostic.average_regret_pct,
                "regret_folds": len(diagnostic.regret_folds),
                "negative_selected_folds": diagnostic.negative_selected_folds,
                "cash_oracle_folds": diagnostic.cash_oracle_folds,
                "policy_compounded_test_return_pct": (
                    diagnostic.policy_result.compounded_test_return_pct
                ),
                "policy_active_positive_fold_fraction": (
                    diagnostic.policy_result.active_positive_fold_fraction
                ),
                "policy_non_negative_fold_fraction": (
                    diagnostic.policy_result.non_negative_fold_fraction
                ),
            }
        )


def write_adaptive_strategy_oracle_folds_csv(
    diagnostic: AdaptiveStrategyOracleDiagnostic,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "test_start",
                "test_end",
                "selected_strategy",
                "oracle_strategy",
                "selected_return_pct",
                "oracle_return_pct",
                "regret_pct",
                "selected_max_drawdown_pct",
                "oracle_max_drawdown_pct",
                "selected_fills",
                "oracle_fills",
                "selected_was_oracle",
                "selected_was_negative",
                "oracle_was_cash",
            ],
        )
        writer.writeheader()
        for fold in diagnostic.folds:
            writer.writerow(
                {
                    "fold": fold.fold_index,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "selected_strategy": fold.selected_strategy,
                    "oracle_strategy": fold.oracle_strategy,
                    "selected_return_pct": fold.selected_return_pct,
                    "oracle_return_pct": fold.oracle_return_pct,
                    "regret_pct": fold.regret_pct,
                    "selected_max_drawdown_pct": fold.selected_max_drawdown_pct,
                    "oracle_max_drawdown_pct": fold.oracle_max_drawdown_pct,
                    "selected_fills": fold.selected_fills,
                    "oracle_fills": fold.oracle_fills,
                    "selected_was_oracle": (
                        "yes" if fold.selected_was_oracle else "no"
                    ),
                    "selected_was_negative": (
                        "yes" if fold.selected_was_negative else "no"
                    ),
                    "oracle_was_cash": "yes" if fold.oracle_was_cash else "no",
                }
            )


def write_adaptive_strategy_oracle_candidates_csv(
    diagnostic: AdaptiveStrategyOracleDiagnostic,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "strategy",
                "train_rank",
                "selected_by_policy",
                "oracle_best",
                "train_return_pct",
                "train_drawdown_adjusted_return_pct",
                "train_max_drawdown_pct",
                "train_sharpe_15m",
                "train_fills",
                "test_return_pct",
                "test_max_drawdown_pct",
                "test_sharpe_15m",
                "test_fills",
            ],
        )
        writer.writeheader()
        for candidate in diagnostic.candidates:
            writer.writerow(
                {
                    "fold": candidate.fold_index,
                    "strategy": candidate.strategy_name,
                    "train_rank": candidate.train_rank,
                    "selected_by_policy": (
                        "yes" if candidate.selected_by_policy else "no"
                    ),
                    "oracle_best": "yes" if candidate.oracle_best else "no",
                    "train_return_pct": candidate.train_return_pct,
                    "train_drawdown_adjusted_return_pct": (
                        candidate.train_drawdown_adjusted_return_pct
                    ),
                    "train_max_drawdown_pct": candidate.train_max_drawdown_pct,
                    "train_sharpe_15m": candidate.train_sharpe_15m,
                    "train_fills": candidate.train_fills,
                    "test_return_pct": candidate.test_return_pct,
                    "test_max_drawdown_pct": candidate.test_max_drawdown_pct,
                    "test_sharpe_15m": candidate.test_sharpe_15m,
                    "test_fills": candidate.test_fills,
                }
            )
