from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.adaptive_strategy_selector import (
    CASH_FALLBACK_LABEL,
    RETURN_EPSILON,
    AdaptiveStrategyCandidate,
    AdaptiveStrategySelectionDecision,
    AdaptiveStrategySelectionFold,
    AdaptiveStrategySelectionResult,
    AdaptiveStrategyTrainScore,
    _candidate_by_label,
    _cash_evaluation,
    _cash_strategy_map,
    _cash_train_score,
    _common_timestamps,
    _config_with_risk_multiplier,
    _passes_train_gate,
    _run_candidate,
    _score_training_candidate,
    _selected_candidates,
    _selected_symbols,
    _slice_prices,
    _slice_quotes,
    _validate_window_sizes,
    build_adaptive_strategy_promotion_audit,
)
from quanthack.backtesting.warmup import (
    WarmupPortfolioEvaluation,
    evaluate_portfolio_after_warmup,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory

DEFAULT_LOSS_COOLDOWNS = (1,)
DEFAULT_MIN_TRAIN_ADJUSTED_RETURNS: tuple[float | None, ...] = (None,)
DEFAULT_TRAIN_FILL_PENALTIES = (0.0,)
DEFAULT_TRANSITION_RISK_MULTIPLIERS = (1.0,)
DEFAULT_CASH_FALLBACK_VALUES = (False,)
DEFAULT_TRAIN_STABILITY_SETTINGS = ((0, False),)


@dataclass(frozen=True)
class AdaptiveStrategyPolicySweepCandidate:
    result: AdaptiveStrategySelectionResult
    decision: AdaptiveStrategySelectionDecision
    selector_score: float

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            _promotion_rank(self.decision.status),
            self.selector_score,
            self.result.active_positive_fold_fraction,
            self.result.non_negative_fold_fraction,
            self.result.compounded_test_return_pct,
            self.result.median_active_test_return_pct,
            -self.result.worst_test_drawdown_pct,
            self.result.average_risk_discipline_score,
            -_cash_fallback_rank(self.result.allow_cash_fallback),
            -float(self.result.loss_cooldown_folds),
        )

    @property
    def selection_counts_text(self) -> str:
        return " ".join(
            f"{count.strategy_name}={count.folds}"
            for count in self.result.selection_counts
        )


@dataclass(frozen=True)
class AdaptiveStrategyPolicySweepResult:
    candidates: tuple[AdaptiveStrategyPolicySweepCandidate, ...]

    @property
    def best(self) -> AdaptiveStrategyPolicySweepCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


@dataclass(frozen=True)
class _CachedPolicyFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_timestamps: tuple[datetime, ...]
    train_scores_by_stability_splits: dict[int, tuple[AdaptiveStrategyTrainScore, ...]]
    evaluation_by_candidate_and_multiplier: dict[
        tuple[str, float],
        tuple[WarmupPortfolioEvaluation, int],
    ]


@dataclass(frozen=True)
class _AdaptiveStrategyPolicyCache:
    strategy_names: tuple[str, ...]
    symbols: tuple[str, ...]
    candidates: tuple[AdaptiveStrategyCandidate, ...]
    folds: tuple[_CachedPolicyFold, ...]
    starting_equity: float


def sweep_adaptive_strategy_policies(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...],
    symbols: tuple[str, ...] | None = None,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    loss_cooldown_values: Sequence[int] = DEFAULT_LOSS_COOLDOWNS,
    min_train_adjusted_return_values: Sequence[
        float | None
    ] = DEFAULT_MIN_TRAIN_ADJUSTED_RETURNS,
    train_fill_penalty_values: Sequence[float] = DEFAULT_TRAIN_FILL_PENALTIES,
    transition_risk_multiplier_values: Sequence[
        float
    ] = DEFAULT_TRANSITION_RISK_MULTIPLIERS,
    cash_fallback_values: Sequence[bool] = DEFAULT_CASH_FALLBACK_VALUES,
    train_stability_settings: Sequence[
        tuple[int, bool]
    ] = DEFAULT_TRAIN_STABILITY_SETTINGS,
    min_train_fills_values: Sequence[int] = (0,),
) -> AdaptiveStrategyPolicySweepResult:
    if not strategy_names:
        raise ValueError("at least one strategy is required")
    loss_cooldowns = tuple(int(value) for value in loss_cooldown_values)
    if not loss_cooldowns:
        raise ValueError("at least one loss cooldown value is required")
    if any(value < 0 for value in loss_cooldowns):
        raise ValueError("loss cooldown values cannot be negative")

    min_train_returns = tuple(min_train_adjusted_return_values)
    if not min_train_returns:
        raise ValueError("at least one min train adjusted return value is required")

    fill_penalties = tuple(float(value) for value in train_fill_penalty_values)
    if not fill_penalties:
        raise ValueError("at least one train fill penalty value is required")
    if any(value < 0 for value in fill_penalties):
        raise ValueError("train fill penalties cannot be negative")

    transition_multipliers = tuple(
        float(value) for value in transition_risk_multiplier_values
    )
    if not transition_multipliers:
        raise ValueError("at least one transition risk multiplier is required")
    if any(value <= 0 or value > 1 for value in transition_multipliers):
        raise ValueError("transition risk multipliers must be in (0, 1]")

    cash_fallbacks = tuple(bool(value) for value in cash_fallback_values)
    if not cash_fallbacks:
        raise ValueError("at least one cash fallback value is required")

    stability_settings = tuple(train_stability_settings)
    if not stability_settings:
        raise ValueError("at least one train stability setting is required")
    for splits, prefer_stability in stability_settings:
        if splits < 0 or splits == 1:
            raise ValueError("train stability splits must be 0 or at least 2")
        if prefer_stability and splits == 0:
            raise ValueError("prefer train stability requires train stability splits")

    min_train_fill_candidates = tuple(int(value) for value in min_train_fills_values)
    if not min_train_fill_candidates:
        raise ValueError("at least one min train fills value is required")
    if any(value < 0 for value in min_train_fill_candidates):
        raise ValueError("min train fills values cannot be negative")

    cache = _build_policy_cache(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_names=strategy_names,
        symbols=symbols,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        stability_splits_values=tuple(
            sorted({splits for splits, _ in stability_settings})
        ),
        transition_risk_multiplier_values=transition_multipliers,
    )
    candidates: list[AdaptiveStrategyPolicySweepCandidate] = []
    for loss_cooldown in loss_cooldowns:
        for min_train_return in min_train_returns:
            for fill_penalty in fill_penalties:
                for transition_multiplier in transition_multipliers:
                    for allow_cash_fallback in cash_fallbacks:
                        for stability_splits, prefer_stability in stability_settings:
                            for min_train_fills in min_train_fill_candidates:
                                result = _run_cached_policy(
                                    cache=cache,
                                    loss_cooldown_folds=loss_cooldown,
                                    min_train_fills=min_train_fills,
                                    min_train_drawdown_adjusted_return_pct=(
                                        min_train_return
                                    ),
                                    train_fill_penalty_pct=fill_penalty,
                                    train_stability_splits=stability_splits,
                                    prefer_train_stability=prefer_stability,
                                    transition_risk_multiplier=transition_multiplier,
                                    allow_cash_fallback=allow_cash_fallback,
                                )
                                decision = build_adaptive_strategy_promotion_audit(
                                    result
                                ).decision
                                candidates.append(
                                    AdaptiveStrategyPolicySweepCandidate(
                                        result=result,
                                        decision=decision,
                                        selector_score=_selector_score(result),
                                    )
                                )
    return AdaptiveStrategyPolicySweepResult(
        candidates=tuple(
            sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        )
    )


def _build_policy_cache(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...],
    symbols: tuple[str, ...] | None,
    train_size: int,
    test_size: int,
    step_size: int,
    stability_splits_values: tuple[int, ...],
    transition_risk_multiplier_values: tuple[float, ...],
) -> _AdaptiveStrategyPolicyCache:
    _validate_window_sizes(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    selected_symbols = _selected_symbols(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
        strategy_names=strategy_names,
    )
    selected_candidates = _selected_candidates(
        strategy_names=strategy_names,
        candidate_maps=(),
        recipe_maps=(),
        symbols=selected_symbols,
    )
    timestamps = _common_timestamps(prices, quotes, selected_symbols)
    if len(timestamps) < train_size + test_size:
        raise ValueError(
            "not enough aligned timestamps for one adaptive strategy-selection fold"
        )

    evaluation_multipliers = tuple(
        sorted({1.0, *transition_risk_multiplier_values}, reverse=True)
    )
    cached_folds: list[_CachedPolicyFold] = []
    for fold_index, start in enumerate(
        range(0, len(timestamps) - train_size - test_size + 1, step_size),
        start=1,
    ):
        train_timestamps = timestamps[start : start + train_size]
        test_timestamps = timestamps[
            start + train_size : start + train_size + test_size
        ]
        combined_timestamps = train_timestamps + test_timestamps
        train_prices = _slice_prices(
            prices,
            symbols=selected_symbols,
            timestamps=train_timestamps,
        )
        train_quotes = _slice_quotes(
            quotes,
            symbols=selected_symbols,
            timestamps=train_timestamps,
        )
        train_scores_by_stability_splits = {
            splits: tuple(
                _score_training_candidate(
                    config=config,
                    prices=train_prices,
                    quotes=train_quotes,
                    candidate=candidate,
                    train_stability_splits=splits,
                )
                for candidate in selected_candidates
            )
            for splits in stability_splits_values
        }
        combined_prices = _slice_prices(
            prices,
            symbols=selected_symbols,
            timestamps=combined_timestamps,
        )
        combined_quotes = _slice_quotes(
            quotes,
            symbols=selected_symbols,
            timestamps=combined_timestamps,
        )
        evaluation_by_candidate_and_multiplier: dict[
            tuple[str, float],
            tuple[WarmupPortfolioEvaluation, int],
        ] = {}
        for candidate in selected_candidates:
            for multiplier in evaluation_multipliers:
                full_result = _run_candidate(
                    config=_config_with_risk_multiplier(
                        config,
                        multiplier=multiplier,
                    ),
                    prices=combined_prices,
                    quotes=combined_quotes,
                    candidate=candidate,
                    target_notional_multiplier=multiplier,
                )
                evaluation_by_candidate_and_multiplier[(candidate.label, multiplier)] = (
                    evaluate_portfolio_after_warmup(
                        full_result,
                        evaluation_start=test_timestamps[0],
                    ),
                    len(full_result.fills),
                )
        cached_folds.append(
            _CachedPolicyFold(
                fold_index=fold_index,
                train_start=train_timestamps[0].isoformat(),
                train_end=train_timestamps[-1].isoformat(),
                test_start=test_timestamps[0].isoformat(),
                test_end=test_timestamps[-1].isoformat(),
                test_timestamps=test_timestamps,
                train_scores_by_stability_splits=train_scores_by_stability_splits,
                evaluation_by_candidate_and_multiplier=(
                    evaluation_by_candidate_and_multiplier
                ),
            )
        )

    return _AdaptiveStrategyPolicyCache(
        strategy_names=tuple(candidate.label for candidate in selected_candidates),
        symbols=selected_symbols,
        candidates=selected_candidates,
        folds=tuple(cached_folds),
        starting_equity=config.competition.starting_equity,
    )


def _run_cached_policy(
    *,
    cache: _AdaptiveStrategyPolicyCache,
    loss_cooldown_folds: int,
    min_train_fills: int,
    min_train_drawdown_adjusted_return_pct: float | None,
    train_fill_penalty_pct: float,
    train_stability_splits: int,
    prefer_train_stability: bool,
    transition_risk_multiplier: float,
    allow_cash_fallback: bool,
) -> AdaptiveStrategySelectionResult:
    folds: list[AdaptiveStrategySelectionFold] = []
    cooldowns = {candidate.label: 0 for candidate in cache.candidates}
    previous_selected_strategy: str | None = None
    for cached_fold in cache.folds:
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
        cooldown_blocked_strategies = tuple(
            strategy_name
            for strategy_name, remaining in cooldowns.items()
            if remaining > 0
        )
        eligible_train_scores = tuple(
            score
            for score in train_scores
            if cooldowns.get(score.strategy_name, 0) <= 0
            and _passes_train_gate(
                score,
                min_train_fills=min_train_fills,
                min_train_drawdown_adjusted_return_pct=(
                    min_train_drawdown_adjusted_return_pct
                ),
            )
        )
        train_gate_blocked_strategies = tuple(
            score.strategy_name
            for score in train_scores
            if cooldowns.get(score.strategy_name, 0) <= 0
            and not _passes_train_gate(
                score,
                min_train_fills=min_train_fills,
                min_train_drawdown_adjusted_return_pct=(
                    min_train_drawdown_adjusted_return_pct
                ),
            )
        )
        selected_strategy = (
            eligible_train_scores[0].strategy_name
            if eligible_train_scores
            else CASH_FALLBACK_LABEL
            if allow_cash_fallback
            else train_scores[0].strategy_name
        )
        selected_strategy_map = (
            _cash_strategy_map(cache.symbols)
            if selected_strategy == CASH_FALLBACK_LABEL
            else _candidate_by_label(cache.candidates, selected_strategy).strategy_by_symbol
        )
        evaluation_risk_multiplier = (
            transition_risk_multiplier
            if (
                previous_selected_strategy is not None
                and selected_strategy != previous_selected_strategy
                and transition_risk_multiplier < 1.0
            )
            else 1.0
        )
        if selected_strategy == CASH_FALLBACK_LABEL:
            fold_train_scores = train_scores + (
                _cash_train_score(
                    symbols=cache.symbols,
                    starting_equity=cache.starting_equity,
                ),
            )
            evaluation = _cash_evaluation(
                cached_fold.test_timestamps,
                starting_equity=cache.starting_equity,
            )
            full_run_fills = 0
        else:
            fold_train_scores = train_scores
            evaluation, full_run_fills = (
                cached_fold.evaluation_by_candidate_and_multiplier[
                    (selected_strategy, evaluation_risk_multiplier)
                ]
            )
        folds.append(
            AdaptiveStrategySelectionFold(
                fold_index=cached_fold.fold_index,
                train_start=cached_fold.train_start,
                train_end=cached_fold.train_end,
                test_start=cached_fold.test_start,
                test_end=cached_fold.test_end,
                selected_strategy=selected_strategy,
                selected_strategy_map=selected_strategy_map,
                cooldown_blocked_strategies=cooldown_blocked_strategies,
                train_gate_blocked_strategies=train_gate_blocked_strategies,
                train_scores=fold_train_scores,
                evaluation=evaluation,
                full_run_fills=full_run_fills,
                evaluation_risk_multiplier=evaluation_risk_multiplier,
            )
        )
        previous_selected_strategy = selected_strategy
        cooldowns = {
            strategy_name: max(0, remaining - 1)
            for strategy_name, remaining in cooldowns.items()
        }
        if evaluation.competition_metrics.return_pct < -RETURN_EPSILON:
            cooldowns[selected_strategy] = max(
                cooldowns.get(selected_strategy, 0),
                loss_cooldown_folds,
            )

    result_strategy_names = cache.strategy_names
    if allow_cash_fallback:
        result_strategy_names = result_strategy_names + (CASH_FALLBACK_LABEL,)
    return AdaptiveStrategySelectionResult(
        strategy_names=result_strategy_names,
        symbols=cache.symbols,
        folds=tuple(folds),
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


def write_adaptive_strategy_policy_sweep_csv(
    sweep: AdaptiveStrategyPolicySweepResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "promotion_status",
                "live_ready",
                "selector_score",
                "strategies",
                "symbols",
                "loss_cooldown_folds",
                "min_train_fills",
                "min_train_adjusted_return_pct",
                "train_fill_penalty_pct",
                "train_stability_splits",
                "prefer_train_stability",
                "transition_risk_multiplier",
                "allow_cash_fallback",
                "folds",
                "positive_fold_fraction",
                "active_fold_fraction",
                "active_positive_fold_fraction",
                "non_negative_fold_fraction",
                "compounded_test_return_pct",
                "median_test_return_pct",
                "median_active_test_return_pct",
                "median_test_sharpe_15m",
                "worst_test_drawdown_pct",
                "average_risk_discipline_score",
                "total_evaluation_fills",
                "selection_counts",
                "promotion_reason",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(sweep.candidates, start=1):
            result = candidate.result
            writer.writerow(
                {
                    "rank": rank,
                    "promotion_status": candidate.decision.status,
                    "live_ready": "yes" if candidate.decision.live_ready else "no",
                    "selector_score": candidate.selector_score,
                    "strategies": " ".join(result.strategy_names),
                    "symbols": " ".join(result.symbols),
                    "loss_cooldown_folds": result.loss_cooldown_folds,
                    "min_train_fills": result.min_train_fills,
                    "min_train_adjusted_return_pct": (
                        ""
                        if result.min_train_drawdown_adjusted_return_pct is None
                        else result.min_train_drawdown_adjusted_return_pct
                    ),
                    "train_fill_penalty_pct": result.train_fill_penalty_pct,
                    "train_stability_splits": result.train_stability_splits,
                    "prefer_train_stability": (
                        "yes" if result.prefer_train_stability else "no"
                    ),
                    "transition_risk_multiplier": result.transition_risk_multiplier,
                    "allow_cash_fallback": (
                        "yes" if result.allow_cash_fallback else "no"
                    ),
                    "folds": len(result.folds),
                    "positive_fold_fraction": result.positive_fold_fraction,
                    "active_fold_fraction": result.active_fold_fraction,
                    "active_positive_fold_fraction": (
                        result.active_positive_fold_fraction
                    ),
                    "non_negative_fold_fraction": result.non_negative_fold_fraction,
                    "compounded_test_return_pct": result.compounded_test_return_pct,
                    "median_test_return_pct": result.median_test_return_pct,
                    "median_active_test_return_pct": (
                        result.median_active_test_return_pct
                    ),
                    "median_test_sharpe_15m": result.median_test_sharpe_15m,
                    "worst_test_drawdown_pct": result.worst_test_drawdown_pct,
                    "average_risk_discipline_score": (
                        result.average_risk_discipline_score
                    ),
                    "total_evaluation_fills": result.total_evaluation_fills,
                    "selection_counts": candidate.selection_counts_text,
                    "promotion_reason": candidate.decision.reason,
                }
            )


def _selector_score(result: AdaptiveStrategySelectionResult) -> float:
    risk_multiplier = min(
        max(result.average_risk_discipline_score / 100.0, 0.0),
        1.0,
    )
    active_cap = min(result.active_fold_fraction, 0.75) / 0.75
    return risk_multiplier * (
        35.0 * result.non_negative_fold_fraction
        + 30.0 * result.active_positive_fold_fraction
        + 15.0 * result.positive_fold_fraction
        + 10.0 * active_cap
        + 350.0 * max(result.compounded_test_return_pct, 0.0)
        + 50.0 * max(result.median_active_test_return_pct, 0.0)
        - 100.0 * result.worst_test_drawdown_pct
    )


def _promotion_rank(status: str) -> float:
    if status == "PROMOTE":
        return 3.0
    if status == "PAPER_ONLY":
        return 2.0
    return 1.0


def _cash_fallback_rank(allow_cash_fallback: bool) -> int:
    return 1 if allow_cash_fallback else 0
