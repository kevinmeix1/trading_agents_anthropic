from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

from quanthack.backtesting.router_optimizer import (
    DEFAULT_ROUTER_BEHAVIOR_PROFILES,
    DEFAULT_ROUTER_WEIGHT_SETS,
    RouterBehaviorProfile,
    RouterOptimizationCandidate,
    RouterOptimizationResult,
    RouterWeightSet,
    optimize_router_weights,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import enabled_symbols, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class PortfolioRouterWalkForwardFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    symbols: tuple[str, ...]
    train_result: RouterOptimizationResult
    test_result: RouterOptimizationResult
    selected_train_candidate: RouterOptimizationCandidate
    selected_test_candidate: RouterOptimizationCandidate
    test_best_candidate: RouterOptimizationCandidate
    stable_candidate: bool

    @property
    def selected_weights(self) -> RouterWeightSet:
        return self.selected_train_candidate.weights

    @property
    def selected_behavior(self) -> RouterBehaviorProfile:
        return self.selected_train_candidate.behavior

    @property
    def selected_was_test_best(self) -> bool:
        return (
            self.selected_weights == self.test_best_candidate.weights
            and self.selected_behavior == self.test_best_candidate.behavior
        )


@dataclass(frozen=True)
class PortfolioRouterWalkForwardSummary:
    folds: tuple[PortfolioRouterWalkForwardFold, ...]
    stable_fold_fraction: float
    median_test_proxy_score: float
    median_test_return_pct: float
    lower_quartile_test_return_pct: float
    median_test_sharpe_15m: float
    worst_test_drawdown_pct: float
    average_risk_discipline_score: float
    total_test_fills: int
    total_test_turnover: float
    most_selected_weights: str
    most_selected_behavior: str
    selected_was_test_best_fraction: float
    eligible: bool


@dataclass(frozen=True)
class RouterPromotionDecision:
    status: str
    live_ready: bool
    reason: str


@dataclass(frozen=True)
class PortfolioRouterWalkForwardResult:
    symbols: tuple[str, ...]
    weight_sets: tuple[RouterWeightSet, ...]
    behavior_profiles: tuple[RouterBehaviorProfile, ...]
    folds: tuple[PortfolioRouterWalkForwardFold, ...]
    summary: PortfolioRouterWalkForwardSummary


def decide_router_promotion(
    summary: PortfolioRouterWalkForwardSummary,
    *,
    min_stable_fold_fraction: float = 0.50,
    min_selected_was_test_best_fraction: float = 0.50,
    min_median_return_pct: float = 0.00005,
    min_lower_quartile_return_pct: float = 0.0,
    max_worst_drawdown_pct: float = 0.03,
    min_risk_discipline_score: float = 90.0,
) -> RouterPromotionDecision:
    if not summary.folds:
        return RouterPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason="no router walk-forward folds were produced",
        )
    if not summary.eligible:
        return RouterPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                "router walk-forward eligibility failed; keep this weight set "
                "in research"
            ),
        )
    if summary.stable_fold_fraction < min_stable_fold_fraction:
        return RouterPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                f"stable fold fraction {summary.stable_fold_fraction:.1%} is below "
                f"{min_stable_fold_fraction:.1%}"
            ),
        )
    if summary.selected_was_test_best_fraction < min_selected_was_test_best_fraction:
        return RouterPromotionDecision(
            status="PAPER_ONLY",
            live_ready=False,
            reason=(
                "train-selected weights were test-best only "
                f"{summary.selected_was_test_best_fraction:.1%} of the time"
            ),
        )
    if summary.median_test_return_pct <= min_median_return_pct:
        return RouterPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                f"median test return {summary.median_test_return_pct:.3%} is not "
                f"above {min_median_return_pct:.3%}"
            ),
        )
    if summary.lower_quartile_test_return_pct < min_lower_quartile_return_pct:
        return RouterPromotionDecision(
            status="PAPER_ONLY",
            live_ready=False,
            reason=(
                "lower-quartile test return "
                f"{summary.lower_quartile_test_return_pct:.3%} is below "
                f"{min_lower_quartile_return_pct:.3%}"
            ),
        )
    if summary.worst_test_drawdown_pct > max_worst_drawdown_pct:
        return RouterPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                f"worst test drawdown {summary.worst_test_drawdown_pct:.3%} is above "
                f"{max_worst_drawdown_pct:.3%}"
            ),
        )
    if summary.average_risk_discipline_score < min_risk_discipline_score:
        return RouterPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                "average risk discipline "
                f"{summary.average_risk_discipline_score:.1f}/100 is below "
                f"{min_risk_discipline_score:.1f}/100"
            ),
        )
    return RouterPromotionDecision(
        status="PROMOTE",
        live_ready=True,
        reason="router walk-forward stability, return, drawdown, and risk gates passed",
    )


def run_portfolio_router_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    weight_sets: tuple[RouterWeightSet, ...] = DEFAULT_ROUTER_WEIGHT_SETS,
    behavior_profiles: tuple[
        RouterBehaviorProfile, ...
    ] = DEFAULT_ROUTER_BEHAVIOR_PROFILES,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
    min_train_fills: int = 1,
    min_train_return_pct: float = 0.0,
    min_test_fills: int = 1,
    min_stable_fold_fraction: float = 0.50,
    max_test_drawdown_pct: float = 0.05,
    min_risk_discipline_score: int = 80,
) -> PortfolioRouterWalkForwardResult:
    _validate_inputs(
        weight_sets=weight_sets,
        behavior_profiles=behavior_profiles,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        min_train_fills=min_train_fills,
        min_train_return_pct=min_train_return_pct,
        min_test_fills=min_test_fills,
        min_stable_fold_fraction=min_stable_fold_fraction,
        max_test_drawdown_pct=max_test_drawdown_pct,
        min_risk_discipline_score=min_risk_discipline_score,
    )
    selected_symbols, timestamps = _selected_symbols_and_common_timestamps(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
    )
    if len(timestamps) < train_size + test_size:
        raise ValueError(
            "not enough aligned timestamps for one portfolio router walk-forward fold"
        )

    folds: list[PortfolioRouterWalkForwardFold] = []
    for fold_index, start in enumerate(
        range(0, len(timestamps) - train_size - test_size + 1, step_size),
        start=1,
    ):
        train_timestamps = timestamps[start : start + train_size]
        test_timestamps = timestamps[start + train_size : start + train_size + test_size]
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
        combined_timestamps = train_timestamps + test_timestamps
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

        train_result = optimize_router_weights(
            config=config,
            prices=train_prices,
            quotes=train_quotes,
            symbols=selected_symbols,
            weight_sets=weight_sets,
            behavior_profiles=behavior_profiles,
        )
        if train_result.best is None:
            raise ValueError("train-window router optimization returned no candidates")

        test_result = optimize_router_weights(
            config=config,
            prices=combined_prices,
            quotes=combined_quotes,
            symbols=selected_symbols,
            weight_sets=weight_sets,
            behavior_profiles=behavior_profiles,
            evaluation_start=test_timestamps[0],
        )
        if test_result.best is None:
            raise ValueError("test-window router optimization returned no candidates")

        selected_train_candidate = _select_train_candidate(
            train_result,
            min_train_fills=min_train_fills,
            min_train_return_pct=min_train_return_pct,
        )
        selected_test_candidate = _matching_candidate(
            test_result,
            selected_train_candidate.weights,
            selected_train_candidate.behavior,
        )
        stable_candidate = _is_stable_candidate(
            selected_test_candidate,
            min_test_fills=min_test_fills,
            max_test_drawdown_pct=max_test_drawdown_pct,
            min_risk_discipline_score=min_risk_discipline_score,
        )
        folds.append(
            PortfolioRouterWalkForwardFold(
                fold_index=fold_index,
                train_start=train_timestamps[0].isoformat(timespec="seconds"),
                train_end=train_timestamps[-1].isoformat(timespec="seconds"),
                test_start=test_timestamps[0].isoformat(timespec="seconds"),
                test_end=test_timestamps[-1].isoformat(timespec="seconds"),
                symbols=selected_symbols,
                train_result=train_result,
                test_result=test_result,
                selected_train_candidate=selected_train_candidate,
                selected_test_candidate=selected_test_candidate,
                test_best_candidate=test_result.best,
                stable_candidate=stable_candidate,
            )
        )

    result_folds = tuple(folds)
    return PortfolioRouterWalkForwardResult(
        symbols=selected_symbols,
        weight_sets=weight_sets,
        behavior_profiles=behavior_profiles,
        folds=result_folds,
        summary=_summarize(
            result_folds,
            min_test_fills=min_test_fills,
            min_stable_fold_fraction=min_stable_fold_fraction,
            max_test_drawdown_pct=max_test_drawdown_pct,
            min_risk_discipline_score=min_risk_discipline_score,
        ),
    )


def write_portfolio_router_walk_forward_summary_csv(
    result: PortfolioRouterWalkForwardResult,
    path: str | Path,
    *,
    min_stable_fold_fraction: float = 0.50,
    max_worst_drawdown_pct: float = 0.03,
    min_risk_discipline_score: float = 90.0,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    promotion = decide_router_promotion(
        summary,
        min_stable_fold_fraction=min_stable_fold_fraction,
        max_worst_drawdown_pct=max_worst_drawdown_pct,
        min_risk_discipline_score=min_risk_discipline_score,
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "eligible",
                "folds",
                "symbols",
                "candidate_weight_sets",
                "candidate_behavior_profiles",
                "promotion_status",
                "promotion_live_ready",
                "promotion_reason",
                "most_selected_weights",
                "most_selected_behavior",
                "stable_fold_fraction",
                "selected_was_test_best_fraction",
                "median_test_proxy_score",
                "median_test_return_pct",
                "lower_quartile_test_return_pct",
                "median_test_sharpe_15m",
                "worst_test_drawdown_pct",
                "average_risk_discipline_score",
                "total_test_fills",
                "total_test_turnover",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "eligible": summary.eligible,
                "folds": len(summary.folds),
                "symbols": " ".join(result.symbols),
                "candidate_weight_sets": len(result.weight_sets),
                "candidate_behavior_profiles": len(result.behavior_profiles),
                "promotion_status": promotion.status,
                "promotion_live_ready": promotion.live_ready,
                "promotion_reason": promotion.reason,
                "most_selected_weights": summary.most_selected_weights,
                "most_selected_behavior": summary.most_selected_behavior,
                "stable_fold_fraction": summary.stable_fold_fraction,
                "selected_was_test_best_fraction": (
                    summary.selected_was_test_best_fraction
                ),
                "median_test_proxy_score": summary.median_test_proxy_score,
                "median_test_return_pct": summary.median_test_return_pct,
                "lower_quartile_test_return_pct": summary.lower_quartile_test_return_pct,
                "median_test_sharpe_15m": summary.median_test_sharpe_15m,
                "worst_test_drawdown_pct": summary.worst_test_drawdown_pct,
                "average_risk_discipline_score": summary.average_risk_discipline_score,
                "total_test_fills": summary.total_test_fills,
                "total_test_turnover": summary.total_test_turnover,
            }
        )


def write_portfolio_router_walk_forward_folds_csv(
    result: PortfolioRouterWalkForwardResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "symbols",
                "selected_weights",
                "selected_behavior",
                "test_best_weights",
                "test_best_behavior",
                "selected_was_test_best",
                "stable_candidate",
                "train_proxy_score",
                "test_proxy_score",
                "train_return_pct",
                "train_drawdown_pct",
                "train_sharpe_15m",
                "train_risk_discipline_score",
                "test_return_pct",
                "test_drawdown_pct",
                "test_sharpe_15m",
                "test_risk_discipline_score",
                "test_final_equity",
                "test_trade_count",
                "test_fills",
                "test_turnover_notional",
                "test_trimmed_allocation_periods",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            train_metrics = fold.selected_train_candidate.competition_metrics
            test_metrics = fold.selected_test_candidate.competition_metrics
            writer.writerow(
                {
                    "fold": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "symbols": " ".join(fold.symbols),
                    "selected_weights": fold.selected_weights.label,
                    "selected_behavior": fold.selected_behavior.label,
                    "test_best_weights": fold.test_best_candidate.weights.label,
                    "test_best_behavior": fold.test_best_candidate.behavior.label,
                    "selected_was_test_best": fold.selected_was_test_best,
                    "stable_candidate": fold.stable_candidate,
                    "train_proxy_score": fold.selected_train_candidate.proxy_score,
                    "test_proxy_score": fold.selected_test_candidate.proxy_score,
                    "train_return_pct": train_metrics.return_pct,
                    "train_drawdown_pct": train_metrics.max_drawdown_pct,
                    "train_sharpe_15m": train_metrics.sharpe_15m,
                    "train_risk_discipline_score": (
                        fold.selected_train_candidate.risk_discipline.score
                    ),
                    "test_return_pct": test_metrics.return_pct,
                    "test_drawdown_pct": test_metrics.max_drawdown_pct,
                    "test_sharpe_15m": test_metrics.sharpe_15m,
                    "test_risk_discipline_score": (
                        fold.selected_test_candidate.risk_discipline.score
                    ),
                    "test_final_equity": test_metrics.final_equity,
                    "test_trade_count": test_metrics.trade_count,
                    "test_fills": len(fold.selected_test_candidate.result.fills),
                    "test_turnover_notional": (
                        fold.selected_test_candidate.result.metrics.turnover_notional
                    ),
                    "test_trimmed_allocation_periods": len(
                        [
                            allocation
                            for allocation in (
                                fold.selected_test_candidate.result.allocation_reports
                            )
                            if allocation.trimmed_targets
                        ]
                    ),
                }
            )


def _selected_symbols_and_common_timestamps(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[datetime, ...]]:
    if symbols:
        requested_symbols = tuple(instrument_for(symbol).symbol for symbol in symbols)
    else:
        requested_symbols = enabled_symbols()

    selected: list[str] = []
    common_timestamps: set[datetime] | None = None
    missing_symbols: list[str] = []
    for symbol in requested_symbols:
        price_timestamps = {bar.timestamp for bar in prices.for_symbol(symbol).bars}
        quote_timestamps = {quote.timestamp for quote in quotes.for_symbol(symbol).quotes}
        timestamps = price_timestamps & quote_timestamps
        if not timestamps:
            if symbols:
                missing_symbols.append(symbol)
            continue
        selected.append(symbol)
        common_timestamps = (
            timestamps
            if common_timestamps is None
            else common_timestamps & timestamps
        )

    if missing_symbols:
        raise ValueError(
            "requested symbols missing aligned price/quote data: "
            f"{', '.join(missing_symbols)}"
        )
    if not selected or not common_timestamps:
        raise ValueError("no supported symbols have aligned price and quote timestamps")
    return tuple(selected), tuple(sorted(common_timestamps))


def _slice_prices(
    prices: PriceHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> PriceHistory:
    timestamp_set = set(timestamps)
    return PriceHistory(
        tuple(
            bar
            for bar in prices.bars
            if bar.symbol in symbols and bar.timestamp in timestamp_set
        )
    )


def _slice_quotes(
    quotes: QuoteHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> QuoteHistory:
    timestamp_set = set(timestamps)
    return QuoteHistory(
        tuple(
            quote
            for quote in quotes.quotes
            if quote.symbol in symbols and quote.timestamp in timestamp_set
        )
    )


def _matching_candidate(
    result: RouterOptimizationResult,
    weights: RouterWeightSet,
    behavior: RouterBehaviorProfile,
) -> RouterOptimizationCandidate:
    for candidate in result.candidates:
        if candidate.weights == weights and candidate.behavior == behavior:
            return candidate
    raise ValueError(
        "selected router candidate not found in test window: "
        f"{weights.label} / {behavior.label}"
    )


def _select_train_candidate(
    result: RouterOptimizationResult,
    *,
    min_train_fills: int,
    min_train_return_pct: float,
) -> RouterOptimizationCandidate:
    eligible = [
        candidate
        for candidate in result.candidates
        if len(candidate.result.fills) >= min_train_fills
        and candidate.competition_metrics.return_pct > min_train_return_pct
    ]
    if eligible:
        return eligible[0]
    if result.best is None:
        raise ValueError("train-window router optimization returned no candidates")
    return result.best


def _is_stable_candidate(
    candidate: RouterOptimizationCandidate,
    *,
    min_test_fills: int,
    max_test_drawdown_pct: float,
    min_risk_discipline_score: int,
) -> bool:
    metrics = candidate.competition_metrics
    return (
        metrics.return_pct > 0
        and metrics.max_drawdown_pct <= max_test_drawdown_pct
        and candidate.risk_discipline.score >= min_risk_discipline_score
        and len(candidate.result.fills) >= min_test_fills
    )


def _summarize(
    folds: tuple[PortfolioRouterWalkForwardFold, ...],
    *,
    min_test_fills: int,
    min_stable_fold_fraction: float,
    max_test_drawdown_pct: float,
    min_risk_discipline_score: int,
) -> PortfolioRouterWalkForwardSummary:
    test_returns = [
        fold.selected_test_candidate.competition_metrics.return_pct for fold in folds
    ]
    test_drawdowns = [
        fold.selected_test_candidate.competition_metrics.max_drawdown_pct
        for fold in folds
    ]
    test_sharpes = [
        fold.selected_test_candidate.competition_metrics.sharpe_15m
        for fold in folds
    ]
    test_proxy_scores = [fold.selected_test_candidate.proxy_score for fold in folds]
    risk_scores = [fold.selected_test_candidate.risk_discipline.score for fold in folds]
    stable_fraction = (
        sum(1 for fold in folds if fold.stable_candidate) / len(folds)
        if folds
        else 0.0
    )
    selected_best_fraction = (
        sum(1 for fold in folds if fold.selected_was_test_best) / len(folds)
        if folds
        else 0.0
    )
    total_fills = sum(len(fold.selected_test_candidate.result.fills) for fold in folds)
    total_turnover = sum(
        fold.selected_test_candidate.result.metrics.turnover_notional
        for fold in folds
    )
    selected_weights = Counter(fold.selected_weights.label for fold in folds)
    selected_behaviors = Counter(fold.selected_behavior.label for fold in folds)
    worst_drawdown = max(test_drawdowns, default=0.0)
    average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    eligible = (
        total_fills >= min_test_fills
        and stable_fraction >= min_stable_fold_fraction
        and worst_drawdown <= max_test_drawdown_pct
        and average_risk_score >= min_risk_discipline_score
    )

    return PortfolioRouterWalkForwardSummary(
        folds=folds,
        stable_fold_fraction=stable_fraction,
        median_test_proxy_score=median(test_proxy_scores) if test_proxy_scores else 0.0,
        median_test_return_pct=median(test_returns) if test_returns else 0.0,
        lower_quartile_test_return_pct=_lower_quartile(test_returns),
        median_test_sharpe_15m=median(test_sharpes) if test_sharpes else 0.0,
        worst_test_drawdown_pct=worst_drawdown,
        average_risk_discipline_score=average_risk_score,
        total_test_fills=total_fills,
        total_test_turnover=total_turnover,
        most_selected_weights=(
            selected_weights.most_common(1)[0][0] if selected_weights else ""
        ),
        most_selected_behavior=(
            selected_behaviors.most_common(1)[0][0] if selected_behaviors else ""
        ),
        selected_was_test_best_fraction=selected_best_fraction,
        eligible=eligible,
    )


def _lower_quartile(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, int(len(sorted_values) * 0.25) - 1)
    return sorted_values[index]


def _validate_inputs(
    *,
    weight_sets: tuple[RouterWeightSet, ...],
    behavior_profiles: tuple[RouterBehaviorProfile, ...],
    train_size: int,
    test_size: int,
    step_size: int,
    min_train_fills: int,
    min_train_return_pct: float,
    min_test_fills: int,
    min_stable_fold_fraction: float,
    max_test_drawdown_pct: float,
    min_risk_discipline_score: int,
) -> None:
    if not weight_sets:
        raise ValueError("at least one router weight set is required")
    if not behavior_profiles:
        raise ValueError("at least one router behavior profile is required")
    if train_size < 2:
        raise ValueError("train_size must be at least 2")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")
    if min_train_fills < 0:
        raise ValueError("min_train_fills cannot be negative")
    if not 0 <= min_train_return_pct <= 1:
        raise ValueError("min_train_return_pct must be between 0 and 1")
    if min_test_fills < 0:
        raise ValueError("min_test_fills cannot be negative")
    if not 0 <= min_stable_fold_fraction <= 1:
        raise ValueError("min_stable_fold_fraction must be between 0 and 1")
    if not 0 <= max_test_drawdown_pct <= 1:
        raise ValueError("max_test_drawdown_pct must be between 0 and 1")
    if not 0 <= min_risk_discipline_score <= 100:
        raise ValueError("min_risk_discipline_score must be between 0 and 100")
