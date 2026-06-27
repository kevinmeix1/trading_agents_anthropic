from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import median

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    PortfolioEquityPoint,
)
from quanthack.backtesting.warmup import (
    WarmupPortfolioEvaluation,
    evaluate_portfolio_after_warmup,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot
from quanthack.strategies.strategy import normalize_strategy_name

RETURN_EPSILON = 1e-12
PER_SYMBOL_ADAPTIVE_LABEL = "per_symbol_adaptive"
CASH_FALLBACK_LABEL = "cash"


@dataclass(frozen=True)
class AdaptiveStrategyCandidate:
    label: str
    strategy_by_symbol: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("adaptive strategy candidate label is required")
        if not self.strategy_by_symbol:
            raise ValueError("adaptive strategy candidate needs at least one symbol")

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
        )

    def strategy_for_symbol(self, symbol: str) -> str:
        canonical = instrument_for(symbol).symbol
        for map_symbol, strategy_name in self.strategy_by_symbol:
            if map_symbol == canonical:
                return strategy_name
        raise KeyError(f"candidate {self.label} has no strategy for {canonical}")


@dataclass(frozen=True)
class AdaptiveStrategyTrainStability:
    splits: int = 0
    active_fraction: float = 0.0
    positive_fraction: float = 0.0
    active_positive_fraction: float = 0.0
    non_negative_fraction: float = 0.0
    median_return_pct: float = 0.0
    median_active_return_pct: float = 0.0


@dataclass(frozen=True)
class AdaptiveStrategyTrainScore:
    strategy_name: str
    strategy_map: tuple[tuple[str, str], ...]
    return_pct: float
    max_drawdown_pct: float
    sharpe_15m: float
    risk_discipline_score: int
    fills: int
    final_equity: float
    stability: AdaptiveStrategyTrainStability = field(
        default_factory=AdaptiveStrategyTrainStability
    )

    @property
    def active(self) -> bool:
        return self.fills > 0 or abs(self.return_pct) > RETURN_EPSILON

    @property
    def drawdown_adjusted_return_pct(self) -> float:
        return self.return_pct - (0.75 * self.max_drawdown_pct)

    @property
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_map
        )

    @property
    def rank_key(self) -> tuple[float, ...]:
        return self.rank_key_with_fill_penalty(0.0)

    def rank_key_with_fill_penalty(self, fill_penalty_pct: float) -> tuple[float, ...]:
        return self.rank_key_with_preferences(
            fill_penalty_pct=fill_penalty_pct,
            prefer_train_stability=False,
        )

    def rank_key_with_preferences(
        self,
        *,
        fill_penalty_pct: float,
        prefer_train_stability: bool,
    ) -> tuple[float, ...]:
        adjusted_return = self.drawdown_adjusted_return_pct - (
            fill_penalty_pct * self.fills
        )
        stability_key: tuple[float, ...] = ()
        if prefer_train_stability:
            stability_key = (
                self.stability.active_positive_fraction,
                self.stability.non_negative_fraction,
                self.stability.median_active_return_pct,
                self.stability.positive_fraction,
                self.stability.active_fraction,
            )
        return (
            1.0 if self.risk_discipline_score >= 95 else 0.0,
            1.0 if self.active else 0.0,
            *stability_key,
            adjusted_return,
            self.sharpe_15m,
            self.return_pct,
            -self.max_drawdown_pct,
            -float(self.fills),
        )


@dataclass(frozen=True)
class AdaptiveStrategySelectionFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_strategy: str
    selected_strategy_map: tuple[tuple[str, str], ...]
    cooldown_blocked_strategies: tuple[str, ...]
    train_gate_blocked_strategies: tuple[str, ...]
    train_scores: tuple[AdaptiveStrategyTrainScore, ...]
    evaluation: WarmupPortfolioEvaluation
    full_run_fills: int
    evaluation_risk_multiplier: float = 1.0

    @property
    def metrics(self) -> CompetitionMetrics:
        return self.evaluation.competition_metrics

    @property
    def risk_discipline(self) -> RiskDisciplineReport:
        return self.evaluation.risk_discipline

    @property
    def selected_train_score(self) -> AdaptiveStrategyTrainScore:
        for score in self.train_scores:
            if score.strategy_name == self.selected_strategy:
                return score
        raise KeyError(f"no train score for selected strategy {self.selected_strategy}")


@dataclass(frozen=True)
class StrategySelectionCount:
    strategy_name: str
    folds: int


@dataclass(frozen=True)
class AdaptiveStrategyStitchedEquityPoint:
    timestamp: str
    equity: float
    drawdown_pct: float
    fold_index: int
    selected_strategy: str
    fold_return_pct: float
    source_fold_equity: float
    source_fold_start_equity: float


@dataclass(frozen=True)
class AdaptiveStrategySelectionResult:
    strategy_names: tuple[str, ...]
    symbols: tuple[str, ...]
    folds: tuple[AdaptiveStrategySelectionFold, ...]
    loss_cooldown_folds: int = 0
    min_train_fills: int = 0
    min_train_drawdown_adjusted_return_pct: float | None = None
    train_fill_penalty_pct: float = 0.0
    train_stability_splits: int = 0
    prefer_train_stability: bool = False
    transition_risk_multiplier: float = 1.0
    allow_cash_fallback: bool = False
    per_symbol_selection: bool = False
    per_symbol_only: bool = False

    @property
    def positive_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        positive = [fold for fold in self.folds if fold.metrics.return_pct > 0]
        return len(positive) / len(self.folds)

    @property
    def active_folds(self) -> tuple[AdaptiveStrategySelectionFold, ...]:
        return tuple(
            fold
            for fold in self.folds
            if len(fold.evaluation.fills) > 0
            or abs(fold.metrics.return_pct) > RETURN_EPSILON
        )

    @property
    def active_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        return len(self.active_folds) / len(self.folds)

    @property
    def active_positive_fold_fraction(self) -> float:
        active_folds = self.active_folds
        if not active_folds:
            return 0.0
        positive = [fold for fold in active_folds if fold.metrics.return_pct > 0]
        return len(positive) / len(active_folds)

    @property
    def non_negative_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        non_negative = [
            fold for fold in self.folds if fold.metrics.return_pct >= -RETURN_EPSILON
        ]
        return len(non_negative) / len(self.folds)

    @property
    def losing_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        return 1.0 - self.non_negative_fold_fraction

    @property
    def compounded_test_return_pct(self) -> float:
        compounded = 1.0
        for fold in self.folds:
            compounded *= 1.0 + fold.metrics.return_pct
        return compounded - 1.0

    @property
    def median_test_return_pct(self) -> float:
        if not self.folds:
            return 0.0
        return median(fold.metrics.return_pct for fold in self.folds)

    @property
    def median_active_test_return_pct(self) -> float:
        active_folds = self.active_folds
        if not active_folds:
            return 0.0
        return median(fold.metrics.return_pct for fold in active_folds)

    @property
    def median_test_sharpe_15m(self) -> float:
        if not self.folds:
            return 0.0
        return median(fold.metrics.sharpe_15m for fold in self.folds)

    @property
    def worst_test_drawdown_pct(self) -> float:
        if not self.folds:
            return 0.0
        return max(fold.metrics.max_drawdown_pct for fold in self.folds)

    @property
    def average_risk_discipline_score(self) -> float:
        if not self.folds:
            return 0.0
        return sum(fold.risk_discipline.score for fold in self.folds) / len(self.folds)

    @property
    def total_evaluation_fills(self) -> int:
        return sum(len(fold.evaluation.fills) for fold in self.folds)

    @property
    def selection_counts(self) -> tuple[StrategySelectionCount, ...]:
        counts = Counter(fold.selected_strategy for fold in self.folds)
        return tuple(
            StrategySelectionCount(strategy_name=strategy_name, folds=counts[strategy_name])
            for strategy_name in self.strategy_names
            if counts[strategy_name] > 0
        )


@dataclass(frozen=True)
class AdaptiveStrategySelectionDecision:
    status: str
    live_ready: bool
    reason: str


@dataclass(frozen=True)
class AdaptivePromotionGate:
    gate_id: str
    category: str
    passed: bool
    value: float
    threshold: float
    comparator: str
    details: str


@dataclass(frozen=True)
class AdaptivePromotionAudit:
    decision: AdaptiveStrategySelectionDecision
    gates: tuple[AdaptivePromotionGate, ...]

    @property
    def failed_gates(self) -> tuple[AdaptivePromotionGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)


def decide_adaptive_strategy_selection_promotion(
    result: AdaptiveStrategySelectionResult,
    *,
    min_positive_fold_fraction: float = 0.50,
    min_active_positive_fold_fraction: float = 0.50,
    min_non_negative_fold_fraction: float = 0.70,
    min_live_positive_fold_fraction: float = 0.67,
    min_live_active_positive_fold_fraction: float = 0.67,
    min_median_active_return_pct: float = 0.0,
    max_worst_drawdown_pct: float = 0.03,
    min_average_risk_discipline_score: float = 95.0,
) -> AdaptiveStrategySelectionDecision:
    return build_adaptive_strategy_promotion_audit(
        result,
        min_positive_fold_fraction=min_positive_fold_fraction,
        min_active_positive_fold_fraction=min_active_positive_fold_fraction,
        min_non_negative_fold_fraction=min_non_negative_fold_fraction,
        min_live_positive_fold_fraction=min_live_positive_fold_fraction,
        min_live_active_positive_fold_fraction=min_live_active_positive_fold_fraction,
        min_median_active_return_pct=min_median_active_return_pct,
        max_worst_drawdown_pct=max_worst_drawdown_pct,
        min_average_risk_discipline_score=min_average_risk_discipline_score,
    ).decision


def build_adaptive_strategy_promotion_audit(
    result: AdaptiveStrategySelectionResult,
    *,
    min_positive_fold_fraction: float = 0.50,
    min_active_positive_fold_fraction: float = 0.50,
    min_non_negative_fold_fraction: float = 0.70,
    min_live_positive_fold_fraction: float = 0.67,
    min_live_active_positive_fold_fraction: float = 0.67,
    min_median_active_return_pct: float = 0.0,
    max_worst_drawdown_pct: float = 0.03,
    min_average_risk_discipline_score: float = 95.0,
) -> AdaptivePromotionAudit:
    gates = _adaptive_promotion_gates(
        result,
        min_positive_fold_fraction=min_positive_fold_fraction,
        min_active_positive_fold_fraction=min_active_positive_fold_fraction,
        min_non_negative_fold_fraction=min_non_negative_fold_fraction,
        min_live_positive_fold_fraction=min_live_positive_fold_fraction,
        min_live_active_positive_fold_fraction=min_live_active_positive_fold_fraction,
        min_median_active_return_pct=min_median_active_return_pct,
        max_worst_drawdown_pct=max_worst_drawdown_pct,
        min_average_risk_discipline_score=min_average_risk_discipline_score,
    )
    if not result.folds:
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason="no adaptive strategy-selection folds were produced",
            ),
            gates=gates,
        )
    if not result.active_folds:
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason="adaptive selector produced no active evaluation folds",
            ),
            gates=gates,
        )
    if result.non_negative_fold_fraction < min_non_negative_fold_fraction:
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason=(
                    f"non-negative fold fraction {result.non_negative_fold_fraction:.1%} "
                    f"is below {min_non_negative_fold_fraction:.1%}"
                ),
            ),
            gates=gates,
        )
    if (
        result.positive_fold_fraction < min_positive_fold_fraction
        and result.active_positive_fold_fraction < min_active_positive_fold_fraction
    ):
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason=(
                    "positive active fold fraction "
                    f"{result.active_positive_fold_fraction:.1%} is below "
                    f"{min_active_positive_fold_fraction:.1%}"
                ),
            ),
            gates=gates,
        )
    if result.median_active_test_return_pct <= min_median_active_return_pct:
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason=(
                    "median active test return "
                    f"{result.median_active_test_return_pct:.3%} is not above "
                    f"{min_median_active_return_pct:.3%}"
                ),
            ),
            gates=gates,
        )
    if result.worst_test_drawdown_pct > max_worst_drawdown_pct:
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason=(
                    f"worst test drawdown {result.worst_test_drawdown_pct:.3%} "
                    f"is above {max_worst_drawdown_pct:.3%}"
                ),
            ),
            gates=gates,
        )
    if result.average_risk_discipline_score < min_average_risk_discipline_score:
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="REJECT",
                live_ready=False,
                reason=(
                    "average risk discipline "
                    f"{result.average_risk_discipline_score:.1f}/100 is below "
                    f"{min_average_risk_discipline_score:.1f}/100"
                ),
            ),
            gates=gates,
        )
    if (
        result.positive_fold_fraction < min_live_positive_fold_fraction
        or result.active_positive_fold_fraction < min_live_active_positive_fold_fraction
    ):
        return AdaptivePromotionAudit(
            decision=AdaptiveStrategySelectionDecision(
                status="PAPER_ONLY",
                live_ready=False,
                reason=(
                    "adaptive selector passed research gates, but live promotion needs "
                    f"{min_live_positive_fold_fraction:.1%} total positive folds and "
                    f"{min_live_active_positive_fold_fraction:.1%} active positive folds"
                ),
            ),
            gates=gates,
        )
    return AdaptivePromotionAudit(
        decision=AdaptiveStrategySelectionDecision(
            status="PROMOTE",
            live_ready=True,
            reason="adaptive selector fold stability, drawdown, and risk gates passed",
        ),
        gates=gates,
    )


def run_adaptive_strategy_selection(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_names: tuple[str, ...],
    candidate_maps: tuple[AdaptiveStrategyCandidate, ...] = (),
    recipe_maps: tuple[AdaptiveStrategyCandidate, ...] = (),
    symbols: tuple[str, ...] | None = None,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    loss_cooldown_folds: int = 0,
    min_train_fills: int = 0,
    min_train_drawdown_adjusted_return_pct: float | None = None,
    train_fill_penalty_pct: float = 0.0,
    train_stability_splits: int = 0,
    prefer_train_stability: bool = False,
    transition_risk_multiplier: float = 1.0,
    allow_cash_fallback: bool = False,
    per_symbol_selection: bool = False,
    per_symbol_only: bool = False,
) -> AdaptiveStrategySelectionResult:
    _validate_window_sizes(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    if loss_cooldown_folds < 0:
        raise ValueError("loss_cooldown_folds cannot be negative")
    if min_train_fills < 0:
        raise ValueError("min_train_fills cannot be negative")
    if train_fill_penalty_pct < 0:
        raise ValueError("train_fill_penalty_pct cannot be negative")
    if train_stability_splits < 0 or train_stability_splits == 1:
        raise ValueError("train_stability_splits must be 0 or at least 2")
    if prefer_train_stability and train_stability_splits == 0:
        raise ValueError("prefer_train_stability requires train_stability_splits")
    if not 0 < transition_risk_multiplier <= 1:
        raise ValueError("transition_risk_multiplier must be in (0, 1]")
    if per_symbol_selection and not strategy_names:
        raise ValueError("per-symbol selection needs at least one base strategy")
    if per_symbol_only and not per_symbol_selection:
        raise ValueError("per_symbol_only requires per_symbol_selection")
    if per_symbol_only and (candidate_maps or recipe_maps):
        raise ValueError("per_symbol_only cannot be combined with candidate or recipe maps")
    selected_symbols = _selected_symbols(
        prices=prices,
        quotes=quotes,
        symbols=symbols,
        recipe_maps=recipe_maps,
        strategy_names=strategy_names,
        candidate_maps=candidate_maps,
    )
    candidates = _selected_candidates(
        strategy_names=strategy_names,
        candidate_maps=candidate_maps,
        recipe_maps=recipe_maps,
        symbols=selected_symbols,
    )
    timestamps = _common_timestamps(prices, quotes, selected_symbols)
    if len(timestamps) < train_size + test_size:
        raise ValueError(
            "not enough aligned timestamps for one adaptive strategy-selection fold"
        )

    folds: list[AdaptiveStrategySelectionFold] = []
    cooldowns = {candidate.label: 0 for candidate in candidates}
    previous_selected_strategy: str | None = None
    for fold_index, start in enumerate(
        range(0, len(timestamps) - train_size - test_size + 1, step_size),
        start=1,
    ):
        train_timestamps = timestamps[start : start + train_size]
        test_timestamps = timestamps[
            start + train_size : start + train_size + test_size
        ]
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
        base_train_scores = tuple(
            sorted(
                (
                    _score_training_candidate(
                        config=config,
                        prices=train_prices,
                        quotes=train_quotes,
                        candidate=candidate,
                        train_stability_splits=train_stability_splits,
                    )
                    for candidate in candidates
                ),
                key=lambda score: score.rank_key_with_preferences(
                    fill_penalty_pct=train_fill_penalty_pct,
                    prefer_train_stability=prefer_train_stability,
                ),
                reverse=True,
            )
        )
        dynamic_candidate: AdaptiveStrategyCandidate | None = None
        if per_symbol_selection:
            dynamic_candidate = _build_per_symbol_adaptive_candidate(
                config=config,
                prices=train_prices,
                quotes=train_quotes,
                symbols=selected_symbols,
                strategy_names=strategy_names,
                train_fill_penalty_pct=train_fill_penalty_pct,
                train_stability_splits=train_stability_splits,
                prefer_train_stability=prefer_train_stability,
            )
            dynamic_score = _score_training_candidate(
                config=config,
                prices=train_prices,
                quotes=train_quotes,
                candidate=dynamic_candidate,
                train_stability_splits=train_stability_splits,
            )
            candidate_scores = (
                (dynamic_score,)
                if per_symbol_only
                else base_train_scores + (dynamic_score,)
            )
            train_scores = tuple(
                sorted(
                    candidate_scores,
                    key=lambda score: score.rank_key_with_preferences(
                        fill_penalty_pct=train_fill_penalty_pct,
                        prefer_train_stability=prefer_train_stability,
                    ),
                    reverse=True,
                )
            )
        else:
            train_scores = base_train_scores
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
            _cash_strategy_map(selected_symbols)
            if selected_strategy == CASH_FALLBACK_LABEL
            else (
                dynamic_candidate.strategy_by_symbol
                if (
                    dynamic_candidate is not None
                    and selected_strategy == dynamic_candidate.label
                )
                else _candidate_by_label(candidates, selected_strategy).strategy_by_symbol
            )
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

        combined_timestamps = train_timestamps + test_timestamps
        if selected_strategy == CASH_FALLBACK_LABEL:
            train_scores = train_scores + (
                _cash_train_score(
                    symbols=selected_symbols,
                    starting_equity=config.competition.starting_equity,
                ),
            )
            evaluation = _cash_evaluation(
                test_timestamps,
                starting_equity=config.competition.starting_equity,
            )
            full_run_fills = 0
        else:
            selected_candidate = (
                dynamic_candidate
                if (
                    dynamic_candidate is not None
                    and selected_strategy == dynamic_candidate.label
                )
                else _candidate_by_label(candidates, selected_strategy)
            )
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
            full_result = _run_candidate(
                config=_config_with_risk_multiplier(
                    config,
                    multiplier=evaluation_risk_multiplier,
                ),
                prices=combined_prices,
                quotes=combined_quotes,
                candidate=selected_candidate,
                target_notional_multiplier=evaluation_risk_multiplier,
            )
            evaluation = evaluate_portfolio_after_warmup(
                full_result,
                evaluation_start=test_timestamps[0],
            )
            full_run_fills = len(full_result.fills)
        folds.append(
            AdaptiveStrategySelectionFold(
                fold_index=fold_index,
                train_start=train_timestamps[0].isoformat(),
                train_end=train_timestamps[-1].isoformat(),
                test_start=test_timestamps[0].isoformat(),
                test_end=test_timestamps[-1].isoformat(),
                selected_strategy=selected_strategy,
                selected_strategy_map=selected_strategy_map,
                cooldown_blocked_strategies=cooldown_blocked_strategies,
                train_gate_blocked_strategies=train_gate_blocked_strategies,
                train_scores=train_scores,
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

    result_strategy_names = (
        (PER_SYMBOL_ADAPTIVE_LABEL,)
        if per_symbol_only
        else tuple(candidate.label for candidate in candidates)
    )
    if per_symbol_selection and not per_symbol_only:
        result_strategy_names = result_strategy_names + (PER_SYMBOL_ADAPTIVE_LABEL,)
    if allow_cash_fallback:
        result_strategy_names = result_strategy_names + (CASH_FALLBACK_LABEL,)

    return AdaptiveStrategySelectionResult(
        strategy_names=result_strategy_names,
        symbols=selected_symbols,
        folds=tuple(folds),
        loss_cooldown_folds=loss_cooldown_folds,
        min_train_fills=min_train_fills,
        min_train_drawdown_adjusted_return_pct=min_train_drawdown_adjusted_return_pct,
        train_fill_penalty_pct=train_fill_penalty_pct,
        train_stability_splits=train_stability_splits,
        prefer_train_stability=prefer_train_stability,
        transition_risk_multiplier=transition_risk_multiplier,
        allow_cash_fallback=allow_cash_fallback,
        per_symbol_selection=per_symbol_selection,
        per_symbol_only=per_symbol_only,
    )


def write_adaptive_strategy_selection_summary_csv(
    result: AdaptiveStrategySelectionResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategies",
                "symbols",
                "folds",
                "loss_cooldown_folds",
                "min_train_fills",
                "min_train_drawdown_adjusted_return_pct",
                "train_fill_penalty_pct",
                "train_stability_splits",
                "prefer_train_stability",
                "transition_risk_multiplier",
                "allow_cash_fallback",
                "per_symbol_selection",
                "per_symbol_only",
                "positive_fold_fraction",
                "active_fold_fraction",
                "active_positive_fold_fraction",
                "non_negative_fold_fraction",
                "losing_fold_fraction",
                "compounded_test_return_pct",
                "median_test_return_pct",
                "median_active_test_return_pct",
                "median_test_sharpe_15m",
                "worst_test_drawdown_pct",
                "average_risk_discipline_score",
                "total_evaluation_fills",
                "selection_counts",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategies": " ".join(result.strategy_names),
                "symbols": " ".join(result.symbols),
                "folds": len(result.folds),
                "loss_cooldown_folds": result.loss_cooldown_folds,
                "min_train_fills": result.min_train_fills,
                "min_train_drawdown_adjusted_return_pct": (
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
                "allow_cash_fallback": "yes" if result.allow_cash_fallback else "no",
                "per_symbol_selection": "yes" if result.per_symbol_selection else "no",
                "per_symbol_only": "yes" if result.per_symbol_only else "no",
                "positive_fold_fraction": result.positive_fold_fraction,
                "active_fold_fraction": result.active_fold_fraction,
                "active_positive_fold_fraction": result.active_positive_fold_fraction,
                "non_negative_fold_fraction": result.non_negative_fold_fraction,
                "losing_fold_fraction": result.losing_fold_fraction,
                "compounded_test_return_pct": result.compounded_test_return_pct,
                "median_test_return_pct": result.median_test_return_pct,
                "median_active_test_return_pct": result.median_active_test_return_pct,
                "median_test_sharpe_15m": result.median_test_sharpe_15m,
                "worst_test_drawdown_pct": result.worst_test_drawdown_pct,
                "average_risk_discipline_score": result.average_risk_discipline_score,
                "total_evaluation_fills": result.total_evaluation_fills,
                "selection_counts": _selection_counts_text(result.selection_counts),
            }
        )


def write_adaptive_strategy_selection_folds_csv(
    result: AdaptiveStrategySelectionResult,
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
                "selected_strategy",
                "selected_strategy_map",
                "cooldown_blocked_strategies",
                "train_gate_blocked_strategies",
                "selected_train_return_pct",
                "selected_train_drawdown_adjusted_return_pct",
                "selected_train_sharpe_15m",
                "selected_train_max_drawdown_pct",
                "selected_train_risk_score",
                "selected_train_stability_splits",
                "selected_train_stability_active_fraction",
                "selected_train_stability_positive_fraction",
                "selected_train_stability_active_positive_fraction",
                "selected_train_stability_non_negative_fraction",
                "selected_train_stability_median_return_pct",
                "selected_train_stability_median_active_return_pct",
                "evaluation_risk_multiplier",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "evaluation_fills",
                "full_run_fills",
                "final_equity",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            metrics = fold.metrics
            selected_score = fold.selected_train_score
            writer.writerow(
                {
                    "fold": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "selected_strategy": fold.selected_strategy,
                    "selected_strategy_map": " ".join(
                        f"{symbol}={strategy}"
                        for symbol, strategy in fold.selected_strategy_map
                    ),
                    "cooldown_blocked_strategies": "|".join(
                        fold.cooldown_blocked_strategies
                    ),
                    "train_gate_blocked_strategies": "|".join(
                        fold.train_gate_blocked_strategies
                    ),
                    "selected_train_return_pct": selected_score.return_pct,
                    "selected_train_drawdown_adjusted_return_pct": (
                        selected_score.drawdown_adjusted_return_pct
                    ),
                    "selected_train_sharpe_15m": selected_score.sharpe_15m,
                    "selected_train_max_drawdown_pct": selected_score.max_drawdown_pct,
                    "selected_train_risk_score": selected_score.risk_discipline_score,
                    "selected_train_stability_splits": selected_score.stability.splits,
                    "selected_train_stability_active_fraction": (
                        selected_score.stability.active_fraction
                    ),
                    "selected_train_stability_positive_fraction": (
                        selected_score.stability.positive_fraction
                    ),
                    "selected_train_stability_active_positive_fraction": (
                        selected_score.stability.active_positive_fraction
                    ),
                    "selected_train_stability_non_negative_fraction": (
                        selected_score.stability.non_negative_fraction
                    ),
                    "selected_train_stability_median_return_pct": (
                        selected_score.stability.median_return_pct
                    ),
                    "selected_train_stability_median_active_return_pct": (
                        selected_score.stability.median_active_return_pct
                    ),
                    "evaluation_risk_multiplier": fold.evaluation_risk_multiplier,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": fold.risk_discipline.score,
                    "evaluation_fills": len(fold.evaluation.fills),
                    "full_run_fills": fold.full_run_fills,
                    "final_equity": metrics.final_equity,
                }
            )


def write_adaptive_strategy_selection_scores_csv(
    result: AdaptiveStrategySelectionResult,
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
                "strategy_map",
                "selected",
                "train_return_pct",
                "train_drawdown_adjusted_return_pct",
                "train_max_drawdown_pct",
                "train_sharpe_15m",
                "train_risk_score",
                "train_fills",
                "train_stability_splits",
                "train_stability_active_fraction",
                "train_stability_positive_fraction",
                "train_stability_active_positive_fraction",
                "train_stability_non_negative_fraction",
                "train_stability_median_return_pct",
                "train_stability_median_active_return_pct",
                "train_gate_passed",
                "train_final_equity",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            for score in fold.train_scores:
                writer.writerow(
                    {
                        "fold": fold.fold_index,
                        "strategy": score.strategy_name,
                        "strategy_map": score.strategy_map_text,
                        "selected": (
                            "yes" if score.strategy_name == fold.selected_strategy else "no"
                        ),
                        "train_return_pct": score.return_pct,
                        "train_drawdown_adjusted_return_pct": (
                            score.drawdown_adjusted_return_pct
                        ),
                        "train_max_drawdown_pct": score.max_drawdown_pct,
                        "train_sharpe_15m": score.sharpe_15m,
                        "train_risk_score": score.risk_discipline_score,
                        "train_fills": score.fills,
                        "train_stability_splits": score.stability.splits,
                        "train_stability_active_fraction": score.stability.active_fraction,
                        "train_stability_positive_fraction": (
                            score.stability.positive_fraction
                        ),
                        "train_stability_active_positive_fraction": (
                            score.stability.active_positive_fraction
                        ),
                        "train_stability_non_negative_fraction": (
                            score.stability.non_negative_fraction
                        ),
                        "train_stability_median_return_pct": (
                            score.stability.median_return_pct
                        ),
                        "train_stability_median_active_return_pct": (
                            score.stability.median_active_return_pct
                        ),
                        "train_gate_passed": (
                            "yes"
                            if _passes_train_gate(
                                score,
                                min_train_fills=result.min_train_fills,
                                min_train_drawdown_adjusted_return_pct=(
                                    result.min_train_drawdown_adjusted_return_pct
                                ),
                            )
                            else "no"
                        ),
                        "train_final_equity": score.final_equity,
                    }
                )


def write_adaptive_strategy_promotion_audit_csv(
    audit: AdaptivePromotionAudit,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "live_ready",
                "decision_reason",
                "gate_id",
                "category",
                "passed",
                "value",
                "threshold",
                "comparator",
                "gap",
                "details",
            ],
        )
        writer.writeheader()
        for gate in audit.gates:
            writer.writerow(
                {
                    "status": audit.decision.status,
                    "live_ready": "yes" if audit.decision.live_ready else "no",
                    "decision_reason": audit.decision.reason,
                    "gate_id": gate.gate_id,
                    "category": gate.category,
                    "passed": "yes" if gate.passed else "no",
                    "value": gate.value,
                    "threshold": gate.threshold,
                    "comparator": gate.comparator,
                    "gap": _gate_gap(gate),
                    "details": gate.details,
                }
            )


def build_adaptive_strategy_stitched_equity_curve(
    result: AdaptiveStrategySelectionResult,
    *,
    starting_equity: float,
) -> tuple[AdaptiveStrategyStitchedEquityPoint, ...]:
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")

    current_equity = starting_equity
    peak_equity = starting_equity
    last_timestamp: datetime | None = None
    points: list[AdaptiveStrategyStitchedEquityPoint] = []
    for fold in result.folds:
        source_points = fold.evaluation.equity_curve
        if not source_points:
            continue
        source_start_equity = source_points[0].equity
        if source_start_equity <= 0:
            raise ValueError("fold source equity must be positive")

        fold_start_equity = current_equity
        last_fold_equity: float | None = None
        for source_point in source_points:
            timestamp = datetime.fromisoformat(source_point.timestamp)
            if last_timestamp is not None and timestamp <= last_timestamp:
                continue
            stitched_equity = (
                fold_start_equity * source_point.equity / source_start_equity
            )
            peak_equity = max(peak_equity, stitched_equity)
            last_timestamp = timestamp
            last_fold_equity = stitched_equity
            points.append(
                AdaptiveStrategyStitchedEquityPoint(
                    timestamp=source_point.timestamp,
                    equity=stitched_equity,
                    drawdown_pct=max(0.0, 1.0 - (stitched_equity / peak_equity)),
                    fold_index=fold.fold_index,
                    selected_strategy=fold.selected_strategy,
                    fold_return_pct=fold.metrics.return_pct,
                    source_fold_equity=source_point.equity,
                    source_fold_start_equity=source_start_equity,
                )
            )
        if last_fold_equity is not None:
            current_equity = last_fold_equity

    return tuple(points)


def write_adaptive_strategy_stitched_equity_csv(
    result: AdaptiveStrategySelectionResult,
    path: str | Path,
    *,
    starting_equity: float,
) -> tuple[AdaptiveStrategyStitchedEquityPoint, ...]:
    curve = build_adaptive_strategy_stitched_equity_curve(
        result,
        starting_equity=starting_equity,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "equity",
                "drawdown_pct",
                "fold",
                "selected_strategy",
                "fold_return_pct",
                "source_fold_equity",
                "source_fold_start_equity",
            ],
        )
        writer.writeheader()
        for point in curve:
            writer.writerow(
                {
                    "timestamp": point.timestamp,
                    "equity": point.equity,
                    "drawdown_pct": point.drawdown_pct,
                    "fold": point.fold_index,
                    "selected_strategy": point.selected_strategy,
                    "fold_return_pct": point.fold_return_pct,
                    "source_fold_equity": point.source_fold_equity,
                    "source_fold_start_equity": point.source_fold_start_equity,
                }
            )
    return curve


def _cash_strategy_map(symbols: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((symbol, CASH_FALLBACK_LABEL) for symbol in symbols)


def _cash_train_score(
    *,
    symbols: tuple[str, ...],
    starting_equity: float,
) -> AdaptiveStrategyTrainScore:
    return AdaptiveStrategyTrainScore(
        strategy_name=CASH_FALLBACK_LABEL,
        strategy_map=_cash_strategy_map(symbols),
        return_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe_15m=0.0,
        risk_discipline_score=100,
        fills=0,
        final_equity=starting_equity,
        stability=AdaptiveStrategyTrainStability(),
    )


def _cash_evaluation(
    timestamps: tuple[datetime, ...],
    *,
    starting_equity: float,
) -> WarmupPortfolioEvaluation:
    if not timestamps:
        raise ValueError("cash fallback needs at least one evaluation timestamp")
    equity_points = tuple(
        PortfolioEquityPoint(
            timestamp=timestamp.isoformat(),
            equity=starting_equity,
            cash=starting_equity,
            gross_notional_usd=0.0,
            net_notional_usd=0.0,
            drawdown_pct=0.0,
            positions=(),
        )
        for timestamp in timestamps
    )
    metrics = build_competition_metrics(
        equity_points=equity_points,
        fills=(),
    )
    risk_discipline = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(equity_points)
    )
    return WarmupPortfolioEvaluation(
        evaluation_start=timestamps[0].isoformat(),
        equity_curve=equity_points,
        fills=(),
        competition_metrics=metrics,
        risk_discipline=risk_discipline,
    )


def _score_training_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    candidate: AdaptiveStrategyCandidate,
    train_stability_splits: int = 0,
) -> AdaptiveStrategyTrainScore:
    result = _run_candidate(
        config=config,
        prices=prices,
        quotes=quotes,
        candidate=candidate,
    )
    metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk_discipline = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(result.equity_curve)
    )
    return AdaptiveStrategyTrainScore(
        strategy_name=candidate.label,
        strategy_map=candidate.strategy_by_symbol,
        return_pct=metrics.return_pct,
        max_drawdown_pct=metrics.max_drawdown_pct,
        sharpe_15m=metrics.sharpe_15m,
        risk_discipline_score=risk_discipline.score,
        fills=metrics.trade_count,
        final_equity=metrics.final_equity,
        stability=_score_training_stability(
            config=config,
            prices=prices,
            quotes=quotes,
            candidate=candidate,
            splits=train_stability_splits,
        ),
    )


def _score_training_stability(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    candidate: AdaptiveStrategyCandidate,
    splits: int,
) -> AdaptiveStrategyTrainStability:
    if splits <= 0:
        return AdaptiveStrategyTrainStability()

    symbols = tuple(symbol for symbol, _ in candidate.strategy_by_symbol)
    timestamps = _common_timestamps(prices, quotes, symbols)
    if len(timestamps) < splits * 2:
        return AdaptiveStrategyTrainStability()

    split_size = len(timestamps) // splits
    returns: list[float] = []
    active_returns: list[float] = []
    for split_index in range(splits):
        start = split_index * split_size
        end = (split_index + 1) * split_size
        if split_index == splits - 1:
            end = len(timestamps)
        sub_timestamps = timestamps[start:end]
        if len(sub_timestamps) < 2:
            continue

        sub_result = _run_candidate(
            config=config,
            prices=_slice_prices(
                prices,
                symbols=symbols,
                timestamps=sub_timestamps,
            ),
            quotes=_slice_quotes(
                quotes,
                symbols=symbols,
                timestamps=sub_timestamps,
            ),
            candidate=candidate,
        )
        metrics = build_competition_metrics(
            equity_points=sub_result.equity_curve,
            fills=sub_result.fills,
        )
        returns.append(metrics.return_pct)
        if metrics.trade_count > 0 or abs(metrics.return_pct) > RETURN_EPSILON:
            active_returns.append(metrics.return_pct)

    if not returns:
        return AdaptiveStrategyTrainStability()

    return AdaptiveStrategyTrainStability(
        splits=len(returns),
        active_fraction=len(active_returns) / len(returns),
        positive_fraction=len([value for value in returns if value > 0]) / len(returns),
        active_positive_fraction=(
            len([value for value in active_returns if value > 0]) / len(active_returns)
            if active_returns
            else 0.0
        ),
        non_negative_fraction=(
            len([value for value in returns if value >= -RETURN_EPSILON])
            / len(returns)
        ),
        median_return_pct=median(returns),
        median_active_return_pct=median(active_returns) if active_returns else 0.0,
    )


def _build_per_symbol_adaptive_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
    strategy_names: tuple[str, ...],
    train_fill_penalty_pct: float,
    train_stability_splits: int,
    prefer_train_stability: bool,
) -> AdaptiveStrategyCandidate:
    normalized_strategies = _normalize_unique_strategy_names(strategy_names)
    if not normalized_strategies:
        raise ValueError("per-symbol selection needs at least one base strategy")

    selected: list[tuple[str, str]] = []
    for symbol in symbols:
        symbol_scores = tuple(
            sorted(
                (
                    _score_training_candidate(
                        config=config,
                        prices=prices,
                        quotes=quotes,
                        candidate=AdaptiveStrategyCandidate(
                            label=strategy_name,
                            strategy_by_symbol=((symbol, strategy_name),),
                        ),
                        train_stability_splits=train_stability_splits,
                    )
                    for strategy_name in normalized_strategies
                ),
                key=lambda score: score.rank_key_with_preferences(
                    fill_penalty_pct=train_fill_penalty_pct,
                    prefer_train_stability=prefer_train_stability,
                ),
                reverse=True,
            )
        )
        selected.append((symbol, symbol_scores[0].strategy_name))

    return AdaptiveStrategyCandidate(
        label=PER_SYMBOL_ADAPTIVE_LABEL,
        strategy_by_symbol=tuple(selected),
    )


def _passes_train_gate(
    score: AdaptiveStrategyTrainScore,
    *,
    min_train_fills: int,
    min_train_drawdown_adjusted_return_pct: float | None,
) -> bool:
    if score.fills < min_train_fills:
        return False
    if (
        min_train_drawdown_adjusted_return_pct is not None
        and score.drawdown_adjusted_return_pct < min_train_drawdown_adjusted_return_pct
    ):
        return False
    return True


def _config_with_risk_multiplier(
    config: AppConfig,
    *,
    multiplier: float,
) -> AppConfig:
    if multiplier == 1.0:
        return config
    return replace(
        config,
        risk=replace(
            config.risk,
            max_gross_leverage=config.risk.max_gross_leverage * multiplier,
            max_symbol_notional_pct=config.risk.max_symbol_notional_pct * multiplier,
        ),
    )


def _run_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    candidate: AdaptiveStrategyCandidate,
    target_notional_multiplier: float = 1.0,
):
    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(
                candidate.strategy_for_symbol(symbol),
                symbol=symbol,
            )
            for symbol, _ in candidate.strategy_by_symbol
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol, _ in candidate.strategy_by_symbol
        },
        clock=config.competition.to_clock(),
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
        target_notional_multiplier=target_notional_multiplier,
    )
    return engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )


def _selected_symbols(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None,
    recipe_maps: tuple[AdaptiveStrategyCandidate, ...] = (),
    strategy_names: tuple[str, ...] = (),
    candidate_maps: tuple[AdaptiveStrategyCandidate, ...] = (),
) -> tuple[str, ...]:
    if symbols:
        selected = tuple(instrument_for(symbol).symbol for symbol in symbols)
    elif recipe_maps and not strategy_names and not candidate_maps:
        selected = tuple(
            dict.fromkeys(
                instrument_for(symbol).symbol
                for recipe in recipe_maps
                for symbol, _ in recipe.strategy_by_symbol
            )
        )
    else:
        selected = tuple(sorted(set(prices.symbols()) & set(quotes.symbols())))
    if not selected:
        raise ValueError("adaptive strategy selection requires at least one symbol")
    missing_prices = sorted(set(selected) - set(prices.symbols()))
    if missing_prices:
        raise ValueError(f"missing price history for: {', '.join(missing_prices)}")
    missing_quotes = sorted(set(selected) - set(quotes.symbols()))
    if missing_quotes:
        raise ValueError(f"missing quote history for: {', '.join(missing_quotes)}")
    return selected


def _selected_candidates(
    *,
    strategy_names: tuple[str, ...],
    candidate_maps: tuple[AdaptiveStrategyCandidate, ...],
    recipe_maps: tuple[AdaptiveStrategyCandidate, ...],
    symbols: tuple[str, ...],
) -> tuple[AdaptiveStrategyCandidate, ...]:
    if not strategy_names and not candidate_maps and not recipe_maps:
        raise ValueError("adaptive strategy selection needs at least one strategy")
    selected: list[str] = []
    candidates: list[AdaptiveStrategyCandidate] = []
    for strategy_name in strategy_names:
        normalized = normalize_strategy_name(strategy_name)
        if normalized not in selected:
            selected.append(normalized)
            candidates.append(
                AdaptiveStrategyCandidate(
                    label=normalized,
                    strategy_by_symbol=tuple(
                        (symbol, normalized) for symbol in symbols
                    ),
                )
            )
    for candidate_map in candidate_maps:
        candidate = _normalize_candidate_map(candidate_map, symbols=symbols)
        if candidate.label in selected:
            raise ValueError(f"duplicate adaptive candidate label {candidate.label!r}")
        selected.append(candidate.label)
        candidates.append(candidate)
    for recipe_map in recipe_maps:
        candidate = _normalize_recipe_map(recipe_map, symbols=symbols)
        if candidate.label in selected:
            raise ValueError(f"duplicate adaptive candidate label {candidate.label!r}")
        selected.append(candidate.label)
        candidates.append(candidate)
    return tuple(candidates)


def _normalize_unique_strategy_names(strategy_names: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for raw_name in strategy_names:
        strategy_name = normalize_strategy_name(raw_name)
        if strategy_name not in selected:
            selected.append(strategy_name)
    return tuple(selected)


def _normalize_candidate_map(
    candidate: AdaptiveStrategyCandidate,
    *,
    symbols: tuple[str, ...],
) -> AdaptiveStrategyCandidate:
    selected_symbols = set(symbols)
    normalized_map = tuple(
        sorted(
            (
                instrument_for(symbol).symbol,
                normalize_strategy_name(strategy_name),
            )
            for symbol, strategy_name in candidate.strategy_by_symbol
        )
    )
    map_symbols = {symbol for symbol, _ in normalized_map}
    missing = sorted(selected_symbols - map_symbols)
    extra = sorted(map_symbols - selected_symbols)
    if missing:
        raise ValueError(
            f"candidate map {candidate.label!r} missing symbols: {', '.join(missing)}"
        )
    if extra:
        raise ValueError(
            f"candidate map {candidate.label!r} has symbols outside selection: "
            f"{', '.join(extra)}"
        )
    return AdaptiveStrategyCandidate(
        label=candidate.label,
        strategy_by_symbol=normalized_map,
    )


def _normalize_recipe_map(
    candidate: AdaptiveStrategyCandidate,
    *,
    symbols: tuple[str, ...],
) -> AdaptiveStrategyCandidate:
    selected_symbols = set(symbols)
    normalized_map = tuple(
        sorted(
            (
                instrument_for(symbol).symbol,
                normalize_strategy_name(strategy_name),
            )
            for symbol, strategy_name in candidate.strategy_by_symbol
        )
    )
    map_symbols = {symbol for symbol, _ in normalized_map}
    extra = sorted(map_symbols - selected_symbols)
    if extra:
        raise ValueError(
            f"recipe map {candidate.label!r} has symbols outside selection: "
            f"{', '.join(extra)}"
        )
    if not normalized_map:
        raise ValueError(f"recipe map {candidate.label!r} cannot be empty")
    return AdaptiveStrategyCandidate(
        label=candidate.label,
        strategy_by_symbol=normalized_map,
    )


def _candidate_by_label(
    candidates: tuple[AdaptiveStrategyCandidate, ...],
    label: str,
) -> AdaptiveStrategyCandidate:
    for candidate in candidates:
        if candidate.label == label:
            return candidate
    raise KeyError(f"no adaptive candidate named {label!r}")


def _adaptive_promotion_gates(
    result: AdaptiveStrategySelectionResult,
    *,
    min_positive_fold_fraction: float,
    min_active_positive_fold_fraction: float,
    min_non_negative_fold_fraction: float,
    min_live_positive_fold_fraction: float,
    min_live_active_positive_fold_fraction: float,
    min_median_active_return_pct: float,
    max_worst_drawdown_pct: float,
    min_average_risk_discipline_score: float,
) -> tuple[AdaptivePromotionGate, ...]:
    active_folds = result.active_folds
    return (
        AdaptivePromotionGate(
            gate_id="folds_present",
            category="research",
            passed=bool(result.folds),
            value=float(len(result.folds)),
            threshold=1.0,
            comparator=">=",
            details="at least one fixed-warmup evaluation fold exists",
        ),
        AdaptivePromotionGate(
            gate_id="active_folds_present",
            category="research",
            passed=bool(active_folds),
            value=float(len(active_folds)),
            threshold=1.0,
            comparator=">=",
            details="at least one fold has trades or non-zero return",
        ),
        AdaptivePromotionGate(
            gate_id="non_negative_fold_fraction",
            category="research",
            passed=result.non_negative_fold_fraction >= min_non_negative_fold_fraction,
            value=result.non_negative_fold_fraction,
            threshold=min_non_negative_fold_fraction,
            comparator=">=",
            details="fraction of folds with return at or above zero",
        ),
        AdaptivePromotionGate(
            gate_id="research_positive_or_active_positive",
            category="research",
            passed=(
                result.positive_fold_fraction >= min_positive_fold_fraction
                or result.active_positive_fold_fraction
                >= min_active_positive_fold_fraction
            ),
            value=max(
                result.positive_fold_fraction,
                result.active_positive_fold_fraction,
            ),
            threshold=min(
                min_positive_fold_fraction,
                min_active_positive_fold_fraction,
            ),
            comparator="positive>=threshold OR active_positive>=threshold",
            details="research pass requires enough positive total or active folds",
        ),
        AdaptivePromotionGate(
            gate_id="median_active_test_return",
            category="research",
            passed=result.median_active_test_return_pct > min_median_active_return_pct,
            value=result.median_active_test_return_pct,
            threshold=min_median_active_return_pct,
            comparator=">",
            details="median return across active folds must be positive",
        ),
        AdaptivePromotionGate(
            gate_id="worst_test_drawdown",
            category="research",
            passed=result.worst_test_drawdown_pct <= max_worst_drawdown_pct,
            value=result.worst_test_drawdown_pct,
            threshold=max_worst_drawdown_pct,
            comparator="<=",
            details="worst out-of-sample fold drawdown stays below guardrail",
        ),
        AdaptivePromotionGate(
            gate_id="average_risk_discipline",
            category="research",
            passed=(
                result.average_risk_discipline_score
                >= min_average_risk_discipline_score
            ),
            value=result.average_risk_discipline_score,
            threshold=min_average_risk_discipline_score,
            comparator=">=",
            details="average fold risk discipline score remains high",
        ),
        AdaptivePromotionGate(
            gate_id="live_positive_fold_fraction",
            category="live",
            passed=result.positive_fold_fraction >= min_live_positive_fold_fraction,
            value=result.positive_fold_fraction,
            threshold=min_live_positive_fold_fraction,
            comparator=">=",
            details="stricter live gate for positive total folds",
        ),
        AdaptivePromotionGate(
            gate_id="live_active_positive_fold_fraction",
            category="live",
            passed=(
                result.active_positive_fold_fraction
                >= min_live_active_positive_fold_fraction
            ),
            value=result.active_positive_fold_fraction,
            threshold=min_live_active_positive_fold_fraction,
            comparator=">=",
            details="stricter live gate for positive active folds",
        ),
    )


def _gate_gap(gate: AdaptivePromotionGate) -> float:
    if gate.passed:
        return 0.0
    if gate.comparator == "<=":
        return gate.value - gate.threshold
    return gate.threshold - gate.value


def _validate_window_sizes(*, train_size: int, test_size: int, step_size: int) -> None:
    if train_size < 1:
        raise ValueError("train_size must be at least 1")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")


def _common_timestamps(
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
) -> tuple[datetime, ...]:
    common: set[datetime] | None = None
    for symbol in symbols:
        price_timestamps = {bar.timestamp for bar in prices.for_symbol(symbol).bars}
        quote_timestamps = {quote.timestamp for quote in quotes.for_symbol(symbol).quotes}
        timestamps = price_timestamps & quote_timestamps
        if not timestamps:
            raise ValueError(f"no aligned price/quote timestamps for {symbol}")
        common = timestamps if common is None else common & timestamps
    if not common:
        raise ValueError("no common timestamps across selected symbols")
    return tuple(sorted(common))


def _slice_prices(
    prices: PriceHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> PriceHistory:
    timestamp_set = set(timestamps)
    return PriceHistory(
        tuple(
            PriceBar(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                close=bar.close,
            )
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
            QuoteSnapshot(
                timestamp=quote.timestamp,
                symbol=quote.symbol,
                bid=quote.bid,
                ask=quote.ask,
            )
            for quote in quotes.quotes
            if quote.symbol in symbols and quote.timestamp in timestamp_set
        )
    )


def _selection_counts_text(counts: tuple[StrategySelectionCount, ...]) -> str:
    return ";".join(f"{count.strategy_name}={count.folds}" for count in counts)
