from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.deployment_profile_selector import (
    DEFAULT_PROFILE_SLOTS,
    DeploymentProfileSelectorResult,
    DeploymentProfileWalkForward,
    build_deployment_profile_selector_result,
    run_deployment_profile_walk_forwards,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


DEFAULT_FALLBACK_SLOTS = ("conservative", "survival")
DEFAULT_MIN_PAST_FOLDS = (1, 2)
DEFAULT_DRAWDOWN_PENALTIES = (0.0, 0.5, 1.0, 2.0)
DEFAULT_RISK_SCORE_FLOORS = (95.0,)


@dataclass(frozen=True)
class DeploymentProfileSelectorSweepCandidate:
    result: DeploymentProfileSelectorResult
    promotion: FixedWarmupPromotionDecision
    selector_score: float

    @property
    def rank_key(self) -> tuple[float, ...]:
        adaptive = self.result.adaptive_result
        return (
            _promotion_rank(self.promotion.status),
            self.selector_score,
            adaptive.active_positive_fold_fraction,
            adaptive.non_negative_fold_fraction,
            adaptive.median_active_test_return_pct,
            1.0 - adaptive.largest_positive_fold_contribution,
            -adaptive.worst_test_drawdown_pct,
            adaptive.average_risk_discipline_score,
            -_fallback_safety_rank(self.result.fallback_slot),
        )

    @property
    def selected_counts_text(self) -> str:
        return " ".join(
            f"{slot}={count}" for slot, count in sorted(self.result.selected_counts.items())
        )

    @property
    def selected_sequence_text(self) -> str:
        return " ".join(selection.selected_slot for selection in self.result.selections)


@dataclass(frozen=True)
class DeploymentProfileSelectorSweepResult:
    fixed_results: tuple[DeploymentProfileWalkForward, ...]
    candidates: tuple[DeploymentProfileSelectorSweepCandidate, ...]

    @property
    def best(self) -> DeploymentProfileSelectorSweepCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def sweep_deployment_profile_selector(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slots: Sequence[str] = DEFAULT_PROFILE_SLOTS,
    fallback_slots: Sequence[str] = DEFAULT_FALLBACK_SLOTS,
    min_past_folds_values: Sequence[int] = DEFAULT_MIN_PAST_FOLDS,
    drawdown_penalties: Sequence[float] = DEFAULT_DRAWDOWN_PENALTIES,
    risk_score_floors: Sequence[float] = DEFAULT_RISK_SCORE_FLOORS,
    require_past_activity_values: Sequence[bool] = (True,),
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> DeploymentProfileSelectorSweepResult:
    fixed_results = run_deployment_profile_walk_forwards(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=profile_pack_json,
        slots=slots,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    fixed_slots = tuple(fixed.profile.slot for fixed in fixed_results)
    fallback_candidates = _validated_values(
        "fallback_slots",
        tuple(fallback_slots),
    )
    for fallback_slot in fallback_candidates:
        if fallback_slot not in fixed_slots:
            raise ValueError(f"fallback slot {fallback_slot!r} is not in selected slots")
    min_past_folds_candidates = tuple(int(value) for value in min_past_folds_values)
    if any(value < 1 for value in min_past_folds_candidates):
        raise ValueError("min_past_folds values must be at least 1")
    drawdown_penalty_candidates = tuple(float(value) for value in drawdown_penalties)
    if any(value < 0 for value in drawdown_penalty_candidates):
        raise ValueError("drawdown penalties cannot be negative")
    risk_score_candidates = tuple(float(value) for value in risk_score_floors)
    if any(value < 0 or value > 100 for value in risk_score_candidates):
        raise ValueError("risk score floors must be between 0 and 100")
    activity_candidates = tuple(require_past_activity_values)
    if not activity_candidates:
        raise ValueError("at least one require_past_activity value is required")

    candidates: list[DeploymentProfileSelectorSweepCandidate] = []
    for fallback_slot in fallback_candidates:
        for min_past_folds in min_past_folds_candidates:
            for drawdown_penalty in drawdown_penalty_candidates:
                for risk_score_floor in risk_score_candidates:
                    for require_past_activity in activity_candidates:
                        result = build_deployment_profile_selector_result(
                            fixed_results=fixed_results,
                            fallback_slot=fallback_slot,
                            min_past_folds=min_past_folds,
                            drawdown_penalty=drawdown_penalty,
                            risk_score_floor=risk_score_floor,
                            require_past_activity=require_past_activity,
                        )
                        promotion = decide_fixed_warmup_promotion(
                            result.adaptive_result
                        )
                        candidates.append(
                            DeploymentProfileSelectorSweepCandidate(
                                result=result,
                                promotion=promotion,
                                selector_score=_selector_score(result),
                            )
                        )
    return DeploymentProfileSelectorSweepResult(
        fixed_results=fixed_results,
        candidates=tuple(
            sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        ),
    )


def write_deployment_profile_selector_sweep_csv(
    sweep: DeploymentProfileSelectorSweepResult,
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
                "selector_score",
                "fallback_slot",
                "min_past_folds",
                "drawdown_penalty",
                "risk_score_floor",
                "require_past_activity",
                "folds",
                "positive_fold_fraction",
                "active_fold_fraction",
                "active_positive_fold_fraction",
                "non_negative_fold_fraction",
                "cumulative_test_return_pct",
                "median_test_return_pct",
                "median_active_test_return_pct",
                "median_test_sharpe_15m",
                "worst_test_drawdown_pct",
                "average_risk_discipline_score",
                "total_evaluation_fills",
                "largest_positive_fold_contribution",
                "selected_counts",
                "selected_sequence",
                "promotion_reason",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(sweep.candidates, start=1):
            adaptive = candidate.result.adaptive_result
            writer.writerow(
                {
                    "rank": rank,
                    "promotion_status": candidate.promotion.status,
                    "selector_score": candidate.selector_score,
                    "fallback_slot": candidate.result.fallback_slot,
                    "min_past_folds": candidate.result.min_past_folds,
                    "drawdown_penalty": candidate.result.drawdown_penalty,
                    "risk_score_floor": candidate.result.risk_score_floor,
                    "require_past_activity": candidate.result.require_past_activity,
                    "folds": len(adaptive.folds),
                    "positive_fold_fraction": adaptive.positive_fold_fraction,
                    "active_fold_fraction": adaptive.active_fold_fraction,
                    "active_positive_fold_fraction": (
                        adaptive.active_positive_fold_fraction
                    ),
                    "non_negative_fold_fraction": adaptive.non_negative_fold_fraction,
                    "cumulative_test_return_pct": _cumulative_test_return_pct(
                        candidate.result
                    ),
                    "median_test_return_pct": adaptive.median_test_return_pct,
                    "median_active_test_return_pct": (
                        adaptive.median_active_test_return_pct
                    ),
                    "median_test_sharpe_15m": adaptive.median_test_sharpe_15m,
                    "worst_test_drawdown_pct": adaptive.worst_test_drawdown_pct,
                    "average_risk_discipline_score": (
                        adaptive.average_risk_discipline_score
                    ),
                    "total_evaluation_fills": adaptive.total_evaluation_fills,
                    "largest_positive_fold_contribution": (
                        adaptive.largest_positive_fold_contribution
                    ),
                    "selected_counts": candidate.selected_counts_text,
                    "selected_sequence": candidate.selected_sequence_text,
                    "promotion_reason": candidate.promotion.reason,
                }
            )


def _selector_score(result: DeploymentProfileSelectorResult) -> float:
    adaptive = result.adaptive_result
    return (
        30.0 * adaptive.active_positive_fold_fraction
        + 20.0 * adaptive.non_negative_fold_fraction
        + 15.0 * min(adaptive.average_risk_discipline_score, 100.0) / 100.0
        + 25.0 * max(_cumulative_test_return_pct(result), 0.0) * 100.0
        + 10.0 * (1.0 - adaptive.largest_positive_fold_contribution)
        + 5.0 * max(adaptive.median_active_test_return_pct, 0.0) * 100.0
        - 10.0 * adaptive.worst_test_drawdown_pct * 100.0
    )


def _cumulative_test_return_pct(result: DeploymentProfileSelectorResult) -> float:
    return sum(fold.metrics.return_pct for fold in result.adaptive_result.folds)


def _promotion_rank(status: str) -> float:
    if status == "PROMOTE":
        return 3.0
    if status == "PAPER_ONLY":
        return 2.0
    return 1.0


def _fallback_safety_rank(slot: str) -> int:
    if slot == "survival":
        return 0
    if slot == "conservative":
        return 1
    return 2


def _validated_values(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = tuple(value.strip() for value in values if value.strip())
    if not cleaned:
        raise ValueError(f"{name} must include at least one value")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} values must be unique")
    return cleaned
