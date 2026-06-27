from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quanthack.backtesting.deployment_profile_selector import (
    DeploymentProfileNextSelection,
    select_next_deployment_profile,
)
from quanthack.backtesting.deployment_profile_selector_sweep import (
    DeploymentProfileSelectorSweepCandidate,
    DeploymentProfileSelectorSweepResult,
    sweep_deployment_profile_selector,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class DeploymentProfileRecommendation:
    recommended_slot: str
    recommended_label: str
    recommendation_reason: str
    promotion_status: str
    promotion_reason: str
    selector_score: float
    fallback_slot: str
    min_past_folds: int
    drawdown_penalty: float
    risk_score_floor: float
    require_past_activity: bool
    completed_folds: int
    positive_fold_fraction: float
    active_positive_fold_fraction: float
    non_negative_fold_fraction: float
    cumulative_test_return_pct: float
    median_active_test_return_pct: float
    worst_test_drawdown_pct: float
    average_risk_discipline_score: float
    largest_positive_fold_contribution: float
    total_evaluation_fills: int
    historical_selected_sequence: str
    historical_selected_counts: str
    data_window_start: str
    data_window_end: str
    past_scores: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class DeploymentProfileRecommendationResult:
    recommendation: DeploymentProfileRecommendation
    sweep: DeploymentProfileSelectorSweepResult
    best_candidate: DeploymentProfileSelectorSweepCandidate
    next_selection: DeploymentProfileNextSelection


def build_deployment_profile_recommendation(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slots: tuple[str, ...],
    fallback_slots: tuple[str, ...],
    min_past_folds_values: tuple[int, ...],
    drawdown_penalties: tuple[float, ...],
    risk_score_floors: tuple[float, ...],
    require_past_activity_values: tuple[bool, ...],
    train_size: int,
    test_size: int,
    step_size: int,
) -> DeploymentProfileRecommendationResult:
    sweep = sweep_deployment_profile_selector(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=profile_pack_json,
        slots=slots,
        fallback_slots=fallback_slots,
        min_past_folds_values=min_past_folds_values,
        drawdown_penalties=drawdown_penalties,
        risk_score_floors=risk_score_floors,
        require_past_activity_values=require_past_activity_values,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    if sweep.best is None:
        raise ValueError("selector sweep produced no candidates")
    best = sweep.best
    next_selection = select_next_deployment_profile(
        fixed_results=sweep.fixed_results,
        fallback_slot=best.result.fallback_slot,
        min_past_folds=best.result.min_past_folds,
        drawdown_penalty=best.result.drawdown_penalty,
        risk_score_floor=best.result.risk_score_floor,
        require_past_activity=best.result.require_past_activity,
    )
    adaptive = best.result.adaptive_result
    recommendation = DeploymentProfileRecommendation(
        recommended_slot=next_selection.selected_slot,
        recommended_label=next_selection.selected_label,
        recommendation_reason=next_selection.selection_reason,
        promotion_status=best.promotion.status,
        promotion_reason=best.promotion.reason,
        selector_score=best.selector_score,
        fallback_slot=best.result.fallback_slot,
        min_past_folds=best.result.min_past_folds,
        drawdown_penalty=best.result.drawdown_penalty,
        risk_score_floor=best.result.risk_score_floor,
        require_past_activity=best.result.require_past_activity,
        completed_folds=next_selection.completed_folds,
        positive_fold_fraction=adaptive.positive_fold_fraction,
        active_positive_fold_fraction=adaptive.active_positive_fold_fraction,
        non_negative_fold_fraction=adaptive.non_negative_fold_fraction,
        cumulative_test_return_pct=sum(
            fold.metrics.return_pct for fold in adaptive.folds
        ),
        median_active_test_return_pct=adaptive.median_active_test_return_pct,
        worst_test_drawdown_pct=adaptive.worst_test_drawdown_pct,
        average_risk_discipline_score=adaptive.average_risk_discipline_score,
        largest_positive_fold_contribution=(
            adaptive.largest_positive_fold_contribution
        ),
        total_evaluation_fills=adaptive.total_evaluation_fills,
        historical_selected_sequence=best.selected_sequence_text,
        historical_selected_counts=best.selected_counts_text,
        data_window_start=(
            "" if not adaptive.folds else adaptive.folds[0].train_start
        ),
        data_window_end="" if not adaptive.folds else adaptive.folds[-1].test_end,
        past_scores=tuple(_past_score_row(score) for score in next_selection.past_scores),
    )
    return DeploymentProfileRecommendationResult(
        recommendation=recommendation,
        sweep=sweep,
        best_candidate=best,
        next_selection=next_selection,
    )


def write_deployment_profile_recommendation_csv(
    result: DeploymentProfileRecommendationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result.recommendation)
    row["past_scores"] = json.dumps(row["past_scores"], sort_keys=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_deployment_profile_recommendation_json(
    result: DeploymentProfileRecommendationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result.recommendation), indent=2) + "\n",
        encoding="utf-8",
    )


def _past_score_row(score) -> dict[str, object]:
    return {
        "slot": score.slot,
        "score": score.score,
        "cumulative_return_pct": score.cumulative_return_pct,
        "worst_drawdown_pct": score.worst_drawdown_pct,
        "average_risk_discipline_score": score.average_risk_discipline_score,
        "active_fold_fraction": score.active_fold_fraction,
        "folds": score.folds,
        "eligible": score.eligible,
        "reason": score.reason,
    }
