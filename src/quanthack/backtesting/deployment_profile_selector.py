from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from quanthack.backtesting.deployment_profile_backtest import (
    LoadedDeploymentProfile,
    load_deployment_profile,
    session_gate_policy_for_profile,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioFold,
    FixedWarmupPortfolioWalkForwardResult,
    RETURN_EPSILON,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.market.market_data import PriceHistory, QuoteHistory


DEFAULT_PROFILE_SLOTS = ("aggressive", "conservative", "survival")


@dataclass(frozen=True)
class DeploymentProfileWalkForward:
    profile: LoadedDeploymentProfile
    walk_forward: FixedWarmupPortfolioWalkForwardResult


@dataclass(frozen=True)
class DeploymentProfilePastScore:
    slot: str
    score: float
    cumulative_return_pct: float
    worst_drawdown_pct: float
    average_risk_discipline_score: float
    active_fold_fraction: float
    folds: int
    eligible: bool
    reason: str


@dataclass(frozen=True)
class DeploymentProfileFoldMetric:
    slot: str
    return_pct: float
    max_drawdown_pct: float
    risk_discipline_score: float
    evaluation_fills: int


@dataclass(frozen=True)
class DeploymentProfileSelection:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_slot: str
    selected_label: str
    selection_score: float | None
    selection_reason: str
    selected_fold: FixedWarmupPortfolioFold
    profile_fold_metrics: tuple[DeploymentProfileFoldMetric, ...]


@dataclass(frozen=True)
class DeploymentProfileNextSelection:
    selected_slot: str
    selected_label: str
    selection_score: float | None
    selection_reason: str
    completed_folds: int
    past_scores: tuple[DeploymentProfilePastScore, ...]


@dataclass(frozen=True)
class DeploymentProfileSelectorResult:
    fixed_results: tuple[DeploymentProfileWalkForward, ...]
    selections: tuple[DeploymentProfileSelection, ...]
    adaptive_result: FixedWarmupPortfolioWalkForwardResult
    fallback_slot: str
    min_past_folds: int
    drawdown_penalty: float
    risk_score_floor: float
    require_past_activity: bool

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(result.profile.slot for result in self.fixed_results)

    @property
    def selected_counts(self) -> dict[str, int]:
        counts = {slot: 0 for slot in self.slots}
        for selection in self.selections:
            counts[selection.selected_slot] = counts.get(selection.selected_slot, 0) + 1
        return counts

    @property
    def fixed_by_slot(self) -> dict[str, DeploymentProfileWalkForward]:
        return {result.profile.slot: result for result in self.fixed_results}


def run_deployment_profile_selector(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slots: Sequence[str] = DEFAULT_PROFILE_SLOTS,
    fallback_slot: str = "conservative",
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    min_past_folds: int = 2,
    drawdown_penalty: float = 0.50,
    risk_score_floor: float = 95.0,
    require_past_activity: bool = True,
) -> DeploymentProfileSelectorResult:
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
    return build_deployment_profile_selector_result(
        fixed_results=fixed_results,
        fallback_slot=fallback_slot,
        min_past_folds=min_past_folds,
        drawdown_penalty=drawdown_penalty,
        risk_score_floor=risk_score_floor,
        require_past_activity=require_past_activity,
    )


def run_deployment_profile_walk_forwards(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slots: Sequence[str] = DEFAULT_PROFILE_SLOTS,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
) -> tuple[DeploymentProfileWalkForward, ...]:
    selected_slots = _validate_slots(slots)
    profiles = tuple(
        load_deployment_profile(
            profile_pack_json=profile_pack_json,
            slot=slot,
        )
        for slot in selected_slots
    )
    symbols = _validate_profile_symbols(profiles)
    fixed_results = tuple(
        _run_profile_walk_forward(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            symbols=symbols,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )
        for profile in profiles
    )
    _validate_aligned_folds(fixed_results)
    return fixed_results


def build_deployment_profile_selector_result(
    *,
    fixed_results: tuple[DeploymentProfileWalkForward, ...],
    fallback_slot: str = "conservative",
    min_past_folds: int = 2,
    drawdown_penalty: float = 0.50,
    risk_score_floor: float = 95.0,
    require_past_activity: bool = True,
) -> DeploymentProfileSelectorResult:
    if not fixed_results:
        raise ValueError("at least one fixed deployment profile result is required")
    slots = tuple(fixed.profile.slot for fixed in fixed_results)
    if fallback_slot not in slots:
        raise ValueError("fallback_slot must be included in fixed_results")
    if min_past_folds < 1:
        raise ValueError("min_past_folds must be at least 1")
    if drawdown_penalty < 0:
        raise ValueError("drawdown_penalty cannot be negative")
    _validate_aligned_folds(fixed_results)
    selections = _build_selections(
        fixed_results=fixed_results,
        fallback_slot=fallback_slot,
        min_past_folds=min_past_folds,
        drawdown_penalty=drawdown_penalty,
        risk_score_floor=risk_score_floor,
        require_past_activity=require_past_activity,
    )
    adaptive_result = FixedWarmupPortfolioWalkForwardResult(
        strategy_name=(
            "adaptive deployment profile selector "
            f"({', '.join(selection.selected_slot for selection in selections)})"
        ),
        symbols=fixed_results[0].walk_forward.symbols,
        folds=tuple(selection.selected_fold for selection in selections),
    )
    return DeploymentProfileSelectorResult(
        fixed_results=fixed_results,
        selections=selections,
        adaptive_result=adaptive_result,
        fallback_slot=fallback_slot,
        min_past_folds=min_past_folds,
        drawdown_penalty=drawdown_penalty,
        risk_score_floor=risk_score_floor,
        require_past_activity=require_past_activity,
    )


def select_next_deployment_profile(
    *,
    fixed_results: tuple[DeploymentProfileWalkForward, ...],
    fallback_slot: str = "conservative",
    min_past_folds: int = 2,
    drawdown_penalty: float = 0.50,
    risk_score_floor: float = 95.0,
    require_past_activity: bool = True,
) -> DeploymentProfileNextSelection:
    if not fixed_results:
        raise ValueError("at least one fixed deployment profile result is required")
    by_slot = {fixed.profile.slot: fixed for fixed in fixed_results}
    if fallback_slot not in by_slot:
        raise ValueError("fallback_slot must be included in fixed_results")
    if min_past_folds < 1:
        raise ValueError("min_past_folds must be at least 1")
    if drawdown_penalty < 0:
        raise ValueError("drawdown_penalty cannot be negative")
    _validate_aligned_folds(fixed_results)
    completed_folds = len(fixed_results[0].walk_forward.folds)
    if completed_folds < min_past_folds:
        fixed = by_slot[fallback_slot]
        return DeploymentProfileNextSelection(
            selected_slot=fallback_slot,
            selected_label=fixed.profile.label,
            selection_score=None,
            selection_reason=(
                f"fallback until {min_past_folds} completed folds exist; "
                f"{completed_folds} available"
            ),
            completed_folds=completed_folds,
            past_scores=(),
        )
    past_scores = tuple(
        _score_past_profile(
            fixed.walk_forward,
            past_fold_count=completed_folds,
            drawdown_penalty=drawdown_penalty,
            risk_score_floor=risk_score_floor,
            require_past_activity=require_past_activity,
            slot=fixed.profile.slot,
        )
        for fixed in fixed_results
    )
    eligible_scores = tuple(score for score in past_scores if score.eligible)
    if not eligible_scores:
        fixed = by_slot[fallback_slot]
        return DeploymentProfileNextSelection(
            selected_slot=fallback_slot,
            selected_label=fixed.profile.label,
            selection_score=None,
            selection_reason=(
                "fallback because no profile met past risk/activity gates "
                f"(risk floor {risk_score_floor:.1f}/100)"
            ),
            completed_folds=completed_folds,
            past_scores=past_scores,
        )
    selected_score = max(
        eligible_scores,
        key=lambda score: (
            score.score,
            score.average_risk_discipline_score,
            score.active_fold_fraction,
            -_slot_rank(fixed_results, score.slot),
        ),
    )
    fixed = by_slot[selected_score.slot]
    return DeploymentProfileNextSelection(
        selected_slot=selected_score.slot,
        selected_label=fixed.profile.label,
        selection_score=selected_score.score,
        selection_reason=(
            f"best past score over {selected_score.folds} completed folds: "
            f"score={selected_score.score:.6f}, "
            f"cumulative_return={selected_score.cumulative_return_pct:.3%}, "
            f"worst_drawdown={selected_score.worst_drawdown_pct:.3%}, "
            "average_risk="
            f"{selected_score.average_risk_discipline_score:.1f}/100"
        ),
        completed_folds=completed_folds,
        past_scores=past_scores,
    )


def write_deployment_profile_selector_summary_csv(
    result: DeploymentProfileSelectorResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "slot",
        "label",
        "selected_count",
        "folds",
        "positive_fold_fraction",
        "active_fold_fraction",
        "active_positive_fold_fraction",
        "non_negative_fold_fraction",
        "median_test_return_pct",
        "median_active_test_return_pct",
        "median_test_sharpe_15m",
        "worst_test_drawdown_pct",
        "average_risk_discipline_score",
        "total_evaluation_fills",
        "largest_positive_fold_contribution",
        "evidence_status",
        "use_case",
        "strategy_map",
        "multiplier_map",
        "allowed_utc_hours",
        "forex_allowed_utc_hours",
        "metal_allowed_utc_hours",
        "crypto_allowed_utc_hours",
        "symbol_allowed_utc_hours",
    ]
    selected_counts = result.selected_counts
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fixed in result.fixed_results:
            writer.writerow(
                _summary_row(
                    mode="fixed",
                    slot=fixed.profile.slot,
                    label=fixed.profile.label,
                    walk_forward=fixed.walk_forward,
                    selected_count=selected_counts.get(fixed.profile.slot, 0),
                    profile=fixed.profile,
                )
            )
        writer.writerow(
            _summary_row(
                mode="adaptive",
                slot="adaptive",
                label=(
                    f"past_{result.min_past_folds}_fold_score"
                    f"_drawdown_penalty_{result.drawdown_penalty:g}"
                    f"_activity_{str(result.require_past_activity).lower()}"
                ),
                walk_forward=result.adaptive_result,
                selected_count=len(result.selections),
                profile=None,
            )
        )


def write_deployment_profile_selector_folds_csv(
    result: DeploymentProfileSelectorResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_metric_fields = [
        field
        for slot in result.slots
        for field in (
            f"{_slot_field_prefix(slot)}_return_pct",
            f"{_slot_field_prefix(slot)}_max_drawdown_pct",
            f"{_slot_field_prefix(slot)}_risk_discipline_score",
            f"{_slot_field_prefix(slot)}_evaluation_fills",
        )
    ]
    fieldnames = [
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "selected_slot",
        "selected_label",
        "selection_score",
        "selection_reason",
        "return_pct",
        "max_drawdown_pct",
        "sharpe_15m",
        "risk_discipline_score",
        "evaluation_fills",
        "full_run_fills",
        "final_equity",
        *fixed_metric_fields,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for selection in result.selections:
            metrics = selection.selected_fold.metrics
            row = {
                "fold": selection.fold_index,
                "train_start": selection.train_start,
                "train_end": selection.train_end,
                "test_start": selection.test_start,
                "test_end": selection.test_end,
                "selected_slot": selection.selected_slot,
                "selected_label": selection.selected_label,
                "selection_score": (
                    "" if selection.selection_score is None else selection.selection_score
                ),
                "selection_reason": selection.selection_reason,
                "return_pct": metrics.return_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "sharpe_15m": metrics.sharpe_15m,
                "risk_discipline_score": selection.selected_fold.risk_discipline.score,
                "evaluation_fills": len(selection.selected_fold.evaluation.fills),
                "full_run_fills": selection.selected_fold.full_run_fills,
                "final_equity": metrics.final_equity,
            }
            for profile_metric in selection.profile_fold_metrics:
                prefix = _slot_field_prefix(profile_metric.slot)
                row[f"{prefix}_return_pct"] = profile_metric.return_pct
                row[f"{prefix}_max_drawdown_pct"] = (
                    profile_metric.max_drawdown_pct
                )
                row[f"{prefix}_risk_discipline_score"] = (
                    profile_metric.risk_discipline_score
                )
                row[f"{prefix}_evaluation_fills"] = (
                    profile_metric.evaluation_fills
                )
            writer.writerow(row)


def _run_profile_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    symbols: tuple[str, ...],
    train_size: int,
    test_size: int,
    step_size: int,
) -> DeploymentProfileWalkForward:
    walk_forward = run_fixed_warmup_portfolio_walk_forward(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_name=profile.strategy_by_symbol[0][1],
        symbols=symbols,
        strategy_by_symbol=dict(profile.strategy_by_symbol),
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        target_notional_multipliers_by_symbol=dict(profile.multipliers_by_symbol),
        session_gate_policy=_profile_session_gate(profile),
    )
    return DeploymentProfileWalkForward(
        profile=profile,
        walk_forward=walk_forward,
    )


def _build_selections(
    *,
    fixed_results: tuple[DeploymentProfileWalkForward, ...],
    fallback_slot: str,
    min_past_folds: int,
    drawdown_penalty: float,
    risk_score_floor: float,
    require_past_activity: bool,
) -> tuple[DeploymentProfileSelection, ...]:
    by_slot = {fixed.profile.slot: fixed for fixed in fixed_results}
    selections: list[DeploymentProfileSelection] = []
    fold_count = len(fixed_results[0].walk_forward.folds)
    for fold_position in range(fold_count):
        if fold_position < min_past_folds:
            selected_slot = fallback_slot
            selected_score: DeploymentProfilePastScore | None = None
            reason = (
                f"fallback until {min_past_folds} completed folds exist; "
                f"{fold_position} available"
            )
        else:
            scores = tuple(
                _score_past_profile(
                    fixed.walk_forward,
                    past_fold_count=fold_position,
                    drawdown_penalty=drawdown_penalty,
                    risk_score_floor=risk_score_floor,
                    require_past_activity=require_past_activity,
                    slot=fixed.profile.slot,
                )
                for fixed in fixed_results
            )
            eligible_scores = tuple(score for score in scores if score.eligible)
            if not eligible_scores:
                selected_slot = fallback_slot
                selected_score = None
                reason = (
                    "fallback because no profile met past risk/activity gates "
                    f"(risk floor {risk_score_floor:.1f}/100)"
                )
            else:
                selected_score = max(
                    eligible_scores,
                    key=lambda score: (
                        score.score,
                        score.average_risk_discipline_score,
                        score.active_fold_fraction,
                        -_slot_rank(fixed_results, score.slot),
                    ),
                )
                selected_slot = selected_score.slot
                reason = (
                    f"best past score over {selected_score.folds} folds: "
                    f"score={selected_score.score:.6f}, "
                    f"cumulative_return={selected_score.cumulative_return_pct:.3%}, "
                    f"worst_drawdown={selected_score.worst_drawdown_pct:.3%}, "
                    "average_risk="
                    f"{selected_score.average_risk_discipline_score:.1f}/100"
                )
        fixed = by_slot[selected_slot]
        selected_fold = fixed.walk_forward.folds[fold_position]
        selections.append(
            DeploymentProfileSelection(
                fold_index=selected_fold.fold_index,
                train_start=selected_fold.train_start,
                train_end=selected_fold.train_end,
                test_start=selected_fold.test_start,
                test_end=selected_fold.test_end,
                selected_slot=selected_slot,
                selected_label=fixed.profile.label,
                selection_score=(
                    None if selected_score is None else selected_score.score
                ),
                selection_reason=reason,
                selected_fold=selected_fold,
                profile_fold_metrics=_profile_fold_metrics(
                    fixed_results,
                    fold_position=fold_position,
                ),
            )
        )
    return tuple(selections)


def _score_past_profile(
    result: FixedWarmupPortfolioWalkForwardResult,
    *,
    past_fold_count: int,
    drawdown_penalty: float,
    risk_score_floor: float,
    require_past_activity: bool,
    slot: str,
) -> DeploymentProfilePastScore:
    past_folds = result.folds[:past_fold_count]
    cumulative_return_pct = sum(fold.metrics.return_pct for fold in past_folds)
    worst_drawdown_pct = max(fold.metrics.max_drawdown_pct for fold in past_folds)
    average_risk_discipline_score = mean(
        fold.risk_discipline.score for fold in past_folds
    )
    active_folds = tuple(
        fold
        for fold in past_folds
        if len(fold.evaluation.fills) > 0
        or abs(fold.metrics.return_pct) > RETURN_EPSILON
    )
    active_fold_fraction = len(active_folds) / len(past_folds)
    score = cumulative_return_pct - drawdown_penalty * worst_drawdown_pct
    meets_risk = average_risk_discipline_score >= risk_score_floor
    meets_activity = (not require_past_activity) or bool(active_folds)
    eligible = meets_risk and meets_activity
    if not meets_risk:
        reason = "below risk floor"
    elif not meets_activity:
        reason = "no active past folds"
    else:
        reason = "eligible"
    return DeploymentProfilePastScore(
        slot=slot,
        score=score,
        cumulative_return_pct=cumulative_return_pct,
        worst_drawdown_pct=worst_drawdown_pct,
        average_risk_discipline_score=average_risk_discipline_score,
        active_fold_fraction=active_fold_fraction,
        folds=len(past_folds),
        eligible=eligible,
        reason=reason,
    )


def _profile_fold_metrics(
    fixed_results: tuple[DeploymentProfileWalkForward, ...],
    *,
    fold_position: int,
) -> tuple[DeploymentProfileFoldMetric, ...]:
    metrics: list[DeploymentProfileFoldMetric] = []
    for fixed in fixed_results:
        fold = fixed.walk_forward.folds[fold_position]
        metrics.append(
            DeploymentProfileFoldMetric(
                slot=fixed.profile.slot,
                return_pct=fold.metrics.return_pct,
                max_drawdown_pct=fold.metrics.max_drawdown_pct,
                risk_discipline_score=fold.risk_discipline.score,
                evaluation_fills=len(fold.evaluation.fills),
            )
        )
    return tuple(metrics)


def _summary_row(
    *,
    mode: str,
    slot: str,
    label: str,
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    selected_count: int,
    profile: LoadedDeploymentProfile | None,
) -> dict[str, object]:
    return {
        "mode": mode,
        "slot": slot,
        "label": label,
        "selected_count": selected_count,
        "folds": len(walk_forward.folds),
        "positive_fold_fraction": walk_forward.positive_fold_fraction,
        "active_fold_fraction": walk_forward.active_fold_fraction,
        "active_positive_fold_fraction": walk_forward.active_positive_fold_fraction,
        "non_negative_fold_fraction": walk_forward.non_negative_fold_fraction,
        "median_test_return_pct": walk_forward.median_test_return_pct,
        "median_active_test_return_pct": walk_forward.median_active_test_return_pct,
        "median_test_sharpe_15m": walk_forward.median_test_sharpe_15m,
        "worst_test_drawdown_pct": walk_forward.worst_test_drawdown_pct,
        "average_risk_discipline_score": (
            walk_forward.average_risk_discipline_score
        ),
        "total_evaluation_fills": walk_forward.total_evaluation_fills,
        "largest_positive_fold_contribution": (
            walk_forward.largest_positive_fold_contribution
        ),
        "evidence_status": "" if profile is None else profile.evidence_status,
        "use_case": "" if profile is None else profile.use_case,
        "strategy_map": (
            "fold-by-fold profile selection" if profile is None else profile.strategy_map_text
        ),
        "multiplier_map": (
            "fold-by-fold profile selection"
            if profile is None
            else profile.multiplier_map_text
        ),
        "allowed_utc_hours": (
            "fold-by-fold profile selection"
            if profile is None
            else profile.allowed_hours_text
        ),
        "forex_allowed_utc_hours": (
            "fold-by-fold profile selection"
            if profile is None
            else profile.forex_hours_text
        ),
        "metal_allowed_utc_hours": (
            "fold-by-fold profile selection"
            if profile is None
            else profile.metal_hours_text
        ),
        "crypto_allowed_utc_hours": (
            "fold-by-fold profile selection" if profile is None else profile.crypto_hours_text
        ),
        "symbol_allowed_utc_hours": (
            "fold-by-fold profile selection"
            if profile is None
            else profile.symbol_hours_text
        ),
    }


def _validate_slots(slots: Sequence[str]) -> tuple[str, ...]:
    selected_slots = tuple(str(slot).strip() for slot in slots if str(slot).strip())
    if not selected_slots:
        raise ValueError("at least one deployment profile slot is required")
    if len(set(selected_slots)) != len(selected_slots):
        raise ValueError("deployment profile slots must be unique")
    return selected_slots


def _validate_profile_symbols(
    profiles: tuple[LoadedDeploymentProfile, ...],
) -> tuple[str, ...]:
    symbols = tuple(symbol for symbol, _ in profiles[0].strategy_by_symbol)
    for profile in profiles[1:]:
        profile_symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
        if profile_symbols != symbols:
            raise ValueError(
                "all selected deployment profiles must use identical symbol order"
            )
    return symbols


def _validate_aligned_folds(
    fixed_results: tuple[DeploymentProfileWalkForward, ...],
) -> None:
    reference = fixed_results[0].walk_forward.folds
    for fixed in fixed_results[1:]:
        folds = fixed.walk_forward.folds
        if len(folds) != len(reference):
            raise ValueError("profile walk-forward folds are not aligned")
        for left, right in zip(reference, folds, strict=True):
            if (
                left.train_start,
                left.train_end,
                left.test_start,
                left.test_end,
            ) != (
                right.train_start,
                right.train_end,
                right.test_start,
                right.test_end,
            ):
                raise ValueError("profile walk-forward folds are not aligned")


def _profile_session_gate(profile: LoadedDeploymentProfile):
    return session_gate_policy_for_profile(profile)


def _slot_rank(
    fixed_results: tuple[DeploymentProfileWalkForward, ...],
    slot: str,
) -> int:
    for index, fixed in enumerate(fixed_results):
        if fixed.profile.slot == slot:
            return index
    return len(fixed_results)


def _slot_field_prefix(slot: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in slot]
    prefix = "".join(chars).strip("_")
    return prefix or "profile"
