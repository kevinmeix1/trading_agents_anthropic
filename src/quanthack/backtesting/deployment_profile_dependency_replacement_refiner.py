from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.competition_score import CompetitionMetrics, RiskDisciplineReport
from quanthack.backtesting.deployment_profile_backtest import (
    LoadedDeploymentProfile,
    load_deployment_profile,
)
from quanthack.backtesting.deployment_profile_dependency_refiner import (
    DependentSymbol,
    DependencyStressRow,
    _ModifiedProfileBacktest,
    _dependency_loss,
    _promotion_rank,
    _raw_profile,
    _run_dependency_stress_rows,
    _run_modified_profile_backtest,
    _run_modified_profile_walk_forward,
    dependent_symbols_from_robustness_csv,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestResult
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


DEFAULT_DEPENDENCY_REPLACEMENT_SCALES = (0.0, 0.25, 0.50)
DEFAULT_REPLACEMENT_REFILL_FRACTIONS = (1.0, 0.75, 0.50)
DEFAULT_REPLACEMENT_BASKET_SIZES = (2, 3, 4)


@dataclass(frozen=True)
class ReplacementSymbol:
    symbol: str
    contribution_score_pct: float
    capacity: float
    base_multiplier: float
    robustness_return_pct: float
    robustness_return_delta_pct: float
    robustness_decision: str


@dataclass(frozen=True)
class DeploymentProfileDependencyReplacementCandidate:
    label: str
    dependency_scale: float
    refill_fraction: float
    basket_size: int
    dependent_symbols: tuple[DependentSymbol, ...]
    replacement_symbols: tuple[ReplacementSymbol, ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    freed_multiplier: float
    refilled_multiplier: float
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None
    promotion: FixedWarmupPromotionDecision | None
    dependency_rows: tuple[DependencyStressRow, ...]
    base_return_pct: float
    base_dependency_loss_pct: float
    min_return_retention: float
    min_dependency_loss_reduction: float

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            _replacement_decision_rank(self.candidate_decision),
            _promotion_rank("" if self.promotion is None else self.promotion.status),
            self.risk_discipline.score,
            self.dependency_loss_reduction,
            self.return_retention_vs_base,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
            self.refilled_multiplier,
        )

    @property
    def multiplier_map_text(self) -> str:
        return " ".join(
            f"{symbol}={multiplier:.3f}"
            for symbol, multiplier in self.multipliers_by_symbol
        )

    @property
    def dependent_symbols_text(self) -> str:
        return " ".join(symbol.symbol for symbol in self.dependent_symbols)

    @property
    def replacement_symbols_text(self) -> str:
        return " ".join(symbol.symbol for symbol in self.replacement_symbols)

    @property
    def unused_multiplier(self) -> float:
        return max(0.0, self.freed_multiplier - self.refilled_multiplier)

    @property
    def worst_dependency_row(self) -> DependencyStressRow | None:
        if not self.dependency_rows:
            return None
        return min(self.dependency_rows, key=lambda row: row.return_delta_pct)

    @property
    def dependency_loss_pct(self) -> float:
        return _dependency_loss(self.dependency_rows)

    @property
    def dependency_loss_reduction(self) -> float:
        if self.base_dependency_loss_pct <= 0:
            return 0.0
        return (
            self.base_dependency_loss_pct - self.dependency_loss_pct
        ) / self.base_dependency_loss_pct

    @property
    def return_retention_vs_base(self) -> float:
        if self.base_return_pct <= 0:
            return 0.0
        return self.competition_metrics.return_pct / self.base_return_pct

    @property
    def candidate_decision(self) -> str:
        if self.risk_discipline.score < 95:
            return "FAIL_RISK"
        if self.competition_metrics.return_pct <= 0:
            return "FAIL_RETURN"
        if self.promotion is not None and self.promotion.status == "REJECT":
            return "REJECTED_BY_WALK_FORWARD"
        if self.dependency_scale == 1.0:
            return "KEEP_BASELINE_DEPENDENCY"
        if (
            self.return_retention_vs_base >= self.min_return_retention
            and self.dependency_loss_reduction >= self.min_dependency_loss_reduction
        ):
            return "BALANCED_REPLACEMENT"
        if self.dependency_loss_reduction > 0 and self.return_retention_vs_base >= 0.50:
            return "WATCHLIST_REPLACEMENT"
        return "REJECT_REPLACEMENT"


@dataclass(frozen=True)
class DeploymentProfileDependencyReplacementResult:
    base_profile: LoadedDeploymentProfile
    robustness_csv: str
    dependent_symbols: tuple[DependentSymbol, ...]
    replacement_pool: tuple[ReplacementSymbol, ...]
    candidates: tuple[DeploymentProfileDependencyReplacementCandidate, ...]

    @property
    def best(self) -> DeploymentProfileDependencyReplacementCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


@dataclass(frozen=True)
class _ReplacementBuild:
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    replacement_symbols: tuple[ReplacementSymbol, ...]
    freed_multiplier: float
    refilled_multiplier: float


@dataclass(frozen=True)
class _RawReplacementCandidate:
    label: str
    dependency_scale: float
    refill_fraction: float
    basket_size: int
    build: _ReplacementBuild
    backtest: _ModifiedProfileBacktest
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None
    promotion: FixedWarmupPromotionDecision | None
    dependency_rows: tuple[DependencyStressRow, ...]


def refine_deployment_profile_dependency_replacement(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    robustness_csv: str | Path,
    dependency_scales: tuple[float, ...] = DEFAULT_DEPENDENCY_REPLACEMENT_SCALES,
    refill_fractions: tuple[float, ...] = DEFAULT_REPLACEMENT_REFILL_FRACTIONS,
    basket_sizes: tuple[int, ...] = DEFAULT_REPLACEMENT_BASKET_SIZES,
    dependency_threshold_pct: float = -0.003,
    min_replacement_score_pct: float = 0.0,
    max_symbol_multiplier: float = 1.0,
    min_return_retention: float = 0.75,
    min_dependency_loss_reduction: float = 0.50,
    include_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> DeploymentProfileDependencyReplacementResult:
    if not dependency_scales:
        raise ValueError("at least one dependency scale is required")
    if any(scale < 0 or scale > 1 for scale in dependency_scales):
        raise ValueError("dependency scales must be between 0 and 1")
    if not refill_fractions:
        raise ValueError("at least one refill fraction is required")
    if any(fraction < 0 or fraction > 1 for fraction in refill_fractions):
        raise ValueError("refill fractions must be between 0 and 1")
    if not basket_sizes:
        raise ValueError("at least one replacement basket size is required")
    if any(size < 1 for size in basket_sizes):
        raise ValueError("replacement basket sizes must be at least 1")
    if not 0 <= max_symbol_multiplier <= 1:
        raise ValueError("max_symbol_multiplier must be between 0 and 1")

    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    dependent_symbols = dependent_symbols_from_robustness_csv(
        robustness_csv=robustness_csv,
        profile=profile,
        dependency_threshold_pct=dependency_threshold_pct,
    )
    replacement_pool = replacement_pool_from_robustness_csv(
        robustness_csv=robustness_csv,
        profile=profile,
        dependent_symbols=dependent_symbols,
        max_symbol_multiplier=max_symbol_multiplier,
        min_replacement_score_pct=min_replacement_score_pct,
    )
    raw_candidates = [
        _evaluate_baseline_candidate(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            dependent_symbols=dependent_symbols,
            include_walk_forward=include_walk_forward,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )
    ]
    for dependency_scale in _unique_values(dependency_scales):
        if dependency_scale == 1.0:
            continue
        for refill_fraction in _unique_values(refill_fractions):
            for basket_size in _unique_int_values(basket_sizes):
                raw_candidates.append(
                    _evaluate_replacement_candidate(
                        config=config,
                        prices=prices,
                        quotes=quotes,
                        profile=profile,
                        dependent_symbols=dependent_symbols,
                        replacement_pool=replacement_pool,
                        dependency_scale=dependency_scale,
                        refill_fraction=refill_fraction,
                        basket_size=basket_size,
                        max_symbol_multiplier=max_symbol_multiplier,
                        include_walk_forward=include_walk_forward,
                        train_size=train_size,
                        test_size=test_size,
                        step_size=step_size,
                    )
                )

    base_raw = raw_candidates[0]
    base_return_pct = base_raw.backtest.competition_metrics.return_pct
    base_dependency_loss_pct = _dependency_loss(base_raw.dependency_rows)
    candidates = tuple(
        DeploymentProfileDependencyReplacementCandidate(
            label=raw.label,
            dependency_scale=raw.dependency_scale,
            refill_fraction=raw.refill_fraction,
            basket_size=raw.basket_size,
            dependent_symbols=dependent_symbols,
            replacement_symbols=raw.build.replacement_symbols,
            multipliers_by_symbol=raw.build.multipliers_by_symbol,
            freed_multiplier=raw.build.freed_multiplier,
            refilled_multiplier=raw.build.refilled_multiplier,
            result=raw.backtest.result,
            competition_metrics=raw.backtest.competition_metrics,
            risk_discipline=raw.backtest.risk_discipline,
            walk_forward=raw.walk_forward,
            promotion=raw.promotion,
            dependency_rows=raw.dependency_rows,
            base_return_pct=base_return_pct,
            base_dependency_loss_pct=base_dependency_loss_pct,
            min_return_retention=min_return_retention,
            min_dependency_loss_reduction=min_dependency_loss_reduction,
        )
        for raw in raw_candidates
    )
    return DeploymentProfileDependencyReplacementResult(
        base_profile=profile,
        robustness_csv=str(robustness_csv),
        dependent_symbols=dependent_symbols,
        replacement_pool=replacement_pool,
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)),
    )


def replacement_pool_from_robustness_csv(
    *,
    robustness_csv: str | Path,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    max_symbol_multiplier: float = 1.0,
    min_replacement_score_pct: float = 0.0,
) -> tuple[ReplacementSymbol, ...]:
    dependent = {row.symbol for row in dependent_symbols}
    base_multipliers = dict(profile.multipliers_by_symbol)
    pool: dict[str, ReplacementSymbol] = {}
    with Path(robustness_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scenario_type",
            "excluded_symbol",
            "return_pct",
            "return_delta_pct",
            "decision",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"robustness CSV missing required columns: {sorted(missing)}")
        for row in reader:
            if row["scenario_type"] != "leave_one_symbol":
                continue
            symbol = instrument_for(row["excluded_symbol"]).symbol
            if symbol in dependent or symbol not in base_multipliers:
                continue
            base_multiplier = base_multipliers[symbol]
            capacity = max(0.0, max_symbol_multiplier - base_multiplier)
            contribution_score = max(0.0, -float(row["return_delta_pct"]))
            if capacity <= 0 or contribution_score <= min_replacement_score_pct:
                continue
            if row["decision"] == "FAIL":
                continue
            pool[symbol] = ReplacementSymbol(
                symbol=symbol,
                contribution_score_pct=contribution_score,
                capacity=capacity,
                base_multiplier=base_multiplier,
                robustness_return_pct=float(row["return_pct"]),
                robustness_return_delta_pct=float(row["return_delta_pct"]),
                robustness_decision=str(row["decision"]),
            )
    return tuple(
        sorted(
            pool.values(),
            key=lambda row: (
                row.contribution_score_pct,
                row.capacity,
                row.symbol,
            ),
            reverse=True,
        )
    )


def write_deployment_profile_dependency_replacement_csv(
    result: DeploymentProfileDependencyReplacementResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "rank",
        "label",
        "base_slot",
        "dependency_scale",
        "refill_fraction",
        "basket_size",
        "dependent_symbols",
        "replacement_symbols",
        "candidate_decision",
        "return_pct",
        "return_retention_vs_base",
        "max_drawdown_pct",
        "sharpe_15m",
        "risk_discipline_score",
        "fills",
        "total_pnl_usd",
        "promotion_status",
        "promotion_reason",
        "wf_positive_fold_fraction",
        "wf_active_positive_fold_fraction",
        "wf_non_negative_fold_fraction",
        "wf_median_active_test_return_pct",
        "wf_worst_test_drawdown_pct",
        "wf_largest_positive_fold_contribution",
        "dependency_symbol",
        "dependency_return_delta_pct",
        "base_dependency_loss_pct",
        "dependency_loss_pct",
        "dependency_loss_reduction",
        "freed_multiplier",
        "refilled_multiplier",
        "unused_multiplier",
        "multiplier_map",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            wf = candidate.walk_forward
            worst = candidate.worst_dependency_row
            writer.writerow(
                {
                    "rank": rank,
                    "label": candidate.label,
                    "base_slot": result.base_profile.slot,
                    "dependency_scale": candidate.dependency_scale,
                    "refill_fraction": candidate.refill_fraction,
                    "basket_size": candidate.basket_size,
                    "dependent_symbols": candidate.dependent_symbols_text,
                    "replacement_symbols": candidate.replacement_symbols_text,
                    "candidate_decision": candidate.candidate_decision,
                    "return_pct": candidate.competition_metrics.return_pct,
                    "return_retention_vs_base": candidate.return_retention_vs_base,
                    "max_drawdown_pct": candidate.competition_metrics.max_drawdown_pct,
                    "sharpe_15m": candidate.competition_metrics.sharpe_15m,
                    "risk_discipline_score": candidate.risk_discipline.score,
                    "fills": len(candidate.result.fills),
                    "total_pnl_usd": candidate.result.total_pnl_usd,
                    "promotion_status": (
                        "" if candidate.promotion is None else candidate.promotion.status
                    ),
                    "promotion_reason": (
                        "" if candidate.promotion is None else candidate.promotion.reason
                    ),
                    "wf_positive_fold_fraction": (
                        "" if wf is None else wf.positive_fold_fraction
                    ),
                    "wf_active_positive_fold_fraction": (
                        "" if wf is None else wf.active_positive_fold_fraction
                    ),
                    "wf_non_negative_fold_fraction": (
                        "" if wf is None else wf.non_negative_fold_fraction
                    ),
                    "wf_median_active_test_return_pct": (
                        "" if wf is None else wf.median_active_test_return_pct
                    ),
                    "wf_worst_test_drawdown_pct": (
                        "" if wf is None else wf.worst_test_drawdown_pct
                    ),
                    "wf_largest_positive_fold_contribution": (
                        "" if wf is None else wf.largest_positive_fold_contribution
                    ),
                    "dependency_symbol": "" if worst is None else worst.symbol,
                    "dependency_return_delta_pct": (
                        "" if worst is None else worst.return_delta_pct
                    ),
                    "base_dependency_loss_pct": candidate.base_dependency_loss_pct,
                    "dependency_loss_pct": candidate.dependency_loss_pct,
                    "dependency_loss_reduction": candidate.dependency_loss_reduction,
                    "freed_multiplier": candidate.freed_multiplier,
                    "refilled_multiplier": candidate.refilled_multiplier,
                    "unused_multiplier": candidate.unused_multiplier,
                    "multiplier_map": candidate.multiplier_map_text,
                }
            )


def write_dependency_replacement_profile_pack_json(
    *,
    source_profile_pack_json: str | Path,
    result: DeploymentProfileDependencyReplacementResult,
    candidate: DeploymentProfileDependencyReplacementCandidate,
    output_json: str | Path,
    refined_slot: str = "dependency_replacement",
) -> None:
    source_path = Path(source_profile_pack_json)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    base_raw = _raw_profile(payload, result.base_profile.slot)
    refined = dict(base_raw)
    refined["slot"] = refined_slot
    refined["label"] = candidate.label
    refined["evidence_status"] = "PAPER_ONLY"
    refined["use_case"] = (
        "Research-only profile that tries to replace fragile single-symbol return "
        "with capped diversified refill symbols; validate on fresh data before live use."
    )
    refined["reason"] = (
        f"Dependency replacement decision={candidate.candidate_decision}; "
        f"dependent={candidate.dependent_symbols_text or 'none'}; "
        f"replacement={candidate.replacement_symbols_text or 'none'}; "
        f"return_retention={candidate.return_retention_vs_base:.1%}; "
        f"dependency_loss_reduction={candidate.dependency_loss_reduction:.1%}"
    )
    refined["return_pct"] = candidate.competition_metrics.return_pct
    refined["max_drawdown_pct"] = candidate.competition_metrics.max_drawdown_pct
    refined["sharpe_15m"] = candidate.competition_metrics.sharpe_15m
    refined["risk_discipline_score"] = candidate.risk_discipline.score
    refined["fold_contribution"] = (
        0.0
        if candidate.walk_forward is None
        else candidate.walk_forward.largest_positive_fold_contribution
    )
    refined["promotion_status"] = (
        "" if candidate.promotion is None else candidate.promotion.status
    )
    refined["promotion_reason"] = (
        "" if candidate.promotion is None else candidate.promotion.reason
    )
    refined["multiplier_map"] = candidate.multiplier_map_text
    refined.setdefault("allowed_utc_hours", result.base_profile.allowed_hours_text)
    refined.setdefault("forex_allowed_utc_hours", result.base_profile.forex_hours_text)
    refined.setdefault("metal_allowed_utc_hours", result.base_profile.metal_hours_text)
    refined.setdefault("crypto_allowed_utc_hours", result.base_profile.crypto_hours_text)
    refined.setdefault(
        "symbol_allowed_utc_hours",
        result.base_profile.symbol_hours_text,
    )
    payload["profiles"] = [
        profile
        for profile in payload.get("profiles", ())
        if profile.get("slot") != refined_slot
    ] + [refined]
    payload["recommended_slot"] = refined_slot
    payload["recommendation_reason"] = (
        "research-only dependency replacement; rerun on fresh data before live use"
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evaluate_baseline_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    include_walk_forward: bool,
    train_size: int,
    test_size: int,
    step_size: int,
) -> _RawReplacementCandidate:
    build = _ReplacementBuild(
        multipliers_by_symbol=profile.multipliers_by_symbol,
        replacement_symbols=(),
        freed_multiplier=0.0,
        refilled_multiplier=0.0,
    )
    return _evaluate_candidate(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        dependent_symbols=dependent_symbols,
        label=f"{profile.slot}_dependency_replacement_baseline",
        dependency_scale=1.0,
        refill_fraction=0.0,
        basket_size=0,
        build=build,
        include_walk_forward=include_walk_forward,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )


def _evaluate_replacement_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    replacement_pool: tuple[ReplacementSymbol, ...],
    dependency_scale: float,
    refill_fraction: float,
    basket_size: int,
    max_symbol_multiplier: float,
    include_walk_forward: bool,
    train_size: int,
    test_size: int,
    step_size: int,
) -> _RawReplacementCandidate:
    build = _replacement_build(
        profile=profile,
        dependent_symbols=dependent_symbols,
        replacement_pool=replacement_pool,
        dependency_scale=dependency_scale,
        refill_fraction=refill_fraction,
        basket_size=basket_size,
        max_symbol_multiplier=max_symbol_multiplier,
    )
    return _evaluate_candidate(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        dependent_symbols=dependent_symbols,
        label=_candidate_label(profile.slot, dependency_scale, refill_fraction, build),
        dependency_scale=dependency_scale,
        refill_fraction=refill_fraction,
        basket_size=basket_size,
        build=build,
        include_walk_forward=include_walk_forward,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )


def _evaluate_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    label: str,
    dependency_scale: float,
    refill_fraction: float,
    basket_size: int,
    build: _ReplacementBuild,
    include_walk_forward: bool,
    train_size: int,
    test_size: int,
    step_size: int,
) -> _RawReplacementCandidate:
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    backtest = _run_modified_profile_backtest(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        symbols=symbols,
        multipliers_by_symbol=build.multipliers_by_symbol,
        slippage_multiplier=1.0,
    )
    dependency_rows = _run_dependency_stress_rows(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        dependent_symbols=dependent_symbols,
        multipliers_by_symbol=build.multipliers_by_symbol,
        baseline=backtest,
    )
    walk_forward = (
        _run_modified_profile_walk_forward(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            multipliers_by_symbol=build.multipliers_by_symbol,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )
        if include_walk_forward
        else None
    )
    promotion = (
        decide_fixed_warmup_promotion(walk_forward)
        if walk_forward is not None
        else None
    )
    return _RawReplacementCandidate(
        label=label,
        dependency_scale=dependency_scale,
        refill_fraction=refill_fraction,
        basket_size=basket_size,
        build=build,
        backtest=backtest,
        walk_forward=walk_forward,
        promotion=promotion,
        dependency_rows=dependency_rows,
    )


def _replacement_build(
    *,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    replacement_pool: tuple[ReplacementSymbol, ...],
    dependency_scale: float,
    refill_fraction: float,
    basket_size: int,
    max_symbol_multiplier: float,
) -> _ReplacementBuild:
    multipliers = dict(profile.multipliers_by_symbol)
    freed_multiplier = 0.0
    for dependent_symbol in dependent_symbols:
        old_multiplier = multipliers.get(dependent_symbol.symbol, 0.0)
        new_multiplier = old_multiplier * dependency_scale
        freed_multiplier += old_multiplier - new_multiplier
        multipliers[dependent_symbol.symbol] = new_multiplier

    selected = replacement_pool[:basket_size]
    refilled = _refill_multipliers(
        multipliers=multipliers,
        selected=selected,
        budget=freed_multiplier * refill_fraction,
        max_symbol_multiplier=max_symbol_multiplier,
    )
    return _ReplacementBuild(
        multipliers_by_symbol=tuple(sorted(multipliers.items())),
        replacement_symbols=selected,
        freed_multiplier=freed_multiplier,
        refilled_multiplier=refilled,
    )


def _refill_multipliers(
    *,
    multipliers: dict[str, float],
    selected: tuple[ReplacementSymbol, ...],
    budget: float,
    max_symbol_multiplier: float,
) -> float:
    remaining = max(0.0, budget)
    refilled = 0.0
    active = [symbol for symbol in selected if symbol.capacity > 0]
    while remaining > 1e-12 and active:
        score_sum = sum(symbol.contribution_score_pct for symbol in active)
        if score_sum <= 0:
            weights = {symbol.symbol: 1.0 / len(active) for symbol in active}
        else:
            weights = {
                symbol.symbol: symbol.contribution_score_pct / score_sum
                for symbol in active
            }
        progressed = False
        next_active: list[ReplacementSymbol] = []
        for symbol in active:
            current = multipliers.get(symbol.symbol, 0.0)
            capacity = max(0.0, max_symbol_multiplier - current)
            if capacity <= 1e-12:
                continue
            proposed = remaining * weights[symbol.symbol]
            applied = min(capacity, proposed)
            if applied > 1e-12:
                multipliers[symbol.symbol] = current + applied
                remaining -= applied
                refilled += applied
                progressed = True
            if multipliers.get(symbol.symbol, 0.0) < max_symbol_multiplier - 1e-12:
                next_active.append(symbol)
        if not progressed:
            break
        active = next_active
    return refilled


def _candidate_label(
    slot: str,
    dependency_scale: float,
    refill_fraction: float,
    build: _ReplacementBuild,
) -> str:
    replacements = "_".join(symbol.symbol for symbol in build.replacement_symbols) or "none"
    return (
        f"{slot}_replace_dep_{dependency_scale:.2f}x_"
        f"refill_{refill_fraction:.2f}x_{replacements}"
    ).replace(".", "p")


def _unique_values(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(dict.fromkeys(values))


def _unique_int_values(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(values))


def _replacement_decision_rank(decision: str) -> float:
    if decision == "BALANCED_REPLACEMENT":
        return 5.0
    if decision == "KEEP_BASELINE_DEPENDENCY":
        return 4.0
    if decision == "WATCHLIST_REPLACEMENT":
        return 3.0
    if decision == "REJECT_REPLACEMENT":
        return 2.0
    return 1.0
