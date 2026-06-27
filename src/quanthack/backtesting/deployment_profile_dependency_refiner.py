from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.deployment_profile_backtest import (
    LoadedDeploymentProfile,
    load_deployment_profile,
    session_gate_policy_for_profile,
)
from quanthack.backtesting.portfolio_backtest import (
    PortfolioBacktestEngine,
    PortfolioBacktestResult,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


DEFAULT_DEPENDENCY_SCALES = (1.0, 0.85, 0.75, 0.50, 0.25, 0.0)


@dataclass(frozen=True)
class DependentSymbol:
    symbol: str
    robustness_return_pct: float
    robustness_return_delta_pct: float
    robustness_decision: str
    base_multiplier: float


@dataclass(frozen=True)
class DependencyStressRow:
    symbol: str
    return_pct: float
    return_delta_pct: float
    max_drawdown_pct: float
    drawdown_delta_pct: float
    sharpe_15m: float
    risk_discipline_score: float
    total_pnl_usd: float
    fills: int
    decision: str
    note: str


@dataclass(frozen=True)
class DeploymentProfileDependencyCandidate:
    label: str
    dependency_scale: float
    dependent_symbols: tuple[DependentSymbol, ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
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
            _candidate_decision_rank(self.candidate_decision),
            _promotion_rank("" if self.promotion is None else self.promotion.status),
            self.risk_discipline.score,
            self.dependency_loss_reduction,
            self.return_retention_vs_base,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
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
    def worst_dependency_row(self) -> DependencyStressRow | None:
        if not self.dependency_rows:
            return None
        return min(self.dependency_rows, key=lambda row: row.return_delta_pct)

    @property
    def dependency_loss_pct(self) -> float:
        worst = self.worst_dependency_row
        if worst is None:
            return 0.0
        return max(0.0, -worst.return_delta_pct)

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
        if (
            self.return_retention_vs_base >= self.min_return_retention
            and self.dependency_loss_reduction >= self.min_dependency_loss_reduction
        ):
            return "BALANCED_REDUCE_DEPENDENCY"
        if self.dependency_scale == 1.0:
            return "KEEP_BASELINE_DEPENDENCY"
        if self.dependency_loss_reduction > 0 and self.return_retention_vs_base >= 0.50:
            return "WATCHLIST_TRADEOFF"
        return "REJECT_TRADEOFF"


@dataclass(frozen=True)
class DeploymentProfileDependencyRefinementResult:
    base_profile: LoadedDeploymentProfile
    robustness_csv: str
    dependent_symbols: tuple[DependentSymbol, ...]
    candidates: tuple[DeploymentProfileDependencyCandidate, ...]

    @property
    def best(self) -> DeploymentProfileDependencyCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


@dataclass(frozen=True)
class _ModifiedProfileBacktest:
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


@dataclass(frozen=True)
class _RawDependencyCandidate:
    label: str
    dependency_scale: float
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    backtest: _ModifiedProfileBacktest
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None
    promotion: FixedWarmupPromotionDecision | None
    dependency_rows: tuple[DependencyStressRow, ...]


def refine_deployment_profile_dependency(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    robustness_csv: str | Path,
    dependency_scales: tuple[float, ...] = DEFAULT_DEPENDENCY_SCALES,
    dependency_threshold_pct: float = -0.003,
    min_return_retention: float = 0.75,
    min_dependency_loss_reduction: float = 0.20,
    include_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> DeploymentProfileDependencyRefinementResult:
    if not dependency_scales:
        raise ValueError("at least one dependency scale is required")
    if any(scale < 0 or scale > 1 for scale in dependency_scales):
        raise ValueError("dependency scales must be between 0 and 1")
    if min_return_retention < 0:
        raise ValueError("min_return_retention must be non-negative")
    if min_dependency_loss_reduction < 0:
        raise ValueError("min_dependency_loss_reduction must be non-negative")

    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    dependent_symbols = dependent_symbols_from_robustness_csv(
        robustness_csv=robustness_csv,
        profile=profile,
        dependency_threshold_pct=dependency_threshold_pct,
    )
    raw_candidates = tuple(
        _evaluate_candidate(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            dependent_symbols=dependent_symbols,
            scale=scale,
            include_walk_forward=include_walk_forward,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )
        for scale in _scale_values(dependency_scales)
    )
    base_raw = _base_raw_candidate(raw_candidates)
    base_return_pct = base_raw.backtest.competition_metrics.return_pct
    base_dependency_loss_pct = _dependency_loss(base_raw.dependency_rows)
    candidates = tuple(
        DeploymentProfileDependencyCandidate(
            label=raw.label,
            dependency_scale=raw.dependency_scale,
            dependent_symbols=dependent_symbols,
            multipliers_by_symbol=raw.multipliers_by_symbol,
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
    return DeploymentProfileDependencyRefinementResult(
        base_profile=profile,
        robustness_csv=str(robustness_csv),
        dependent_symbols=dependent_symbols,
        candidates=tuple(sorted(candidates, key=lambda row: row.rank_key, reverse=True)),
    )


def dependent_symbols_from_robustness_csv(
    *,
    robustness_csv: str | Path,
    profile: LoadedDeploymentProfile,
    dependency_threshold_pct: float = -0.003,
) -> tuple[DependentSymbol, ...]:
    base_multipliers = dict(profile.multipliers_by_symbol)
    dependent: dict[str, DependentSymbol] = {}
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
            if symbol not in base_multipliers:
                continue
            return_delta_pct = float(row["return_delta_pct"])
            decision = str(row["decision"])
            if decision not in {"FRAGILE", "FAIL"} and return_delta_pct > dependency_threshold_pct:
                continue
            dependent[symbol] = DependentSymbol(
                symbol=symbol,
                robustness_return_pct=float(row["return_pct"]),
                robustness_return_delta_pct=return_delta_pct,
                robustness_decision=decision,
                base_multiplier=base_multipliers[symbol],
            )
    return tuple(
        sorted(
            dependent.values(),
            key=lambda row: (row.robustness_return_delta_pct, row.symbol),
        )
    )


def write_deployment_profile_dependency_refinement_csv(
    result: DeploymentProfileDependencyRefinementResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "rank",
        "label",
        "base_slot",
        "dependency_scale",
        "dependent_symbols",
        "dependent_symbol_count",
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
        "dependency_return_pct",
        "dependency_return_delta_pct",
        "base_dependency_loss_pct",
        "dependency_loss_pct",
        "dependency_loss_reduction",
        "dependency_decision",
        "dependency_note",
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
                    "dependent_symbols": candidate.dependent_symbols_text,
                    "dependent_symbol_count": len(candidate.dependent_symbols),
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
                    "dependency_return_pct": "" if worst is None else worst.return_pct,
                    "dependency_return_delta_pct": (
                        "" if worst is None else worst.return_delta_pct
                    ),
                    "base_dependency_loss_pct": candidate.base_dependency_loss_pct,
                    "dependency_loss_pct": candidate.dependency_loss_pct,
                    "dependency_loss_reduction": candidate.dependency_loss_reduction,
                    "dependency_decision": "" if worst is None else worst.decision,
                    "dependency_note": "" if worst is None else worst.note,
                    "multiplier_map": candidate.multiplier_map_text,
                }
            )


def write_dependency_refined_profile_pack_json(
    *,
    source_profile_pack_json: str | Path,
    result: DeploymentProfileDependencyRefinementResult,
    candidate: DeploymentProfileDependencyCandidate,
    output_json: str | Path,
    refined_slot: str = "dependency_refined",
) -> None:
    source_path = Path(source_profile_pack_json)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    base_raw = _raw_profile(payload, result.base_profile.slot)
    refined = dict(base_raw)
    refined["slot"] = refined_slot
    refined["label"] = candidate.label
    refined["evidence_status"] = "PAPER_ONLY"
    refined["use_case"] = (
        "Research-only profile that balances backtest return against "
        "single-symbol dependency risk; validate on fresh official data before live use."
    )
    if candidate.dependency_scale == 1.0:
        action = "Kept base dependency-symbol multipliers after the dependency sweep"
    else:
        action = (
            f"Scaled dependency symbols {candidate.dependent_symbols_text or 'none'} "
            f"to {candidate.dependency_scale:.2f}x"
        )
    refined["reason"] = (
        f"{action}; decision={candidate.candidate_decision}; "
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
    profiles = [
        profile
        for profile in payload.get("profiles", ())
        if profile.get("slot") != refined_slot
    ]
    profiles.append(refined)
    payload["profiles"] = profiles
    payload["recommended_slot"] = refined_slot
    payload["recommendation_reason"] = (
        "research-only dependency refinement; rerun on fresh data before live use"
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evaluate_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    scale: float,
    include_walk_forward: bool,
    train_size: int,
    test_size: int,
    step_size: int,
) -> _RawDependencyCandidate:
    multipliers = _scaled_multipliers(
        profile=profile,
        dependent_symbols=dependent_symbols,
        scale=scale,
    )
    backtest = _run_modified_profile_backtest(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        symbols=tuple(symbol for symbol, _ in profile.strategy_by_symbol),
        multipliers_by_symbol=multipliers,
        slippage_multiplier=1.0,
    )
    dependency_rows = _run_dependency_stress_rows(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        dependent_symbols=dependent_symbols,
        multipliers_by_symbol=multipliers,
        baseline=backtest,
    )
    walk_forward = (
        _run_modified_profile_walk_forward(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            multipliers_by_symbol=multipliers,
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
    return _RawDependencyCandidate(
        label=_candidate_label(profile.slot, scale, dependent_symbols),
        dependency_scale=scale,
        multipliers_by_symbol=multipliers,
        backtest=backtest,
        walk_forward=walk_forward,
        promotion=promotion,
        dependency_rows=dependency_rows,
    )


def _run_dependency_stress_rows(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    multipliers_by_symbol: tuple[tuple[str, float], ...],
    baseline: _ModifiedProfileBacktest,
) -> tuple[DependencyStressRow, ...]:
    all_symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    rows: list[DependencyStressRow] = []
    for dependent_symbol in dependent_symbols:
        kept_symbols = tuple(
            symbol for symbol in all_symbols if symbol != dependent_symbol.symbol
        )
        if not kept_symbols:
            continue
        stress = _run_modified_profile_backtest(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            symbols=kept_symbols,
            multipliers_by_symbol=tuple(
                (symbol, multiplier)
                for symbol, multiplier in multipliers_by_symbol
                if symbol in kept_symbols
            ),
            slippage_multiplier=1.0,
        )
        rows.append(
            _dependency_row(
                symbol=dependent_symbol.symbol,
                stress=stress,
                baseline=baseline,
            )
        )
    return tuple(rows)


def _run_modified_profile_backtest(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    symbols: tuple[str, ...],
    multipliers_by_symbol: tuple[tuple[str, float], ...],
    slippage_multiplier: float,
) -> _ModifiedProfileBacktest:
    strategy_map = dict(profile.strategy_by_symbol)
    multiplier_map = dict(multipliers_by_symbol)
    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(strategy_map[symbol], symbol=symbol)
            for symbol in symbols
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in symbols
        },
        clock=config.competition.to_clock(),
        fill_model=FillModel(
            slippage_bps=config.backtest.slippage_bps * slippage_multiplier
        ),
        periods_per_year=config.backtest.periods_per_year,
        target_notional_multipliers_by_symbol={
            symbol: multiplier_map.get(symbol, 1.0) for symbol in symbols
        },
        session_gate_policy=session_gate_policy_for_profile(profile),
    )
    result = engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )
    metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(result.equity_curve)
    )
    return _ModifiedProfileBacktest(
        result=result,
        competition_metrics=metrics,
        risk_discipline=risk,
    )


def _run_modified_profile_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    multipliers_by_symbol: tuple[tuple[str, float], ...],
    train_size: int,
    test_size: int,
    step_size: int,
) -> FixedWarmupPortfolioWalkForwardResult:
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    return run_fixed_warmup_portfolio_walk_forward(
        config=config,
        prices=prices,
        quotes=quotes,
        strategy_name=profile.strategy_by_symbol[0][1],
        symbols=symbols,
        strategy_by_symbol=dict(profile.strategy_by_symbol),
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        session_gate_policy=session_gate_policy_for_profile(profile),
        target_notional_multipliers_by_symbol=dict(multipliers_by_symbol),
    )


def _dependency_row(
    *,
    symbol: str,
    stress: _ModifiedProfileBacktest,
    baseline: _ModifiedProfileBacktest,
) -> DependencyStressRow:
    metrics = stress.competition_metrics
    baseline_metrics = baseline.competition_metrics
    return_delta = metrics.return_pct - baseline_metrics.return_pct
    drawdown_delta = metrics.max_drawdown_pct - baseline_metrics.max_drawdown_pct
    decision, note = _dependency_decision(
        symbol=symbol,
        return_pct=metrics.return_pct,
        return_delta=return_delta,
        drawdown_delta=drawdown_delta,
        risk_score=stress.risk_discipline.score,
        baseline_return=baseline_metrics.return_pct,
    )
    return DependencyStressRow(
        symbol=symbol,
        return_pct=metrics.return_pct,
        return_delta_pct=return_delta,
        max_drawdown_pct=metrics.max_drawdown_pct,
        drawdown_delta_pct=drawdown_delta,
        sharpe_15m=metrics.sharpe_15m,
        risk_discipline_score=stress.risk_discipline.score,
        total_pnl_usd=stress.result.total_pnl_usd,
        fills=len(stress.result.fills),
        decision=decision,
        note=note,
    )


def _dependency_decision(
    *,
    symbol: str,
    return_pct: float,
    return_delta: float,
    drawdown_delta: float,
    risk_score: float,
    baseline_return: float,
) -> tuple[str, str]:
    if risk_score < 95:
        return "FAIL", "risk discipline below 95/100"
    if return_pct <= 0:
        return "FAIL", "dependency stress return is not positive"
    if baseline_return > 0 and return_delta <= -0.50 * baseline_return:
        return "FRAGILE", f"excluding {symbol} removes at least half of candidate return"
    if drawdown_delta > 0.01:
        return "FRAGILE", "drawdown increases by more than 1 percentage point"
    if return_delta < 0:
        return "PASS_WEAKER", "positive without dependency symbol but weaker"
    return "PASS", "positive without dependency symbol"


def _scaled_multipliers(
    *,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    scale: float,
) -> tuple[tuple[str, float], ...]:
    dependent = {row.symbol for row in dependent_symbols}
    return tuple(
        sorted(
            (
                symbol,
                multiplier * scale if symbol in dependent else multiplier,
            )
            for symbol, multiplier in profile.multipliers_by_symbol
        )
    )


def _scale_values(scales: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(dict.fromkeys((1.0,) + scales))


def _base_raw_candidate(
    raw_candidates: tuple[_RawDependencyCandidate, ...],
) -> _RawDependencyCandidate:
    for candidate in raw_candidates:
        if candidate.dependency_scale == 1.0:
            return candidate
    raise ValueError("dependency refinement must include a 1.0 baseline scale")


def _dependency_loss(rows: tuple[DependencyStressRow, ...]) -> float:
    if not rows:
        return 0.0
    return max(0.0, -min(row.return_delta_pct for row in rows))


def _candidate_label(
    slot: str,
    scale: float,
    dependent_symbols: tuple[DependentSymbol, ...],
) -> str:
    if not dependent_symbols:
        return f"{slot}_no_dependent_symbols"
    return f"{slot}_dependency_symbols_{scale:.2f}x".replace(".", "p")


def _raw_profile(payload: dict, slot: str) -> dict:
    for raw_profile in payload.get("profiles", ()):
        if raw_profile.get("slot") == slot:
            return raw_profile
    raise ValueError(f"profile slot {slot!r} not found in source pack")


def _candidate_decision_rank(decision: str) -> float:
    if decision == "BALANCED_REDUCE_DEPENDENCY":
        return 4.0
    if decision == "KEEP_BASELINE_DEPENDENCY":
        return 3.0
    if decision == "WATCHLIST_TRADEOFF":
        return 2.0
    return 1.0


def _promotion_rank(status: str) -> float:
    if status == "PROMOTE":
        return 3.0
    if status == "PAPER_ONLY":
        return 2.0
    if not status:
        return 1.5
    return 1.0
