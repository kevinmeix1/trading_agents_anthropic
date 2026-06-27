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
from quanthack.backtesting.deployment_profile_dependency_refiner import (
    DependentSymbol,
    DependencyStressRow,
    _dependency_loss,
    _promotion_rank,
    dependent_symbols_from_robustness_csv,
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
from quanthack.backtesting.portfolio_symbol_evidence import SymbolEvidenceGatePolicy
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


DEFAULT_PROBE_MULTIPLIERS = (0.25, 0.50, 0.75)
DEFAULT_STALE_AFTER_BARS = (0, 16, 32)


@dataclass(frozen=True)
class DeploymentProfileEvidencePolicySpec:
    label: str
    target_symbols: tuple[str, ...]
    allow_without_history: bool
    no_history_target_multiplier: float
    failed_evidence_target_multiplier: float
    lookback_closed_events: int = 1
    min_closed_events: int = 1
    min_realized_pnl_usd: float = 0.0
    min_win_rate: float = 0.0
    stale_after_bars: int | None = None

    @property
    def policy(self) -> SymbolEvidenceGatePolicy | None:
        if self.label == "baseline_no_gate":
            return None
        return SymbolEvidenceGatePolicy(
            lookback_closed_events=self.lookback_closed_events,
            min_closed_events=self.min_closed_events,
            min_realized_pnl_usd=self.min_realized_pnl_usd,
            min_win_rate=self.min_win_rate,
            allow_without_history=self.allow_without_history,
            stale_after_bars=self.stale_after_bars,
            target_symbols=self.target_symbols,
            no_history_target_multiplier=self.no_history_target_multiplier,
            failed_evidence_target_multiplier=self.failed_evidence_target_multiplier,
        )


@dataclass(frozen=True)
class DeploymentProfileEvidenceGateCandidate:
    spec: DeploymentProfileEvidencePolicySpec
    dependent_symbols: tuple[DependentSymbol, ...]
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
            _evidence_decision_rank(self.candidate_decision),
            _promotion_rank("" if self.promotion is None else self.promotion.status),
            self.risk_discipline.score,
            self.dependency_loss_reduction,
            self.return_retention_vs_base,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
        )

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
    def target_symbols_text(self) -> str:
        return " ".join(self.spec.target_symbols)

    @property
    def candidate_decision(self) -> str:
        if self.spec.label == "baseline_no_gate":
            return "KEEP_BASELINE"
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
            return "PROMOTE_EVIDENCE_GATE"
        if self.dependency_loss_reduction > 0 and self.return_retention_vs_base >= 0.50:
            return "WATCHLIST_EVIDENCE_GATE"
        return "REJECT_EVIDENCE_GATE"


@dataclass(frozen=True)
class DeploymentProfileEvidenceGateResult:
    base_profile: LoadedDeploymentProfile
    robustness_csv: str
    dependent_symbols: tuple[DependentSymbol, ...]
    candidates: tuple[DeploymentProfileEvidenceGateCandidate, ...]

    @property
    def best(self) -> DeploymentProfileEvidenceGateCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


@dataclass(frozen=True)
class _BacktestRun:
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


def refine_deployment_profile_symbol_evidence_gate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    robustness_csv: str | Path,
    target_symbols: tuple[str, ...] | None = None,
    probe_multipliers: tuple[float, ...] = DEFAULT_PROBE_MULTIPLIERS,
    stale_after_bars_values: tuple[int, ...] = DEFAULT_STALE_AFTER_BARS,
    dependency_threshold_pct: float = -0.003,
    min_return_retention: float = 0.75,
    min_dependency_loss_reduction: float = 0.20,
    include_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> DeploymentProfileEvidenceGateResult:
    if any(multiplier < 0 or multiplier > 1 for multiplier in probe_multipliers):
        raise ValueError("probe multipliers must be between 0 and 1")
    if any(value < 0 for value in stale_after_bars_values):
        raise ValueError("stale_after_bars values cannot be negative")
    profile = load_deployment_profile(profile_pack_json=profile_pack_json, slot=slot)
    dependent_symbols = dependent_symbols_from_robustness_csv(
        robustness_csv=robustness_csv,
        profile=profile,
        dependency_threshold_pct=dependency_threshold_pct,
    )
    selected_symbols = tuple(
        instrument_for(symbol).symbol
        for symbol in (target_symbols or tuple(row.symbol for row in dependent_symbols))
    )
    specs = _policy_specs(
        target_symbols=selected_symbols,
        probe_multipliers=probe_multipliers,
        stale_after_bars_values=stale_after_bars_values,
    )
    raw_candidates = tuple(
        _evaluate_candidate(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            dependent_symbols=dependent_symbols,
            spec=spec,
            include_walk_forward=include_walk_forward,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )
        for spec in specs
    )
    baseline = raw_candidates[0]
    base_return_pct = baseline.competition_metrics.return_pct
    base_dependency_loss_pct = _dependency_loss(baseline.dependency_rows)
    candidates = tuple(
        DeploymentProfileEvidenceGateCandidate(
            spec=candidate.spec,
            dependent_symbols=dependent_symbols,
            result=candidate.result,
            competition_metrics=candidate.competition_metrics,
            risk_discipline=candidate.risk_discipline,
            walk_forward=candidate.walk_forward,
            promotion=candidate.promotion,
            dependency_rows=candidate.dependency_rows,
            base_return_pct=base_return_pct,
            base_dependency_loss_pct=base_dependency_loss_pct,
            min_return_retention=min_return_retention,
            min_dependency_loss_reduction=min_dependency_loss_reduction,
        )
        for candidate in raw_candidates
    )
    return DeploymentProfileEvidenceGateResult(
        base_profile=profile,
        robustness_csv=str(robustness_csv),
        dependent_symbols=dependent_symbols,
        candidates=tuple(sorted(candidates, key=lambda row: row.rank_key, reverse=True)),
    )


def write_deployment_profile_symbol_evidence_gate_csv(
    result: DeploymentProfileEvidenceGateResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "rank",
        "label",
        "target_symbols",
        "candidate_decision",
        "allow_without_history",
        "no_history_target_multiplier",
        "failed_evidence_target_multiplier",
        "lookback_closed_events",
        "min_closed_events",
        "min_realized_pnl_usd",
        "min_win_rate",
        "stale_after_bars",
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
        "dependency_symbol",
        "dependency_return_delta_pct",
        "base_dependency_loss_pct",
        "dependency_loss_pct",
        "dependency_loss_reduction",
        "gate_applied_count",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            wf = candidate.walk_forward
            worst = candidate.worst_dependency_row
            spec = candidate.spec
            writer.writerow(
                {
                    "rank": rank,
                    "label": spec.label,
                    "target_symbols": candidate.target_symbols_text,
                    "candidate_decision": candidate.candidate_decision,
                    "allow_without_history": spec.allow_without_history,
                    "no_history_target_multiplier": spec.no_history_target_multiplier,
                    "failed_evidence_target_multiplier": spec.failed_evidence_target_multiplier,
                    "lookback_closed_events": spec.lookback_closed_events,
                    "min_closed_events": spec.min_closed_events,
                    "min_realized_pnl_usd": spec.min_realized_pnl_usd,
                    "min_win_rate": spec.min_win_rate,
                    "stale_after_bars": "" if spec.stale_after_bars is None else spec.stale_after_bars,
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
                    "dependency_symbol": "" if worst is None else worst.symbol,
                    "dependency_return_delta_pct": (
                        "" if worst is None else worst.return_delta_pct
                    ),
                    "base_dependency_loss_pct": candidate.base_dependency_loss_pct,
                    "dependency_loss_pct": candidate.dependency_loss_pct,
                    "dependency_loss_reduction": candidate.dependency_loss_reduction,
                    "gate_applied_count": sum(
                        1 for report in candidate.result.symbol_evidence_reports if report.applied
                    ),
                }
            )


def write_deployment_profile_symbol_evidence_gate_json(
    result: DeploymentProfileEvidenceGateResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = result.best
    payload = {
        "base_slot": result.base_profile.slot,
        "base_label": result.base_profile.label,
        "dependent_symbols": [row.symbol for row in result.dependent_symbols],
        "recommended": None if best is None else _candidate_json(best),
        "candidates": [_candidate_json(candidate) for candidate in result.candidates],
        "warning": (
            "Paper-only evidence gate recommendation. Re-run on the latest official "
            "data before using in live MT5 operation."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _candidate_json(candidate: DeploymentProfileEvidenceGateCandidate) -> dict:
    spec = candidate.spec
    worst = candidate.worst_dependency_row
    return {
        "label": spec.label,
        "decision": candidate.candidate_decision,
        "target_symbols": list(spec.target_symbols),
        "allow_without_history": spec.allow_without_history,
        "no_history_target_multiplier": spec.no_history_target_multiplier,
        "failed_evidence_target_multiplier": spec.failed_evidence_target_multiplier,
        "lookback_closed_events": spec.lookback_closed_events,
        "min_closed_events": spec.min_closed_events,
        "min_realized_pnl_usd": spec.min_realized_pnl_usd,
        "min_win_rate": spec.min_win_rate,
        "stale_after_bars": spec.stale_after_bars,
        "return_pct": candidate.competition_metrics.return_pct,
        "return_retention_vs_base": candidate.return_retention_vs_base,
        "max_drawdown_pct": candidate.competition_metrics.max_drawdown_pct,
        "sharpe_15m": candidate.competition_metrics.sharpe_15m,
        "risk_discipline_score": candidate.risk_discipline.score,
        "fills": len(candidate.result.fills),
        "total_pnl_usd": candidate.result.total_pnl_usd,
        "promotion_status": "" if candidate.promotion is None else candidate.promotion.status,
        "dependency_symbol": "" if worst is None else worst.symbol,
        "dependency_return_delta_pct": None if worst is None else worst.return_delta_pct,
        "dependency_loss_reduction": candidate.dependency_loss_reduction,
        "gate_applied_count": sum(
            1 for report in candidate.result.symbol_evidence_reports if report.applied
        ),
    }


def _policy_specs(
    *,
    target_symbols: tuple[str, ...],
    probe_multipliers: tuple[float, ...],
    stale_after_bars_values: tuple[int, ...],
) -> tuple[DeploymentProfileEvidencePolicySpec, ...]:
    specs = [
        DeploymentProfileEvidencePolicySpec(
            label="baseline_no_gate",
            target_symbols=target_symbols,
            allow_without_history=True,
            no_history_target_multiplier=0.0,
            failed_evidence_target_multiplier=0.0,
        )
    ]
    for probe_multiplier in tuple(dict.fromkeys(probe_multipliers)):
        for stale_after_bars in tuple(dict.fromkeys(stale_after_bars_values)):
            stale_value = None if stale_after_bars == 0 else stale_after_bars
            stale_label = "none" if stale_value is None else str(stale_value)
            specs.append(
                DeploymentProfileEvidencePolicySpec(
                    label=(
                        f"target_probe_{probe_multiplier:.2f}x_stale_{stale_label}"
                    ).replace(".", "p"),
                    target_symbols=target_symbols,
                    allow_without_history=False,
                    no_history_target_multiplier=probe_multiplier,
                    failed_evidence_target_multiplier=0.0,
                    stale_after_bars=stale_value,
                )
            )
    return tuple(specs)


@dataclass(frozen=True)
class _RawEvidenceCandidate:
    spec: DeploymentProfileEvidencePolicySpec
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None
    promotion: FixedWarmupPromotionDecision | None
    dependency_rows: tuple[DependencyStressRow, ...]


def _evaluate_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    spec: DeploymentProfileEvidencePolicySpec,
    include_walk_forward: bool,
    train_size: int,
    test_size: int,
    step_size: int,
) -> _RawEvidenceCandidate:
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    backtest = _run_profile(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        symbols=symbols,
        policy=spec.policy,
    )
    dependency_rows = _dependency_rows(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        dependent_symbols=dependent_symbols,
        policy=spec.policy,
        baseline=backtest,
    )
    walk_forward = (
        _run_walk_forward(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            policy=spec.policy,
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
    return _RawEvidenceCandidate(
        spec=spec,
        result=backtest.result,
        competition_metrics=backtest.competition_metrics,
        risk_discipline=backtest.risk_discipline,
        walk_forward=walk_forward,
        promotion=promotion,
        dependency_rows=dependency_rows,
    )


def _run_profile(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    symbols: tuple[str, ...],
    policy: SymbolEvidenceGatePolicy | None,
) -> _BacktestRun:
    strategy_map = dict(profile.strategy_by_symbol)
    multiplier_map = dict(profile.multipliers_by_symbol)
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
        fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
        periods_per_year=config.backtest.periods_per_year,
        target_notional_multipliers_by_symbol={
            symbol: multiplier_map.get(symbol, 1.0) for symbol in symbols
        },
        session_gate_policy=session_gate_policy_for_profile(profile),
        symbol_evidence_gate_policy=policy,
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
    return _BacktestRun(
        result=result,
        competition_metrics=metrics,
        risk_discipline=risk,
    )


def _dependency_rows(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    dependent_symbols: tuple[DependentSymbol, ...],
    policy: SymbolEvidenceGatePolicy | None,
    baseline: _BacktestRun,
) -> tuple[DependencyStressRow, ...]:
    all_symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    rows: list[DependencyStressRow] = []
    for dependent_symbol in dependent_symbols:
        kept_symbols = tuple(symbol for symbol in all_symbols if symbol != dependent_symbol.symbol)
        if not kept_symbols:
            continue
        stress = _run_profile(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            symbols=kept_symbols,
            policy=policy,
        )
        rows.append(
            _dependency_row(
                symbol=dependent_symbol.symbol,
                stress=stress,
                baseline=baseline,
            )
        )
    return tuple(rows)


def _dependency_row(
    *,
    symbol: str,
    stress: _BacktestRun,
    baseline: _BacktestRun,
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


def _run_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    policy: SymbolEvidenceGatePolicy | None,
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
        target_notional_multipliers_by_symbol=dict(profile.multipliers_by_symbol),
        symbol_evidence_gate_policy=policy,
    )


def _evidence_decision_rank(decision: str) -> float:
    if decision == "PROMOTE_EVIDENCE_GATE":
        return 5.0
    if decision == "KEEP_BASELINE":
        return 4.0
    if decision == "WATCHLIST_EVIDENCE_GATE":
        return 3.0
    if decision == "REJECT_EVIDENCE_GATE":
        return 2.0
    return 1.0
