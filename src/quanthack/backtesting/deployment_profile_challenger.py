from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.deployment_profile_backtest import (
    DeploymentProfileBacktestResult,
    LoadedDeploymentProfile,
    run_deployment_profile_backtest,
    session_gate_policy_for_profile,
)
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class DeploymentProfileCandidateSpec:
    label: str
    profile_pack_json: str
    slot: str


@dataclass(frozen=True)
class DeploymentProfileChallengerRow:
    label: str
    slot: str
    profile_label: str
    profile_pack_json: str
    rank_score: float
    decision: str
    reason: str
    backtest_return_pct: float
    backtest_return_delta_pct: float
    backtest_max_drawdown_pct: float
    backtest_drawdown_delta_pct: float
    backtest_sharpe_15m: float
    backtest_sharpe_delta: float
    total_pnl_usd: float
    total_pnl_delta_usd: float
    fills: int
    risk_discipline_score: float
    promotion_status: str
    promotion_reason: str
    wf_positive_fold_fraction: float
    wf_active_positive_fold_fraction: float
    wf_non_negative_fold_fraction: float
    wf_median_active_test_return_pct: float
    wf_worst_test_drawdown_pct: float
    wf_largest_positive_fold_contribution: float
    gate_complexity: int
    strategy_map: str
    multiplier_map: str
    allowed_utc_hours: str
    forex_allowed_utc_hours: str
    metal_allowed_utc_hours: str
    crypto_allowed_utc_hours: str
    symbol_allowed_utc_hours: str

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            _decision_rank(self.decision),
            self.rank_score,
            self.backtest_return_pct,
            self.backtest_sharpe_15m,
            -self.backtest_max_drawdown_pct,
            -self.gate_complexity,
        )


@dataclass(frozen=True)
class DeploymentProfileChallengerResult:
    rows: tuple[DeploymentProfileChallengerRow, ...]

    @property
    def best(self) -> DeploymentProfileChallengerRow | None:
        if not self.rows:
            return None
        return self.rows[0]


def compare_deployment_profile_challengers(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    candidates: tuple[DeploymentProfileCandidateSpec, ...],
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> DeploymentProfileChallengerResult:
    if not candidates:
        raise ValueError("at least one deployment profile candidate is required")
    evaluated = tuple(
        _evaluate_candidate(
            config=config,
            prices=prices,
            quotes=quotes,
            spec=spec,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )
        for spec in candidates
    )
    baseline = evaluated[0]
    rows = tuple(
        _build_row(
            spec=spec,
            backtest=backtest,
            walk_forward=walk_forward,
            promotion=promotion,
            baseline=baseline[0],
            is_baseline=index == 0,
        )
        for index, (spec, (backtest, walk_forward, promotion)) in enumerate(
            zip(candidates, evaluated, strict=True)
        )
    )
    return DeploymentProfileChallengerResult(
        rows=tuple(sorted(rows, key=lambda row: row.rank_key, reverse=True))
    )


def write_deployment_profile_challenger_csv(
    result: DeploymentProfileChallengerResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "rank",
        "label",
        "slot",
        "profile_label",
        "profile_pack_json",
        "rank_score",
        "decision",
        "reason",
        "backtest_return_pct",
        "backtest_return_delta_pct",
        "backtest_max_drawdown_pct",
        "backtest_drawdown_delta_pct",
        "backtest_sharpe_15m",
        "backtest_sharpe_delta",
        "total_pnl_usd",
        "total_pnl_delta_usd",
        "fills",
        "risk_discipline_score",
        "promotion_status",
        "promotion_reason",
        "wf_positive_fold_fraction",
        "wf_active_positive_fold_fraction",
        "wf_non_negative_fold_fraction",
        "wf_median_active_test_return_pct",
        "wf_worst_test_drawdown_pct",
        "wf_largest_positive_fold_contribution",
        "gate_complexity",
        "strategy_map",
        "multiplier_map",
        "allowed_utc_hours",
        "forex_allowed_utc_hours",
        "metal_allowed_utc_hours",
        "crypto_allowed_utc_hours",
        "symbol_allowed_utc_hours",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(result.rows, start=1):
            payload = row.__dict__.copy()
            payload["rank"] = rank
            writer.writerow(payload)


def _evaluate_candidate(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    spec: DeploymentProfileCandidateSpec,
    train_size: int,
    test_size: int,
    step_size: int,
) -> tuple[
    DeploymentProfileBacktestResult,
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
]:
    backtest = run_deployment_profile_backtest(
        config=config,
        prices=prices,
        quotes=quotes,
        profile_pack_json=spec.profile_pack_json,
        slot=spec.slot,
    )
    profile = backtest.profile
    walk_forward = _run_profile_walk_forward(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    return backtest, walk_forward, decide_fixed_warmup_promotion(walk_forward)


def _run_profile_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
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
    )


def _build_row(
    *,
    spec: DeploymentProfileCandidateSpec,
    backtest: DeploymentProfileBacktestResult,
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    promotion: FixedWarmupPromotionDecision,
    baseline: DeploymentProfileBacktestResult,
    is_baseline: bool,
) -> DeploymentProfileChallengerRow:
    metrics = backtest.competition_metrics
    baseline_metrics = baseline.competition_metrics
    gate_complexity = _gate_complexity(backtest.profile)
    rank_score = _rank_score(
        backtest=backtest,
        walk_forward=walk_forward,
        promotion=promotion,
        gate_complexity=gate_complexity,
    )
    decision, reason = _decision(
        is_baseline=is_baseline,
        backtest=backtest,
        walk_forward=walk_forward,
        promotion=promotion,
        baseline=baseline,
        gate_complexity=gate_complexity,
        baseline_gate_complexity=_gate_complexity(baseline.profile),
    )
    return DeploymentProfileChallengerRow(
        label=spec.label,
        slot=backtest.profile.slot,
        profile_label=backtest.profile.label,
        profile_pack_json=spec.profile_pack_json,
        rank_score=rank_score,
        decision=decision,
        reason=reason,
        backtest_return_pct=metrics.return_pct,
        backtest_return_delta_pct=metrics.return_pct - baseline_metrics.return_pct,
        backtest_max_drawdown_pct=metrics.max_drawdown_pct,
        backtest_drawdown_delta_pct=(
            metrics.max_drawdown_pct - baseline_metrics.max_drawdown_pct
        ),
        backtest_sharpe_15m=metrics.sharpe_15m,
        backtest_sharpe_delta=metrics.sharpe_15m - baseline_metrics.sharpe_15m,
        total_pnl_usd=backtest.result.total_pnl_usd,
        total_pnl_delta_usd=backtest.result.total_pnl_usd - baseline.result.total_pnl_usd,
        fills=len(backtest.result.fills),
        risk_discipline_score=backtest.risk_discipline.score,
        promotion_status=promotion.status,
        promotion_reason=promotion.reason,
        wf_positive_fold_fraction=walk_forward.positive_fold_fraction,
        wf_active_positive_fold_fraction=walk_forward.active_positive_fold_fraction,
        wf_non_negative_fold_fraction=walk_forward.non_negative_fold_fraction,
        wf_median_active_test_return_pct=walk_forward.median_active_test_return_pct,
        wf_worst_test_drawdown_pct=walk_forward.worst_test_drawdown_pct,
        wf_largest_positive_fold_contribution=(
            walk_forward.largest_positive_fold_contribution
        ),
        gate_complexity=gate_complexity,
        strategy_map=backtest.profile.strategy_map_text,
        multiplier_map=backtest.profile.multiplier_map_text,
        allowed_utc_hours=backtest.profile.allowed_hours_text,
        forex_allowed_utc_hours=backtest.profile.forex_hours_text,
        metal_allowed_utc_hours=backtest.profile.metal_hours_text,
        crypto_allowed_utc_hours=backtest.profile.crypto_hours_text,
        symbol_allowed_utc_hours=backtest.profile.symbol_hours_text,
    )


def _decision(
    *,
    is_baseline: bool,
    backtest: DeploymentProfileBacktestResult,
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    promotion: FixedWarmupPromotionDecision,
    baseline: DeploymentProfileBacktestResult,
    gate_complexity: int,
    baseline_gate_complexity: int,
) -> tuple[str, str]:
    if is_baseline:
        return "BASELINE", "reference candidate"
    if promotion.status != "PROMOTE":
        return "REJECT", f"walk-forward status {promotion.status}: {promotion.reason}"
    if backtest.risk_discipline.score < 100:
        return "REJECT", "risk discipline is below 100/100"
    if backtest.competition_metrics.return_pct <= baseline.competition_metrics.return_pct:
        return "REJECT", "does not improve baseline full-sample return"
    if (
        backtest.competition_metrics.max_drawdown_pct
        > baseline.competition_metrics.max_drawdown_pct + 0.001
    ):
        return "PAPER_CHALLENGER", "improves return but increases drawdown"
    if gate_complexity > baseline_gate_complexity + 16:
        return "PAPER_CHALLENGER", "improves return but adds many session restrictions"
    if walk_forward.largest_positive_fold_contribution > 0.80:
        return "PAPER_CHALLENGER", "positive fold contribution remains concentrated"
    return "PROMOTE_CHALLENGER", "improves baseline while passing promotion and risk gates"


def _rank_score(
    *,
    backtest: DeploymentProfileBacktestResult,
    walk_forward: FixedWarmupPortfolioWalkForwardResult,
    promotion: FixedWarmupPromotionDecision,
    gate_complexity: int,
) -> float:
    promotion_bonus = 1.0 if promotion.status == "PROMOTE" else 0.0
    return (
        promotion_bonus
        + 100.0 * backtest.competition_metrics.return_pct
        + 20.0 * max(backtest.competition_metrics.sharpe_15m, 0.0)
        + 2.0 * walk_forward.non_negative_fold_fraction
        + 2.0 * walk_forward.active_positive_fold_fraction
        - 50.0 * backtest.competition_metrics.max_drawdown_pct
        - 0.01 * gate_complexity
    )


def _gate_complexity(profile: LoadedDeploymentProfile) -> int:
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    asset_counts = {
        asset_class: sum(
            1 for symbol in symbols if instrument_for(symbol).asset_class == asset_class
        )
        for asset_class in AssetClass
    }
    complexity = 0
    if profile.allowed_utc_hours is not None:
        complexity += len(symbols) * max(0, 24 - len(profile.allowed_utc_hours))
    if profile.forex_allowed_utc_hours is not None:
        complexity += asset_counts[AssetClass.FOREX] * max(
            0, 24 - len(profile.forex_allowed_utc_hours)
        )
    if profile.metal_allowed_utc_hours is not None:
        complexity += asset_counts[AssetClass.METAL] * max(
            0, 24 - len(profile.metal_allowed_utc_hours)
        )
    if profile.crypto_allowed_utc_hours is not None:
        complexity += asset_counts[AssetClass.CRYPTO] * max(
            0, 24 - len(profile.crypto_allowed_utc_hours)
        )
    for _, hours in profile.symbol_allowed_utc_hours:
        complexity += max(0, 24 - len(hours))
    return complexity


def _decision_rank(decision: str) -> float:
    if decision == "PROMOTE_CHALLENGER":
        return 4.0
    if decision == "BASELINE":
        return 3.0
    if decision == "PAPER_CHALLENGER":
        return 2.0
    return 1.0


def parse_candidate_spec(text: str) -> DeploymentProfileCandidateSpec:
    parts = text.split(",", 2)
    if len(parts) != 3:
        raise ValueError("candidate must be LABEL,PROFILE_PACK_JSON,SLOT")
    label, profile_pack_json, slot = (part.strip() for part in parts)
    if not label or not profile_pack_json or not slot:
        raise ValueError("candidate must include non-empty label, profile pack, and slot")
    return DeploymentProfileCandidateSpec(
        label=label,
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
