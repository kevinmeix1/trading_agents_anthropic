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


DEFAULT_WEAK_SYMBOL_SCALES = (1.0, 0.75, 0.50, 0.25, 0.0)


@dataclass(frozen=True)
class WeakSymbolAttribution:
    symbol: str
    total_pnl_usd: float
    fills: int
    base_multiplier: float


@dataclass(frozen=True)
class DeploymentProfileRefinementCandidate:
    label: str
    weak_symbol_scale: float
    weak_symbols: tuple[WeakSymbolAttribution, ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    walk_forward: FixedWarmupPortfolioWalkForwardResult | None
    promotion: FixedWarmupPromotionDecision | None

    @property
    def rank_key(self) -> tuple[float, ...]:
        wf = self.walk_forward
        promotion_rank = 0.0 if self.promotion is None else _promotion_rank(self.promotion.status)
        return (
            promotion_rank,
            self.competition_metrics.return_pct,
            0.0 if wf is None else wf.active_positive_fold_fraction,
            0.0 if wf is None else wf.non_negative_fold_fraction,
            self.risk_discipline.score,
            -self.competition_metrics.max_drawdown_pct,
            0.0 if wf is None else -wf.worst_test_drawdown_pct,
            0.0 if wf is None else -wf.largest_positive_fold_contribution,
        )

    @property
    def multiplier_map_text(self) -> str:
        return " ".join(
            f"{symbol}={multiplier:.3f}"
            for symbol, multiplier in self.multipliers_by_symbol
        )

    @property
    def weak_symbols_text(self) -> str:
        return " ".join(symbol.symbol for symbol in self.weak_symbols)


@dataclass(frozen=True)
class DeploymentProfileRefinementResult:
    base_profile: LoadedDeploymentProfile
    attribution_csv: str
    weak_symbols: tuple[WeakSymbolAttribution, ...]
    candidates: tuple[DeploymentProfileRefinementCandidate, ...]

    @property
    def best(self) -> DeploymentProfileRefinementCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def refine_deployment_profile_from_attribution(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    attribution_csv: str | Path,
    weak_symbol_scales: tuple[float, ...] = DEFAULT_WEAK_SYMBOL_SCALES,
    weak_pnl_threshold_usd: float = 0.0,
    min_symbol_fills: int = 1,
    include_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> DeploymentProfileRefinementResult:
    if not weak_symbol_scales:
        raise ValueError("at least one weak symbol scale is required")
    if any(scale < 0 or scale > 1 for scale in weak_symbol_scales):
        raise ValueError("weak symbol scales must be between 0 and 1")
    if min_symbol_fills < 1:
        raise ValueError("min_symbol_fills must be at least 1")
    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    weak_symbols = _weak_symbols_from_attribution(
        attribution_csv=attribution_csv,
        profile=profile,
        weak_pnl_threshold_usd=weak_pnl_threshold_usd,
        min_symbol_fills=min_symbol_fills,
    )
    candidates = []
    for scale in tuple(dict.fromkeys(weak_symbol_scales)):
        multipliers = _scaled_multipliers(
            profile=profile,
            weak_symbols=weak_symbols,
            scale=scale,
        )
        label = _candidate_label(profile.slot, scale, weak_symbols)
        backtest = _run_modified_profile_backtest(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            multipliers_by_symbol=multipliers,
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
        candidates.append(
            DeploymentProfileRefinementCandidate(
                label=label,
                weak_symbol_scale=scale,
                weak_symbols=weak_symbols,
                multipliers_by_symbol=multipliers,
                result=backtest.result,
                competition_metrics=backtest.competition_metrics,
                risk_discipline=backtest.risk_discipline,
                walk_forward=walk_forward,
                promotion=promotion,
            )
        )
    return DeploymentProfileRefinementResult(
        base_profile=profile,
        attribution_csv=str(attribution_csv),
        weak_symbols=weak_symbols,
        candidates=tuple(
            sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        ),
    )


def write_deployment_profile_refinement_csv(
    result: DeploymentProfileRefinementResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "label",
                "base_slot",
                "weak_symbol_scale",
                "weak_symbols",
                "weak_symbol_count",
                "return_pct",
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
                "multiplier_map",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            wf = candidate.walk_forward
            writer.writerow(
                {
                    "rank": rank,
                    "label": candidate.label,
                    "base_slot": result.base_profile.slot,
                    "weak_symbol_scale": candidate.weak_symbol_scale,
                    "weak_symbols": candidate.weak_symbols_text,
                    "weak_symbol_count": len(candidate.weak_symbols),
                    "return_pct": candidate.competition_metrics.return_pct,
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
                    "multiplier_map": candidate.multiplier_map_text,
                }
            )


def write_refined_profile_pack_json(
    *,
    source_profile_pack_json: str | Path,
    result: DeploymentProfileRefinementResult,
    candidate: DeploymentProfileRefinementCandidate,
    output_json: str | Path,
    refined_slot: str = "refined",
) -> None:
    source_path = Path(source_profile_pack_json)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    base_raw = _raw_profile(payload, result.base_profile.slot)
    refined = dict(base_raw)
    refined["slot"] = refined_slot
    refined["label"] = candidate.label
    refined["evidence_status"] = "PAPER_ONLY"
    refined["use_case"] = (
        "Research-only profile with attribution-scaled weak symbols; validate "
        "on new official data before live use."
    )
    refined["reason"] = (
        f"Scaled weak attribution symbols {candidate.weak_symbols_text or 'none'} "
        f"to {candidate.weak_symbol_scale:.2f}x"
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
    profiles = [profile for profile in payload.get("profiles", ()) if profile.get("slot") != refined_slot]
    profiles.append(refined)
    payload["profiles"] = profiles
    payload["recommended_slot"] = refined_slot
    payload["recommendation_reason"] = (
        "research-only attribution refinement; rerun on fresh data before live use"
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _ModifiedProfileBacktest:
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


def _run_modified_profile_backtest(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    multipliers_by_symbol: tuple[tuple[str, float], ...],
) -> _ModifiedProfileBacktest:
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(strategy, symbol=symbol)
            for symbol, strategy in profile.strategy_by_symbol
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
        target_notional_multipliers_by_symbol=dict(multipliers_by_symbol),
        session_gate_policy=_session_gate_policy(profile),
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
        session_gate_policy=_session_gate_policy(profile),
        target_notional_multipliers_by_symbol=dict(multipliers_by_symbol),
    )


def _weak_symbols_from_attribution(
    *,
    attribution_csv: str | Path,
    profile: LoadedDeploymentProfile,
    weak_pnl_threshold_usd: float,
    min_symbol_fills: int,
) -> tuple[WeakSymbolAttribution, ...]:
    grouped: dict[str, dict[str, float | int]] = {}
    with Path(attribution_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "fills", "total_pnl_usd"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"attribution CSV missing required columns: {sorted(missing)}")
        for row in reader:
            symbol = instrument_for(row["symbol"]).symbol
            bucket = grouped.setdefault(symbol, {"pnl": 0.0, "fills": 0})
            bucket["pnl"] = float(bucket["pnl"]) + float(row["total_pnl_usd"])
            bucket["fills"] = int(bucket["fills"]) + int(float(row["fills"]))
    base_multipliers = dict(profile.multipliers_by_symbol)
    weak = tuple(
        WeakSymbolAttribution(
            symbol=symbol,
            total_pnl_usd=float(values["pnl"]),
            fills=int(values["fills"]),
            base_multiplier=base_multipliers.get(symbol, 1.0),
        )
        for symbol, values in grouped.items()
        if float(values["pnl"]) < weak_pnl_threshold_usd
        and int(values["fills"]) >= min_symbol_fills
        and symbol in base_multipliers
    )
    return tuple(sorted(weak, key=lambda row: row.total_pnl_usd))


def _scaled_multipliers(
    *,
    profile: LoadedDeploymentProfile,
    weak_symbols: tuple[WeakSymbolAttribution, ...],
    scale: float,
) -> tuple[tuple[str, float], ...]:
    weak = {row.symbol for row in weak_symbols}
    return tuple(
        sorted(
            (
                symbol,
                multiplier * scale if symbol in weak else multiplier,
            )
            for symbol, multiplier in profile.multipliers_by_symbol
        )
    )


def _session_gate_policy(profile: LoadedDeploymentProfile):
    return session_gate_policy_for_profile(profile)


def _candidate_label(
    slot: str,
    scale: float,
    weak_symbols: tuple[WeakSymbolAttribution, ...],
) -> str:
    if not weak_symbols:
        return f"{slot}_no_weak_symbols"
    return f"{slot}_weak_symbols_{scale:.2f}x".replace(".", "p")


def _raw_profile(payload: dict, slot: str) -> dict:
    for raw_profile in payload.get("profiles", ()):
        if raw_profile.get("slot") == slot:
            return raw_profile
    raise ValueError(f"profile slot {slot!r} not found in source pack")


def _promotion_rank(status: str) -> float:
    if status == "PROMOTE":
        return 3.0
    if status == "PAPER_ONLY":
        return 2.0
    return 1.0
