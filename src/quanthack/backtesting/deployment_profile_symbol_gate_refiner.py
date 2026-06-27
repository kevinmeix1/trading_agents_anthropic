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
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


ALL_UTC_HOURS = tuple(range(24))


@dataclass(frozen=True)
class SymbolHourAttribution:
    symbol: str
    utc_hour: int
    total_pnl_usd: float
    fills: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.symbol, self.utc_hour)


@dataclass(frozen=True)
class DeploymentProfileSymbolGateCandidate:
    label: str
    dropped_symbol_hours: tuple[SymbolHourAttribution, ...]
    symbol_allowed_utc_hours: tuple[tuple[str, tuple[int, ...]], ...]
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
            -len(self.dropped_symbol_hours),
        )

    @property
    def dropped_symbol_hours_text(self) -> str:
        if not self.dropped_symbol_hours:
            return ""
        return " ".join(
            f"{row.symbol}:{row.utc_hour}" for row in self.dropped_symbol_hours
        )

    @property
    def symbol_hours_text(self) -> str:
        return _symbol_hours_text(self.symbol_allowed_utc_hours)


@dataclass(frozen=True)
class DeploymentProfileSymbolGateRefinementResult:
    base_profile: LoadedDeploymentProfile
    attribution_csv: str
    weak_symbol_hours: tuple[SymbolHourAttribution, ...]
    candidates: tuple[DeploymentProfileSymbolGateCandidate, ...]

    @property
    def best(self) -> DeploymentProfileSymbolGateCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def refine_deployment_profile_symbol_gates(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    attribution_csv: str | Path,
    max_dropped_symbol_hours: int = 5,
    weak_pnl_threshold_usd: float = 0.0,
    min_hour_fills: int = 1,
    include_walk_forward: bool = True,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> DeploymentProfileSymbolGateRefinementResult:
    if max_dropped_symbol_hours < 0:
        raise ValueError("max_dropped_symbol_hours cannot be negative")
    if min_hour_fills < 1:
        raise ValueError("min_hour_fills must be at least 1")
    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    profile_symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    weak_symbol_hours = _weak_symbol_hours_from_attribution(
        attribution_csv=attribution_csv,
        profile_symbols=profile_symbols,
        weak_pnl_threshold_usd=weak_pnl_threshold_usd,
        min_hour_fills=min_hour_fills,
    )
    candidates: list[DeploymentProfileSymbolGateCandidate] = []
    max_count = min(max_dropped_symbol_hours, len(weak_symbol_hours))
    for drop_count in range(max_count + 1):
        dropped = weak_symbol_hours[:drop_count]
        candidate_profile = _profile_with_dropped_symbol_hours(
            profile=profile,
            dropped=dropped,
        )
        if candidate_profile is None:
            continue
        label = _candidate_label(profile.slot, dropped)
        backtest = _run_profile_backtest(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=candidate_profile,
        )
        walk_forward = (
            _run_profile_walk_forward(
                config=config,
                prices=prices,
                quotes=quotes,
                profile=candidate_profile,
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
            DeploymentProfileSymbolGateCandidate(
                label=label,
                dropped_symbol_hours=dropped,
                symbol_allowed_utc_hours=candidate_profile.symbol_allowed_utc_hours,
                result=backtest.result,
                competition_metrics=backtest.competition_metrics,
                risk_discipline=backtest.risk_discipline,
                walk_forward=walk_forward,
                promotion=promotion,
            )
        )
    return DeploymentProfileSymbolGateRefinementResult(
        base_profile=profile,
        attribution_csv=str(attribution_csv),
        weak_symbol_hours=weak_symbol_hours,
        candidates=tuple(
            sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)
        ),
    )


def write_deployment_profile_symbol_gate_refinement_csv(
    result: DeploymentProfileSymbolGateRefinementResult,
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
                "dropped_symbol_hours",
                "dropped_symbol_hour_count",
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
                "symbol_allowed_utc_hours",
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
                    "dropped_symbol_hours": candidate.dropped_symbol_hours_text,
                    "dropped_symbol_hour_count": len(candidate.dropped_symbol_hours),
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
                    "symbol_allowed_utc_hours": candidate.symbol_hours_text,
                }
            )


def write_symbol_gated_profile_pack_json(
    *,
    source_profile_pack_json: str | Path,
    result: DeploymentProfileSymbolGateRefinementResult,
    candidate: DeploymentProfileSymbolGateCandidate,
    output_json: str | Path,
    refined_slot: str = "symbol_refined",
) -> None:
    payload = json.loads(Path(source_profile_pack_json).read_text(encoding="utf-8"))
    base_raw = _raw_profile(payload, result.base_profile.slot)
    refined = dict(base_raw)
    refined["slot"] = refined_slot
    refined["label"] = candidate.label
    refined["evidence_status"] = "PAPER_ONLY"
    refined["use_case"] = (
        "Research-only deployment profile with attribution-derived symbol UTC-hour "
        "gates; validate on fresh official data before live use."
    )
    refined["reason"] = (
        "Dropped weak symbol hours "
        f"{candidate.dropped_symbol_hours_text or 'none'} from session attribution"
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
    refined["symbol_allowed_utc_hours"] = candidate.symbol_hours_text
    profiles = [
        profile
        for profile in payload.get("profiles", ())
        if profile.get("slot") != refined_slot
    ]
    profiles.append(refined)
    payload["profiles"] = profiles
    payload["recommended_slot"] = refined_slot
    payload["recommendation_reason"] = (
        "research-only symbol-hour refinement; rerun on fresh data before live use"
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _ProfileBacktest:
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport


def _run_profile_backtest(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
) -> _ProfileBacktest:
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
        target_notional_multipliers_by_symbol=dict(profile.multipliers_by_symbol),
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
    return _ProfileBacktest(
        result=result,
        competition_metrics=metrics,
        risk_discipline=risk,
    )


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


def _weak_symbol_hours_from_attribution(
    *,
    attribution_csv: str | Path,
    profile_symbols: tuple[str, ...],
    weak_pnl_threshold_usd: float,
    min_hour_fills: int,
) -> tuple[SymbolHourAttribution, ...]:
    profile_symbol_set = set(profile_symbols)
    grouped: dict[tuple[str, int], dict[str, float | int]] = {}
    with Path(attribution_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "utc_hour", "fills", "total_pnl_usd"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"attribution CSV missing required columns: {sorted(missing)}")
        for row in reader:
            symbol = instrument_for(row["symbol"]).symbol
            if symbol not in profile_symbol_set:
                continue
            hour = int(float(row["utc_hour"]))
            key = (symbol, hour)
            bucket = grouped.setdefault(key, {"pnl": 0.0, "fills": 0})
            bucket["pnl"] = float(bucket["pnl"]) + float(row["total_pnl_usd"])
            bucket["fills"] = int(bucket["fills"]) + int(float(row["fills"]))
    weak = tuple(
        SymbolHourAttribution(
            symbol=symbol,
            utc_hour=hour,
            total_pnl_usd=float(values["pnl"]),
            fills=int(values["fills"]),
        )
        for (symbol, hour), values in grouped.items()
        if float(values["pnl"]) < weak_pnl_threshold_usd
        and int(values["fills"]) >= min_hour_fills
    )
    return tuple(sorted(weak, key=lambda row: row.total_pnl_usd))


def _profile_with_dropped_symbol_hours(
    *,
    profile: LoadedDeploymentProfile,
    dropped: tuple[SymbolHourAttribution, ...],
) -> LoadedDeploymentProfile | None:
    symbol_hours = dict(profile.symbol_allowed_utc_hours)
    for row in dropped:
        current_hours = set(symbol_hours.get(row.symbol, _base_hours_for_symbol(profile, row.symbol)))
        current_hours.discard(row.utc_hour)
        if not current_hours:
            return None
        symbol_hours[row.symbol] = tuple(sorted(current_hours))
    return replace(profile, symbol_allowed_utc_hours=tuple(sorted(symbol_hours.items())))


def _base_hours_for_symbol(
    profile: LoadedDeploymentProfile,
    symbol: str,
) -> tuple[int, ...]:
    if profile.allowed_utc_hours is not None:
        global_hours = profile.allowed_utc_hours
    else:
        global_hours = ALL_UTC_HOURS
    asset_class = instrument_for(symbol).asset_class
    if asset_class == AssetClass.FOREX and profile.forex_allowed_utc_hours is not None:
        return profile.forex_allowed_utc_hours
    if asset_class == AssetClass.METAL and profile.metal_allowed_utc_hours is not None:
        return profile.metal_allowed_utc_hours
    if asset_class == AssetClass.CRYPTO and profile.crypto_allowed_utc_hours is not None:
        return profile.crypto_allowed_utc_hours
    return global_hours


def _candidate_label(
    slot: str,
    dropped: tuple[SymbolHourAttribution, ...],
) -> str:
    if not dropped:
        return f"{slot}_symbol_session_base"
    return f"{slot}_symbol_session_drop_{len(dropped)}h"


def _raw_profile(payload: dict, slot: str) -> dict:
    for raw_profile in payload.get("profiles", ()):
        if raw_profile.get("slot") == slot:
            return raw_profile
    raise ValueError(f"profile slot {slot!r} not found in source pack")


def _hours_text(hours: tuple[int, ...]) -> str:
    return "|".join(str(hour) for hour in hours)


def _symbol_hours_text(symbol_hours: tuple[tuple[str, tuple[int, ...]], ...]) -> str:
    if not symbol_hours:
        return ""
    return " ".join(f"{symbol}={_hours_text(hours)}" for symbol, hours in symbol_hours)


def _promotion_rank(status: str) -> float:
    if status == "PROMOTE":
        return 3.0
    if status == "PAPER_ONLY":
        return 2.0
    return 1.0
