from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.crypto_overlay_sizing_compare import (
    CryptoOverlaySizingSpec,
    _aggressive_strategy_map,
    _multipliers_for_spec,
    _run_strategy_map,
    _selected_symbols,
    _session_policy_for_spec,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestResult
from quanthack.backtesting.portfolio_fixed_warmup_walk_forward import (
    FixedWarmupPortfolioWalkForwardResult,
    FixedWarmupPromotionDecision,
    decide_fixed_warmup_promotion,
    run_fixed_warmup_portfolio_walk_forward,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.strategies.strategy import normalize_strategy_name


@dataclass(frozen=True)
class AssetClassStabilitySpec:
    label: str
    fx_multiplier: float
    metal_multiplier: float
    crypto_spec: CryptoOverlaySizingSpec

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("asset-class stability label is required")
        for name, value in (
            ("fx_multiplier", self.fx_multiplier),
            ("metal_multiplier", self.metal_multiplier),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class AssetClassStabilityCandidate:
    spec: AssetClassStabilitySpec
    strategy_by_symbol: tuple[tuple[str, str], ...]
    multipliers_by_symbol: tuple[tuple[str, float], ...]
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    return_retention: float
    stability_score: float
    stability_status: str
    walk_forward: FixedWarmupPortfolioWalkForwardResult
    promotion: FixedWarmupPromotionDecision

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            self.stability_score,
            self.return_retention,
            1.0 - self.walk_forward.largest_positive_fold_contribution,
            self.walk_forward.active_positive_fold_fraction,
            self.walk_forward.non_negative_fold_fraction,
            self.walk_forward.median_active_test_return_pct,
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
    def strategy_map_text(self) -> str:
        return " ".join(
            f"{symbol}={strategy}" for symbol, strategy in self.strategy_by_symbol
        )

    @property
    def crypto_hours_text(self) -> str:
        hours = self.spec.crypto_spec.crypto_allowed_utc_hours
        if hours is None:
            return "all"
        return "|".join(str(hour) for hour in hours)


@dataclass(frozen=True)
class AssetClassStabilityOptimization:
    official_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    base_strategy: str
    candidates: tuple[AssetClassStabilityCandidate, ...]

    @property
    def best(self) -> AssetClassStabilityCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def default_asset_class_stability_specs() -> tuple[AssetClassStabilitySpec, ...]:
    profiles = (
        (
            "current_london",
            CryptoOverlaySizingSpec(
                label="current_london",
                crypto_multiplier=0.75,
                btc_multiplier=0.75,
                sol_multiplier=1.0,
                crypto_allowed_utc_hours=tuple(range(7, 17)),
            ),
        ),
        (
            "current_asia",
            CryptoOverlaySizingSpec(
                label="current_asia",
                crypto_multiplier=0.75,
                btc_multiplier=0.75,
                sol_multiplier=1.0,
                crypto_allowed_utc_hours=tuple(range(0, 9)),
            ),
        ),
        (
            "current_all",
            CryptoOverlaySizingSpec(
                label="current_all",
                crypto_multiplier=0.75,
                btc_multiplier=0.75,
                sol_multiplier=1.0,
            ),
        ),
        (
            "soft_london",
            CryptoOverlaySizingSpec(
                label="soft_london",
                crypto_multiplier=0.50,
                btc_multiplier=0.50,
                sol_multiplier=0.75,
                reversion_crypto_multiplier=0.50,
                crypto_allowed_utc_hours=tuple(range(7, 17)),
            ),
        ),
        (
            "soft_asia",
            CryptoOverlaySizingSpec(
                label="soft_asia",
                crypto_multiplier=0.50,
                btc_multiplier=0.50,
                sol_multiplier=0.75,
                reversion_crypto_multiplier=0.50,
                crypto_allowed_utc_hours=tuple(range(0, 9)),
            ),
        ),
    )
    asset_scales = (
        ("full", 1.00, 1.00),
        ("metal75", 1.00, 0.75),
        ("metal50", 1.00, 0.50),
        ("metal25", 1.00, 0.25),
        ("fx75_metal75", 0.75, 0.75),
        ("fx75_metal50", 0.75, 0.50),
    )
    return tuple(
        AssetClassStabilitySpec(
            label=f"{profile_label}_{asset_label}",
            fx_multiplier=fx_multiplier,
            metal_multiplier=metal_multiplier,
            crypto_spec=crypto_spec,
        )
        for profile_label, crypto_spec in profiles
        for asset_label, fx_multiplier, metal_multiplier in asset_scales
    )


DEFAULT_ASSET_CLASS_STABILITY_SPECS = default_asset_class_stability_specs()


def optimize_asset_class_stability(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    base_strategy: str = "macd_momentum",
    symbols: tuple[str, ...] | None = None,
    specs: tuple[AssetClassStabilitySpec, ...] = DEFAULT_ASSET_CLASS_STABILITY_SPECS,
    train_size: int = 96,
    test_size: int = 48,
    step_size: int = 48,
) -> AssetClassStabilityOptimization:
    if not specs:
        raise ValueError("asset-class stability optimizer needs at least one spec")
    selected_symbols = _selected_symbols(prices=prices, quotes=quotes, symbols=symbols)
    official_symbols = tuple(
        symbol
        for symbol in selected_symbols
        if instrument_for(symbol).asset_class != AssetClass.CRYPTO
    )
    crypto_symbols = tuple(
        symbol
        for symbol in selected_symbols
        if instrument_for(symbol).asset_class == AssetClass.CRYPTO
    )
    normalized_base = normalize_strategy_name(base_strategy)
    strategy_map = _aggressive_strategy_map(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        base_strategy=normalized_base,
    )

    candidates: list[AssetClassStabilityCandidate] = []
    for spec in specs:
        multipliers = _multipliers_for_asset_spec(
            spec=spec,
            symbols=selected_symbols,
            crypto_symbols=crypto_symbols,
            strategy_by_symbol=strategy_map,
        )
        session_policy = _session_policy_for_spec(spec.crypto_spec)
        result = _run_strategy_map(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_by_symbol=strategy_map,
            multipliers_by_symbol=multipliers,
            session_gate_policy=session_policy,
        )
        metrics = build_competition_metrics(
            equity_points=result.equity_curve,
            fills=result.fills,
        )
        risk_discipline = build_risk_discipline_report(
            risk_samples_from_portfolio_equity(result.equity_curve)
        )
        walk_forward = run_fixed_warmup_portfolio_walk_forward(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_name=normalized_base,
            symbols=tuple(symbol for symbol, _ in sorted(strategy_map.items())),
            strategy_by_symbol=strategy_map,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            target_notional_multipliers_by_symbol=multipliers,
            session_gate_policy=session_policy,
        )
        promotion = decide_fixed_warmup_promotion(walk_forward)
        candidates.append(
            AssetClassStabilityCandidate(
                spec=spec,
                strategy_by_symbol=tuple(sorted(strategy_map.items())),
                multipliers_by_symbol=tuple(sorted(multipliers.items())),
                result=result,
                competition_metrics=metrics,
                risk_discipline=risk_discipline,
                return_retention=0.0,
                stability_score=0.0,
                stability_status="UNSCORED",
                walk_forward=walk_forward,
                promotion=promotion,
            )
        )

    scored = _score_candidates(tuple(candidates))
    return AssetClassStabilityOptimization(
        official_symbols=official_symbols,
        crypto_symbols=crypto_symbols,
        base_strategy=normalized_base,
        candidates=tuple(sorted(scored, key=lambda candidate: candidate.rank_key, reverse=True)),
    )


def write_asset_class_stability_csv(
    result: AssetClassStabilityOptimization,
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
                "stability_status",
                "stability_score",
                "return_retention",
                "fx_multiplier",
                "metal_multiplier",
                "crypto_profile",
                "crypto_allowed_utc_hours",
                "official_symbols",
                "crypto_symbols",
                "base_strategy",
                "strategy_map",
                "multiplier_map",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "trade_count",
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
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            metrics = candidate.competition_metrics
            walk_forward = candidate.walk_forward
            writer.writerow(
                {
                    "rank": rank,
                    "label": candidate.spec.label,
                    "stability_status": candidate.stability_status,
                    "stability_score": candidate.stability_score,
                    "return_retention": candidate.return_retention,
                    "fx_multiplier": candidate.spec.fx_multiplier,
                    "metal_multiplier": candidate.spec.metal_multiplier,
                    "crypto_profile": candidate.spec.crypto_spec.label,
                    "crypto_allowed_utc_hours": candidate.crypto_hours_text,
                    "official_symbols": " ".join(result.official_symbols),
                    "crypto_symbols": " ".join(result.crypto_symbols),
                    "base_strategy": result.base_strategy,
                    "strategy_map": candidate.strategy_map_text,
                    "multiplier_map": candidate.multiplier_map_text,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": candidate.risk_discipline.score,
                    "trade_count": metrics.trade_count,
                    "fills": len(candidate.result.fills),
                    "total_pnl_usd": candidate.result.total_pnl_usd,
                    "promotion_status": candidate.promotion.status,
                    "promotion_reason": candidate.promotion.reason,
                    "wf_positive_fold_fraction": (
                        walk_forward.positive_fold_fraction
                    ),
                    "wf_active_positive_fold_fraction": (
                        walk_forward.active_positive_fold_fraction
                    ),
                    "wf_non_negative_fold_fraction": (
                        walk_forward.non_negative_fold_fraction
                    ),
                    "wf_median_active_test_return_pct": (
                        walk_forward.median_active_test_return_pct
                    ),
                    "wf_worst_test_drawdown_pct": walk_forward.worst_test_drawdown_pct,
                    "wf_largest_positive_fold_contribution": (
                        walk_forward.largest_positive_fold_contribution
                    ),
                }
            )


def _multipliers_for_asset_spec(
    *,
    spec: AssetClassStabilitySpec,
    symbols: tuple[str, ...],
    crypto_symbols: tuple[str, ...],
    strategy_by_symbol: dict[str, str],
) -> dict[str, float]:
    crypto_multipliers = _multipliers_for_spec(
        spec=spec.crypto_spec,
        crypto_symbols=crypto_symbols,
        strategy_by_symbol=strategy_by_symbol,
    )
    multipliers: dict[str, float] = {}
    for symbol in symbols:
        asset_class = instrument_for(symbol).asset_class
        if asset_class == AssetClass.FOREX:
            multipliers[symbol] = spec.fx_multiplier
        elif asset_class == AssetClass.METAL:
            multipliers[symbol] = spec.metal_multiplier
        else:
            multipliers[symbol] = crypto_multipliers.get(symbol, 1.0)
    return multipliers


def _score_candidates(
    candidates: tuple[AssetClassStabilityCandidate, ...],
) -> tuple[AssetClassStabilityCandidate, ...]:
    best_return = max(
        (candidate.competition_metrics.return_pct for candidate in candidates),
        default=0.0,
    )
    best_positive_return = max(best_return, 0.0)
    scored: list[AssetClassStabilityCandidate] = []
    for candidate in candidates:
        return_retention = (
            0.0
            if best_positive_return <= 0
            else max(candidate.competition_metrics.return_pct, 0.0)
            / best_positive_return
        )
        scored.append(
            AssetClassStabilityCandidate(
                spec=candidate.spec,
                strategy_by_symbol=candidate.strategy_by_symbol,
                multipliers_by_symbol=candidate.multipliers_by_symbol,
                result=candidate.result,
                competition_metrics=candidate.competition_metrics,
                risk_discipline=candidate.risk_discipline,
                return_retention=return_retention,
                stability_score=_stability_score(candidate, return_retention),
                stability_status=_stability_status(candidate),
                walk_forward=candidate.walk_forward,
                promotion=candidate.promotion,
            )
        )
    return tuple(scored)


def _stability_score(
    candidate: AssetClassStabilityCandidate,
    return_retention: float,
) -> float:
    metrics = candidate.competition_metrics
    walk_forward = candidate.walk_forward
    risk_score = min(max(candidate.risk_discipline.score, 0.0), 100.0) / 100.0
    drawdown_score = 1.0 - min(metrics.max_drawdown_pct / 0.03, 1.0)
    trade_score = min(metrics.trade_count / 40.0, 1.0)
    concentration_score = 1.0 - min(
        max(walk_forward.largest_positive_fold_contribution, 0.0),
        1.0,
    )
    base = (
        0.18 * return_retention
        + 0.20 * walk_forward.active_positive_fold_fraction
        + 0.17 * walk_forward.non_negative_fold_fraction
        + 0.20 * concentration_score
        + 0.10 * drawdown_score
        + 0.10 * risk_score
        + 0.05 * trade_score
    )
    penalty = 0.0
    if metrics.return_pct <= 0:
        penalty += 40.0
    if walk_forward.median_active_test_return_pct <= 0:
        penalty += 20.0
    return 100.0 * base - penalty


def _stability_status(candidate: AssetClassStabilityCandidate) -> str:
    metrics = candidate.competition_metrics
    walk_forward = candidate.walk_forward
    if metrics.return_pct <= 0 or candidate.risk_discipline.score < 95:
        return "REJECT"
    if (
        walk_forward.non_negative_fold_fraction < 0.70
        or walk_forward.median_active_test_return_pct <= 0
    ):
        return "REJECT"
    if (
        walk_forward.largest_positive_fold_contribution <= 0.80
        and walk_forward.positive_fold_fraction >= 0.67
        and walk_forward.active_positive_fold_fraction >= 0.67
    ):
        return "STABLE_PROFILE"
    return "FRAGILE_PROFILE"
