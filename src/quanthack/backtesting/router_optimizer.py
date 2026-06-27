from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    official_composite_score,
    risk_samples_from_portfolio_equity,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.market.market_quality import MarketQualityLimits
from quanthack.backtesting.portfolio_allocator import AllocationPolicy
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine, PortfolioBacktestResult
from quanthack.backtesting.warmup import evaluate_portfolio_after_warmup
from quanthack.strategies.strategy import AlphaRouterConfig, build_strategy


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class RouterWeightSet:
    momentum_weight: float
    moving_average_weight: float
    breakout_weight: float
    mean_reversion_weight: float
    session_breakout_weight: float = 0.25
    cross_rate_weight: float = 0.0
    relative_strength_weight: float = 0.0
    volatility_squeeze_weight: float = 0.0
    dual_squeeze_weight: float = 0.0
    macd_momentum_weight: float = 0.0
    kalman_trend_weight: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("momentum_weight", self.momentum_weight),
            ("moving_average_weight", self.moving_average_weight),
            ("breakout_weight", self.breakout_weight),
            ("mean_reversion_weight", self.mean_reversion_weight),
            ("session_breakout_weight", self.session_breakout_weight),
            ("cross_rate_weight", self.cross_rate_weight),
            ("relative_strength_weight", self.relative_strength_weight),
            ("volatility_squeeze_weight", self.volatility_squeeze_weight),
            ("dual_squeeze_weight", self.dual_squeeze_weight),
            ("macd_momentum_weight", self.macd_momentum_weight),
            ("kalman_trend_weight", self.kalman_trend_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.total <= 0:
            raise ValueError("at least one router weight must be positive")

    @property
    def total(self) -> float:
        return (
            self.momentum_weight
            + self.moving_average_weight
            + self.breakout_weight
            + self.mean_reversion_weight
            + self.session_breakout_weight
            + self.cross_rate_weight
            + self.relative_strength_weight
            + self.volatility_squeeze_weight
            + self.dual_squeeze_weight
            + self.macd_momentum_weight
            + self.kalman_trend_weight
        )

    @property
    def label(self) -> str:
        return (
            f"mom={self.momentum_weight:.2f};"
            f"ma={self.moving_average_weight:.2f};"
            f"breakout={self.breakout_weight:.2f};"
            f"reversion={self.mean_reversion_weight:.2f};"
            f"session={self.session_breakout_weight:.2f};"
            f"xrate={self.cross_rate_weight:.2f};"
            f"rel={self.relative_strength_weight:.2f};"
            f"squeeze={self.volatility_squeeze_weight:.2f};"
            f"dual={self.dual_squeeze_weight:.2f};"
            f"macd={self.macd_momentum_weight:.2f};"
            f"kalman={self.kalman_trend_weight:.2f}"
        )


@dataclass(frozen=True)
class RouterBehaviorProfile:
    entry_score: float = 0.35
    min_signal_confidence: float = 0.20
    cost_buffer: float = 1.20
    conflict_penalty: float = 0.50
    primary_signal_override_enabled: bool = True
    exit_score: float | None = None

    def __post_init__(self) -> None:
        _validate_positive("entry_score", self.entry_score)
        if not 0 <= self.min_signal_confidence <= 1:
            raise ValueError("min_signal_confidence must be between 0 and 1")
        _validate_positive("cost_buffer", self.cost_buffer)
        if not 0 <= self.conflict_penalty <= 1:
            raise ValueError("conflict_penalty must be between 0 and 1")
        if self.exit_score is not None:
            _validate_non_negative("exit_score", self.exit_score)
            if self.exit_score >= self.entry_score:
                raise ValueError("exit_score must be below entry_score")

    @property
    def resolved_exit_score(self) -> float:
        if self.exit_score is not None:
            return self.exit_score
        return min(0.12, self.entry_score * 0.45)

    @property
    def label(self) -> str:
        override = "on" if self.primary_signal_override_enabled else "off"
        return (
            f"entry={self.entry_score:.2f};"
            f"exit={self.resolved_exit_score:.2f};"
            f"conf={self.min_signal_confidence:.2f};"
            f"cost={self.cost_buffer:.2f};"
            f"conflict={self.conflict_penalty:.2f};"
            f"override={override}"
        )


DEFAULT_ROUTER_WEIGHT_SETS: tuple[RouterWeightSet, ...] = (
    RouterWeightSet(0.30, 0.15, 0.15, 0.35, 0.25, 0.00),
    RouterWeightSet(0.40, 0.20, 0.35, 0.25, 0.25, 0.00),
    RouterWeightSet(0.50, 0.25, 0.35, 0.10, 0.15, 0.00),
    RouterWeightSet(0.30, 0.30, 0.25, 0.25, 0.15, 0.00),
    RouterWeightSet(0.25, 0.20, 0.20, 0.45, 0.10, 0.00),
    RouterWeightSet(0.35, 0.40, 0.15, 0.10, 0.15, 0.00),
    RouterWeightSet(0.20, 0.20, 0.50, 0.10, 0.20, 0.00),
    RouterWeightSet(0.25, 0.15, 0.15, 0.35, 0.20, 0.10),
    RouterWeightSet(0.20, 0.15, 0.10, 0.40, 0.15, 0.20),
    RouterWeightSet(0.40, 0.20, 0.25, 0.15, 0.20, 0.05),
    RouterWeightSet(0.25, 0.15, 0.10, 0.35, 0.15, 0.00, 0.05),
    RouterWeightSet(0.20, 0.10, 0.10, 0.40, 0.15, 0.05, 0.10),
    RouterWeightSet(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
    RouterWeightSet(0.05, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.80),
    RouterWeightSet(0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.80),
    RouterWeightSet(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
    RouterWeightSet(0.05, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.85),
    RouterWeightSet(0.15, 0.05, 0.05, 0.15, 0.10, 0.00, 0.00, 0.00, 0.00, 0.60, 0.20),
    RouterWeightSet(0.10, 0.05, 0.05, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00, 0.40, 0.40),
    RouterWeightSet(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00),
    RouterWeightSet(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
)


DEFAULT_ROUTER_BEHAVIOR_PROFILES: tuple[RouterBehaviorProfile, ...] = (
    RouterBehaviorProfile(),
)


CONSERVATIVE_ROUTER_BEHAVIOR_PROFILES: tuple[RouterBehaviorProfile, ...] = (
    RouterBehaviorProfile(),
    RouterBehaviorProfile(0.55, 0.20, 1.20, 0.70, False),
    RouterBehaviorProfile(0.55, 0.20, 2.20, 0.70, False),
    RouterBehaviorProfile(0.55, 0.35, 1.20, 0.70, False),
    RouterBehaviorProfile(0.55, 0.35, 2.20, 0.70, False),
    RouterBehaviorProfile(0.75, 0.20, 1.20, 0.70, False),
)


@dataclass(frozen=True)
class RouterOptimizationCandidate:
    weights: RouterWeightSet
    behavior: RouterBehaviorProfile
    result: PortfolioBacktestResult
    competition_metrics: CompetitionMetrics
    risk_discipline: RiskDisciplineReport
    return_rank: float = 0.0
    drawdown_rank: float = 0.0
    sharpe_rank: float = 0.0
    proxy_score: float = 0.0

    @property
    def rank_key(self) -> tuple[float, int, float, float, float]:
        return (
            self.proxy_score,
            self.risk_discipline.score,
            self.competition_metrics.return_pct,
            self.competition_metrics.sharpe_15m,
            -self.competition_metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class RouterOptimizationResult:
    symbols: tuple[str, ...]
    candidates: tuple[RouterOptimizationCandidate, ...]

    @property
    def best(self) -> RouterOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def optimize_router_weights(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    weight_sets: tuple[RouterWeightSet, ...] = DEFAULT_ROUTER_WEIGHT_SETS,
    behavior_profiles: tuple[
        RouterBehaviorProfile, ...
    ] = DEFAULT_ROUTER_BEHAVIOR_PROFILES,
    allocation_policy: AllocationPolicy | None = None,
    evaluation_start: datetime | str | None = None,
) -> RouterOptimizationResult:
    if not weight_sets:
        raise ValueError("at least one router weight set is required")
    if not behavior_profiles:
        raise ValueError("at least one router behavior profile is required")
    selected_symbols = _selected_symbols(prices=prices, quotes=quotes, symbols=symbols)

    candidates: list[RouterOptimizationCandidate] = []
    for weights in weight_sets:
        for behavior in behavior_profiles:
            router_config = _router_config(config.alpha_router, weights, behavior)
            engine = PortfolioBacktestEngine(
                strategies={
                    symbol: _build_router_strategy(config, router_config, symbol)
                    for symbol in selected_symbols
                },
                risk_limits=config.risk,
                quality_limits=config.market_quality,
                quality_limits_by_symbol={
                    symbol: _quality_limits_for_symbol(config.market_quality, symbol)
                    for symbol in selected_symbols
                },
                allocation_policy=allocation_policy,
                clock=config.competition.to_clock(),
                fill_model=FillModel(slippage_bps=config.backtest.slippage_bps),
                periods_per_year=config.backtest.periods_per_year,
            )
            result = engine.run(
                prices=prices,
                quotes=quotes,
                starting_equity=config.competition.starting_equity,
            )
            if evaluation_start is None:
                competition_metrics = build_competition_metrics(
                    equity_points=result.equity_curve,
                    fills=result.fills,
                )
                risk_discipline = build_risk_discipline_report(
                    risk_samples_from_portfolio_equity(result.equity_curve)
                )
            else:
                evaluation = evaluate_portfolio_after_warmup(
                    result,
                    evaluation_start=evaluation_start,
                )
                competition_metrics = evaluation.competition_metrics
                risk_discipline = evaluation.risk_discipline
            candidates.append(
                RouterOptimizationCandidate(
                    weights=weights,
                    behavior=behavior,
                    result=result,
                    competition_metrics=competition_metrics,
                    risk_discipline=risk_discipline,
                )
            )

    ranked = _attach_proxy_scores(tuple(candidates))
    ranked = tuple(sorted(ranked, key=lambda item: item.rank_key, reverse=True))
    return RouterOptimizationResult(symbols=selected_symbols, candidates=ranked)


def write_router_optimization_csv(
    result: RouterOptimizationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "symbols",
                "weights",
                "behavior",
                "proxy_score",
                "momentum_weight",
                "moving_average_weight",
                "breakout_weight",
                "mean_reversion_weight",
                "session_breakout_weight",
                "cross_rate_weight",
                "relative_strength_weight",
                "volatility_squeeze_weight",
                "dual_squeeze_weight",
                "macd_momentum_weight",
                "kalman_trend_weight",
                "entry_score",
                "exit_score",
                "min_signal_confidence",
                "cost_buffer",
                "conflict_penalty",
                "primary_signal_override_enabled",
                "return_rank",
                "drawdown_rank",
                "sharpe_rank",
                "risk_discipline_score",
                "final_equity",
                "official_return_pct",
                "official_max_drawdown_pct",
                "official_15m_sharpe",
                "trade_count",
                "fills",
                "turnover_notional",
                "trimmed_allocation_periods",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            metrics = candidate.competition_metrics
            writer.writerow(
                {
                    "rank": rank,
                    "symbols": " ".join(result.symbols),
                    "weights": candidate.weights.label,
                    "behavior": candidate.behavior.label,
                    "proxy_score": candidate.proxy_score,
                    "momentum_weight": candidate.weights.momentum_weight,
                    "moving_average_weight": candidate.weights.moving_average_weight,
                    "breakout_weight": candidate.weights.breakout_weight,
                    "mean_reversion_weight": candidate.weights.mean_reversion_weight,
                    "session_breakout_weight": (
                        candidate.weights.session_breakout_weight
                    ),
                    "cross_rate_weight": candidate.weights.cross_rate_weight,
                    "relative_strength_weight": (
                        candidate.weights.relative_strength_weight
                    ),
                    "volatility_squeeze_weight": (
                        candidate.weights.volatility_squeeze_weight
                    ),
                    "dual_squeeze_weight": candidate.weights.dual_squeeze_weight,
                    "macd_momentum_weight": candidate.weights.macd_momentum_weight,
                    "kalman_trend_weight": candidate.weights.kalman_trend_weight,
                    "entry_score": candidate.behavior.entry_score,
                    "exit_score": candidate.behavior.resolved_exit_score,
                    "min_signal_confidence": candidate.behavior.min_signal_confidence,
                    "cost_buffer": candidate.behavior.cost_buffer,
                    "conflict_penalty": candidate.behavior.conflict_penalty,
                    "primary_signal_override_enabled": (
                        candidate.behavior.primary_signal_override_enabled
                    ),
                    "return_rank": candidate.return_rank,
                    "drawdown_rank": candidate.drawdown_rank,
                    "sharpe_rank": candidate.sharpe_rank,
                    "risk_discipline_score": candidate.risk_discipline.score,
                    "final_equity": metrics.final_equity,
                    "official_return_pct": metrics.return_pct,
                    "official_max_drawdown_pct": metrics.max_drawdown_pct,
                    "official_15m_sharpe": metrics.sharpe_15m,
                    "trade_count": metrics.trade_count,
                    "fills": candidate.competition_metrics.trade_count,
                    "turnover_notional": candidate.result.metrics.turnover_notional,
                    "trimmed_allocation_periods": len(
                        [
                            report
                            for report in candidate.result.allocation_reports
                            if report.trimmed_targets
                        ]
                    ),
                }
            )


def _router_config(
    base: AlphaRouterConfig,
    weights: RouterWeightSet,
    behavior: RouterBehaviorProfile,
) -> AlphaRouterConfig:
    return replace(
        base,
        momentum_weight=weights.momentum_weight,
        moving_average_weight=weights.moving_average_weight,
        breakout_weight=weights.breakout_weight,
        mean_reversion_weight=weights.mean_reversion_weight,
        session_breakout_weight=weights.session_breakout_weight,
        cross_rate_weight=weights.cross_rate_weight,
        relative_strength_weight=weights.relative_strength_weight,
        volatility_squeeze_weight=weights.volatility_squeeze_weight,
        dual_squeeze_weight=weights.dual_squeeze_weight,
        macd_momentum_weight=weights.macd_momentum_weight,
        kalman_trend_weight=weights.kalman_trend_weight,
        entry_score=behavior.entry_score,
        exit_score=behavior.resolved_exit_score,
        min_signal_confidence=behavior.min_signal_confidence,
        cost_buffer=behavior.cost_buffer,
        conflict_penalty=behavior.conflict_penalty,
        primary_signal_override_enabled=behavior.primary_signal_override_enabled,
    )


def _build_router_strategy(
    config: AppConfig,
    router_config: AlphaRouterConfig,
    symbol: str,
):
    instrument = instrument_for(symbol)
    return build_strategy(
        "alpha_router",
        simple_momentum=replace(
            config.simple_momentum,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        ma_crossover=replace(
            config.ma_crossover,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        macd_momentum=replace(
            config.macd_momentum,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        kalman_trend=replace(
            config.kalman_trend,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        breakout=replace(
            config.breakout,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        volatility_squeeze=replace(
            config.volatility_squeeze,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        dual_squeeze=replace(
            config.dual_squeeze,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        session_breakout=replace(
            config.session_breakout,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        mean_reversion=replace(
            config.mean_reversion,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        relative_strength=replace(
            config.relative_strength,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        alpha_router=replace(
            router_config,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
        cross_rate_reversion=replace(
            config.cross_rate_reversion,
            symbol=instrument.symbol,
            max_spread_bps=instrument.max_spread_bps,
        ),
    )


def _quality_limits_for_symbol(
    base_limits: MarketQualityLimits,
    symbol: str,
) -> MarketQualityLimits:
    return replace(
        base_limits,
        max_spread_bps=instrument_for(symbol).max_spread_bps,
    )


def _selected_symbols(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if symbols:
        selected = tuple(instrument_for(symbol).symbol for symbol in symbols)
    else:
        selected = tuple(sorted(set(prices.symbols()) & set(quotes.symbols())))
    if not selected:
        raise ValueError("no symbols found in both price and quote data")
    return selected


def _attach_proxy_scores(
    candidates: tuple[RouterOptimizationCandidate, ...],
) -> tuple[RouterOptimizationCandidate, ...]:
    return_ranks = _percentile_scores(
        [item.competition_metrics.return_pct for item in candidates],
        higher_is_better=True,
    )
    drawdown_ranks = _percentile_scores(
        [item.competition_metrics.max_drawdown_pct for item in candidates],
        higher_is_better=False,
    )
    sharpe_ranks = _percentile_scores(
        [item.competition_metrics.sharpe_15m for item in candidates],
        higher_is_better=True,
    )

    ranked: list[RouterOptimizationCandidate] = []
    for item, return_rank, drawdown_rank, sharpe_rank in zip(
        candidates,
        return_ranks,
        drawdown_ranks,
        sharpe_ranks,
        strict=True,
    ):
        proxy_score = official_composite_score(
            return_rank=return_rank,
            drawdown_rank=drawdown_rank,
            sharpe_rank=sharpe_rank,
            risk_discipline_score=item.risk_discipline.score,
            sharpe_rank_cap=item.competition_metrics.sharpe_rank_cap,
        )
        ranked.append(
            replace(
                item,
                return_rank=return_rank,
                drawdown_rank=drawdown_rank,
                sharpe_rank=sharpe_rank,
                proxy_score=proxy_score,
            )
        )
    return tuple(ranked)


def _percentile_scores(
    values: list[float],
    *,
    higher_is_better: bool,
) -> tuple[float, ...]:
    if len(values) <= 1 or len(set(values)) == 1:
        return tuple(100.0 for _ in values)
    denominator = len(values) - 1
    scores: list[float] = []
    for value in values:
        if higher_is_better:
            worse = sum(1 for other in values if other < value)
        else:
            worse = sum(1 for other in values if other > value)
        scores.append((worse / denominator) * 100)
    return tuple(scores)
