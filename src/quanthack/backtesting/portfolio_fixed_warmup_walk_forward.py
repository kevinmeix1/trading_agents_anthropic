from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine
from quanthack.backtesting.portfolio_regime import RegimeTiltPolicy
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.backtesting.portfolio_symbol_evidence import SymbolEvidenceGatePolicy
from quanthack.backtesting.portfolio_volatility import VolatilityTargetingPolicy
from quanthack.backtesting.warmup import (
    WarmupPortfolioEvaluation,
    evaluate_portfolio_after_warmup,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceBar, PriceHistory, QuoteHistory, QuoteSnapshot

RETURN_EPSILON = 1e-12


@dataclass(frozen=True)
class FixedWarmupPortfolioFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    evaluation: WarmupPortfolioEvaluation
    full_run_fills: int

    @property
    def metrics(self) -> CompetitionMetrics:
        return self.evaluation.competition_metrics

    @property
    def risk_discipline(self) -> RiskDisciplineReport:
        return self.evaluation.risk_discipline


@dataclass(frozen=True)
class FixedWarmupPortfolioWalkForwardResult:
    strategy_name: str
    symbols: tuple[str, ...]
    folds: tuple[FixedWarmupPortfolioFold, ...]

    @property
    def positive_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        positive = [fold for fold in self.folds if fold.metrics.return_pct > 0]
        return len(positive) / len(self.folds)

    @property
    def active_folds(self) -> tuple[FixedWarmupPortfolioFold, ...]:
        return tuple(
            fold
            for fold in self.folds
            if len(fold.evaluation.fills) > 0
            or abs(fold.metrics.return_pct) > RETURN_EPSILON
        )

    @property
    def active_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        return len(self.active_folds) / len(self.folds)

    @property
    def active_positive_fold_fraction(self) -> float:
        active_folds = self.active_folds
        if not active_folds:
            return 0.0
        positive = [fold for fold in active_folds if fold.metrics.return_pct > 0]
        return len(positive) / len(active_folds)

    @property
    def non_negative_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        non_negative = [
            fold for fold in self.folds if fold.metrics.return_pct >= -RETURN_EPSILON
        ]
        return len(non_negative) / len(self.folds)

    @property
    def losing_fold_fraction(self) -> float:
        if not self.folds:
            return 0.0
        return 1.0 - self.non_negative_fold_fraction

    @property
    def median_test_return_pct(self) -> float:
        if not self.folds:
            return 0.0
        return median(fold.metrics.return_pct for fold in self.folds)

    @property
    def median_active_test_return_pct(self) -> float:
        active_folds = self.active_folds
        if not active_folds:
            return 0.0
        return median(fold.metrics.return_pct for fold in active_folds)

    @property
    def median_test_sharpe_15m(self) -> float:
        if not self.folds:
            return 0.0
        return median(fold.metrics.sharpe_15m for fold in self.folds)

    @property
    def worst_test_drawdown_pct(self) -> float:
        if not self.folds:
            return 0.0
        return max(fold.metrics.max_drawdown_pct for fold in self.folds)

    @property
    def average_risk_discipline_score(self) -> float:
        if not self.folds:
            return 0.0
        return sum(fold.risk_discipline.score for fold in self.folds) / len(self.folds)

    @property
    def total_evaluation_fills(self) -> int:
        return sum(len(fold.evaluation.fills) for fold in self.folds)

    @property
    def positive_return_sum_pct(self) -> float:
        return sum(max(fold.metrics.return_pct, 0.0) for fold in self.folds)

    @property
    def largest_positive_fold_return_pct(self) -> float:
        if not self.folds:
            return 0.0
        return max(max(fold.metrics.return_pct, 0.0) for fold in self.folds)

    @property
    def largest_positive_fold_contribution(self) -> float:
        positive_sum = self.positive_return_sum_pct
        if positive_sum <= 0:
            return 0.0
        return self.largest_positive_fold_return_pct / positive_sum


@dataclass(frozen=True)
class FixedWarmupPromotionDecision:
    status: str
    live_ready: bool
    reason: str


def decide_fixed_warmup_promotion(
    result: FixedWarmupPortfolioWalkForwardResult,
    *,
    min_positive_fold_fraction: float = 0.50,
    min_active_positive_fold_fraction: float = 0.50,
    min_non_negative_fold_fraction: float = 0.70,
    min_live_positive_fold_fraction: float = 0.67,
    min_live_active_positive_fold_fraction: float = 0.67,
    min_median_return_pct: float = 0.0,
    max_worst_drawdown_pct: float = 0.03,
    min_average_risk_discipline_score: float = 95.0,
    max_largest_positive_fold_contribution: float = 0.80,
) -> FixedWarmupPromotionDecision:
    if not result.folds:
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason="no fixed-warmup walk-forward folds were produced",
        )
    if not result.active_folds:
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason="strategy produced no active fixed-warmup evaluation folds",
        )
    if result.non_negative_fold_fraction < min_non_negative_fold_fraction:
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                f"non-negative fold fraction {result.non_negative_fold_fraction:.1%} "
                f"is below {min_non_negative_fold_fraction:.1%}"
            ),
        )
    if (
        result.positive_fold_fraction < min_positive_fold_fraction
        and result.active_positive_fold_fraction < min_active_positive_fold_fraction
    ):
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                "positive active fold fraction "
                f"{result.active_positive_fold_fraction:.1%} is below "
                f"{min_active_positive_fold_fraction:.1%}"
            ),
        )
    if result.median_active_test_return_pct <= min_median_return_pct:
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                "median active test return "
                f"{result.median_active_test_return_pct:.3%} is not above "
                f"{min_median_return_pct:.3%}"
            ),
        )
    if result.worst_test_drawdown_pct > max_worst_drawdown_pct:
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                f"worst test drawdown {result.worst_test_drawdown_pct:.3%} is above "
                f"{max_worst_drawdown_pct:.3%}"
            ),
        )
    if result.average_risk_discipline_score < min_average_risk_discipline_score:
        return FixedWarmupPromotionDecision(
            status="REJECT",
            live_ready=False,
            reason=(
                "average risk discipline "
                f"{result.average_risk_discipline_score:.1f}/100 is below "
                f"{min_average_risk_discipline_score:.1f}/100"
            ),
        )
    if result.largest_positive_fold_contribution > max_largest_positive_fold_contribution:
        return FixedWarmupPromotionDecision(
            status="PAPER_ONLY",
            live_ready=False,
            reason=(
                "largest positive fold contributes "
                f"{result.largest_positive_fold_contribution:.1%} of positive "
                "walk-forward return"
            ),
        )
    if (
        result.positive_fold_fraction < min_live_positive_fold_fraction
        or result.active_positive_fold_fraction < min_live_active_positive_fold_fraction
    ):
        return FixedWarmupPromotionDecision(
            status="PAPER_ONLY",
            live_ready=False,
            reason=(
                "selective walk-forward gates passed, but live promotion needs "
                f"{min_live_positive_fold_fraction:.1%} total positive folds and "
                f"{min_live_active_positive_fold_fraction:.1%} active positive folds"
            ),
        )
    return FixedWarmupPromotionDecision(
        status="PROMOTE",
        live_ready=True,
        reason="fixed-warmup return, drawdown, fold stability, and risk gates passed",
    )


def run_fixed_warmup_portfolio_walk_forward(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    strategy_name: str,
    symbols: tuple[str, ...],
    strategy_by_symbol: Mapping[str, str] | None = None,
    train_size: int = 960,
    test_size: int = 192,
    step_size: int = 192,
    regime_tilt_policy: RegimeTiltPolicy | None = None,
    session_gate_policy: SessionGatePolicy | None = None,
    volatility_targeting_policy: VolatilityTargetingPolicy | None = None,
    symbol_evidence_gate_policy: SymbolEvidenceGatePolicy | None = None,
    target_notional_multipliers_by_symbol: Mapping[str, float] | None = None,
) -> FixedWarmupPortfolioWalkForwardResult:
    _validate_window_sizes(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    if not symbols:
        raise ValueError("fixed warmup walk-forward requires at least one symbol")
    strategy_overrides = dict(strategy_by_symbol or {})

    timestamps = _common_timestamps(prices, quotes, symbols)
    if len(timestamps) < train_size + test_size:
        raise ValueError(
            "not enough aligned timestamps for one warmup walk-forward fold"
        )

    folds: list[FixedWarmupPortfolioFold] = []
    for fold_index, start in enumerate(
        range(0, len(timestamps) - train_size - test_size + 1, step_size),
        start=1,
    ):
        train_timestamps = timestamps[start : start + train_size]
        test_timestamps = timestamps[
            start + train_size : start + train_size + test_size
        ]
        combined_timestamps = train_timestamps + test_timestamps
        fold_prices = _slice_prices(
            prices,
            symbols=symbols,
            timestamps=combined_timestamps,
        )
        fold_quotes = _slice_quotes(
            quotes,
            symbols=symbols,
            timestamps=combined_timestamps,
        )
        engine = PortfolioBacktestEngine(
            strategies={
                symbol: config.build_strategy(
                    strategy_overrides.get(symbol, strategy_name),
                    symbol=symbol,
                )
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
            regime_tilt_policy=regime_tilt_policy,
            session_gate_policy=session_gate_policy,
            volatility_targeting_policy=volatility_targeting_policy,
            symbol_evidence_gate_policy=symbol_evidence_gate_policy,
            target_notional_multipliers_by_symbol=(
                target_notional_multipliers_by_symbol
            ),
        )
        full_result = engine.run(
            prices=fold_prices,
            quotes=fold_quotes,
            starting_equity=config.competition.starting_equity,
        )
        evaluation = evaluate_portfolio_after_warmup(
            full_result,
            evaluation_start=test_timestamps[0],
        )
        folds.append(
            FixedWarmupPortfolioFold(
                fold_index=fold_index,
                train_start=train_timestamps[0].isoformat(),
                train_end=train_timestamps[-1].isoformat(),
                test_start=test_timestamps[0].isoformat(),
                test_end=test_timestamps[-1].isoformat(),
                evaluation=evaluation,
                full_run_fills=len(full_result.fills),
            )
        )

    return FixedWarmupPortfolioWalkForwardResult(
        strategy_name=_strategy_display(strategy_name, strategy_overrides),
        symbols=symbols,
        folds=tuple(folds),
    )


def write_fixed_warmup_summary_csv(
    result: FixedWarmupPortfolioWalkForwardResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "symbols",
                "folds",
                "positive_fold_fraction",
                "active_fold_fraction",
                "active_positive_fold_fraction",
                "non_negative_fold_fraction",
                "losing_fold_fraction",
                "median_test_return_pct",
                "median_active_test_return_pct",
                "median_test_sharpe_15m",
                "worst_test_drawdown_pct",
                "average_risk_discipline_score",
                "total_evaluation_fills",
                "positive_return_sum_pct",
                "largest_positive_fold_return_pct",
                "largest_positive_fold_contribution",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": result.strategy_name,
                "symbols": " ".join(result.symbols),
                "folds": len(result.folds),
                "positive_fold_fraction": result.positive_fold_fraction,
                "active_fold_fraction": result.active_fold_fraction,
                "active_positive_fold_fraction": (
                    result.active_positive_fold_fraction
                ),
                "non_negative_fold_fraction": result.non_negative_fold_fraction,
                "losing_fold_fraction": result.losing_fold_fraction,
                "median_test_return_pct": result.median_test_return_pct,
                "median_active_test_return_pct": (
                    result.median_active_test_return_pct
                ),
                "median_test_sharpe_15m": result.median_test_sharpe_15m,
                "worst_test_drawdown_pct": result.worst_test_drawdown_pct,
                "average_risk_discipline_score": result.average_risk_discipline_score,
                "total_evaluation_fills": result.total_evaluation_fills,
                "positive_return_sum_pct": result.positive_return_sum_pct,
                "largest_positive_fold_return_pct": (
                    result.largest_positive_fold_return_pct
                ),
                "largest_positive_fold_contribution": (
                    result.largest_positive_fold_contribution
                ),
            }
        )


def write_fixed_warmup_folds_csv(
    result: FixedWarmupPortfolioWalkForwardResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "return_pct",
                "max_drawdown_pct",
                "sharpe_15m",
                "risk_discipline_score",
                "evaluation_fills",
                "full_run_fills",
                "final_equity",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            metrics = fold.metrics
            writer.writerow(
                {
                    "fold": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "return_pct": metrics.return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_15m": metrics.sharpe_15m,
                    "risk_discipline_score": fold.risk_discipline.score,
                    "evaluation_fills": len(fold.evaluation.fills),
                    "full_run_fills": fold.full_run_fills,
                    "final_equity": metrics.final_equity,
                }
            )


def _validate_window_sizes(*, train_size: int, test_size: int, step_size: int) -> None:
    if train_size < 1:
        raise ValueError("train_size must be at least 1")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")


def _strategy_display(
    fallback_strategy: str,
    strategy_by_symbol: Mapping[str, str],
) -> str:
    if not strategy_by_symbol:
        return fallback_strategy
    overrides = ", ".join(
        f"{symbol}={strategy}" for symbol, strategy in sorted(strategy_by_symbol.items())
    )
    return f"{fallback_strategy} with overrides ({overrides})"


def _common_timestamps(
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...],
) -> tuple[datetime, ...]:
    common: set[datetime] | None = None
    for symbol in symbols:
        price_timestamps = {bar.timestamp for bar in prices.for_symbol(symbol).bars}
        quote_timestamps = {quote.timestamp for quote in quotes.for_symbol(symbol).quotes}
        timestamps = price_timestamps & quote_timestamps
        if not timestamps:
            raise ValueError(f"no aligned price/quote timestamps for {symbol}")
        common = timestamps if common is None else common & timestamps
    if not common:
        raise ValueError("no common timestamps across selected symbols")
    return tuple(sorted(common))


def _slice_prices(
    prices: PriceHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> PriceHistory:
    timestamp_set = set(timestamps)
    return PriceHistory(
        tuple(
            PriceBar(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                close=bar.close,
            )
            for bar in prices.bars
            if bar.symbol in symbols and bar.timestamp in timestamp_set
        )
    )


def _slice_quotes(
    quotes: QuoteHistory,
    *,
    symbols: tuple[str, ...],
    timestamps: tuple[datetime, ...],
) -> QuoteHistory:
    timestamp_set = set(timestamps)
    return QuoteHistory(
        tuple(
            QuoteSnapshot(
                timestamp=quote.timestamp,
                symbol=quote.symbol,
                bid=quote.bid,
                ask=quote.ask,
            )
            for quote in quotes.quotes
            if quote.symbol in symbols and quote.timestamp in timestamp_set
        )
    )
